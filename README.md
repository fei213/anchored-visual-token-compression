# Anchored Visual Token Compression for Efficient Video Large Multimodal Models

Official implementation of **Anchored Visual Token Compression** (ST-Rare), a
learned visual-token compressor for video LLMs that keeps the temporal/visual
budget low while preserving generation quality.

The method replaces the heuristic token *selection* of prior work with a
learned *representation* compressor:

$$Z \;=\; A \;+\; \alpha \cdot C_{\theta}(X)$$

where $X \in \mathbb{R}^{T \times N \times D}$ are the pooled visual tokens,
$A$ are the **Uniform anchors** (a selection known to work), $C_\theta$ is a
lightweight cross-attention compressor over a bank of learnable queries, and
$\alpha$ is a learned scalar initialized at $0$. At $\alpha = 0$ the compressor
*is exactly* the Uniform baseline, so training can only improve on a safe
starting point rather than falling off a cliff. A diversity regularizer keeps
the compressed tokens from collapsing to rank $\sim 1$.

Two learned variants are released: **Anchor** ($\alpha$ learned, no regularizer)
and **Anchor+Div** ($\alpha$ learned + diversity regularizer). Both are evaluated
against **Full** (uncompressed) and **Uniform** under a matched-budget protocol
on MVBench, VideoMME, and SurveillanceVQA.

---

## Repository layout

```
anchored-visual-token-compression/
├── llava/
│   ├── compression/            # core implementation (the plugin)
│   │   ├── learned.py          #   LatentCompressor + ResidualAnchoredCompressor
│   │   ├── selectors.py        #   heuristic baselines (uniform/random/st_rare/...)
│   │   ├── packing.py          #   pack selected tokens back into the LLM sequence
│   │   ├── spatial.py          #   spatial rarity
│   │   ├── temporal.py         #   temporal novelty
│   │   ├── budget.py           #   token-budget accounting
│   │   ├── config.py           #   config load/validate
│   │   ├── integration.py      #   optional wrapper (see "Early integration hook" below)
│   │   └── hook.py             #   convenience entry points
│   └── model/
│       └── kmeans.py           # spherical k-means medoid (kmeans_medoid baseline)
├── benchmark/st_rare/
│   ├── strict_video_protocol.py        # controlled single-video path (select/generate)
│   ├── train_latent_compressor.py      # train Anchor / Anchor+Div
│   ├── eval_latent_compressor.py       # evaluate a trained compressor vs Uniform
│   ├── efficiency_profile.py           # Full/Uniform/Anchor/Anchor+Div timing + memory
│   ├── run_ratesweep.py                # zero-shot rate-distortion sweep
│   ├── mvbench_strict.py               # MVBench evaluation
│   ├── videomme_strict.py              # VideoMME evaluation
│   ├── surveillancevqa_smoke.py        # SurveillanceVQA generation
│   ├── surveillancevqa_judge.py        # 4-dim judge scoring (multithreaded)
│   ├── surveillancevqa_aggregate.py    # judge-score aggregation
│   ├── surveillancevqa_judge_prompts.json  # official judge prompts (verbatim)
│   ├── paired_bootstrap.py             # paired bootstrap (matched items)
│   ├── final_tables.py                 # result tables + CIs
│   ├── task_categories.py              # MVBench task categories
│   └── video_utils.py                  # video frame loading
├── docs/
│   ├── REPRODUCTION.md
│   └── DATA_PREPARATION.md
├── OPEN_SOURCE_AUDIT.md
└── requirements.txt
```

This repository is a **plugin / overlay** for
[LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) (the code behind
`lmms-lab/LLaVA-Video-7B-Qwen2`). It contains only our compression module and
evaluation scripts; the upstream `llava` package (model builder, image
processing, conversation templates, constants) is **not** vendored here.

---

## Installation

### 1. Install upstream LLaVA-NeXT

```bash
git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git
cd LLaVA-NeXT
pip install -e .
cd ..
```

This provides `llava.model.builder`, `llava.mm_utils`, `llava.conversation`,
`llava.constants`, and the `llava_qwen` model class used by the evaluation
scripts.

### 2. Install this repository

```bash
git clone https://github.com/<your-account>/anchored-visual-token-compression.git
cd anchored-visual-token-compression
pip install -r requirements.txt
```

### 3. Overlay our module onto the upstream `llava` package

Our `llava.compression` and `llava.model.kmeans` need to be importable from the
same `llava` namespace that the upstream package provides. Symlink them into the
upstream checkout (kept in sync automatically):

```bash
LLAVA_NEXT=/path/to/LLaVA-NeXT
ln -s "$(pwd)/llava/compression" "$LLAVA_NEXT/llava/compression"
mkdir -p "$LLAVA_NEXT/llava/model"
ln -s "$(pwd)/llava/model/kmeans.py" "$LLAVA_NEXT/llava/model/kmeans.py"
```

The evaluation scripts also add this repository's root to `sys.path`, so the
`benchmark.st_rare` package resolves directly.

> **Why an overlay?** The four configurations are evaluated through a
> *controlled* path (`benchmark/st_rare/strict_video_protocol.py`) that calls the
> upstream vision tower / projector / 2D-pool and then injects the compressed
> visual tokens straight into the language model's input embeddings. It therefore
> needs no patch to the upstream `llava_arch.py`; only the standard upstream APIs
> listed above.

---

## Reproducing the four configurations

All four methods share the same frozen 16-frame, matched-budget protocol. The
method is a string passed to `select_visual`:

| Method      | `method` name      | Implemented by |
|-------------|--------------------|----------------|
| Full        | `controlled_full`  | protocol (no compression) |
| Uniform     | `uniform`          | `llava.compression.selectors` |
| Anchor      | `learned_anchor`   | `ResidualAnchoredCompressor` (registered) |
| Anchor+Div  | `learned_div`      | `ResidualAnchoredCompressor` (registered, div-trained) |

### Train the learned compressors

```bash
# Anchor (no diversity regularizer)
python -m benchmark.st_rare.train_latent_compressor \
    --out-root artifacts/st_rare/latent_compressor_anchor \
    --manifest /path/to/train_manifest \
    --diversity-weight 0.0

# Anchor+Div (diversity regularizer)
python -m benchmark.st_rare.train_latent_compressor \
    --out-root artifacts/st_rare/latent_compressor_anchor_div \
    --manifest /path/to/train_manifest \
    --diversity-weight <lambda>
```

The checkpoints are small `state_dict` files (a few MB); they are not shipped
with this repo.

### Evaluate a trained compressor vs Uniform

```bash
python -m benchmark.st_rare.eval_latent_compressor \
    --checkpoint artifacts/st_rare/latent_compressor_anchor_div/compressor_epoch2.pt \
    --anchor \
    --manifest /path/to/eval_manifest \
    --output artifacts/st_rare/eval_anchor_div \
    --methods uniform learned
```

### End-to-end benchmark evaluation

```bash
# MVBench (16-frame multiple choice)
python -m benchmark.st_rare.mvbench_strict \
    --dataset /path/to/MVBench --output artifacts/mvbench

# VideoMME (16-frame multiple choice)
python -m benchmark.st_rare.videomme_strict \
    --dataset /path/to/VideoMME --output artifacts/videomme

# Efficiency (Full / Uniform / Anchor / Anchor+Div timing + peak memory)
python -m benchmark.st_rare.efficiency_profile \
    --primary-ckpt artifacts/.../anchor_div.pt \
    --secondary-ckpt artifacts/.../anchor.pt \
    --manifest /path/to/manifest --out-root artifacts/efficiency

# Rate-distortion (same checkpoint, switched ratio at deploy time)
python -m benchmark.st_rare.run_ratesweep \
    --primary-ckpt artifacts/.../anchor_div.pt \
    --secondary-ckpt artifacts/.../anchor.pt \
    --manifest /path/to/manifest --out-root artifacts/ratesweep
```

### SurveillanceVQA (4-dim judge protocol)

```bash
# 1. Generate predictions (smoke test plumbing first)
export ZHIPUAI_API_KEY=<your key>
python -m benchmark.st_rare.surveillancevqa_smoke \
    --bench /path/to/SurveillanceVQA \
    --media /path/to/SurveillanceVQA_videos \
    --anchor-ckpt artifacts/.../anchor.pt \
    --div-ckpt artifacts/.../anchor_div.pt

# 2. Judge predictions (CI / DO / CU / TU, equal-weight mean, multithreaded)
python -m benchmark.st_rare.surveillancevqa_judge \
    --preds artifacts/.../predictions.jsonl \
    --output artifacts/.../judged.jsonl \
    --api-key "$ZHIPUAI_API_KEY"

# 3. Aggregate + paired bootstrap
python -m benchmark.st_rare.surveillancevqa_aggregate --judged artifacts/.../judged.jsonl
```

The judge prompts in `surveillancevqa_judge_prompts.json` are the official
SurveillanceVQA prompts, reproduced verbatim; the four dimension scores are
combined with an unweighted mean (matching the official paper's Avg).

---

## Notes

- **Early integration hook**: `llava/compression/integration.py` and `hook.py`
  provide an optional wrapper that could be called from `llava_arch.py`'s
  `prepare_inputs_labels_for_multimodal` (the original ST-Rare integration).
  The results in the paper use the controlled protocol instead, which does not
  require this hook.
- **Offline mode**: `run_balanced_dev.py` sets `HF_HUB_OFFLINE` /
  `TRANSFORMERS_OFFLINE` (via `setdefault`, so you can override). Model weights
  must be pre-downloaded to the Hugging Face cache.

---

## Citation

```bibtex
@article{qiao2025anchored,
  title   = {Anchored Visual Token Compression for Efficient Video Large Multimodal Models},
  author  = {Pengfei Qiao and Tongtong Yuan},
  journal = {arXiv preprint},
  year    = {2025}
}
```

## License

[Apache License 2.0](LICENSE).
