# Reproduction Guide

Steps to reproduce the four configurations (Full / Uniform / Anchor /
Anchor+Div) end to end. All commands are run from the repository root.

## 0. Environment

Follow `README.md` "Installation" to (1) install upstream LLaVA-NeXT, (2) install
this repo's `requirements.txt`, and (3) symlink `llava/compression` and
`llava/model/kmeans.py` into the upstream checkout. Verify:

```bash
python -c "import llava.compression.learned, benchmark.st_rare.strict_video_protocol; print('ok')"
```

`benchmark.st_rare.strict_video_protocol` imports the upstream `llava` package
lazily, so this also verifies the overlay.

## 1. Build a frozen manifest

Every experiment starts from a **manifest** (a `manifest.jsonl` plus a
`protocol.json` and `COMPLETE` marker). For MVBench:

```bash
python -m benchmark.st_rare.mvbench_strict build-manifest \
    --dataset /path/to/MVBench \
    --output artifacts/mvbench_manifest
```

The manifest freezes the item list, so different methods are always evaluated on
the **same** items (the matched-budget protocol).

## 2. Train the learned compressors

```bash
# Anchor: residual around the Uniform anchors, no diversity regularizer
python -m benchmark.st_rare.train_latent_compressor \
    --train-manifest artifacts/mvbench_manifest/manifest.jsonl \
    --feature-cache artifacts/latent_features \
    --anchor --div-lambda 0.0 \
    --output artifacts/latent_compressor_anchor

# Anchor+Div: residual + diversity regularizer
python -m benchmark.st_rare.train_latent_compressor \
    --train-manifest artifacts/mvbench_manifest/manifest.jsonl \
    --feature-cache artifacts/latent_features \
    --anchor --div-lambda <lambda> \
    --output artifacts/latent_compressor_anchor_div
```

Outputs: `compressor_epoch<e>.pt` state dicts (a few MB). At `--div-lambda 0.0`
the compressor is the **Anchor** variant; a positive lambda adds the diversity
regularizer to produce **Anchor+Div**. The module's scalar `alpha` starts at 0,
so the untrained compressor is exactly the Uniform baseline.

## 3. Evaluate a compressor vs Uniform (matched items)

```bash
python -m benchmark.st_rare.eval_latent_compressor \
    --checkpoint artifacts/latent_compressor_anchor_div/compressor_epoch2.pt \
    --anchor \
    --manifest artifacts/mvbench_manifest \
    --output artifacts/eval_anchor_div \
    --methods uniform learned
```

This reuses `mvbench_strict.run` (the identical generation/scoring path every
baseline uses) and only swaps the selector, so a `learned` vs `uniform` gap
cannot come from scorer or prompt drift.

## 4. End-to-end benchmark evaluation

```bash
# MVBench
python -m benchmark.st_rare.mvbench_strict run \
    --manifest artifacts/mvbench_manifest \
    --output artifacts/mvbench_results \
    --methods controlled_full uniform learned_anchor learned_div

# VideoMME
python -m benchmark.st_rare.videomme_strict \
    --dataset /path/to/VideoMME \
    --output artifacts/videomme_results \
    --methods controlled_full uniform
```

> The learned methods (`learned_anchor` / `learned_div`) must be registered with
> `register_learned(...)` before they can be selected. The eval / efficiency /
> rate-sweep / SurveillanceVQA scripts do this registration internally; when
> calling `mvbench_strict run` directly, register them first (see
> `eval_latent_compressor.py` for the pattern).

## 5. SurveillanceVQA (4-dim judge protocol)

```bash
export ZHIPUAI_API_KEY=<your key>

# generate predictions (Full / Uniform / Anchor / Anchor+Div on the same items)
python -m benchmark.st_rare.surveillancevqa_smoke \
    --bench /path/to/SurveillanceVQA \
    --media /path/to/SurveillanceVQA_media \
    --anchor-ckpt artifacts/latent_compressor_anchor/compressor_epoch2.pt \
    --div-ckpt artifacts/latent_compressor_anchor_div/compressor_epoch2.pt

# judge (CI / DO / CU / TU, equal-weight mean, multithreaded, resumable)
python -m benchmark.st_rare.surveillancevqa_judge \
    --preds artifacts/surveillancevqa/smoke.jsonl \
    --output artifacts/surveillancevqa/judged.jsonl

# aggregate + paired bootstrap
python -m benchmark.st_rare.surveillancevqa_aggregate \
    --judged artifacts/surveillancevqa/judged.jsonl
```

The judge prompts are the official SurveillanceVQA prompts, reproduced verbatim
in `surveillancevqa_judge_prompts.json`.

## 6. Efficiency and rate-distortion

```bash
# Efficiency: timing + peak memory for the four methods
python -m benchmark.st_rare.efficiency_profile \
    --manifest artifacts/mvbench_manifest \
    --primary-ckpt artifacts/latent_compressor_anchor_div/compressor_epoch2.pt \
    --secondary-ckpt artifacts/latent_compressor_anchor/compressor_epoch2.pt \
    --output artifacts/efficiency

# Rate-distortion: same checkpoint, switched ratio at deploy time (no retrain)
python -m benchmark.st_rare.run_ratesweep \
    --primary-ckpt artifacts/latent_compressor_anchor_div/compressor_epoch2.pt \
    --secondary-ckpt artifacts/latent_compressor_anchor/compressor_epoch2.pt \
    --manifest artifacts/mvbench_manifest \
    --out-root artifacts/ratesweep
```

## 7. Result tables + paired bootstrap

```bash
# Paired bootstrap for one method vs a reference (matched items)
python -m benchmark.st_rare.paired_bootstrap \
    --results artifacts/mvbench_results/results.jsonl \
    --reference uniform --n-boot 10000

# Markdown + CSV tables with per-method accuracy, CI, category breakdown
python -m benchmark.st_rare.final_tables \
    --results artifacts/mvbench_results/results.jsonl \
    --reference uniform \
    --proposed learned_div --secondary learned_anchor
```

The paired bootstrap resamples matched items with replacement, reports the mean
accuracy delta with a 95% percentile CI and a one-sided `P(delta <= 0)`.

---

## Reproducibility notes

- **Matched budget**: all methods at a ratio share the exact token budget; the
  protocol asserts the packed sequence length equals the target for every method.
- **Same items**: manifests freeze the item order and content, so every method is
  scored on the identical set.
- **Frozen backbone**: the vision tower / projector / LLM are never fine-tuned;
  only the small compressor (`LatentCompressor` inside `ResidualAnchoredCompressor`)
  is trained, in `float32`, then cast back to the backbone dtype at the boundary.
- **Deploy-time rate switch**: the rate-distortion sweep evaluates the *same*
  25%-trained checkpoint at 25% / 12.5% / 6.25%; nothing is retrained.
