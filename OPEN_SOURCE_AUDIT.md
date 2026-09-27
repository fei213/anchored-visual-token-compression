# Open-Source Audit

This document records what was included in, and excluded from, this public
release, and the checks performed to make the repository safe to publish.

**Scope**: this repository (`anchored-visual-token-compression`) is a fresh,
self-contained overlay for [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT).
It was built from a clean copy of the source tree, **not** from the original
development repository's git history.

---

## 1. Why the original git history is not published

The original development repository's `.git` is **not** carried over. Reasons:

- **Size**: the `.git` directory is ~460 MB across ~421 commits.
- **Large historical blobs**: the largest historical object is an ~84 MB
  `results.jsonl` (experiment output) that should never be committed, let alone
  published.
- **Leaked credential**: an early commit (`c35de14`) tracks
  `scripts/archived/entry_cmd.sh`, which contains a Hugging Face token
  (`HF_TOKEN=…`) and a private network-storage path.
  That token has been flagged for revocation. A fresh history is the only way to
  guarantee the token does not ship.

This repository is initialized as a brand-new git repository containing only the
files listed below.

---

## 2. Included (core implementation + evaluation)

| Path | Purpose |
|------|---------|
| `llava/compression/learned.py` | `LatentCompressor`, `ResidualAnchoredCompressor` (the paper's method) |
| `llava/compression/selectors.py` | heuristic baselines (`uniform`, `random`, `kmeans_medoid`, `rare_only`, `temporal_only`, `st_rare`) |
| `llava/compression/packing.py` | repack selected/compressed tokens into the LLM sequence |
| `llava/compression/spatial.py`, `temporal.py`, `budget.py`, `config.py` | supporting modules |
| `llava/compression/integration.py`, `hook.py` | optional wrapper for the early `llava_arch.py` integration |
| `llava/model/kmeans.py` | spherical k-means medoid (`kmeans_medoid` baseline) |
| `benchmark/st_rare/*` | controlled protocol, training, evaluation, judge, bootstrap, tables |

## 3. Excluded (datasets, weights, caches, private/dev-only)

- **Datasets** (MVBench, VideoMME, SurveillanceVQA) — never committed.
- **Model weights / checkpoints** (`*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`,
  `falling_features.pt`, etc.).
- **Caches / run outputs** (`artifacts/`, `*.jsonl`, `*.npy`, `*.npz`, `wandb/`).
- **Vendored third-party code** that shadows pip packages (`transformers/`,
  `trl/`).
- **Exploratory selectors** not used by the paper:
  `rare_refill_selector`, `frame_merge`, `residual_keep_selector`,
  `residual_merge`, `context_tube_selector`, `scaffold_rare_selector`,
  `st_rare_global_selector`, `gstep_vendor`, `dynamic_temporal_selector`,
  `coverage_rare_selector`, `local_rare_selector`, and
  `llava/compression/temporal_v2.py` / `temporal_v3.py` (no longer referenced).
- **Private / env-dependent files**: `llava/stc_utils.py` (loads `.env`),
  `llava/model/fusion.py` (depends on `stc_utils`), and the internal
  `Internal-EVA02-CLIP-*.json` configs.
- **Playground / scripts / training configs** (`playground/`, `scripts/`,
  `configs/`, `tests/`) — development-only and not part of the paper's
  reproduction path.

## 4. Sensitive-information removal

| Item | Original location | Action |
|------|-------------------|--------|
| ZhipuAI API key | `benchmark/st_rare/surveillancevqa_judge.py` (a machine-local key file) | replaced with `--api-key` arg + `ZHIPUAI_API_KEY` env var |
| ZhipuAI API key | `benchmark/st_rare/surveillancevqa_smoke.py` (a machine-local key file) | replaced with `--api-key` arg + `ZHIPUAI_API_KEY` env var |
| Hugging Face token | `scripts/archived/entry_cmd.sh` | file excluded (whole `scripts/` tree); token flagged for revocation |
| Private network-storage path | `scripts/archived/entry_cmd.sh` | file excluded |

A scan of the published tree for credential patterns (`sk-…`, `hf_…`, `ghp_…`,
`AIza…`, inline `api_key =`) finds **no** hard-coded secrets. The only `api_key`
references are function/CLI parameters populated from the environment variable.

## 5. Machine-path parameterization

All machine-local absolute paths were removed.
Specific changes:

| Original | Now |
|----------|-----|
| `videomme_strict.py` `QUESTION_JSON` / `FRAMES_ROOT` (a machine-local benchmark path) | deleted (dead constants); `--dataset` is a required arg, paths derived from it |
| `videomme_strict.py` `--dataset` default is machine-local | `--dataset` required |
| `surveillancevqa_smoke.py` `BENCH` / `MEDIA` (machine-local paths) | `--bench` / `--media` required args |
| `surveillancevqa_smoke.py` checkpoint paths | `--anchor-ckpt` / `--div-ckpt` required args |
| `run_experiments.py` video paths | file removed (superseded by `video_utils.load_video`) |

A final grep of the published *code* for machine-local absolute paths returns
**0** matches. The path descriptions that remain in this audit document name no
concrete machine; they state only what was removed.

## 6. Verification

- `python -m py_compile` passes for every `.py` file in the tree.
- `import llava.compression` succeeds standalone (no dependency on the upstream
  `llava` package's model/builder modules).
- No residual references to any excluded module remain (verified by grep).
- `OPEN_SOURCE_AUDIT.md` (this file) records the audit for reviewers.

---

*Prepared for the public release of the ST-Rare / Anchored Visual Token
Compression implementation.*
