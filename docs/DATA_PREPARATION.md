# Data Preparation

The evaluation scripts expect the standard directory layouts below. Datasets and
model weights are **not** distributed with this repository; obtain them from
their official sources and adjust the paths via the CLI arguments.

---

## Model weights

The base video LLM is `lmms-lab/LLaVA-Video-7B-Qwen2` (a LLaVA-NeXT checkpoint).
Download it into your Hugging Face cache:

```bash
huggingface-cli download lmms-lab/LLaVA-Video-7B-Qwen2 --local-dir /path/to/models/LLaVA-Video-7B-Qwen2
```

The evaluation scripts load it by its Hugging Face id
(`lmms-lab/LLaVA-Video-7B-Qwen2`), so either pre-populate the cache or point
`HF_HOME` at your model directory. `run_balanced_dev.py` sets
`HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` via `setdefault`, so the weights
must already be cached.

---

## MVBench

```
MVBench/
├── json/                       # annotation files: <task>.json
└── video/                      # videos/frames, one directory per task source
    ├── star/Charades_v1_480/...
    ├── ssv2_video/...
    ├── perception/videos/...
    ├── clevrer/video_validation/...
    └── ...                     # see task map in mvbench_strict.py
```

- Source: <https://github.com/OpenGVLab/Ask-Anything> (MVBench).
- `mvbench_strict.py build-manifest --dataset /path/to/MVBench --output ...`
  resolves each item's video via the task-to-directory map and writes a frozen
  `manifest.jsonl`.

---

## VideoMME

```
VideoMME/
├── question.json                       # VideoMME question/annotation file
└── 16frames/
    └── 16_frames/
        └── frames/
            └── <videoID>/              # one directory of frame images per video
                ├── 000000.jpg
                └── ...
```

- Source: <https://github.com/BradyFU/Video-MME>.
- `videomme_strict.py --dataset /path/to/VideoMME` reads `question.json` and the
  pre-extracted 16-frame directories.

---

## SurveillanceVQA

```
SurveillanceVQA/                        # --bench
└── test_datasets/
    ├── <category>/
    │   └── <stem>.json                 # each contains the *_qa_pairs fields
    └── ...

SurveillanceVQA_media/                  # --media
├── UCF/videos_cut/...
├── MSAD/videos_cut/...
├── MEVA/videos_cut/...
└── NWPU/videos_cut/...
```

- The four QA families are `detection_qa_pairs`, `classification_qa_pairs`,
  `subject_qa_pairs`, `description_qa_pairs`, `cause_qa_pairs`,
  `result_qa_pairs`.
- `surveillancevqa_smoke.py` indexes media by stem and loads the first
  `--videos` videos that have at least one QA pair.

---

## Compressor training data

`train_latent_compressor.py` consumes a **train manifest** (`--train-manifest`,
a `manifest.jsonl` produced by `mvbench_strict.py build-manifest` or
`run_balanced_dev.py`) and a **feature cache** (`--feature-cache`), which stores
the pooled `[16, N, D]` visual features as `<task>__<stem>.pt` per item. The
cache is produced by any of the feature-extraction scripts (e.g. the manifest
builders that write `.pt` feature files), so training never re-runs the frozen
vision tower.
