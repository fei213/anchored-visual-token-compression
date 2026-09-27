"""VideoMME inference under the frozen 16-frame protocol (videomme_config.md).

Frozen evaluation config (see experiments/videomme_config.md):
  - 16 frames sampled uniformly from 16frames/16_frames/frames/<videoID>/
  - multiple-choice, `Answer with only the option letter.`
  - native exact-match accuracy on the option letter
  - reported as Overall + Short / Medium / Long

Matched evaluation: each unique video's vision features are extracted ONCE and
reused for all of that video's questions (VideoMME has 900 videos / 2700 QA).
No hyperparameter search; the method config is frozen on MVBench first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path = [p for p in sys.path if Path(p or ".").resolve() != ROOT]
sys.path.append(str(ROOT))

from benchmark.st_rare.run_balanced_dev import load_pretrained_model
from benchmark.st_rare.strict_video_protocol import generate_from_visual, select_visual
from benchmark.st_rare.mvbench_strict import parse_choice, LETTERS
from benchmark.st_rare.paired_bootstrap import load_rows
from llava.mm_utils import process_images

DURATIONS = ["short", "medium", "long"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_items(dataset: Path) -> list[dict]:
    """One row per question, ordered by video for matched feature reuse."""
    rows = json.loads((dataset / "question.json").read_text())
    items = []
    for r in rows:
        if not isinstance(r.get("options"), list) or r["answer"] not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            raise ValueError(f"invalid VideoMME item: {r.get('question_id')}")
        items.append(dict(video_id=r["videoID"], question_id=r["question_id"],
                          duration=r["duration"], domain=r["domain"], sub_category=r["sub_category"],
                          task_type=r["task_type"], question=r["question"],
                          options=r["options"], answer=r["answer"]))
    items.sort(key=lambda r: (r["video_id"], r["question_id"]))
    return items


def prompt(row: dict) -> str:
    # VideoMME options already carry their own "A. " / "B. " prefixes; do not add
    # a second letter. Guard that the prefix order really matches LETTERS.
    for i, value in enumerate(row["options"]):
        if not value.startswith(f"{LETTERS[i]}.") and not value.startswith(f"{LETTERS[i]} ."):
            raise ValueError(f"option {i} of {row['question_id']} lacks its letter prefix: {value!r}")
    options = "\n".join(row["options"])
    return f"{row['question']}\nOptions:\n{options}\nAnswer with only the option letter."


def load_videomme_frames(dataset: Path, video_id: str, frames: int = 16) -> list[Image.Image]:
    d = dataset / "16frames" / "16_frames" / "frames" / video_id
    paths = sorted(p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        raise FileNotFoundError(f"no frames for {video_id}")
    idx = torch.linspace(0, len(paths) - 1, frames, dtype=torch.int64).tolist()
    return [Image.open(paths[i]).convert("RGB") for i in idx]


def summarize_runs(rows: list[dict], methods: list[str]) -> dict:
    """Overall and per-duration accuracy for each method.

    A method with no rows is reported with samples=0 / accuracy=None instead of
    raising ZeroDivisionError, so an aborted run still yields a usable summary.
    """
    summary = {}
    for method in methods:
        values = [r for r in rows if r["method"] == method]
        if not values:
            summary[method] = dict(samples=0, accuracy=None, valid_rate=None,
                                   by_duration={d: None for d in DURATIONS})
            continue
        by_dur = {d: (sum(r["correct"] for r in values if r["duration"] == d) /
                      max(1, sum(r["duration"] == d for r in values))) for d in DURATIONS}
        summary[method] = dict(samples=len(values),
                               accuracy=sum(r["correct"] for r in values) / len(values),
                               valid_rate=sum(r["success"] for r in values) / len(values),
                               by_duration=by_dur)
    return summary


def run(dataset: Path, output: Path, methods: list[str], ratio: float, limit: int | None, max_new_tokens: int) -> None:
    if output.exists():
        raise FileExistsError(output)
    items = build_items(dataset)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        items = items[:limit]
    output.mkdir(parents=True)

    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    tokenizer, model, processor, _ = load_pretrained_model(
        "lmms-lab/LLaVA-Video-7B-Qwen2", None, "llava_qwen", device_map="auto", attn_implementation="sdpa")
    model.eval()
    model.config.image_aspect_ratio = "square"
    model.config.image_grid_pinpoints = None

    protocol = dict(version="videomme_strict_inference_v1", dataset=str(dataset),
                    question_sha256=sha256(dataset / "question.json"),
                    methods=methods, samples=len(items), frames=16, ratio=ratio,
                    max_new_tokens=max_new_tokens, seed=0, smoke_limit=limit,
                    scoring="native exact-match on option letter; letter parsed first, then unique option-text fallback",
                    protocol_name="controlled_frame_newline_v1", code_sha256=sha256(Path(__file__)))
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n")

    results = output / "results.jsonl"
    vision_seconds = 0.0
    cache_id = None
    with torch.inference_mode(), results.open("w", encoding="utf-8") as handle:
        for number, row in enumerate(items, start=1):
            started = time.monotonic()
            if row["video_id"] != cache_id:
                frames = load_videomme_frames(dataset, row["video_id"], 16)
                pixels = process_images(frames, processor, model.config)
                if isinstance(pixels, list):
                    pixels = torch.stack(pixels)
                tower, projector = model.get_vision_tower(), model.get_model().mm_projector
                pixels = pixels.to(model.device, dtype=next(tower.parameters()).dtype)
                vision = tower(pixels)
                visual_features = model.get_2dPool(projector(vision.to(next(projector.parameters()).dtype)))
                cache_id = row["video_id"]
                vision_seconds = time.monotonic() - started
            for method in methods:
                t0 = time.monotonic()
                visual, metadata = select_visual(visual_features, model.model.image_newline, method, ratio)
                selection_seconds = time.monotonic() - t0
                t0 = time.monotonic()
                answer, counts = generate_from_visual(model, tokenizer, prompt(row), visual, max_tokens=max_new_tokens)
                index = parse_choice(answer, row["options"])
                predicted = LETTERS[index] if index is not None else None
                result = {**row, **counts, **metadata, "method": method, "prediction": answer,
                          "predicted_letter": predicted, "correct": (predicted == row["answer"]),
                          "success": (predicted is not None),
                          "selection_seconds": selection_seconds, "generation_seconds": time.monotonic() - t0,
                          "vision_seconds": vision_seconds}
                handle.write(json.dumps(result, ensure_ascii=False) + "\n"); handle.flush()
            print(f"DONE sample={number}/{len(items)} video={row['video_id']}", flush=True)

    all_rows, skipped = load_rows(results)
    if skipped:
        print(f"警告: 跳过 {skipped} 行无法解析（疑似中断写入的尾部）", flush=True)
    summary = summarize_runs(all_rows, methods)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (output / "COMPLETE").write_text(str(len(all_rows)) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True,
                        help="VideoMME dataset root (contains question.json and 16frames/)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=["controlled_full", "uniform"])
    parser.add_argument("--ratio", type=float, default=0.25)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    run(args.dataset, args.output, args.methods, args.ratio, args.limit, args.max_new_tokens)


if __name__ == "__main__":
    main()
