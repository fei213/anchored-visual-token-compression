"""Reproducible MVBench inference under the controlled 16-frame protocol.

This runner deliberately uses MVBench's native multiple-choice accuracy.  The
manifest is frozen before model loading; use --limit only for a smoke test, not
for reporting benchmark results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path = [p for p in sys.path if Path(p or ".").resolve() != ROOT]
sys.path.append(str(ROOT))

from benchmark.st_rare.run_balanced_dev import load_pretrained_model
from benchmark.st_rare.strict_video_protocol import generate_from_visual, select_visual
from llava.mm_utils import process_images
from decord import VideoReader, cpu
from PIL import Image

DEFAULT_METHODS = ["controlled_full", "uniform", "rare_only", "temporal_only", "st_rare", "dynamic_temporal"]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
TASK_SPECS = {
    "action_sequence": ("star/Charades_v1_480", "video"), "action_prediction": ("star/Charades_v1_480", "video"),
    "action_antonym": ("ssv2_video", "video"), "fine_grained_action": ("Moments_in_Time_Raw/videos", "video"),
    "unexpected_action": ("FunQA_test/test", "video"), "object_existence": ("clevrer/video_validation", "video"),
    "object_interaction": ("star/Charades_v1_480", "video"), "object_shuffle": ("perception/videos", "video"),
    "moving_direction": ("clevrer/video_validation", "video"), "action_localization": ("sta/sta_video", "video"),
    "scene_transition": ("scene_qa/video", "video"), "action_count": ("perception/videos", "video"),
    "moving_count": ("clevrer/video_validation", "video"), "moving_attribute": ("clevrer/video_validation", "video"),
    "state_change": ("perception/videos", "video"), "fine_grained_pose": ("nturgbd", "video"),
    "character_order": ("perception/videos", "video"), "egocentric_navigation": ("vlnqa", "video"),
    "episodic_reasoning": ("tvqa/frames_fps3_hq", "frames"), "counterfactual_inference": ("clevrer/video_validation", "video"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_video(root: Path, task: str, value: str) -> tuple[Path, str]:
    if task not in TASK_SPECS:
        raise ValueError(f"No official media mapping for {task}")
    subdir, media_kind = TASK_SPECS[task]
    # The official layout is video/<source>/..., but the distributed archive
    # places a subset (including STAR) below video/data0613/<source>/....
    candidates = [root / subdir / value, root / "data0613" / subdir / value]
    # OpenGVLab distributes the NTU subset required by fine_grained_pose as
    # converted MP4 media in video/nturgbd_convert/, while its annotation
    # retains the original NTU .avi filenames.  Preserve the downloaded
    # layout and resolve by the shared filename stem instead of renaming data.
    if task == "fine_grained_pose":
        candidates.append(root / "nturgbd_convert" / f"{Path(value).stem}.mp4")
    for path in candidates:
        if (media_kind == "video" and path.is_file()) or (media_kind == "frames" and path.is_dir()):
            return path, media_kind
    raise FileNotFoundError(f"{task} requires one of {candidates}; MVBench is incomplete")


def build_manifest(dataset: Path, output: Path, tasks: set[str] | None) -> None:
    if output.exists():
        raise FileExistsError(output)
    json_dir, video_dir = dataset / "json", dataset / "video"
    if not json_dir.is_dir() or not video_dir.is_dir():
        raise ValueError("dataset must contain json/ and video/")
    output.mkdir(parents=True)
    rows, source_hashes = [], {}
    for annotation in sorted(json_dir.glob("*.json")):
        task = annotation.stem
        if tasks is not None and task not in tasks:
            continue
        source_hashes[annotation.name] = sha256(annotation)
        items = json.loads(annotation.read_text())
        if not isinstance(items, list):
            raise ValueError(f"annotation is not a list: {annotation}")
        for item_index, item in enumerate(items):
            candidates = item.get("candidates")
            answer = item.get("answer")
            if not isinstance(candidates, list) or not candidates or answer not in candidates:
                raise ValueError(f"invalid multiple choice item: {annotation}:{item_index}")
            path, media_kind = resolve_video(video_dir, task, item["video"])
            rows.append(dict(task=task, item_index=item_index, video=item["video"], video_path=str(path),
                             question=item["question"], candidates=candidates, answer=answer,
                             answer_index=candidates.index(answer), media_kind=media_kind,
                             start=item.get("start"), end=item.get("end"), fps=item.get("fps")))
    if not rows:
        raise ValueError("no MVBench rows")
    protocol = dict(version="mvbench_strict_manifest_v2", dataset=str(dataset), samples=len(rows),
                    task_counts={task: sum(r["task"] == task for r in rows) for task in sorted({r["task"] for r in rows})},
                    annotation_sha256=source_hashes,
                    resolver="official task-to-directory mapping; fine_grained_pose accepts OpenGVLab nturgbd_convert MP4 files by filename stem", frames=16,
                    code_sha256=sha256(Path(__file__)))
    (output / "manifest.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n")
    (output / "COMPLETE").write_text(str(len(rows)) + "\n")
    print(json.dumps(protocol, indent=2), flush=True)


def prompt(row: dict) -> str:
    options = "\n".join(f"({LETTERS[i]}) {value}" for i, value in enumerate(row["candidates"]))
    return f"{row['question']}\nOptions:\n{options}\nAnswer with only the option letter."


def parse_choice(text: str, candidates: list[str]) -> int | None:
    for match in re.finditer(r"(?<![A-Z])([A-Z])(?![A-Z])", text.upper()):
        index = LETTERS.index(match.group(1))
        if index < len(candidates):
            return index
    normalized = " ".join(text.lower().split())
    hits = [i for i, candidate in enumerate(candidates) if " ".join(candidate.lower().split()) in normalized]
    return hits[0] if len(hits) == 1 else None


def load_mvbench_frames(row: dict, frames: int = 16) -> list[Image.Image]:
    if row["media_kind"] == "frames":
        paths = sorted(path for path in Path(row["video_path"]).iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        fps = float(row.get("fps") or 3.0)
        first = max(0, round(float(row.get("start") or 0.0) * fps))
        last = min(len(paths) - 1, round(float(row.get("end") or len(paths) / fps) * fps))
        if not paths or last < first:
            raise RuntimeError(f"invalid frame sequence: {row['video_path']}")
        return [Image.open(paths[i]).convert("RGB") for i in torch.linspace(first, last, frames, dtype=torch.int64).tolist()]
    vr = VideoReader(row["video_path"], ctx=cpu(0), num_threads=1)
    fps = float(vr.get_avg_fps())
    first = max(0, round(float(row.get("start") or 0.0) * fps))
    last = min(len(vr) - 1, round(float(row.get("end") or len(vr) / fps) * fps))
    if last < first:
        raise RuntimeError(f"invalid video range: {row['video_path']}")
    return [Image.fromarray(frame) for frame in vr.get_batch(torch.linspace(first, last, frames, dtype=torch.int64).tolist()).asnumpy()]


def run(manifest: Path, output: Path, methods: list[str], limit: int | None, task: str | None, max_new_tokens: int, ratio: float = 0.25) -> None:
    if not (manifest / "COMPLETE").exists():
        raise ValueError("manifest is not frozen and complete")
    if output.exists():
        raise FileExistsError(output)
    rows = [json.loads(line) for line in (manifest / "manifest.jsonl").read_text().splitlines() if line]
    if task is not None:
        rows = [row for row in rows if row["task"] == task]
        if not rows:
            raise ValueError(f"task not present in frozen manifest: {task}")
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    output.mkdir(parents=True)
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    tokenizer, model, processor, _ = load_pretrained_model("lmms-lab/LLaVA-Video-7B-Qwen2", None, "llava_qwen", device_map="auto", attn_implementation="sdpa")
    model.eval(); model.config.image_aspect_ratio = "square"; model.config.image_grid_pinpoints = None
    protocol = dict(version="mvbench_strict_inference_v1", manifest_sha256=sha256(manifest / "manifest.jsonl"),
                    methods=methods, samples=len(rows), frames=16, ratio=ratio, max_new_tokens=max_new_tokens, seed=0,
                    smoke_limit=limit, selected_task=task, scoring="native exact multiple-choice accuracy; letter parsed first, then unique candidate-text fallback",
                    protocol_name="controlled_frame_newline_v1", code_sha256=sha256(Path(__file__)))
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n")
    results = output / "results.jsonl"
    with torch.inference_mode(), results.open("w", encoding="utf-8") as handle:
        for number, row in enumerate(rows, start=1):
            frames = load_mvbench_frames(row, 16)
            pixels = process_images(frames, processor, model.config)
            if isinstance(pixels, list):
                pixels = torch.stack(pixels)
            tower, projector = model.get_vision_tower(), model.get_model().mm_projector
            pixels = pixels.to(model.device, dtype=next(tower.parameters()).dtype)
            vision = tower(pixels)
            features = model.get_2dPool(projector(vision.to(next(projector.parameters()).dtype)))
            for method in methods:
                started = time.monotonic()
                visual, metadata = select_visual(features, model.model.image_newline, method, ratio)
                selection_seconds = time.monotonic() - started
                started = time.monotonic()
                answer, counts = generate_from_visual(model, tokenizer, prompt(row), visual, max_tokens=max_new_tokens)
                predicted = parse_choice(answer, row["candidates"])
                # `metadata` contains the selector's method name; keep the
                # benchmark row's explicit method field single-valued.
                result = {**row, **counts, **metadata, "method": method, "prediction": answer, "predicted_index": predicted,
                          "correct": (predicted == row["answer_index"]), "success": (predicted is not None),
                          "selection_seconds": selection_seconds, "generation_seconds": time.monotonic() - started}
                handle.write(json.dumps(result, ensure_ascii=False) + "\n"); handle.flush()
            print(f"DONE sample={number}/{len(rows)}", flush=True)
    all_rows = [json.loads(line) for line in results.read_text().splitlines()]
    summary = {}
    for method in methods:
        values = [row for row in all_rows if row["method"] == method]
        task_accuracy = {task: sum(row["correct"] for row in values if row["task"] == task) / sum(row["task"] == task for row in values)
                         for task in sorted({row["task"] for row in values})}
        summary[method] = dict(samples=len(values), accuracy=sum(row["correct"] for row in values) / len(values),
                               valid_rate=sum(row["success"] for row in values) / len(values),
                               task_macro_accuracy=sum(task_accuracy.values()) / len(task_accuracy), task_accuracy=task_accuracy)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "COMPLETE").write_text(str(len(all_rows)) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build-manifest"); p.add_argument("--dataset", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--tasks", nargs="+")
    p = sub.add_parser("run"); p.add_argument("--manifest", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS); p.add_argument("--limit", type=int); p.add_argument("--task"); p.add_argument("--max-new-tokens", type=int, default=32); p.add_argument("--ratio", type=float, default=0.25)
    args = parser.parse_args()
    if args.command == "build-manifest": build_manifest(args.dataset, args.output, set(args.tasks) if args.tasks else None)
    else: run(args.manifest, args.output, args.methods, args.limit, args.task, args.max_new_tokens, args.ratio)


if __name__ == "__main__":
    main()
