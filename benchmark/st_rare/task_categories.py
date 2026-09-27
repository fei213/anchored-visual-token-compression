"""Per-category rollup of a *_strict run (CHECKLIST §8 Go condition, §5.1).

§8's Go condition asks for a clear advantage on "at least some temporal / motion
tasks", not just overall accuracy, so the 20 MVBench tasks are grouped into
single-assignment categories here.

NOTE: this is the classification adopted for THIS project, not an official
MVBench grouping -- each task is assigned to exactly one primary category so the
rollup cannot double-count.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from benchmark.st_rare.paired_bootstrap import load_rows

TASK_CATEGORY = {
    # Action: what is being done
    "action_sequence": "Action", "action_prediction": "Action", "action_antonym": "Action",
    "fine_grained_action": "Action", "unexpected_action": "Action",
    "action_localization": "Action", "action_count": "Action",
    # Motion: how things move
    "moving_direction": "Motion", "moving_count": "Motion",
    "moving_attribute": "Motion", "fine_grained_pose": "Motion",
    # Temporal: ordering / change over time
    "state_change": "Temporal", "character_order": "Temporal", "scene_transition": "Temporal",
    # Spatial: where things are / navigation
    "egocentric_navigation": "Spatial",
    # Object: object-centric
    "object_existence": "Object", "object_interaction": "Object", "object_shuffle": "Object",
    # Reasoning: episodic / counterfactual
    "episodic_reasoning": "Reasoning", "counterfactual_inference": "Reasoning",
}
CATEGORY_ORDER = ["Action", "Motion", "Temporal", "Spatial", "Object", "Reasoning"]


def category_accuracy(rows: list[dict]) -> tuple[dict[str, float], dict[str, int]]:
    """Accuracy per category for one method's rows."""
    buckets: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        cat = TASK_CATEGORY.get(r["task"])
        if cat is None:
            raise KeyError(f"no category for task {r['task']!r}; add it to TASK_CATEGORY")
        buckets[cat].append(bool(r["correct"]))
    acc = {c: sum(v) / len(v) for c, v in buckets.items()}
    n = {c: len(v) for c, v in buckets.items()}
    return acc, n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--proposed", required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    rows, skipped = load_rows(args.results)
    if skipped:
        print(f"警告: 跳过 {skipped} 行无法解析（疑似中断写入的尾部）", flush=True)
    methods = sorted({r["method"] for r in rows})
    out = args.out_dir or args.results.parent
    out.mkdir(parents=True, exist_ok=True)

    by_method = {m: category_accuracy([r for r in rows if r["method"] == m]) for m in methods}

    print(f"{'category':10s} " + " ".join(f"{m[:13]:>13s}" for m in methods))
    for cat in CATEGORY_ORDER:
        cells = []
        for m in methods:
            acc, n = by_method[m]
            cells.append(f"{acc[cat]:13.4f}" if cat in acc else f"{'-':>13s}")
        print(f"{cat:10s} " + " ".join(cells))

    with (out / "category_delta.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["baseline", "category", "n", "proposed_acc", "baseline_acc", "delta"])
        p_acc, p_n = by_method[args.proposed]
        for m in methods:
            if m == args.proposed:
                continue
            b_acc, _ = by_method[m]
            for cat in CATEGORY_ORDER:
                if cat in p_acc and cat in b_acc:
                    w.writerow([m, cat, p_n[cat], p_acc[cat], b_acc[cat], p_acc[cat] - b_acc[cat]])
    print(f"\nwrote {out / 'category_delta.csv'}")


if __name__ == "__main__":
    main()
