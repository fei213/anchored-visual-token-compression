"""Paired bootstrap for matched-budget method comparison (CHECKLIST §14, §20 step 10).

Compares methods evaluated on the SAME items (MVBench sample-paired). Resamples
items with replacement, recomputes the accuracy difference, and reports the mean
delta with a 95% percentile CI and a one-sided p-value (P(delta <= 0)).

Usage:
    python -m benchmark.st_rare.paired_bootstrap --results <results.jsonl> \
        --reference uniform [--n-boot 10000]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_rows(results_path: Path) -> tuple[list[dict], int]:
    """Parse a results.jsonl, skipping malformed lines.

    An interrupted run can leave a truncated final line; crashing on it would
    discard every result already written. Returns (rows, n_skipped).
    """
    rows, skipped = [], 0
    for line in results_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            skipped += 1
    return rows, skipped


def item_key(row: dict) -> tuple:
    """Pairing key for matched comparison.

    MVBench rows pair on (task, item_index); VideoMME rows on
    (video_id, question_id) -- the same question on the same video.
    """
    if "task" in row and "item_index" in row:
        return (row["task"], row["item_index"])
    if "video_id" in row and "question_id" in row:
        return (row["video_id"], row["question_id"])
    raise KeyError(f"cannot derive a pairing key from keys {sorted(row)}")


def load_per_item(results_path: Path) -> dict[tuple, dict[str, bool]]:
    """Return {item_key: {method: correct}} for a *_strict results.jsonl."""
    rows, skipped = load_rows(results_path)
    if skipped:
        print(f"  [load_per_item] 跳过 {skipped} 行（疑似中断写入的尾部）", flush=True)
    per_item: dict[tuple, dict[str, bool]] = defaultdict(dict)
    for row in rows:
        per_item[item_key(row)][row["method"]] = bool(row["correct"])
    return dict(per_item)


def paired_bootstrap(per_item: dict[tuple, dict[str, bool]], method_a: str, method_b: str,
                     n_boot: int = 10000, seed: int = 0) -> dict:
    """Paired bootstrap of acc(method_a) - acc(method_b), resampling items."""
    items = [k for k, v in per_item.items() if method_a in v and method_b in v]
    if not items:
        raise ValueError(f"no shared items for {method_a} vs {method_b}")
    correct_a = np.array([per_item[k][method_a] for k in items], dtype=float)
    correct_b = np.array([per_item[k][method_b] for k in items], dtype=float)
    rng = np.random.default_rng(seed)
    n = len(items)
    deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = correct_a[idx].mean() - correct_b[idx].mean()
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    # NOTE: mean_delta is the EXACT observed difference, not the mean of the
    # bootstrap deltas -- the latter carries ~1e-4 Monte-Carlo noise, which an
    # independent audit flagged as a spurious 4th-decimal discrepancy. The CI and
    # p-value still come from the bootstrap (and are exact functionals of the data).
    return dict(
        method_a=method_a, method_b=method_b, n_items=n, n_boot=n_boot,
        acc_a=float(correct_a.mean()), acc_b=float(correct_b.mean()),
        mean_delta=float(correct_a.mean() - correct_b.mean()),
        mean_delta_boot=float(deltas.mean()),
        ci95=[float(ci_lo), float(ci_hi)],
        p_one_sided_le0=float((deltas <= 0).mean()),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--reference", required=True, help="method treated as 'Proposed' (A)")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    per_item = load_per_item(args.results)
    methods = sorted({m for v in per_item.values() for m in v})
    print(f"items={len(per_item)} methods={methods}")

    rows = []
    for other in methods:
        if other == args.reference:
            continue
        rows.append(paired_bootstrap(per_item, args.reference, other, args.n_boot, args.seed))

    for r in rows:
        ci = r["ci95"]
        print(f"{r['method_a']} - {r['method_b']} = {r['mean_delta']:+.4f} "
              f"95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}] "
              f"P(delta<=0)={r['p_one_sided_le0']:.4f}  "
              f"(acc {r['acc_a']:.4f} vs {r['acc_b']:.4f}, n={r['n_items']})")

    out = args.results.with_name("bootstrap.json")
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
