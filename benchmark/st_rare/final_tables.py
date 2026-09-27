"""Final result tables (paired bootstrap over matched items).

Emits two markdown tables:

  Table 1: Method | Overall | vs Uniform delta | CI | Temporal | Motion | Spatial
           | Compression overhead
  Table 2: Task | Uniform | Anchor | Anchor+Div | deltas

The verdict is read from Table 1's overall paired comparison
(Anchor+Div > Uniform, ideally > Anchor); the category columns and Table 2 are
mechanism analysis only and do not decide pass/fail.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from benchmark.st_rare.paired_bootstrap import item_key, load_rows, paired_bootstrap
from benchmark.st_rare.task_categories import CATEGORY_ORDER, TASK_CATEGORY

METHOD_ORDER = ["controlled_full", "uniform", "st_rare", "learned_anchor", "learned_div"]
DISPLAY = {"controlled_full": "Full", "uniform": "Uniform", "st_rare": "ST-Rare",
           "learned_anchor": "Anchor", "learned_div": "Anchor+Div"}
CAT_COLS = ["Temporal", "Motion", "Spatial"]


def cell(x, fmt="{:.4f}"):
    return fmt.format(x) if x is not None else "—"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--reference", default="uniform", help="baseline for the vs-Δ column")
    ap.add_argument("--proposed", default="learned_div")
    ap.add_argument("--secondary", default="learned_anchor")
    ap.add_argument("--overhead-ref", default="controlled_full")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    rows, skipped = load_rows(args.results)
    if skipped:
        print(f"警告: 跳过 {skipped} 行无法解析", flush=True)
    methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in rows)]
    methods += sorted({r["method"] for r in rows} - set(methods))
    per_item: dict[tuple, dict] = {}          # key -> method -> full row
    for r in rows:
        per_item.setdefault(item_key(r), {})[r["method"]] = r
    # paired_bootstrap expects {key: {method: correct(bool)}}, not the row dicts
    per_item_correct = {k: {m: bool(v["correct"]) for m, v in d.items()}
                        for k, d in per_item.items()}

    def acc(sub):  # accuracy of a list of rows
        return sum(bool(r["correct"]) for r in sub) / len(sub) if sub else None

    def cat_acc(m, cat):
        sub = [r for r in rows if r["method"] == m and TASK_CATEGORY.get(r["task"]) == cat]
        return acc(sub)

    # compression overhead: selection time / generation time saved vs the uncompressed ref
    overhead = {}
    if args.overhead_ref in methods:
        shared = [k for k, v in per_item.items() if len(v) == len(methods)]
        for m in methods:
            sel = np.mean([per_item[k][m]["selection_seconds"] for k in shared])
            gen = np.mean([per_item[k][m]["generation_seconds"] for k in shared])
            ref = np.mean([per_item[k][args.overhead_ref]["generation_seconds"] for k in shared])
            saved = ref - gen
            overhead[m] = (sel / saved) if saved > 0 else None

    boots = {m: paired_bootstrap(per_item_correct, m, args.reference, args.n_boot)
             for m in methods if m != args.reference and args.reference in methods}

    # ---------- Table 1 ----------
    hdr = ["Method", "Overall", f"vs {DISPLAY.get(args.reference, args.reference)} Δ", "95% CI"] \
          + CAT_COLS + ["Compression overhead"]
    t1 = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for m in methods:
        a = acc([r for r in rows if r["method"] == m])
        b = boots.get(m)
        d = cell(b["mean_delta"], "{:+.4f}") if b else "—"
        ci = f"[{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]" if b else "—"
        over = cell(overhead.get(m), "{:.3f}") if m != args.overhead_ref else "—"
        t1.append("| " + " | ".join([DISPLAY.get(m, m), cell(a)] + [d, ci] +
                                    [cell(cat_acc(m, c)) for c in CAT_COLS] + [over]) + " |")

    # ---------- Table 2 ----------
    tasks = sorted({r["task"] for r in rows}, key=lambda t: (CATEGORY_ORDER.index(TASK_CATEGORY[t]), t))
    cols = ["uniform", "learned_anchor", args.proposed]
    present = [c for c in cols if c in methods]
    hdr2 = ["Task", "Category"] + [DISPLAY.get(c, c) for c in present] + \
           [f"{DISPLAY.get(args.proposed, args.proposed)}−{DISPLAY.get(c, c)}" for c in present
            if c != args.proposed]
    t2 = ["| " + " | ".join(hdr2) + " |", "|" + "---|" * len(hdr2)]
    for t in tasks:
        cells, base = [], {}
        for c in present:
            sub = [r for r in rows if r["method"] == c and r["task"] == t]
            base[c] = acc(sub)
            cells.append(cell(base[c]))
        deltas = [cell(base[args.proposed] - base[c], "{:+.4f}") for c in present if c != args.proposed]
        t2.append("| " + " | ".join([t, TASK_CATEGORY[t]] + cells + deltas) + " |")

    out = args.out_dir or args.results.parent
    out.mkdir(parents=True, exist_ok=True)
    md = ("# Table 1 — overall\n\n" + "\n".join(t1) +
          "\n\n# Table 2 — per task\n\n" + "\n".join(t2) + "\n")
    (out / "final_tables.md").write_text(md)

    for name, tbl, hds in (("table1_overall.csv", t1, hdr), ("table2_per_task.csv", t2, hdr2)):
        with (out / name).open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(hds)
            for line in tbl[2:]:
                w.writerow([c.strip() for c in line.strip("| ").split(" | ")])

    print("\n".join(t1))
    if args.proposed in boots:
        b = boots[args.proposed]
        print(f"\n【主判据】{DISPLAY.get(args.proposed)} − {DISPLAY.get(args.reference)} = "
              f"{b['mean_delta']:+.4f}  95% CI [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]  "
              f"P(Δ≤0)={b['p_one_sided_le0']:.4f}")
    # NOTE: `boots` holds (method - reference) for every method, so the secondary
    # criterion (proposed - secondary) needs its own comparison -- reusing
    # boots[secondary] would silently report (secondary - reference) instead.
    if args.secondary in methods and args.secondary != args.proposed:
        b = paired_bootstrap(per_item_correct, args.proposed, args.secondary, args.n_boot)
        print(f"【次判据】{DISPLAY.get(args.proposed)} − {DISPLAY.get(args.secondary)} = "
              f"{b['mean_delta']:+.4f}  95% CI [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]  "
              f"P(Δ≤0)={b['p_one_sided_le0']:.4f}")
    print()
    print("\n".join(t2))
    print(f"\nwrote final_tables.md / table1_overall.csv / table2_per_task.csv to {out}")


if __name__ == "__main__":
    main()
