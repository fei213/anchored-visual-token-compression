"""SurveillanceVQA aggregation (pre-registered in CHECKLIST §25, written BEFORE scores seen).

Primary metric  = equal-weight mean of the four judge dimensions (CI/DO/CU/TU).
Primary compare = Delta_Div (Anchor+Div - Uniform) and Delta_Anchor (Anchor - Uniform),
paired on (video, family). Per-domain / per-family are exploratory only.

SAFETY: refuses to report unless the record set is complete (or --allow-partial given),
so partial results cannot be read by accident.
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np

METHODS = ["full", "uniform", "anchor", "div"]
EXPECTED = 720 * 6 * 4          # 720 QA-bearing videos x 6 families x 4 methods


def load(p: Path) -> list[dict]:
    out = []
    for l in p.read_text().splitlines():
        if l.strip():
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError:
                pass
    return out


def paired_boot(items_a: list[float], items_b: list[float], n_boot=10000, seed=0):
    a, b = np.asarray(items_a, float), np.asarray(items_b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    n = len(d)
    boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return dict(n=n, mean=float(d.mean()), ci95=[float(np.percentile(boots, 2.5)),
                                                float(np.percentile(boots, 97.5))],
                p_le0=float((boots <= 0).mean()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()

    recs = load(args.judged)
    if len(recs) < EXPECTED and not args.allow_partial:
        print(f"  ⛔ 记录数 {len(recs)} < 期望 {EXPECTED}：按 §25 预注册纪律拒绝汇总"
              f"（如确需，显式传 --allow-partial）")
        return
    scored = [r for r in recs if r.get("avg") is not None]
    print(f"  记录 {len(recs)} 条，含有效分数 {len(scored)} 条")

    by_item = defaultdict(dict)                      # (stem,family) -> method -> avg
    for r in scored:
        by_item[(r["stem"], r["family"])][r["method"]] = float(r["avg"])

    summary = {}
    for m in METHODS:
        vals = [v[m] for v in by_item.values() if m in v]
        summary[m] = dict(n_items=len(vals), mean_score=float(np.mean(vals)) if vals else None)
    print("\n| method | items | mean score (4-dim equal-weight) |")
    print("|---|---:|---:|")
    for m in METHODS:
        s = summary[m]
        print(f"| {m} | {s['n_items']} | {s['mean_score']:.4f} |" if s["mean_score"] is not None
              else f"| {m} | 0 | — |")

    for prop in ["div", "anchor"]:
        pairs = [(v[prop], v["uniform"]) for v in by_item.values() if prop in v and "uniform" in v]
        if not pairs:
            continue
        pb = paired_boot([p[0] for p in pairs], [p[1] for p in pairs])
        tag = "Δ_Div" if prop == "div" else "Δ_Anchor"
        print(f"\n  {tag} = {pb['mean']:+.4f}  CI95 [{pb['ci95'][0]:+.4f}, {pb['ci95'][1]:+.4f}]  "
              f"P(<=0)={pb['p_le0']:.4f}  n={pb['n']}")
        summary[tag] = pb

    # exploratory：按源域与家族
    for field in ["category", "family"]:
        groups = defaultdict(list)
        for r in scored:
            groups[r[field]].append(r)
        print(f"\n  === 探索性：按 {field} ===")
        for g in sorted(groups):
            gs = groups[g]
            dm = {m: np.mean([float(x["avg"]) for x in gs if x["method"] == m]) for m in METHODS}
            print(f"    {g:26s} n={len(gs):5d}  full={dm['full']:.2f} uniform={dm['uniform']:.2f} "
                  f"anchor={dm['anchor']:.2f} div={dm['div']:.2f}")
            summary.setdefault(f"by_{field}", {})[g] = dm

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
