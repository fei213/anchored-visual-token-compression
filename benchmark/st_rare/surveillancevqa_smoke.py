"""SurveillanceVQA smoke test (Priority 2, perry 2026-09-27).

GOAL = plumbing only, NOT accuracy. Checks: video decode, prompt construction from the
dataset's own open-ended question, ground-truth parsing, model-output validity, that all
methods see the SAME items, judge determinism, and that paired comparison is feasible.

Frozen configuration only: Full / Uniform 25% / Anchor 25% / Anchor+Div 25%.
Scoring uses SurveillanceVQA's OWN judge prompts (verbatim, extracted to
surveillancevqa_judge_prompts.json); the 4 dimensions are combined with an EQUAL-WEIGHT
mean (perry 2026-09-27; the official paper's Avg is likewise unweighted).
"""
from __future__ import annotations
import argparse, json, os, random, re, sys, time
from pathlib import Path

import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]
sys.path = [p for p in sys.path if Path(p or ".").resolve() != ROOT]; sys.path.append(str(ROOT))

FAMS = ["detection_qa_pairs", "classification_qa_pairs", "subject_qa_pairs",
        "description_qa_pairs", "cause_qa_pairs", "result_qa_pairs"]
METHODS = ["full", "uniform", "anchor", "div"]


def build_media_index(media_root: Path) -> dict:
    idx = {}
    for src in ["UCF", "MSAD", "MEVA", "NWPU"]:
        for dp, _, fs in os.walk(media_root / src):
            for f in fs:
                if f.lower().endswith((".mp4", ".avi")):
                    idx.setdefault(f.rsplit(".", 1)[0], os.path.join(dp, f))
    return idx


def load_sample(bench_root: Path, limit: int, seed: int) -> list[dict]:
    rows = []
    for cat in sorted(os.listdir(bench_root / "test_datasets")):
        p = bench_root / "test_datasets" / cat
        if not p.is_dir():
            continue
        for f in sorted(os.listdir(p)):
            if f.endswith(".json"):
                rows.append({"category": cat, "stem": f[:-5], "path": str(p / f)})
    random.seed(seed)
    random.shuffle(rows)
    withqa = []
    for r in rows:                                  # 先筛出真正含 QA 的视频
        d = json.load(open(r["path"], encoding="utf-8"))
        if sum(len(d.get(x) or []) for x in FAMS) > 0:
            withqa.append(r)
    picked, seen = [], set()
    for r in withqa:                                # 第一段：每个类别至多取 1 个
        if r["category"] not in seen:
            seen.add(r["category"]); picked.append(r)
        if len(picked) == limit:
            return picked
    for r in withqa:                                # 第二段：再按原序填充到 limit
        if r not in picked:
            picked.append(r)
        if len(picked) == limit:
            break
    return picked[:limit]


def decode_frames(path: str, n: int = 16):
    import decord
    from PIL import Image
    vr = decord.VideoReader(path, num_threads=1)
    if len(vr) == 0:
        raise RuntimeError("empty video")
    idx = np.linspace(0, len(vr) - 1, n).astype(int)
    return [Image.fromarray(f).convert("RGB") for f in vr.get_batch(idx).asnumpy()]


def judge(question: str, ref: str, pred: str, dim: str, prompts: dict, api_key: str):
    from zhipuai import ZhipuAI
    sysmsg = prompts[dim]["prompt_texts"][0]
    body = "\n".join(prompts[dim]["prompt_texts"][1:])
    client = ZhipuAI(api_key=api_key)
    msg = (f"{sysmsg}\n{body}\n\nQuestion:\n{question}\n\nCorrect answer:\n{ref}\n\n"
           f"Predicted answer:\n{pred}\n\nProvide ONLY the integer 0-5.")
    r = client.chat.completions.create(model="glm-4-flash",
        messages=[{"role": "user", "content": msg}], temperature=0.0)
    m = re.search(r"\b([0-5])\b", r.choices[0].message.content or "")
    return int(m.group(1)) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=Path, required=True, help="SurveillanceVQA benchmark root")
    ap.add_argument("--media", type=Path, required=True, help="SurveillanceVQA media root")
    ap.add_argument("--anchor-ckpt", type=Path, required=True, help="Anchor compressor checkpoint (.pt)")
    ap.add_argument("--div-ckpt", type=Path, required=True, help="Anchor+Div compressor checkpoint (.pt)")
    ap.add_argument("--videos", type=int, default=6)
    ap.add_argument("--families", type=int, default=6, help="每题评测的家族数（默认全部 6 个）")
    ap.add_argument("--judge-pairs", type=int, default=10**9, help="最多评判多少对（默认全部）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, default=ROOT / "artifacts/st_rare/surveillancevqa_smoke_v1")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--api-key", default=os.environ.get("ZHIPUAI_API_KEY"),
                    help="ZhipuAI API key (or set ZHIPUAI_API_KEY)")
    args = ap.parse_args()

    from benchmark.st_rare.run_balanced_dev import load_pretrained_model
    from benchmark.st_rare.strict_video_protocol import select_visual, generate_from_visual, register_learned
    from llava.compression.learned import (ResidualAnchoredCompressor,
                                           compress_and_pack_anchored, count_trainable)
    from benchmark.st_rare.eval_latent_compressor import load_compressor_state
    from llava.mm_utils import process_images

    idx = build_media_index(args.media); sample = load_sample(args.bench, args.videos, args.seed)
    print(f"抽样 {len(sample)} 个有 QA 的视频，类别: {[r['category'] for r in sample]}")
    miss = [r["stem"] for r in sample if r["stem"] not in idx]
    print(f"  媒体命中: {len(sample)-len(miss)}/{len(sample)}" + (f"  未命中={miss}" if miss else ""))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    def load_anchored(p):
        st = load_compressor_state(p)
        c = ResidualAnchoredCompressor(dim=st["core.queries"].shape[1],
            num_frames=st["core.frame_emb"].shape[0],
            max_tokens_per_frame=st["core.queries"].shape[0], num_heads=8).to(device)
        c.load_state_dict(st, strict=False); c.eval(); return c
    anc = load_anchored(args.anchor_ckpt)
    div = load_anchored(args.div_ckpt)
    for nm, c in [("learned_anchor", anc), ("learned_div", div)]:
        def mk(comp=c):
            def sel(f, nl, ct):
                with torch.no_grad(): return compress_and_pack_anchored(comp, f, nl, ct)
            return sel
        register_learned(nm, mk())
    dispatcher = {"full": "controlled_full", "uniform": "uniform",
                  "anchor": "learned_anchor", "div": "learned_div"}

    tokenizer, model, processor, _ = load_pretrained_model(
        "lmms-lab/LLaVA-Video-7B-Qwen2", None, "llava_qwen", device_map="auto", attn_implementation="sdpa")
    model.eval(); model.config.image_aspect_ratio = "square"; model.config.image_grid_pinpoints = None
    prompts = json.loads((ROOT/"benchmark/st_rare/surveillancevqa_judge_prompts.json").read_text())

    args.output.mkdir(parents=True, exist_ok=True)
    out_path = args.output / "smoke.jsonl"
    recs, plumbing = [], {"decode_fail": [], "empty_pred": [], "items_per_method": {}}
    done = set()
    if out_path.exists():                       # 断点续跑：跳过已完成的 (stem, family, method)
        for l in out_path.read_text().splitlines():
            if l.strip():
                try:
                    x = json.loads(l); recs.append(x); done.add((x["stem"], x["family"], x["method"]))
                except json.JSONDecodeError:
                    pass
        print(f"  resume: 已有 {len(done)} 条记录，跳过它们")
    handle = out_path.open("a", encoding="utf-8")
    with torch.inference_mode():
        for r in sample:
            if r["stem"] not in idx:
                plumbing["decode_fail"].append((r["category"], r["stem"], "no media")); continue
            try:
                frames = decode_frames(idx[r["stem"]], 16)
            except Exception as e:
                plumbing["decode_fail"].append((r["category"], r["stem"], f"{type(e).__name__}")); continue
            px = process_images(frames, processor, model.config)
            if isinstance(px, list): px = torch.stack(px)
            tower, proj = model.get_vision_tower(), model.get_model().mm_projector
            feat = model.get_2dPool(proj(tower(px.to(model.device, dtype=next(tower.parameters()).dtype))
                                        .to(next(proj.parameters()).dtype)))
            d = json.load(open(r["path"], encoding="utf-8"))
            for fam in FAMS[: args.families]:
                for it in (d.get(fam) or []):
                    q, ref = str(it.get("Q","")).strip(), str(it.get("A","")).strip()
                    for m in METHODS:
                        if (r["stem"], fam, m) in done:
                            continue
                        vis, meta = select_visual(feat, model.model.image_newline, dispatcher[m], 0.25)
                        txt, counts = generate_from_visual(model, tokenizer, q, vis, max_tokens=48)
                        rec = {"category": r["category"], "stem": r["stem"], "family": fam,
                               "method": m, "question": q, "reference": ref,
                               "prediction": txt.strip(), "tokens": meta["actual_visual_tokens"],
                               "generated_tokens": counts["generated_tokens"]}
                        recs.append(rec)
                        handle.write(json.dumps(rec, ensure_ascii=False) + "\n"); handle.flush()
                        if not txt.strip(): plumbing["empty_pred"].append((r["stem"], fam, m))
            print(f"  done {r['category']}/{r['stem']}", flush=True)

    for m in METHODS:
        plumbing["items_per_method"][m] = len({(x["stem"], x["family"]) for x in recs if x["method"] == m})
    print("\n=== 管路检查（不报 accuracy）===")
    print(f"  解码失败: {plumbing['decode_fail'] or '无'}")
    print(f"  空输出: {len(plumbing['empty_pred'])} 例")
    print(f"  各方法样本数（应相等=同一批）: {plumbing['items_per_method']}")
    print(f"  每题方法齐全: {len({(x['stem'],x['family']) for x in recs})} 题，"
          f"齐全者 {sum(1 for k in {(x['stem'],x['family']) for x in recs} if len({x['method'] for x in recs if (x['stem'],x['family'])==k})==len(METHODS))}")

    if not args.no_judge:
        api_key = args.api_key
        if not api_key:
            print("  ⚠ 未提供 API key（--api-key 或 ZHIPUAI_API_KEY），跳过评判")
        else:
            judged, fails = [], 0
            for x in recs[: args.judge_pairs]:
                sc = {}
                for dim in ["CI","DO","CU","TU"]:
                    try: sc[dim] = judge(x["question"], x["reference"], x["prediction"], dim, prompts, api_key)
                    except Exception as e: sc[dim] = None; fails += 1
                x["judge"] = sc
                x["avg"] = None if any(v is None for v in sc.values()) else sum(sc.values())/4
                judged.append(x)
            print(f"  评判 {len(judged)} 对 × 4 维（失败 {fails}）")
            for x in judged[:3]:
                print(f"    {x['method']:8s} {x['family'][:20]:20s} CI/DO/CU/TU={x['judge']} avg={x['avg']}")
            if len(judged) >= 2:                      # 确定性：重评同一对
                a = judged[0]
                sc2 = {d: judge(a["question"], a["reference"], a["prediction"], d, prompts, api_key)
                       for d in ["CI","DO","CU","TU"]}
                print(f"  评判确定性: 首次={a['judge']} 重评={sc2} 一致={a['judge']==sc2}")

    print(f"\nwrote {args.output/'smoke.jsonl'} ({len(recs)} 条)")


if __name__ == "__main__":
    main()
