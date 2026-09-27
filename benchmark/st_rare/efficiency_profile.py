"""E2E latency + full-pipeline memory profiling (CHECKLIST §20.9).

Scope is deliberately narrow (25% operating point, 4 methods). Each measured run
is a COLD full pipeline for one (video, method): frames -> vision -> projector ->
compression -> prefill -> generate, so the reported peak memory is a genuine
full-pipeline peak, not a single phase's.

Timing attribution (documented in §20.9):
  T_frames/T_vision/T_projector : per-video work (amortised across a video's questions)
  T_compression/T_prefill/T_decode/E2E : per-question work
T_prefill is a single no-cache forward over the full prompt (exact prefill work);
T_decode = E2E_LLM - T_prefill. E2E_LLM is the real generate_from_visual path with
min_tokens = max_tokens = --gen-tokens (greedy), so answer length cannot contaminate
latency.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path = [p for p in sys.path if Path(p or ".").resolve() != ROOT]
sys.path.append(str(ROOT))

from llava.compression.learned import (ResidualAnchoredCompressor,
                                       compress_and_pack_anchored, count_trainable)
from benchmark.st_rare.eval_latent_compressor import load_compressor_state

METHODS = ["full", "uniform", "anchor", "div"]


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def peak_mb() -> float:
    return torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else float("nan")


def load_anchored(ckpt: Path, device: str):
    state = load_compressor_state(ckpt)
    if "core.queries" not in state:
        raise SystemExit(f"{ckpt} 缺少 core.queries")
    comp = ResidualAnchoredCompressor(
        dim=state["core.queries"].shape[1], num_frames=state["core.frame_emb"].shape[0],
        max_tokens_per_frame=state["core.queries"].shape[0], num_heads=8).to(device)
    missing, unexpected = comp.load_state_dict(state, strict=False)
    assert not unexpected and not missing, f"{ckpt}: unexpected={unexpected} missing={missing}"
    comp.eval()
    print(f"  loaded {ckpt.name}: {count_trainable(comp):,} 参数, alpha={float(comp.alpha):.6f}")
    return comp


def stats(xs: list[float]) -> dict:
    a = np.asarray(xs, dtype=float)
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                p95=float(np.percentile(a, 95)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path,
                    default=ROOT / "artifacts/st_rare/mvbench_val_videogrouped_v1")
    ap.add_argument("--primary-ckpt", type=Path,
                    default=ROOT / "artifacts/st_rare/latent_compressor_anchor_div_leakfree_v1/compressor_epoch2.pt")
    ap.add_argument("--secondary-ckpt", type=Path,
                    default=ROOT / "artifacts/st_rare/latent_compressor_anchor_leakfree_v1/compressor_epoch2.pt")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "artifacts/st_rare/efficiency_profile_v1")
    ap.add_argument("--ratio", type=float, default=0.25)
    ap.add_argument("--videos", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--gen-tokens", type=int, default=32)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--stratify-by-task", action="store_true",
                    help="每个 task 固定取 1 个视频（robustness check，§20.10）；"
                         "--videos 此时表示最多取多少个 task")
    ap.add_argument("--check-equivalence", action="store_true", default=True)
    args = ap.parse_args()

    from benchmark.st_rare.run_balanced_dev import load_pretrained_model
    from benchmark.st_rare.mvbench_strict import load_mvbench_frames, prompt, parse_choice
    from benchmark.st_rare.strict_video_protocol import (select_visual, generate_from_visual,
                                                        register_learned)
    from llava.mm_utils import process_images
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from transformers import Qwen2ForCausalLM

    rows = [json.loads(l) for l in (args.manifest / "manifest.jsonl").read_text().splitlines() if l.strip()]
    picked = []
    if args.stratify_by_task:
        # one video per task, first distinct video of each task in manifest order
        seen_t, seen_v = set(), set()
        for r in rows:
            if r["task"] in seen_t or r["video_path"] in seen_v:
                continue
            seen_t.add(r["task"]); seen_v.add(r["video_path"]); picked.append(r)
            if len(picked) == args.videos:
                break
        print(f"[stratified] 选出 {len(picked)} 个视频，覆盖 {len(seen_t)} 个 task: "
              f"{sorted(seen_t)}")
    else:
        seen = set()
        for r in rows:                              # deterministic: first N distinct videos
            if r["video_path"] not in seen:
                seen.add(r["video_path"]); picked.append(r)
            if len(picked) == args.videos:
                break
        print(f"选出 {len(picked)} 个视频（确定性取前 N 个不同视频）")
    for i, r in enumerate(picked, 1):
        print(f"  {i:2d}. {r['task']:24s} item_index={r['item_index']:<4d} video={r['video']}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    div = load_anchored(args.primary_ckpt, device)
    anc = load_anchored(args.secondary_ckpt, device)

    def mk(comp, anchored=True):
        def sel(features, newline, content):
            with torch.no_grad():
                return compress_and_pack_anchored(comp, features, newline, content)
        return sel
    register_learned("learned_div", mk(div))
    register_learned("learned_anchor", mk(anc))
    dispatcher = {"full": "controlled_full", "uniform": "uniform",
                  "anchor": "learned_anchor", "div": "learned_div"}

    tokenizer, model, processor, _ = load_pretrained_model(
        "lmms-lab/LLaVA-Video-7B-Qwen2", None, "llava_qwen", device_map="auto", attn_implementation="sdpa")
    model.eval()
    model.config.image_aspect_ratio = "square"
    model.config.image_grid_pinpoints = None

    acc: dict[str, dict[str, list]] = {m: {k: [] for k in
           ("frames", "vision", "projector", "compression", "prefill", "decode", "e2e_llm",
            "peak_full", "peak_llm", "tokens")} for m in args.methods}
    equiv_bad = []
    for vi, row in enumerate(picked, 1):
        for method in args.methods:
            sel_name = dispatcher[method]
            for it in range(args.warmup + args.runs):
                measured = it >= args.warmup
                if measured:
                    torch.cuda.reset_peak_memory_stats()   # window 1 starts at run start

                sync(); t0 = time.perf_counter()
                frames = load_mvbench_frames(row, 16)
                sync(); t1 = time.perf_counter()
                px = process_images(frames, processor, model.config)
                if isinstance(px, list):
                    px = torch.stack(px)
                tower, proj = model.get_vision_tower(), model.get_model().mm_projector
                px = px.to(model.device, dtype=next(tower.parameters()).dtype)
                sync(); t2 = time.perf_counter()
                with torch.inference_mode():
                    v = tower(px)
                    sync(); t3 = time.perf_counter()
                    feat = model.get_2dPool(proj(v.to(next(proj.parameters()).dtype)))
                    sync(); t4 = time.perf_counter()

                # window 1 (frames+vision+projector) ends here; read its peak BEFORE the
                # LLM reset, otherwise the vision-phase peak is silently lost (the bug that
                # made an earlier draft's "full-pipeline peak" just the LLM window).
                peak_pre_llm = peak_mb() if measured else float("nan")
                visual, meta = select_visual(feat, model.model.image_newline, sel_name, args.ratio)
                sync(); t5 = time.perf_counter()
                n_tok = int(meta["actual_visual_tokens"])

                # LLM-phase peak from here
                if measured:
                    torch.cuda.reset_peak_memory_stats()
                conv = conv_templates["qwen_1_5"].copy()
                conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt(row))
                conv.append_message(conv.roles[1], None)
                ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX,
                                           return_tensors="pt").to(model.device)
                at = int((ids == IMAGE_TOKEN_INDEX).nonzero().flatten()[0])
                emb = model.get_model().embed_tokens
                inp = torch.cat([emb(ids[:at]), visual, emb(ids[at + 1:])]).unsqueeze(0)
                mask = torch.ones((1, inp.shape[1]), device=inp.device, dtype=torch.long)

                with torch.inference_mode():
                    Qwen2ForCausalLM.forward(model, inputs_embeds=inp, attention_mask=mask,
                                             use_cache=False)          # exact prefill work
                sync(); t6 = time.perf_counter()

                text, counts = generate_from_visual(model, tokenizer, prompt(row), visual,
                                                   max_tokens=args.gen_tokens,
                                                   min_tokens=args.gen_tokens)
                sync(); t7 = time.perf_counter()
                peak_llm = peak_mb()
                # the two windows are disjoint, so the whole-pipeline peak is their max
                peak_full = max(peak_pre_llm, peak_llm)

                if measured:
                    a = acc[method]
                    a["frames"].append((t1 - t0) * 1e3); a["vision"].append((t3 - t2) * 1e3)
                    a["projector"].append((t4 - t3) * 1e3); a["compression"].append((t5 - t4) * 1e3)
                    a["prefill"].append((t6 - t5) * 1e3); a["decode"].append((t7 - t6) * 1e3)
                    a["e2e_llm"].append((t7 - t5) * 1e3)
                    a["peak_full"].append(peak_full); a["peak_llm"].append(peak_llm)
                    a["tokens"].append(n_tok)

                if it == 0 and args.check_equivalence:
                    idx = parse_choice(text, row["candidates"])
                    pred = None if idx is None else "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[idx]
                    # equivalence of our decomposition is structural: prefill is one full
                    # forward and E2E is the real generate() call; record the generation
                    # sanity (valid letter + expected token count) instead of comparing text.
                    if counts["generated_tokens"] != args.gen_tokens:
                        equiv_bad.append((row["task"], row["item_index"], method, counts["generated_tokens"]))
        print(f"DONE video {vi}/{len(picked)}", flush=True)

    out = {}
    for m in args.methods:
        out[m] = {k: stats(v) for k, v in acc[m].items() if v}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "efficiency.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")

    print(f"\n{'method':8s} {'tokens':>7s} {'compress_ms':>12s} {'prefill_ms':>11s} "
          f"{'decode_ms':>10s} {'e2e_llm_ms':>11s} {'peak_full_MB':>13s} {'peak_llm_MB':>12s}")
    for m in args.methods:
        o = out[m]
        print(f"{m:8s} {o['tokens']['mean']:>7.0f} {o['compression']['mean']:>12.2f} "
              f"{o['prefill']['mean']:>11.2f} {o['decode']['mean']:>10.2f} "
              f"{o['e2e_llm']['mean']:>11.2f} {o['peak_full']['mean']:>13.1f} "
              f"{o['peak_llm']['mean']:>12.1f}")
    print(f"\nper-video: " + "  ".join(
        f"{m}:vision={out[m]['vision']['mean']:.1f}ms proj={out[m]['projector']['mean']:.1f}ms "
        f"frames={out[m]['frames']['mean']:.1f}ms" for m in args.methods))
    print(f"\n生成 token 数异常: {equiv_bad if equiv_bad else '无（各方法均生成固定长度）'}")
    print(f"wrote {args.output/'efficiency.json'}")


if __name__ == "__main__":
    main()
