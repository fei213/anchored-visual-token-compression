"""Zero-shot rate-distortion sweep (CHECKLIST §20.7).

Frozen 25%-trained checkpoints, deployment-time rate switch only: the SAME
checkpoint is evaluated at 25% / 12.5% / 6.25%, nothing is retrained or tuned.
Scan set = the video-grouped val (never used for method selection). The
confirmatory final is NOT touched here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path = [p for p in sys.path if Path(p or ".").resolve() != ROOT]
sys.path.append(str(ROOT))

from llava.compression.learned import (ResidualAnchoredCompressor,
                                       compress_and_pack_anchored, count_trainable)
from benchmark.st_rare.eval_latent_compressor import load_compressor_state

METHODS = ["controlled_full", "uniform", "learned_div", "learned_anchor"]
RATIOS = [0.25, 0.125, 0.0625]


def load_anchored(ckpt: Path, device: str):
    state = load_compressor_state(ckpt)
    if "core.queries" not in state:
        raise SystemExit(f"{ckpt} 缺少 core.queries（应传 ResidualAnchoredCompressor checkpoint）")
    comp = ResidualAnchoredCompressor(
        dim=state["core.queries"].shape[1], num_frames=state["core.frame_emb"].shape[0],
        max_tokens_per_frame=state["core.queries"].shape[0], num_heads=8).to(device)
    missing, unexpected = comp.load_state_dict(state, strict=False)
    assert not unexpected and not missing, f"{ckpt}: unexpected={unexpected} missing={missing}"
    comp.eval()
    print(f"  loaded {ckpt.name}: {count_trainable(comp):,} 参数, alpha={float(comp.alpha):.6f}")
    return comp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary-ckpt", type=Path, required=True, help="Anchor+Div (25% trained)")
    ap.add_argument("--secondary-ckpt", type=Path, required=True, help="Anchor (25% trained)")
    ap.add_argument("--manifest", type=Path, required=True, help="video-grouped val manifest DIR")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--ratios", nargs="+", type=float, default=RATIOS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--smoke", type=int, default=None,
                    help="只跑第一个 ratio 的前 N 条做通路验证")
    args = ap.parse_args()

    from benchmark.st_rare import mvbench_strict as M
    from benchmark.st_rare.strict_video_protocol import register_learned

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("loading compressors (frozen, 25%-trained):")
    div = load_anchored(args.primary_ckpt, device)
    anc = load_anchored(args.secondary_ckpt, device)

    def make_selector(comp):
        def selector(features, newline, content):
            with torch.no_grad():
                return compress_and_pack_anchored(comp, features, newline, content)
        return selector

    register_learned("learned_div", make_selector(div))
    register_learned("learned_anchor", make_selector(anc))
    print(f"registered learned_div(primary) + learned_anchor(secondary)")

    ratios = args.ratios
    if args.smoke is not None:
        ratios = ratios[:1]
        print(f"[SMOKE] 只跑 ratio={ratios[0]} 前 {args.smoke} 条（不作结果）")

    for ratio in ratios:
        out = args.out_root / f"ratio_{ratio:g}"
        print(f"\n########## ratio={ratio}  ->  {out} ##########")
        if out.exists():
            raise SystemExit(f"{out} 已存在，拒绝覆盖")
        M.run(args.manifest, out, METHODS, args.smoke or args.limit, None,
              args.max_new_tokens, ratio=ratio)
        print(f"ratio={ratio} 完成")


if __name__ == "__main__":
    main()
