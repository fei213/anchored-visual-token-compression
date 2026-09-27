"""Evaluate a trained latent compressor against Uniform on the SAME items.

Deliberately reuses benchmark.st_rare.mvbench_strict.run -- the identical
generation/scoring path every baseline used -- and only swaps the selector via
the protocol's additive learned-selector hook. So a `learned` vs `uniform` gap
cannot come from scorer or prompt drift.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path = [p for p in sys.path if Path(p or ".").resolve() != ROOT]
sys.path.append(str(ROOT))

from llava.compression.learned import (LatentCompressor, ResidualAnchoredCompressor,
                                       compress_and_pack, compress_and_pack_anchored,
                                       count_trainable)


def load_compressor_state(path) -> dict:
    """Load a compressor checkpoint.

    PyTorch >= 2.6 defaults to weights_only=True; early checkpoints also stored the
    argparse namespace, whose Path values are not allowlisted, so they need the
    explicit fallback. Both branches are our own locally produced files.
    """
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:                                  # noqa: BLE001
        print(f"  注意: weights_only=True 加载失败 ({type(e).__name__})；"
              f"回退到 weights_only=False（本地自有 checkpoint）")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True, help="manifest DIR (needs COMPLETE)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--methods", nargs="+", default=["uniform", "learned"])
    ap.add_argument("--ratio", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--anchor", action="store_true",
                    help="checkpoint 是 ResidualAnchoredCompressor（Z = A + alpha*C(X)）")
    args = ap.parse_args()

    from benchmark.st_rare import mvbench_strict as M
    from benchmark.st_rare.strict_video_protocol import register_learned

    # NOTE: do NOT load the model here. mvbench_strict.run() loads it internally;
    # loading a second copy would put two 7B models on a 32 GB card. The compressor's
    # shapes are read from the checkpoint instead.
    state = load_compressor_state(args.checkpoint)
    # anchored checkpoints nest the core module under "core." -- handle both
    pre = "core." if args.anchor else ""
    key_q = f"{pre}queries"
    if key_q not in state:
        raise SystemExit(f"checkpoint 缺少 {key_q}（是否忘了/多写了 --anchor？可用键示例: {list(state)[:4]}）")
    dim = state[key_q].shape[1]
    k_max = state[key_q].shape[0]
    n_frames = state[f"{pre}frame_emb"].shape[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cls = ResidualAnchoredCompressor if args.anchor else LatentCompressor
    comp = cls(dim=dim, num_frames=n_frames, max_tokens_per_frame=k_max, num_heads=8).to(device)
    missing, unexpected = comp.load_state_dict(state, strict=False)
    assert not unexpected, f"checkpoint 含未知键（是否忘了 --anchor?）: {unexpected}"
    assert not missing, f"checkpoint 缺键: {missing}"
    comp.eval()
    print(f"loaded {'ANCHORED ' if args.anchor else ''}compressor: {count_trainable(comp):,} 参数, "
          f"dim={dim}, K={k_max}, frames={n_frames}, device={device}, ckpt={args.checkpoint}")
    if args.anchor:
        print(f"  alpha = {float(comp.alpha):.6f}")

    packer = compress_and_pack_anchored if args.anchor else compress_and_pack

    def learned_selector(features, newline, content):
        with torch.no_grad():
            return packer(comp, features, newline, content)

    register_learned("learned", learned_selector)
    print(f"registered learned selector; methods={args.methods}")

    M.run(args.manifest, args.output, args.methods, args.limit, None,
          args.max_new_tokens, ratio=args.ratio)


if __name__ == "__main__":
    main()
