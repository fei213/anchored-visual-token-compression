"""Is the latent bottleneck actually used? (perry's first NO-GO check)

If the compressor collapses -- its K output tokens near-identical, or its outputs
nearly frame-invariant -- then the bottleneck is not being exploited and low
accuracy says nothing about the compression idea. Reported per checkpoint:

  within-frame collapse : mean pairwise cosine among a frame's K tokens
                          (→1 means the K tokens are redundant with each other)
  effective rank        : participation ratio of the singular-value spectrum of Z
                          (K means all K directions are used; 1 means one direction)
  cross-frame variety   : mean cosine between the same query index across frames
                          (→1 means the output ignores temporal position)

CPU-only: runs on cached features, so no GPU is needed.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path = [p for p in sys.path if Path(p or ".").resolve() != ROOT]
sys.path.append(str(ROOT))

from llava.compression.learned import (LatentCompressor, ResidualAnchoredCompressor,
                                       anchors_for, frame_quotas)
from benchmark.st_rare.eval_latent_compressor import load_compressor_state


def pairwise_cosine_metrics(z: np.ndarray) -> dict:
    """z: [T, K, D] -> frame-internal and cross-frame cosine statistics.

    Cast to float32: the visual features arrive as bfloat16/float16, which numpy
    linalg does not support (this crashed the control group on raw tokens).
    """
    z = np.asarray(z, dtype=np.float32)
    t, k, _ = z.shape
    zn = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
    within = [float((zn[i] @ zn[i].T)[np.triu_indices(k, 1)].mean()) for i in range(t)]
    cross = [float((zn[:, j] @ zn[:, j].T)[np.triu_indices(t, 1)].mean()) for j in range(k)]
    return dict(within_frame_cosine=float(np.mean(within)),
                cross_frame_cosine=float(np.mean(cross)), k=k, t=t)


def participation_ratio(z: np.ndarray) -> float:
    """(Σλ)² / Σλ² of the covariance spectrum: 1 = rank-1, d = isotropic."""
    x = np.asarray(z, dtype=np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(x, compute_uv=False)
    lam = s ** 2
    return float(lam.sum() ** 2 / max(1e-12, (lam ** 2).sum()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--feature-cache", type=Path,
                    default=ROOT / "artifacts/st_rare/latent_compressor_features_v1")
    ap.add_argument("--num-files", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--anchor", action="store_true", help="checkpoint 为 ResidualAnchoredCompressor")
    ap.add_argument("--ratio", type=float, default=0.25, help="仅 --anchor 时用于计算每帧配额")
    args = ap.parse_args()

    files = sorted(args.feature_cache.glob("*.pt"))[: args.num_files]
    if not files:
        raise SystemExit(f"没有特征缓存: {args.feature_cache}")
    state = load_compressor_state(args.checkpoint)
    pre = "core." if args.anchor else ""
    if args.anchor and f"{pre}queries" not in state:
        raise SystemExit("state 里没有 core.queries —— 是否忘了 --anchor？")
    dim = state[f"{pre}queries"].shape[1]
    cls = ResidualAnchoredCompressor if args.anchor else LatentCompressor
    mod = cls(dim=dim, num_frames=state[f"{pre}frame_emb"].shape[0],
              max_tokens_per_frame=state[f"{pre}queries"].shape[0], num_heads=8)
    mod.load_state_dict(state)
    mod.eval()
    if args.anchor:
        print(f"  锚定模式: alpha={float(mod.alpha):.6f}")

    per, ctrl = [], []
    with torch.no_grad():
        for f in files:
            x = torch.load(f, map_location="cpu")
            if args.anchor:                          # 度量 LLM 实际看到的 Z（含锚点）
                t_, n_, d_ = x.shape
                quotas = frame_quotas(int(math.floor(t_ * (n_ + 1) * args.ratio) - t_), t_)
                a = anchors_for(x, quotas)
                with torch.no_grad():
                    z = (a.float() + mod.alpha * mod(x)[:, :a.shape[1]]).numpy()
            else:
                z = mod(x).numpy()                   # [T, K, D]
            t, k, _ = z.shape
            rec = pairwise_cosine_metrics(z)
            rec["effective_rank"] = participation_ratio(z.reshape(t * k, -1))
            rec["file"] = f.name
            per.append(rec)
            # CONTROL: the same metrics on the raw visual tokens this frame feeds
            xn = x.numpy()
            c = pairwise_cosine_metrics(xn)
            c["effective_rank"] = participation_ratio(xn.reshape(xn.shape[0] * xn.shape[1], -1))
            ctrl.append(c)

    keys = ("within_frame_cosine", "cross_frame_cosine", "effective_rank")
    agg = {k: float(np.mean([p[k] for p in per])) for k in keys}
    cag = {k: float(np.mean([c[k] for c in ctrl])) for k in keys}
    K = per[0]["k"]
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  样本文件 {len(per)} 个   K={K}（对照的原始 token 数 {ctrl[0]['k']}）  T={per[0]['t']}")
    print(f"  {'指标':<26}{'压缩输出':>12}{'原始token(对照)':>18}")
    print(f"  {'帧内两两余弦':<26}{agg['within_frame_cosine']:>12.4f}{cag['within_frame_cosine']:>18.4f}")
    print(f"  {'跨帧同槽位余弦':<26}{agg['cross_frame_cosine']:>12.4f}{cag['cross_frame_cosine']:>18.4f}")
    print(f"  {'有效秩':<26}{agg['effective_rank']:>12.2f}{cag['effective_rank']:>18.2f}")

    # 保守判读：只在与对照相比"明显更冗余/秩更低"时才提示，避免误报。
    # 绝对阈值需要真实训练后的 checkpoint 才能校准，故此处只给强条件。
    excess = agg["within_frame_cosine"] - cag["within_frame_cosine"]
    rank_ratio = agg["effective_rank"] / max(1e-9, cag["effective_rank"])
    flags = []
    if excess > 0.15:
        flags.append(f"帧内冗余显著高于原始 token（+{excess:.3f}）")
    if rank_ratio < 0.1:
        flags.append(f"有效秩不足对照的 10%（{rank_ratio:.3f}×）")
    verdict = ("疑似 bottleneck 退化: " + "; ".join(flags)) if flags else \
              "未见相对原始 token 的明显退化（绝对阈值待真实 checkpoint 校准）"
    print(f"  判读: {verdict}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(dict(aggregate=agg, control=cag, verdict=verdict,
                                            per_file=per, control_per_file=ctrl),
                                       indent=2, ensure_ascii=False) + "\n")
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
