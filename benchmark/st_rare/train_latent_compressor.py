"""Train the minimal latent compressor (new phase, perry 2026-09-25 spec).

Frozen: vision encoder, projector, LLM. Trainable: the LatentCompressor only.

Objectives (pick ONE first -- do not stack losses):
  A task   : cross-entropy on the multiple-choice answer
  B logit  : A + lambda * KL(p_full || p_comp)
  C hidden : A + lambda * ||H_full - H_comp||^2   (at --hidden-layer)

Protocol: MVBench Dev for training, MVBench Val for the Go/No-Go read, 25% matched
budget, compared against Uniform on the same items.

`--dry-run` validates the loop mechanics (freezing, losses, optimizer wiring) on
synthetic tensors without loading the model, so the code is testable on CPU.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path = [p for p in sys.path if Path(p or ".").resolve() != ROOT]
sys.path.append(str(ROOT))

from llava.compression.learned import (LatentCompressor, ResidualAnchoredCompressor,
                                       compress_and_pack, compress_and_pack_anchored,
                                       count_trainable, frame_quotas, anchors_for)


def freeze_backbone(model) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()


def objective_loss(objective: str, logits_comp, labels, logits_full=None,
                   hidden_full=None, hidden_comp=None, lam: float = 1.0):
    """Compute the training loss. Teacher tensors are detached (frozen teacher)."""
    loss = F.cross_entropy(logits_comp, labels)
    parts = {"task": float(loss.detach())}
    if objective == "logit":
        if logits_full is None:
            raise ValueError("objective=logit needs logits_full")
        p_full = F.log_softmax(logits_full.detach().float(), dim=-1)
        log_p_comp = F.log_softmax(logits_comp.float(), dim=-1)
        kl = F.kl_div(log_p_comp, p_full, log_target=True, reduction="batchmean")
        loss = loss + lam * kl
        parts["kl"] = float(kl.detach())
    elif objective == "hidden":
        if hidden_full is None or hidden_comp is None:
            raise ValueError("objective=hidden needs hidden_full/hidden_comp")
        mse = F.mse_loss(hidden_comp.float(), hidden_full.detach().float())
        loss = loss + lam * mse
        parts["mse"] = float(mse.detach())
    elif objective != "task":
        raise ValueError(objective)
    return loss, parts


def module_output_for_metrics(module, x, content, anchored: bool):
    """The tensor the LLM actually sees, for health metrics (Z, incl. anchor path)."""
    if not anchored:
        return module(x).float()
    t, n, d = x.shape
    quotas = frame_quotas(content, t)
    a = anchors_for(x, quotas)
    delta = module(x)[:, :a.shape[1]]
    return (a.float() + module.alpha * delta)


def diversity_loss(z: torch.Tensor) -> torch.Tensor:
    """Mean squared off-diagonal cosine between the compressed tokens (perry 2026-09-25).

    z: [T, K, D] -> normalize each token, G = z_hat z_hat^T, penalize off-diagonal
    entries. NOTE: the target is NOT rank ~ K -- raw visual tokens only have an
    effective rank of ~8.6, so pushing K=64 outputs to full rank would manufacture
    meaningless orthogonal noise. The reference is the matched Uniform selection,
    and lambda is the knob.
    """
    zn = F.normalize(z.float(), dim=-1)
    k = zn.shape[1]
    if k < 2:
        return z.new_zeros(())
    g = zn @ zn.transpose(-1, -2)
    off = g - torch.eye(k, device=g.device, dtype=g.dtype)[None]
    return (off ** 2).sum(dim=(-1, -2)).mean() / (k * (k - 1))

def health_metrics(module, feature_files, content, anchored: bool) -> dict:
    """Rank/cosine/norm health of the compressed representation (no LLM needed)."""
    import numpy as np
    from benchmark.st_rare.diagnose_latent_bottleneck import (
        pairwise_cosine_metrics, participation_ratio)
    rows = []
    with torch.no_grad():
        dev = next(module.parameters()).device
        for f in feature_files:
            # the module lives on cuda while training; features load on cpu, so move
            # them -- feeding cpu tensors to cuda params raises the device RuntimeError
            # (same failure class as the earlier D1 crash).
            x = torch.load(f, map_location="cpu").to(dev)
            z = module_output_for_metrics(module, x, content, anchored).float().cpu().numpy()
            t, k, _ = z.shape
            m = pairwise_cosine_metrics(z)
            m["effective_rank"] = participation_ratio(z.reshape(t * k, -1))
            nrm = np.linalg.norm(z.reshape(-1, z.shape[-1]), axis=-1)
            m["norm_mean"], m["norm_std"] = float(nrm.mean()), float(nrm.std())
            rows.append(m)
    keys = ("within_frame_cosine", "cross_frame_cosine", "effective_rank", "norm_mean", "norm_std")
    return {k: float(sum(r[k] for r in rows) / len(rows)) for k in keys}


def health_verdict(h: dict, rank_min: float, cos_max: float) -> tuple[bool, list]:
    bad = []
    if h["effective_rank"] < rank_min:
        bad.append(f"有效秩 {h['effective_rank']:.2f} < {rank_min}")
    if h["within_frame_cosine"] > cos_max:
        bad.append(f"帧内余弦 {h['within_frame_cosine']:.3f} > {cos_max}")
    if h["cross_frame_cosine"] > 0.995:
        bad.append(f"跨帧余弦 {h['cross_frame_cosine']:.4f} > 0.995")
    return (not bad), bad

def dry_run(args) -> None:
    """Validate mechanics on synthetic tensors: freezing, all three objectives."""
    print("=== dry-run（合成张量，验证机制，不加载模型）===")
    dim, T, N, K, n_cls = 64, 4, 196, 64, 5
    comp = LatentCompressor(dim=dim, num_frames=T, max_tokens_per_frame=K, num_heads=4)
    backbone = nn.Sequential(nn.Linear(8, 8))
    freeze_backbone(backbone)
    print(f"  compressor 可训练参数 {count_trainable(comp):,}；"
          f"backbone 可训练 {sum(p.requires_grad for p in backbone.parameters())} 个张量")

    feats = torch.randn(T, N, dim)
    nl = torch.randn(dim)
    content = 40
    packed, meta = compress_and_pack(comp, feats, nl, content)
    assert packed.shape == (content + T, dim)
    print(f"  压缩输出 {tuple(packed.shape)}（= content {content} + {T} 分隔符）✓")

    B = 3
    logits_comp = torch.randn(B, n_cls, requires_grad=True)
    labels = torch.randint(0, n_cls, (B,))
    logits_full = torch.randn(B, n_cls)
    h_full, h_comp = torch.randn(B, 8), torch.randn(B, 8, requires_grad=True)
    for obj in ("task", "logit", "hidden"):
        loss, parts = objective_loss(obj, logits_comp, labels, logits_full, h_full, h_comp, lam=0.5)
        loss.backward(retain_graph=True)
        finite = math.isfinite(float(loss))
        print(f"  objective={obj:<7} loss={float(loss):.4f} 分量={ {k: round(v,4) for k,v in parts.items()} } 有限={finite}")
        assert finite
    # div 接线：镜像主循环同一表达式（loss = loss + lambda*div），
    # 确保 λ>0 的无人值守训练不会因接线错误而白跑
    for _anch, _cls, _mod in ((False, LatentCompressor, comp),):
        _pk = compress_and_pack
        _packed, _meta, _z = _pk(_mod, feats, nl, content, return_z=True)
        _task = torch.tensor(1.0, requires_grad=True)
        _dl = diversity_loss(_z)
        _comb = _task + 0.05 * _dl
        assert torch.isfinite(_comb)
        _comb.backward()
        assert _task.grad is not None and float(_task.grad) == 1.0
        print(f"  div 接线: diversity_loss={float(_dl):.4f}  task+0.05*div={float(_comb):.4f}  有限且可反传 ✓")
    _am_local = ResidualAnchoredCompressor(dim=dim, num_frames=T,
                                           max_tokens_per_frame=K, num_heads=4)
    for _lam in (0.02, 0.05, 0.10):
        _pk = compress_and_pack_anchored
        _packed, _meta, _z = _pk(_am_local, feats, nl, content, return_z=True)
        _dl = diversity_loss(_z)
        assert torch.isfinite(_dl) and float(_dl) > 0
        print(f"  锚定 + λ={_lam}: diversity_loss={float(_dl):.4f}  z 形状={tuple(_z.shape)} ✓")

    # anchor 路径：alpha=0 时输出应与 uniform 锚点一致
    am = ResidualAnchoredCompressor(dim=dim, num_frames=T, max_tokens_per_frame=K, num_heads=4)
    assert float(am.alpha) == 0.0
    out0, meta0 = compress_and_pack_anchored(am, feats, nl, content)
    z0 = module_output_for_metrics(am, feats, content, True)
    import numpy as _np
    print(f"  anchor: alpha={float(am.alpha)} 输出形状={tuple(out0.shape)} 每帧={meta0['per_frame_count'][:3]}")

    # 真实参数梯度（对压缩输出做一次反向）
    comp.zero_grad()
    compress_and_pack(comp, feats, nl, content)[0].float().pow(2).mean().backward()
    n_grad = sum(p.grad is not None for p in comp.parameters())
    assert n_grad == len(list(comp.parameters()))
    print(f"  compressor 全部 {n_grad} 个参数组收到梯度 ✓")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", choices=["task", "logit", "hidden"], default="task")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--hidden-layer", type=int, default=16)
    ap.add_argument("--ratio", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-manifest", type=Path,
                    default=ROOT / "artifacts/st_rare/mvbench_dev_val_final_v1/dev.jsonl")
    ap.add_argument("--eval-manifest", type=Path,
                    default=ROOT / "artifacts/st_rare/mvbench_val_manifest_v1/manifest.jsonl")
    ap.add_argument("--feature-cache", type=Path,
                    default=ROOT / "artifacts/st_rare/latent_compressor_features_v1")
    ap.add_argument("--output", type=Path, default=ROOT / "artifacts/st_rare/latent_compressor_v1")
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-eval", type=int, default=None)
    ap.add_argument("--anchor", action="store_true",
                    help="用 ResidualAnchoredCompressor（Z = A + alpha*C(X)，alpha 初值 0）")
    ap.add_argument("--div-lambda", type=float, default=0.0,
                    help="diversity 正则权重；0 表示关闭（默认，与现行为逐位一致）")
    ap.add_argument("--health-files", type=int, default=3,
                    help="计算健康指标用的缓存特征文件数")
    ap.add_argument("--stop-rank-min", type=float, default=2.0)
    ap.add_argument("--stop-cos-max", type=float, default=0.95)
    ap.add_argument("--no-early-stop", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        dry_run(args)
        return

    if not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用（训练需要 GPU）；可用 --dry-run 只验证机制")

    from benchmark.st_rare.run_balanced_dev import load_pretrained_model
    from benchmark.st_rare.mvbench_strict import load_mvbench_frames, prompt, parse_choice, LETTERS
    from benchmark.st_rare.strict_video_protocol import select_visual
    from llava.mm_utils import process_images

    args.output.mkdir(parents=True, exist_ok=True)
    args.feature_cache.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in args.train_manifest.read_text().splitlines() if l.strip()]
    if args.limit_train:
        rows = rows[: args.limit_train]

    tokenizer, model, processor, _ = load_pretrained_model(
        "lmms-lab/LLaVA-Video-7B-Qwen2", None, "llava_qwen", device_map="auto", attn_implementation="sdpa")
    # 答案字母的 token id：logits 必须按字母取，不能取"词表前 C 个"
    letter_ids = []
    for L in LETTERS:
        enc = tokenizer.encode(L, add_special_tokens=False)
        assert len(enc) == 1, f"字母 {L} 不是单 token: {enc}"
        letter_ids.append(enc[0])
    print(f"答案字母 token id: {dict(zip(LETTERS[:5], letter_ids[:5]))}")
    model.config.image_aspect_ratio = "square"
    model.config.image_grid_pinpoints = None
    freeze_backbone(model)
    # 压缩对象是 projector 之后、2D pool 之后的张量 => dim 为 LLM 的 hidden_size，
    # 而不是视觉塔的 mm_hidden_size（1152）。
    dim = model.config.hidden_size
    cls = ResidualAnchoredCompressor if args.anchor else LatentCompressor
    compressor = cls(dim=dim, num_frames=16, max_tokens_per_frame=64, num_heads=8).to(model.device)
    packer = compress_and_pack_anchored if args.anchor else compress_and_pack
    print(f"{'ANCHORED ' if args.anchor else ''}latent compressor: "
          f"{count_trainable(compressor):,} 可训练参数, dim={dim}"
          + (f", alpha_init={float(compressor.alpha):.4f}" if args.anchor else ""))

    def features_for(row, frames=16):
        key = args.feature_cache / f"{row['task']}__{Path(row['video']).stem}.pt"
        if key.exists():
            return torch.load(key, map_location="cpu")
        px = process_images(load_mvbench_frames(row, frames), processor, model.config)
        if isinstance(px, list):
            px = torch.stack(px)
        tower, proj = model.get_vision_tower(), model.get_model().mm_projector
        with torch.inference_mode():
            v = tower(px.to(model.device, dtype=next(tower.parameters()).dtype))
            feat = model.get_2dPool(proj(v.to(next(proj.parameters()).dtype)))
        torch.save(feat.cpu(), key)
        return feat.cpu()

    def build_inputs(visual, row):
        """Assemble inputs_embeds + attention_mask (mirrors the proven eval path,
        which calls the BASE class generate; the LLaVA wrapper rejects inputs_embeds)."""
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        conv = conv_templates["qwen_1_5"].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt(row))
        conv.append_message(conv.roles[1], None)
        ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").to(model.device)
        at = int((ids == IMAGE_TOKEN_INDEX).nonzero().flatten()[0])
        emb = model.get_model().embed_tokens
        inp = torch.cat([emb(ids[:at]), visual, emb(ids[at + 1:])]).unsqueeze(0)
        mask = torch.ones((1, inp.shape[1]), device=inp.device, dtype=torch.long)
        return inp, mask

    opt = torch.optim.AdamW([p for p in compressor.parameters() if p.requires_grad], lr=args.lr)
    log = []
    for epoch in range(args.epochs):
        compressor.train()
        running = 0.0
        for i, row_ in enumerate(rows, 1):
            feat = features_for(row_).to(model.device)
            t, n, _ = feat.shape
            content = math.floor(t * (n + 1) * args.ratio) - t
            newline = model.model.image_newline
            packed, meta, z_out = packer(compressor, feat, newline, content, return_z=True)
            from transformers import Qwen2ForCausalLM
            inp_comp, mask_comp = build_inputs(packed, row_)
            label = torch.tensor([row_["answer_index"]], device=model.device)
            cand_ids = torch.tensor([letter_ids[i] for i in range(len(row_["candidates"]))],
                                    device=model.device)
            out_comp = Qwen2ForCausalLM.forward(
                model, inputs_embeds=inp_comp, attention_mask=mask_comp, use_cache=False,
                output_hidden_states=(args.objective == "hidden"))
            logits_comp = out_comp.logits[0, -1, cand_ids].unsqueeze(0)
            logits_full = hidden_full = hidden_comp = None
            if args.objective in ("logit", "hidden"):
                with torch.inference_mode():
                    vis_full, _ = select_visual(feat, newline, "controlled_full", args.ratio)
                    inp_full, mask_full = build_inputs(vis_full, row_)
                    out_full = Qwen2ForCausalLM.forward(
                        model, inputs_embeds=inp_full, attention_mask=mask_full, use_cache=False,
                        output_hidden_states=True)
                logits_full = out_full.logits[0, -1, cand_ids].unsqueeze(0)
                if args.objective == "hidden":
                    hidden_full = out_full.hidden_states[args.hidden_layer][0, -1, :].unsqueeze(0)
                    hidden_comp = out_comp.hidden_states[args.hidden_layer][0, -1, :].unsqueeze(0)
            loss, parts = objective_loss(args.objective, logits_comp, label, logits_full,
                                         hidden_full, hidden_comp, args.lam)
            if args.div_lambda > 0:
                dl = diversity_loss(z_out)
                loss = loss + args.div_lambda * dl
                parts["div"] = float(dl.detach())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in compressor.parameters() if p.requires_grad], 1.0)
            opt.step()
            running += float(loss.detach())
            if i % 20 == 0 or i == len(rows):
                print(f"  epoch {epoch+1} {i}/{len(rows)} loss={running/i:.4f} {parts}", flush=True)
        # ---- 健康检查（perry 2026-09-25 门禁：退化 checkpoint 不再跑完 Val）----
        # Restrict health files to TRAINING-domain keys: the feature cache is SHARED with
        # eval runs, so a bare sorted(glob)[:n] can silently pick val-domain files (filename
        # order, not split). Verified 2026-09-26 that the previous selection happened to
        # be the three dev files action_antonym__10773/108710/114567 (nothing sorts before
        # them), so this change is behaviour-preserving for every recorded result.
        train_keys = {f"{r['task']}__{Path(r['video']).stem}" for r in rows}
        hfiles = [q for q in sorted(args.feature_cache.glob("*.pt")) if q.stem in train_keys][: args.health_files]
        if not hfiles:
            raise SystemExit("健康检查需要训练域缓存特征，但 feature_cache 中没有匹配训练清单的文件")
        content_h = math.floor(16 * (196 + 1) * args.ratio) - 16
        health = health_metrics(compressor, hfiles, content_h, args.anchor) if hfiles else {}
        ok, reasons = health_verdict(health, args.stop_rank_min, args.stop_cos_max) if health else (True, [])
        entry = dict(epoch=epoch + 1, mean_loss=running / max(1, len(rows)), health=health,
                     healthy=ok, reasons=reasons)
        log.append(entry)
        print(f"  [health] epoch{epoch+1}: 秩={health.get('effective_rank', float('nan')):.2f} "
              f"帧内余弦={health.get('within_frame_cosine', float('nan')):.3f} "
              f"跨帧余弦={health.get('cross_frame_cosine', float('nan')):.4f} "
              f"范数={health.get('norm_mean', float('nan')):.2f}  "
              f"=> {'健康' if ok else '不健康: ' + '; '.join(reasons)}", flush=True)
        # store only primitives so the checkpoint loads under weights_only=True
        safe_args = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
        torch.save(dict(state_dict=compressor.state_dict(), args=safe_args),
                   args.output / f"compressor_epoch{epoch+1}.pt")
        if health:
            (args.output / "health.json").write_text(json.dumps(log, indent=2, ensure_ascii=False) + "\n")
        if health and not ok and not args.no_early_stop:
            (args.output / "HEALTH_FAILED").write_text("; ".join(reasons) + "\n")
            print(f"  ⛔ early stop：{' ; '.join(reasons)}（不写 HEALTH_OK，故不会跑完整 Val）", flush=True)
            break
        if health and ok:
            (args.output / "HEALTH_OK").write_text(json.dumps(health, ensure_ascii=False) + "\n")
    (args.output / "train_log.json").write_text(json.dumps(log, indent=2) + "\n")
    print(f"训练完成，checkpoint 写入 {args.output}")


if __name__ == "__main__":
    main()
