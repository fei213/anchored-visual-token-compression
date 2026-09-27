#1.聚类，对于聚类操作，还需要修改def get_2dPool(self, image_feature, stride=2):
import torch
from typing import Dict, List, Any, Union

def cluster_tokens_spherical_kmeans(
    image_features: torch.Tensor,
    K: int = 64,
    max_iters: int = 20,
    tol: float = 1e-3,
    seed: int = 0,
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    在归一化特征空间上做 spherical k-means（余弦），选每簇 medoid 作为代表。
    - 不使用空间正则、权重或后处理
    - 需要每簇统计量(size/dispersion等)
    - 若输入为 (B, N, D)，逐帧聚类并返回长度为 B 的结果列表
      输入为 (N, D) 时返回单个结果字典

    返回（单帧时的字典字段，批量时为该字典列表）：
      - rep_indices: LongTensor [K_eff]                 # 各簇 medoid 的原始索引
      - rep_features: Tensor [K_eff, D]                 # 各 medoid 的特征（已 L2 归一化）
      - centroids: Tensor [K_eff, D]                    # 簇中心向量（单位范数）
      - labels: LongTensor [N]                          # 每个 token 的簇指派
      - cluster_stats: List[dict]，每簇包含：
          * size: int
          * dispersion_mean: float                      # 平均(1 - cos)
          * dispersion_p95: float                       # 95% 分位的(1 - cos)
          * intra_cosine_mean: float                    # 平均 cos
          * top3_members: List[int]                     # 按 cos 排序的前 3 个成员索引
    """
    def _l2_normalize(X: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return X / (X.norm(dim=-1, keepdim=True) + eps)

    def _kmeans_plus_plus_init(X: torch.Tensor, K_eff: int, gen: torch.Generator) -> torch.Tensor:
        """
        在单位球面上做 k-means++ 初始化，距离度量使用 1 - cos（等价于最大化与最近中心的角距离）。
        返回初始中心 C: [K_eff, D]（已归一化）
        """
        N, D = X.shape
        device = X.device
        # 首个中心
        first = torch.randint(low=0, high=N, size=(1,), generator=gen, device=device)
        centers_idx = [first.item()]
        C = X[first].clone()  # [1, D]

        while len(centers_idx) < K_eff:
            # 与现有中心的最大相似度
            sim = X @ C.T  # [N, t]
            max_sim, _ = sim.max(dim=1)  # [N]
            # 概率 ∝ 距离（可用 squared，但在球面上直接用 (1 - cos) 也可）
            dist = (1.0 - max_sim).clamp_min(1e-12)
            probs = dist / dist.sum()
            nxt = torch.multinomial(probs, num_samples=1, replacement=False, generator=gen)
            nxt_i = nxt.item()
            centers_idx.append(nxt_i)
            C = torch.cat([C, X[nxt]], dim=0)

        # 保险归一化
        C = _l2_normalize(C)
        return C

    def _quantile(t: torch.Tensor, q: float) -> torch.Tensor:
        # 兼容老版本 PyTorch：使用排序近似分位数
        if hasattr(torch, "quantile"):
            return torch.quantile(t, q)
        k = max(0, min(t.numel() - 1, int((t.numel() * q) - 1)))
        vals, _ = torch.sort(t)
        return vals[k]

    def safe_argsort(x, ascending=True, dim=-1):
        """
        对输入 x 执行 argsort，自动兼容 torch.Tensor / numpy.ndarray。
        PyTorch 用 descending 参数；NumPy 用 flip 实现降序。
        """
        if isinstance(x, torch.Tensor):
            # torch 版本使用 descending 参数
            return torch.argsort(x, dim=dim, descending=not ascending)
        else:
            # 转成 numpy 数组再排序
            arr = np.asarray(x)
            idx = np.argsort(arr, axis=dim)
            if not ascending:
                idx = np.flip(idx, axis=dim)
            return idx

    def _cluster_single_frame(X_in: torch.Tensor) -> Dict[str, Any]:
        """
        对单帧 (N, D) 聚类
        """
        device, dtype = X_in.device, X_in.dtype
        N, D = X_in.shape
        K_eff = min(K, N)

        # L2 归一化（spherical k-means 前置）
        X = _l2_normalize(X_in)

        # 退化场景
        if N == 0:
            return {
                "rep_indices": torch.empty(0, dtype=torch.long, device=device),
                "rep_features": torch.empty(0, D, dtype=dtype, device=device),
                "centroids": torch.empty(0, D, dtype=dtype, device=device),
                "labels": torch.empty(0, dtype=torch.long, device=device),
                "cluster_stats": [],
            }
        if K_eff == 1:
            # 全部归为一簇，medoid 为与均值方向最相似的点
            centroid = _l2_normalize(X.mean(dim=0, keepdim=True))  # [1, D]
            sim = (X @ centroid.T).squeeze(1)                      # [N]
            medoid = int(sim.argmax())
            stats = {
                "size": int(N),
                "dispersion_mean": float((1.0 - sim).mean().item()),
                "dispersion_p95": float(_quantile(1.0 - sim, 0.95).item()),
                "intra_cosine_mean": float(sim.mean().item()),
                "top3_members": torch.topk(sim, min(3, N)).indices.tolist(),
            }
            return {
                "rep_indices": torch.tensor([medoid], dtype=torch.long, device=device),
                "rep_features": X[medoid].unsqueeze(0),
                "centroids": centroid,
                "labels": torch.zeros(N, dtype=torch.long, device=device),
                "cluster_stats": [stats],
            }

        # k-means++ 初始化
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        C = _kmeans_plus_plus_init(X, K_eff, gen)  # [K_eff, D]

        # 迭代优化
        J_prev = None
        for it in range(max_iters):
            sim = X @ C.T                                 # [N, K_eff]
            labels = sim.argmax(dim=1)                    # [N]
            chosen_sim = sim[torch.arange(N, device=device), labels]  # [N]
            J = (1.0 - chosen_sim).sum()                  # 标量：总失真

            # 更新中心
            C_new = torch.zeros_like(C)
            empty_buckets = []
            for k in range(K_eff):
                idx = (labels == k).nonzero(as_tuple=False).squeeze(1)
                if idx.numel() == 0:
                    empty_buckets.append(k)
                    continue
                m = X.index_select(0, idx).mean(dim=0)    # [D]
                C_new[k] = _l2_normalize(m.unsqueeze(0)).squeeze(0)

            # 处理空簇：用“当前与任一中心的最大相似度最小”的点填充
            if empty_buckets:
                max_sim_all, _ = sim.max(dim=1)           # [N]
                # farthest_order = torch.argsort(max_sim_all, ascending=True)
                farthest_order = safe_argsort(max_sim_all, ascending=True)
                fill_ptr = 0
                for k in empty_buckets:
                    # 选尚未用作中心的“最远”样本
                    while fill_ptr < N:
                        cand = farthest_order[fill_ptr]
                        fill_ptr += 1
                        C_new[k] = X[cand]
                        break
                C_new = _l2_normalize(C_new)

            # 早停
            if J_prev is not None:
                rel_impr = ((J_prev - J) / (J_prev + 1e-12)).abs().item()
                C = C_new
                if rel_impr < tol:
                    break
            else:
                C = C_new
            J_prev = J

        # 最终指派与 medoid
        sim = X @ C.T
        labels = sim.argmax(dim=1)
        rep_indices = torch.empty(K_eff, dtype=torch.long, device=device)
        cluster_stats: List[Dict[str, Any]] = []

        for k in range(K_eff):
            idx = (labels == k).nonzero(as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                # 极少见：仍为空簇，fallback 到全局对第 k 个中心最相似的样本
                sims_all = sim[:, k]
                medoid = int(sims_all.argmax())
                rep_indices[k] = medoid
                cluster_stats.append({
                    "size": 0,
                    "dispersion_mean": float("nan"),
                    "dispersion_p95": float("nan"),
                    "intra_cosine_mean": float("nan"),
                    "top3_members": [medoid],
                })
            else:
                sims = sim.index_select(0, idx)[:, k]     # [|c|]
                medoid_local = int(sims.argmax())
                medoid = int(idx[medoid_local])
                rep_indices[k] = medoid

                disp = 1.0 - sims
                p95 = _quantile(disp.float(), 0.95).item()
                topk = min(3, idx.numel())
                top3 = idx[torch.topk(sims, topk).indices].tolist()

                cluster_stats.append({
                    "size": int(idx.numel()),
                    "dispersion_mean": float(disp.mean().item()),
                    "dispersion_p95": float(p95),
                    "intra_cosine_mean": float(sims.mean().item()),
                    "top3_members": [int(t) for t in top3],
                })

        return {
            "rep_indices": rep_indices,
            "rep_features": X_in.index_select(0, rep_indices),  # 原始未归一化特征
            "centroids": C,                                  # 单位范数中心
            "labels": labels,
            "cluster_stats": cluster_stats,
        }

    # --- 入口：单帧或多帧 ---
    if image_features.dim() == 3:
        # (B, N, D)
        B = image_features.size(0)
        return [_cluster_single_frame(image_features[b]) for b in range(B)]
    elif image_features.dim() == 2:
        # (N, D)
        return _cluster_single_frame(image_features)
    else:
        raise ValueError(f"Expected image_features of shape (N, D) or (B, N, D), got {tuple(image_features.shape)}")