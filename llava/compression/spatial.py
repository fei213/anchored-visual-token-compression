"""
Spatial Rarity Scoring Module

Computes spatial rarity scores based on cluster sizes from spherical k-means.
Implements coverage token selection strategy.
"""

import torch
from typing import Dict, List, Tuple, Any


def compute_spatial_rarity(
    cluster_outputs: List[Dict[str, Any]],
    mode: str = "inverse_sqrt"
) -> List[torch.Tensor]:
    """
    计算每个token的空间稀有度分数

    Args:
        cluster_outputs: 聚类结果列表，每帧一个字典，包含：
            - rep_indices: [K_eff] 代表token索引
            - labels: [N] 每个token的簇指派
            - cluster_stats: 簇统计信息列表
        mode: 稀有度计算模式
            - "inverse_sqrt": 1/sqrt(cluster_size) (默认)
            - "inverse": 1/cluster_size
            - "log": -log(cluster_size/N)

    Returns:
        rarity_scores: 每帧token的稀有度分数列表 [T个tensor, 每个shape=[N]]
    """
    rarity_scores = []

    for frame_idx, out in enumerate(cluster_outputs):
        labels = out["labels"]  # [N]
        cluster_stats = out["cluster_stats"]
        N = labels.shape[0]

        # 构建簇大小映射
        cluster_sizes = torch.tensor(
            [stat["size"] for stat in cluster_stats],
            dtype=torch.float32,
            device=labels.device
        )

        # 计算每个token的稀有度（基于其所属簇的大小）
        token_cluster_sizes = cluster_sizes[labels]  # [N]

        if mode == "inverse_sqrt":
            # rarity = 1/sqrt(size)
            rarity = 1.0 / torch.sqrt(token_cluster_sizes.clamp(min=1.0))
        elif mode == "inverse":
            # rarity = 1/size
            rarity = 1.0 / token_cluster_sizes.clamp(min=1.0)
        elif mode == "log":
            # rarity = -log(size/N)
            rarity = -torch.log((token_cluster_sizes / N).clamp(min=1e-6))
        else:
            raise ValueError(f"Unknown rarity mode: {mode}")

        rarity_scores.append(rarity)

    return rarity_scores


def select_coverage_tokens(
    cluster_outputs: List[Dict[str, Any]],
    m: int = 4
) -> List[torch.Tensor]:
    """
    为每帧选择覆盖token（最大簇的medoid）

    Args:
        cluster_outputs: 聚类结果列表
        m: 每帧覆盖token数量

    Returns:
        coverage_indices: 每帧的覆盖token索引列表 [T个tensor, 每个shape=[m]]
    """
    coverage_indices = []

    for frame_idx, out in enumerate(cluster_outputs):
        rep_indices = out["rep_indices"]  # [K_eff]
        cluster_stats = out["cluster_stats"]
        labels = out["labels"]  # [N]
        N = labels.shape[0]

        # 按簇大小降序排序
        cluster_sizes = [(i, stat["size"]) for i, stat in enumerate(cluster_stats)]
        cluster_sizes.sort(key=lambda x: (-x[1], x[0]))  # 降序，并列按索引

        # 选择前m个非空簇的medoid
        selected = []
        for cluster_id, size in cluster_sizes:
            if size > 0:
                selected.append(rep_indices[cluster_id].item())
            if len(selected) >= m:
                break

        # 如果非空簇不足m个，从未选token中按索引补充
        if len(selected) < m:
            all_tokens = set(range(N))
            selected_set = set(selected)
            remaining = sorted(all_tokens - selected_set)
            selected.extend(remaining[:m - len(selected)])

        coverage_indices.append(
            torch.tensor(selected, dtype=torch.long, device=rep_indices.device)
        )

    return coverage_indices


def get_cluster_medoids(
    cluster_outputs: List[Dict[str, Any]]
) -> List[torch.Tensor]:
    """
    提取每帧的所有簇medoid索引

    Args:
        cluster_outputs: 聚类结果列表

    Returns:
        medoid_indices: 每帧的medoid索引列表 [T个tensor, 每个shape=[K_eff]]
    """
    medoid_indices = []

    for out in cluster_outputs:
        rep_indices = out["rep_indices"]  # [K_eff]

        # 过滤空簇（size=0）
        cluster_stats = out["cluster_stats"]
        non_empty_mask = torch.tensor(
            [stat["size"] > 0 for stat in cluster_stats],
            dtype=torch.bool,
            device=rep_indices.device
        )

        valid_medoids = rep_indices[non_empty_mask]
        medoid_indices.append(valid_medoids)

    return medoid_indices


def compute_cluster_dispersion(
    cluster_outputs: List[Dict[str, Any]]
) -> Dict[str, List[float]]:
    """
    计算聚类质量指标

    Args:
        cluster_outputs: 聚类结果列表

    Returns:
        metrics: 包含各项指标的字典
            - mean_dispersion: 每帧平均离散度
            - p95_dispersion: 每帧95%分位离散度
            - num_singletons: 每帧单例簇数量
            - num_clusters: 每帧实际簇数
    """
    metrics = {
        "mean_dispersion": [],
        "p95_dispersion": [],
        "num_singletons": [],
        "num_clusters": []
    }

    for out in cluster_outputs:
        cluster_stats = out["cluster_stats"]

        mean_disp = sum(s["dispersion_mean"] for s in cluster_stats) / len(cluster_stats)
        p95_disp = sum(s["dispersion_p95"] for s in cluster_stats) / len(cluster_stats)
        num_singletons = sum(1 for s in cluster_stats if s["size"] == 1)
        num_clusters = len([s for s in cluster_stats if s["size"] > 0])

        metrics["mean_dispersion"].append(mean_disp)
        metrics["p95_dispersion"].append(p95_disp)
        metrics["num_singletons"].append(num_singletons)
        metrics["num_clusters"].append(num_clusters)

    return metrics


def get_small_cluster_mask(
    cluster_outputs: List[Dict[str, Any]],
    max_size: int = 2
) -> List[torch.Tensor]:
    """
    获取小簇token的mask（用于消融实验）

    Args:
        cluster_outputs: 聚类结果列表
        max_size: 小簇定义（size <= max_size）

    Returns:
        small_cluster_masks: 每帧的小簇mask [T个tensor, 每个shape=[N]]
    """
    masks = []

    for out in cluster_outputs:
        labels = out["labels"]  # [N]
        cluster_stats = out["cluster_stats"]

        # 构建小簇ID集合
        small_cluster_ids = {
            i for i, stat in enumerate(cluster_stats)
            if stat["size"] <= max_size and stat["size"] > 0
        }

        # 创建mask
        mask = torch.tensor(
            [labels[i].item() in small_cluster_ids for i in range(labels.shape[0])],
            dtype=torch.bool,
            device=labels.device
        )

        masks.append(mask)

    return masks
