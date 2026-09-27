"""
Temporal Novelty Scoring Module

Computes temporal novelty scores based on cosine similarity with previous frames' medoids.
Implements sliding window memory mechanism.
"""

import torch
from typing import Dict, List, Any, Optional


def compute_temporal_novelty(
    image_features: torch.Tensor,
    cluster_outputs: List[Dict[str, Any]],
    memory_frames: int = 10,
    novelty_weight: float = 1.0,
    chunk_size: int = 256,
    reset_per_video: bool = True
) -> List[torch.Tensor]:
    """
    计算每个token的时间新颖度分数

    Args:
        image_features: 原始特征 [T, N, D]
        cluster_outputs: 聚类结果列表
        memory_frames: 记忆窗口大小（前W帧）
        novelty_weight: 新颖度权重（用于综合评分）
        chunk_size: 相似度计算时的批次大小（控制显存）
        reset_per_video: 是否每个视频重置记忆

    Returns:
        novelty_scores: 每帧token的新颖度分数 [T个tensor, 每个shape=[N]]
    """
    T, N, D = image_features.shape
    device = image_features.device
    dtype = image_features.dtype

    novelty_scores = []
    memory_medoids = []  # 存储前W帧的medoid特征

    for t in range(T):
        current_features = image_features[t]  # [N, D]

        if t == 0 or len(memory_medoids) == 0:
            # 第一帧没有历史，novelty=0
            novelty = torch.zeros(N, dtype=torch.float32, device=device)
        else:
            # 将记忆中的medoid拼接
            memory_tensor = torch.cat(memory_medoids, dim=0)  # [M, D], M为记忆中的medoid总数

            # 分块计算相似度（避免显存溢出）
            max_cosines = []
            for i in range(0, N, chunk_size):
                end_i = min(i + chunk_size, N)
                chunk = current_features[i:end_i]  # [chunk_size, D]

                # 归一化并计算余弦相似度
                chunk_norm = torch.nn.functional.normalize(chunk, p=2, dim=-1)
                memory_norm = torch.nn.functional.normalize(memory_tensor, p=2, dim=-1)

                sim = torch.mm(chunk_norm, memory_norm.T)  # [chunk_size, M]
                max_sim, _ = sim.max(dim=-1)  # [chunk_size]
                max_cosines.append(max_sim)

            max_cosine = torch.cat(max_cosines, dim=0)  # [N]

            # novelty = 1 - max_cosine, 并clamp到[0, 2]
            novelty = torch.clamp(1.0 - max_cosine, min=0.0, max=2.0)

        novelty_scores.append(novelty)

        # 更新记忆：提取当前帧的medoid
        rep_indices = cluster_outputs[t]["rep_indices"]  # [K_eff]
        cluster_stats = cluster_outputs[t]["cluster_stats"]

        # 只保留非空簇的medoid
        non_empty_mask = torch.tensor(
            [stat["size"] > 0 for stat in cluster_stats],
            dtype=torch.bool,
            device=device
        )
        valid_medoid_indices = rep_indices[non_empty_mask]

        if valid_medoid_indices.numel() > 0:
            current_medoids = current_features[valid_medoid_indices]  # [K_valid, D]
            memory_medoids.append(current_medoids)

            # 保持窗口大小：只保留最近W帧
            if len(memory_medoids) > memory_frames:
                memory_medoids.pop(0)

    return novelty_scores


def compute_combined_score(
    rarity_scores: List[torch.Tensor],
    novelty_scores: List[torch.Tensor],
    novelty_weight: float = 1.0
) -> List[torch.Tensor]:
    """
    计算综合评分：score = rarity * (1 + w * novelty)

    Args:
        rarity_scores: 空间稀有度分数 [T个tensor]
        novelty_scores: 时间新颖度分数 [T个tensor]
        novelty_weight: 新颖度权重

    Returns:
        combined_scores: 综合分数 [T个tensor]
    """
    assert len(rarity_scores) == len(novelty_scores), "rarity和novelty长度必须相同"

    combined_scores = []
    for rarity, novelty in zip(rarity_scores, novelty_scores):
        score = rarity * (1.0 + novelty_weight * novelty)
        combined_scores.append(score)

    return combined_scores


def get_memory_statistics(
    image_features: torch.Tensor,
    cluster_outputs: List[Dict[str, Any]],
    memory_frames: int = 10
) -> Dict[str, List[float]]:
    """
    计算记忆相关统计信息（用于分析）

    Args:
        image_features: 原始特征 [T, N, D]
        cluster_outputs: 聚类结果列表
        memory_frames: 记忆窗口大小

    Returns:
        stats: 统计信息字典
            - memory_size: 每帧记忆中的token数量
            - avg_max_similarity: 平均最大相似度
            - novelty_mean: 平均新颖度
    """
    T, N, D = image_features.shape
    device = image_features.device

    stats = {
        "memory_size": [],
        "avg_max_similarity": [],
        "novelty_mean": []
    }

    memory_medoids = []

    for t in range(T):
        current_features = image_features[t]

        if t == 0:
            stats["memory_size"].append(0)
            stats["avg_max_similarity"].append(0.0)
            stats["novelty_mean"].append(0.0)
        else:
            memory_tensor = torch.cat(memory_medoids, dim=0)
            stats["memory_size"].append(memory_tensor.shape[0])

            # 计算平均最大相似度
            current_norm = torch.nn.functional.normalize(current_features, p=2, dim=-1)
            memory_norm = torch.nn.functional.normalize(memory_tensor, p=2, dim=-1)
            sim = torch.mm(current_norm, memory_norm.T)
            max_sim, _ = sim.max(dim=-1)

            stats["avg_max_similarity"].append(max_sim.mean().item())
            stats["novelty_mean"].append((1.0 - max_sim).mean().item())

        # 更新记忆
        rep_indices = cluster_outputs[t]["rep_indices"]
        cluster_stats = cluster_outputs[t]["cluster_stats"]
        non_empty_mask = torch.tensor(
            [stat["size"] > 0 for stat in cluster_stats],
            dtype=torch.bool,
            device=device
        )
        valid_medoid_indices = rep_indices[non_empty_mask]

        if valid_medoid_indices.numel() > 0:
            current_medoids = current_features[valid_medoid_indices]
            memory_medoids.append(current_medoids)
            if len(memory_medoids) > memory_frames:
                memory_medoids.pop(0)

    return stats


def compute_frame_level_novelty(
    image_features: torch.Tensor,
    cluster_outputs: List[Dict[str, Any]],
    memory_frames: int = 10
) -> torch.Tensor:
    """
    计算帧级别的新颖度（用于帧选择策略）

    Args:
        image_features: 原始特征 [T, N, D]
        cluster_outputs: 聚类结果列表
        memory_frames: 记忆窗口大小

    Returns:
        frame_novelty: 每帧的新颖度分数 [T]
    """
    T = image_features.shape[0]
    device = image_features.device

    frame_novelty = torch.zeros(T, dtype=torch.float32, device=device)
    memory_medoids = []

    for t in range(T):
        if t == 0:
            frame_novelty[t] = 0.0
        else:
            # 当前帧的medoid
            rep_indices = cluster_outputs[t]["rep_indices"]
            cluster_stats = cluster_outputs[t]["cluster_stats"]
            non_empty_mask = torch.tensor(
                [stat["size"] > 0 for stat in cluster_stats],
                dtype=torch.bool,
                device=device
            )
            valid_medoid_indices = rep_indices[non_empty_mask]

            if valid_medoid_indices.numel() > 0:
                current_medoids = image_features[t][valid_medoid_indices]

                # 与记忆对比
                memory_tensor = torch.cat(memory_medoids, dim=0)
                current_norm = torch.nn.functional.normalize(current_medoids, p=2, dim=-1)
                memory_norm = torch.nn.functional.normalize(memory_tensor, p=2, dim=-1)
                sim = torch.mm(current_norm, memory_norm.T)
                max_sim, _ = sim.max(dim=-1)

                # 帧级新颖度 = 平均token新颖度
                frame_novelty[t] = (1.0 - max_sim).mean()

        # 更新记忆
        rep_indices = cluster_outputs[t]["rep_indices"]
        cluster_stats = cluster_outputs[t]["cluster_stats"]
        non_empty_mask = torch.tensor(
            [stat["size"] > 0 for stat in cluster_stats],
            dtype=torch.bool,
            device=device
        )
        valid_medoid_indices = rep_indices[non_empty_mask]

        if valid_medoid_indices.numel() > 0:
            current_medoids = image_features[t][valid_medoid_indices]
            memory_medoids.append(current_medoids)
            if len(memory_medoids) > memory_frames:
                memory_medoids.pop(0)

    return frame_novelty
