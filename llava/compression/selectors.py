"""
Token Selection Methods

Implements all baseline and proposed methods:
- native_full: 原模型全部token
- layout_control: 全部token + 每帧1个newline
- uniform: 均匀采样
- random: 随机采样
- kmeans_medoid: 仅选K个medoid
- rare_only: 仅空间稀有度
- temporal_only: 仅时间新颖度
- st_rare: 完整ST-Rare方法
"""

import torch
from typing import Dict, List, Tuple, Any, Optional

# 延迟导入避免循环依赖
def _import_kmeans():
    from llava.model.kmeans import cluster_tokens_spherical_kmeans
    return cluster_tokens_spherical_kmeans

from .spatial import (
    compute_spatial_rarity,
    select_coverage_tokens,
    get_cluster_medoids
)
from .temporal import (
    compute_temporal_novelty,
    compute_combined_score
)
from .budget import (
    compute_budget,
    check_budget_feasibility,
    select_tokens_global,
    verify_selection
)
from .packing import pack_tokens


class TokenSelector:
    """Token选择器基类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.method = config.get("method", "st_rare")

    def select(
        self,
        image_features: torch.Tensor,
        budget_ratio: float,
        image_newline: torch.Tensor,
        pooled_height: int,
        pooled_width: int
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        选择token并打包

        Args:
            image_features: [T, N, D]
            budget_ratio: 预算比例
            image_newline: 帧分隔符 [D]
            pooled_height: 池化后高度
            pooled_width: 池化后宽度

        Returns:
            packed_features: [B+T, D]
            metadata: 选择元数据
        """
        raise NotImplementedError


class NativeFullSelector(TokenSelector):
    """原模型：不压缩"""

    def select(self, image_features, budget_ratio, image_newline, pooled_height, pooled_width):
        T, N, D = image_features.shape

        # 所有token
        all_indices = [torch.arange(N, device=image_features.device) for _ in range(T)]

        packed_features, metadata = pack_tokens(
            image_features,
            all_indices,
            image_newline,
            pooled_height,
            pooled_width,
            order="frame_row_col"
        , add_separators=False)

        metadata["method"] = "native_full"
        metadata["budget_ratio"] = 1.0

        return packed_features, metadata


class LayoutControlSelector(TokenSelector):
    """Layout Control：全部token + 每帧1个newline"""

    def select(self, image_features, budget_ratio, image_newline, pooled_height, pooled_width):
        # 与native_full相同，主要用于对比不同的newline策略
        return NativeFullSelector(self.config).select(
            image_features, budget_ratio, image_newline, pooled_height, pooled_width
        )


class UniformSelector(TokenSelector):
    """均匀采样"""

    def select(self, image_features, budget_ratio, image_newline, pooled_height, pooled_width):
        T, N, D = image_features.shape
        device = image_features.device

        # 计算预算
        V0 = T * N + T
        V_target, B = compute_budget(V0, budget_ratio, T, rounding="floor")

        # 检查可行性
        feasible, reason = check_budget_feasibility(B, T, N, min_tokens_per_frame=1)
        if not feasible:
            raise ValueError(f"预算不可行: {reason}")

        # 每帧分配
        base = B // T
        remainder = B % T

        selected_indices = []
        for t in range(T):
            budget_t = base + (1 if t < remainder else 0)

            # 均匀采样
            if budget_t >= N:
                indices = torch.arange(N, device=device)
            else:
                # 计算均匀间隔的索引
                step = N / budget_t
                indices = torch.tensor(
                    [int((j + 0.5) * step) for j in range(budget_t)],
                    dtype=torch.long,
                    device=device
                )

            selected_indices.append(indices)

        # 打包
        packed_features, metadata = pack_tokens(
            image_features,
            selected_indices,
            image_newline,
            pooled_height,
            pooled_width,
            order="frame_row_col"
        , add_separators=False)

        metadata["method"] = "uniform"
        metadata["budget_ratio"] = budget_ratio
        metadata["V0"] = V0
        metadata["V_target"] = V_target
        metadata["B"] = B

        return packed_features, metadata


class RandomSelector(TokenSelector):
    """随机采样（固定种子）"""

    def select(self, image_features, budget_ratio, image_newline, pooled_height, pooled_width):
        T, N, D = image_features.shape
        device = image_features.device

        # 计算预算
        V0 = T * N + T
        V_target, B = compute_budget(V0, budget_ratio, T, rounding="floor")

        feasible, reason = check_budget_feasibility(B, T, N, min_tokens_per_frame=1)
        if not feasible:
            raise ValueError(f"预算不可行: {reason}")

        # 每帧分配
        base = B // T
        remainder = B % T

        # 固定种子
        seed = self.config.get("seed", 0)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        selected_indices = []
        for t in range(T):
            budget_t = base + (1 if t < remainder else 0)

            if budget_t >= N:
                indices = torch.arange(N, device=device)
            else:
                # 无放回随机采样
                perm = torch.randperm(N, generator=generator, device=device)
                indices = torch.sort(perm[:budget_t])[0]  # 排序

            selected_indices.append(indices)

        # 打包
        packed_features, metadata = pack_tokens(
            image_features,
            selected_indices,
            image_newline,
            pooled_height,
            pooled_width,
            order="frame_row_col"
        , add_separators=False)

        metadata["method"] = "random"
        metadata["budget_ratio"] = budget_ratio
        metadata["seed"] = seed
        metadata["V0"] = V0
        metadata["V_target"] = V_target
        metadata["B"] = B

        return packed_features, metadata


class KMeansMedoidSelector(TokenSelector):
    """仅选K个medoid"""

    def select(self, image_features, budget_ratio, image_newline, pooled_height, pooled_width):
        cluster_tokens_spherical_kmeans = _import_kmeans()

        T, N, D = image_features.shape
        device = image_features.device

        # 计算预算
        V0 = T * N + T
        V_target, B = compute_budget(V0, budget_ratio, T, rounding="floor")

        feasible, reason = check_budget_feasibility(B, T, N, min_tokens_per_frame=1)
        if not feasible:
            raise ValueError(f"预算不可行: {reason}")

        # 每帧的K值
        base = B // T
        remainder = B % T

        selected_indices = []
        cluster_outputs = []

        for t in range(T):
            K_t = base + (1 if t < remainder else 0)

            # 聚类
            out = cluster_tokens_spherical_kmeans(
                image_features[t],  # [N, D]
                K=min(K_t, N),
                max_iters=self.config.get("kmeans_max_iters", 20),
                tol=self.config.get("kmeans_tolerance", 1e-3),
                seed=self.config.get("seed", 0)
            )

            cluster_outputs.append(out)

            # 选择非空簇的medoid
            rep_indices = out["rep_indices"]
            cluster_stats = out["cluster_stats"]
            non_empty_mask = torch.tensor(
                [stat["size"] > 0 for stat in cluster_stats],
                dtype=torch.bool,
                device=device
            )

            valid_medoids = rep_indices[non_empty_mask]

            # 如果不足K_t，从剩余token补充
            if valid_medoids.numel() < K_t:
                all_tokens = set(range(N))
                selected_set = set(valid_medoids.tolist())
                remaining = sorted(all_tokens - selected_set)
                additional = torch.tensor(
                    remaining[:K_t - valid_medoids.numel()],
                    dtype=torch.long,
                    device=device
                )
                indices = torch.cat([valid_medoids, additional])
            else:
                indices = valid_medoids[:K_t]

            indices = torch.sort(indices)[0]
            selected_indices.append(indices)

        # 打包
        packed_features, metadata = pack_tokens(
            image_features,
            selected_indices,
            image_newline,
            pooled_height,
            pooled_width,
            order="frame_row_col"
        , add_separators=False)

        metadata["method"] = "kmeans_medoid"
        metadata["budget_ratio"] = budget_ratio
        metadata["V0"] = V0
        metadata["V_target"] = V_target
        metadata["B"] = B
        metadata["cluster_outputs"] = cluster_outputs

        return packed_features, metadata


class RareOnlySelector(TokenSelector):
    """仅使用空间稀有度"""

    def select(self, image_features, budget_ratio, image_newline, pooled_height, pooled_width):
        cluster_tokens_spherical_kmeans = _import_kmeans()

        T, N, D = image_features.shape
        device = image_features.device

        # 计算预算
        V0 = T * N + T
        V_target, B = compute_budget(V0, budget_ratio, T, rounding="floor")

        feasible, reason = check_budget_feasibility(B, T, N, min_tokens_per_frame=1)
        if not feasible:
            raise ValueError(f"预算不可行: {reason}")

        # 聚类
        K = self.config.get("kmeans_k", 64)
        cluster_outputs = cluster_tokens_spherical_kmeans(
            image_features,
            K=K,
            max_iters=self.config.get("kmeans_max_iters", 20),
            tol=self.config.get("kmeans_tolerance", 1e-3),
            seed=self.config.get("seed", 0)
        )

        # 计算稀有度
        rarity_scores = compute_spatial_rarity(
            cluster_outputs,
            mode=self.config.get("rarity_mode", "inverse_sqrt")
        )

        # 覆盖token
        m = self.config.get("min_tokens_per_frame", 4)
        coverage_indices = select_coverage_tokens(cluster_outputs, m=m)

        # 全局选择
        selected_indices, selection_metadata = select_tokens_global(
            rarity_scores,
            B, T, m,
            coverage_indices
        )

        # 打包
        packed_features, metadata = pack_tokens(
            image_features,
            selected_indices,
            image_newline,
            pooled_height,
            pooled_width,
            order="frame_row_col"
        , add_separators=False)

        metadata["method"] = "rare_only"
        metadata["budget_ratio"] = budget_ratio
        metadata["V0"] = V0
        metadata["V_target"] = V_target
        metadata["B"] = B
        metadata["selection"] = selection_metadata
        metadata["cluster_outputs"] = cluster_outputs

        return packed_features, metadata


class TemporalOnlySelector(TokenSelector):
    """仅使用时间新颖度"""

    def select(self, image_features, budget_ratio, image_newline, pooled_height, pooled_width):
        cluster_tokens_spherical_kmeans = _import_kmeans()

        T, N, D = image_features.shape
        device = image_features.device

        # 计算预算
        V0 = T * N + T
        V_target, B = compute_budget(V0, budget_ratio, T, rounding="floor")

        feasible, reason = check_budget_feasibility(B, T, N, min_tokens_per_frame=1)
        if not feasible:
            raise ValueError(f"预算不可行: {reason}")

        # 聚类（用于建立记忆）
        K = self.config.get("kmeans_k", 64)
        cluster_outputs = cluster_tokens_spherical_kmeans(
            image_features,
            K=K,
            max_iters=self.config.get("kmeans_max_iters", 20),
            tol=self.config.get("kmeans_tolerance", 1e-3),
            seed=self.config.get("seed", 0)
        )

        # 计算新颖度
        novelty_scores = compute_temporal_novelty(
            image_features,
            cluster_outputs,
            memory_frames=self.config.get("temporal_memory_frames", 10),
            chunk_size=self.config.get("temporal_chunk_size", 256)
        )

        # 覆盖token
        m = self.config.get("min_tokens_per_frame", 4)
        coverage_indices = select_coverage_tokens(cluster_outputs, m=m)

        # 全局选择
        selected_indices, selection_metadata = select_tokens_global(
            novelty_scores,
            B, T, m,
            coverage_indices
        )

        # 打包
        packed_features, metadata = pack_tokens(
            image_features,
            selected_indices,
            image_newline,
            pooled_height,
            pooled_width,
            order="frame_row_col"
        , add_separators=False)

        metadata["method"] = "temporal_only"
        metadata["budget_ratio"] = budget_ratio
        metadata["V0"] = V0
        metadata["V_target"] = V_target
        metadata["B"] = B
        metadata["selection"] = selection_metadata
        metadata["cluster_outputs"] = cluster_outputs

        return packed_features, metadata


class STRareSelector(TokenSelector):
    """完整ST-Rare方法"""

    def select(self, image_features, budget_ratio, image_newline, pooled_height, pooled_width):
        cluster_tokens_spherical_kmeans = _import_kmeans()

        T, N, D = image_features.shape
        device = image_features.device

        # 计算预算
        V0 = T * N + T
        V_target, B = compute_budget(V0, budget_ratio, T, rounding="floor")

        feasible, reason = check_budget_feasibility(B, T, N, min_tokens_per_frame=1)
        if not feasible:
            raise ValueError(f"预算不可行: {reason}")

        # 聚类
        K = self.config.get("kmeans_k", 64)
        cluster_outputs = cluster_tokens_spherical_kmeans(
            image_features,
            K=K,
            max_iters=self.config.get("kmeans_max_iters", 20),
            tol=self.config.get("kmeans_tolerance", 1e-3),
            seed=self.config.get("seed", 0)
        )

        # 计算稀有度
        rarity_scores = compute_spatial_rarity(
            cluster_outputs,
            mode=self.config.get("rarity_mode", "inverse_sqrt")
        )

        # 计算新颖度
        novelty_scores = compute_temporal_novelty(
            image_features,
            cluster_outputs,
            memory_frames=self.config.get("temporal_memory_frames", 10),
            novelty_weight=self.config.get("novelty_weight", 1.0),
            chunk_size=self.config.get("temporal_chunk_size", 256)
        )
        temporal_metadata = {}

        # 综合评分
        combined_scores = compute_combined_score(
            rarity_scores,
            novelty_scores,
            novelty_weight=self.config.get("novelty_weight", 1.0)
        )

        # 覆盖token
        m = self.config.get("min_tokens_per_frame", 4)
        coverage_indices = select_coverage_tokens(cluster_outputs, m=m)

        # 全局选择
        selected_indices, selection_metadata = select_tokens_global(
            combined_scores,
            B, T, m,
            coverage_indices
        )

        # 验证
        valid, report = verify_selection(selected_indices, B, T, N)
        if not valid:
            print(f"警告：选择验证失败: {report}")

        # 打包
        packed_features, metadata = pack_tokens(
            image_features,
            selected_indices,
            image_newline,
            pooled_height,
            pooled_width,
            order="frame_row_col"
        , add_separators=False)

        metadata["method"] = "st_rare"
        metadata["budget_ratio"] = budget_ratio
        metadata["V0"] = V0
        metadata["V_target"] = V_target
        metadata["B"] = B
        metadata["selection"] = selection_metadata
        metadata["verification"] = report
        metadata["cluster_outputs"] = cluster_outputs
        metadata["rarity_scores"] = [r.cpu() for r in rarity_scores]
        metadata["novelty_scores"] = [n.cpu() for n in novelty_scores]
        metadata["combined_scores"] = [c.cpu() for c in combined_scores]

        return packed_features, metadata


class FirstFrameSelector(TokenSelector):
    """仅选择第一帧"""

    def select(self, image_features, budget_ratio, image_newline, pooled_height, pooled_width):
        T, N, D = image_features.shape
        device = image_features.device

        # 计算预算
        V0 = T * N + T
        V_target, B = compute_budget(V0, budget_ratio, T, rounding="floor")

        # 仅选择第一帧的所有token
        selected_indices = [torch.arange(N, device=device)]  # 第一帧全选
        for t in range(1, T):
            selected_indices.append(torch.tensor([], dtype=torch.long, device=device))  # 其他帧不选

        # 打包
        packed_features, metadata = pack_tokens(
            image_features,
            selected_indices,
            image_newline,
            pooled_height,
            pooled_width,
            order="frame_row_col"
        , add_separators=False)

        metadata["method"] = "first_frame"
        metadata["budget_ratio"] = budget_ratio
        metadata["V0"] = V0
        metadata["V_target"] = V_target
        metadata["B"] = N  # 实际只使用了第一帧的N个token

        return packed_features, metadata


class LastFrameSelector(TokenSelector):
    """仅选择最后一帧"""

    def select(self, image_features, budget_ratio, image_newline, pooled_height, pooled_width):
        T, N, D = image_features.shape
        device = image_features.device

        # 计算预算
        V0 = T * N + T
        V_target, B = compute_budget(V0, budget_ratio, T, rounding="floor")

        # 仅选择最后一帧的所有token
        selected_indices = []
        for t in range(T - 1):
            selected_indices.append(torch.tensor([], dtype=torch.long, device=device))  # 前面的帧不选
        selected_indices.append(torch.arange(N, device=device))  # 最后一帧全选

        # 打包
        packed_features, metadata = pack_tokens(
            image_features,
            selected_indices,
            image_newline,
            pooled_height,
            pooled_width,
            order="frame_row_col"
        , add_separators=False)

        metadata["method"] = "last_frame"
        metadata["budget_ratio"] = budget_ratio
        metadata["V0"] = V0
        metadata["V_target"] = V_target
        metadata["B"] = N  # 实际只使用了最后一帧的N个token

        return packed_features, metadata


def create_selector(method: str, config: Dict[str, Any]) -> TokenSelector:
    """工厂函数：创建对应的选择器"""

    selector_map = {
        "native_full": NativeFullSelector,
        "layout_control": LayoutControlSelector,
        "uniform": UniformSelector,
        "random": RandomSelector,
        "kmeans_medoid": KMeansMedoidSelector,
        "kmeans": KMeansMedoidSelector,  # 别名，兼容实验脚本
        "first_frame": FirstFrameSelector,
        "last_frame": LastFrameSelector,
        "rare_only": RareOnlySelector,
        "temporal_only": TemporalOnlySelector,
        "st_rare": STRareSelector
    }

    if method not in selector_map:
        raise ValueError(f"未知方法: {method}. 可选: {list(selector_map.keys())}")

    return selector_map[method](config)
