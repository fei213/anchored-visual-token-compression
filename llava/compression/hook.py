"""
ST-Rare Compression Hook for LLaVA

提供简单的集成接口，在llava_arch.py中调用
"""

import torch
from typing import Dict, Any, Optional, Tuple


def apply_st_rare_compression(
    pooled_features: torch.Tensor,
    image_newline: torch.Tensor,
    compression_config: Optional[Dict[str, Any]] = None
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    对池化后的特征应用ST-Rare压缩

    这是一个简化的接口，可以直接在llava_arch.py中调用

    Args:
        pooled_features: [T, N, D] 池化后的视频特征
        image_newline: [D] 帧分隔符token
        compression_config: 配置字典，如果为None则使用默认配置
            {
                "enabled": True/False,
                "method": "st_rare" | "uniform" | "random" | ...,
                "budget_ratio": 0.25,
                "kmeans_k": 64,
                "novelty_weight": 1.0,
                "seed": 0
            }

    Returns:
        processed_features: [V_output, D] 处理后的特征序列
        metadata: 元数据字典
    """
    # 延迟导入，避免循环依赖
    from llava.compression.integration import apply_compression_to_pooled_features

    T, N, D = pooled_features.shape

    # 计算池化后的网格尺寸
    pooled_height = int(N ** 0.5)
    pooled_width = pooled_height

    if pooled_height * pooled_width != N:
        # 如果不是完全平方数，尝试其他分解
        import math
        pooled_height = int(math.sqrt(N))
        pooled_width = N // pooled_height
        if pooled_height * pooled_width != N:
            raise ValueError(f"无法将N={N}分解为height×width")

    # 调用compression
    processed_features, metadata = apply_compression_to_pooled_features(
        pooled_features,
        image_newline,
        pooled_height,
        pooled_width,
        compression_config
    )

    return processed_features, metadata


def get_default_compression_config() -> Dict[str, Any]:
    """获取默认compression配置"""
    return {
        "enabled": False,  # 默认禁用
        "method": "st_rare",
        "budget_ratio": 0.25,
        "kmeans_k": 64,
        "kmeans_max_iters": 20,
        "kmeans_tolerance": 1e-3,
        "rarity_mode": "inverse_sqrt",
        "temporal_memory_frames": 10,
        "temporal_chunk_size": 256,
        "novelty_weight": 1.0,
        "min_tokens_per_frame": 4,
        "seed": 0,
        "count_tokens": True,
        "save_metadata": False
    }


def quick_compress(
    pooled_features: torch.Tensor,
    image_newline: torch.Tensor,
    method: str = "st_rare",
    budget_ratio: float = 0.25
) -> torch.Tensor:
    """
    快速压缩接口（不返回元数据）

    Args:
        pooled_features: [T, N, D]
        image_newline: [D]
        method: 压缩方法
        budget_ratio: 预算比例

    Returns:
        processed_features: [V_output, D]
    """
    config = {
        "enabled": True,
        "method": method,
        "budget_ratio": budget_ratio
    }

    processed, _ = apply_st_rare_compression(
        pooled_features,
        image_newline,
        config
    )

    return processed
