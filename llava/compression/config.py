"""
Configuration Module for ST-Rare Compression

Handles configuration loading and validation.
"""

from typing import Dict, Any, Optional
import yaml
from pathlib import Path


def load_compression_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载compression配置

    Args:
        config_path: 配置文件路径，如果为None则使用默认配置

    Returns:
        配置字典
    """
    default_config = {
        "enabled": False,
        "method": "st_rare",
        "budget_ratio": 0.25,
        "kmeans_k": 64,
        "kmeans_max_iters": 20,
        "kmeans_tolerance": 1e-3,
        "rarity_mode": "inverse_sqrt",
        "temporal_memory_frames": 10,
        "temporal_chunk_size": 256,
        "novelty_weight": 1.0,
        "min_tokens_per_frame": 20,  # 提高从4到20，保证每帧至少10%的token
        "seed": 0,
        "count_tokens": True,
        "save_metadata": False
    }

    if config_path is not None and Path(config_path).exists():
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
            compression_config = yaml_config.get("compression", {})
            default_config.update(compression_config)

    return default_config


def validate_config(config: Dict[str, Any]) -> bool:
    """验证配置有效性"""
    required_fields = ["enabled", "method"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"缺少必需配置项: {field}")

    valid_methods = [
        "native_full", "layout_control", "uniform", "random",
        "kmeans_medoid", "kmeans", "first_frame", "last_frame",
        "rare_only", "temporal_only", "st_rare"
    ]

    if config["method"] not in valid_methods:
        raise ValueError(f"无效方法: {config['method']}, 可选: {valid_methods}")

    if config.get("budget_ratio", 0.25) <= 0 or config.get("budget_ratio", 0.25) > 1.0:
        raise ValueError(f"budget_ratio必须在(0, 1]范围内")

    return True
