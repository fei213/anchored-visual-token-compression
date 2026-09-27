"""
ST-Rare Integration Module for LLaVA

集成compression模块到LLaVA模型，支持在prepare_inputs_labels_for_multimodal中调用。
"""

import torch
from typing import Dict, Any, Optional, Tuple
from llava.compression.config import load_compression_config, validate_config
from llava.compression.selectors import create_selector


class CompressionWrapper:
    """Compression模块的包装器，用于集成到LLaVA"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化compression包装器

        Args:
            config: compression配置，如果为None则使用默认配置
        """
        if config is None:
            config = load_compression_config()

        validate_config(config)
        self.config = config
        self.enabled = config.get("enabled", False)
        self.method = config.get("method", "st_rare")
        self.budget_ratio = config.get("budget_ratio", 0.25)
        self.count_tokens = config.get("count_tokens", True)
        self.save_metadata = config.get("save_metadata", False)

        # 创建selector
        if self.enabled:
            self.selector = create_selector(self.method, config)
        else:
            self.selector = None

        # Token计数器
        self.token_counts = {}

    def process_video_features(
        self,
        image_features: torch.Tensor,
        image_newline: torch.Tensor,
        pooled_height: int,
        pooled_width: int
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        处理视频特征（压缩或原样返回）

        Args:
            image_features: 池化后的特征 [T, N, D]
            image_newline: 帧分隔符 [D]
            pooled_height: 池化后高度
            pooled_width: 池化后宽度

        Returns:
            processed_features: 处理后的特征
            metadata: 元数据
        """
        T, N, D = image_features.shape

        # 记录原始token数
        if self.count_tokens:
            V0 = T * N + T  # 原生序列长度（含separators）
            self.token_counts["n_encoder"] = T * 729  # 假设encoder输出729
            self.token_counts["n_pooled"] = T * N
            self.token_counts["n_native_separators"] = T
            self.token_counts["V0"] = V0

        metadata = {
            "compression_enabled": self.enabled,
            "method": self.method if self.enabled else "native_full",
            "original_shape": (T, N, D)
        }

        if not self.enabled or self.method == "native_full":
            # 禁用压缩或原生模式：按原样返回
            packed_features = []
            for t in range(T):
                packed_features.append(image_features[t])  # [N, D]
                packed_features.append(image_newline.unsqueeze(0))  # [1, D]

            final_features = torch.cat(packed_features, dim=0)  # [T*N + T, D]

            if self.count_tokens:
                self.token_counts["V_output"] = final_features.shape[0]

            metadata.update({
                "V0": T * N + T,
                "V_output": final_features.shape[0],
                "compression_ratio": 1.0
            })

            return final_features, metadata

        # 启用压缩：调用selector
        try:
            packed_features, selector_metadata = self.selector.select(
                image_features,
                self.budget_ratio,
                image_newline,
                pooled_height,
                pooled_width
            )

            # 合并元数据
            metadata.update(selector_metadata)

            if self.count_tokens:
                self.token_counts["V_target"] = selector_metadata.get("V_target", 0)
                self.token_counts["B"] = selector_metadata.get("B", 0)
                self.token_counts["V_output"] = packed_features.shape[0]

            return packed_features, metadata

        except Exception as e:
            print(f"⚠️ Compression失败: {e}")
            print("回退到原生模式...")

            # 回退到原生模式
            packed_features = []
            for t in range(T):
                packed_features.append(image_features[t])
                packed_features.append(image_newline.unsqueeze(0))

            final_features = torch.cat(packed_features, dim=0)

            metadata.update({
                "compression_failed": True,
                "error": str(e),
                "V_output": final_features.shape[0]
            })

            return final_features, metadata

    def get_token_counts(self) -> Dict[str, int]:
        """获取token计数"""
        return self.token_counts.copy()

    def reset_counts(self):
        """重置计数器"""
        self.token_counts = {}


def apply_compression_to_pooled_features(
    pooled_features: torch.Tensor,
    image_newline: torch.Tensor,
    pooled_height: int,
    pooled_width: int,
    compression_config: Optional[Dict[str, Any]] = None
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    对池化后的特征应用compression

    这是一个便捷函数，可以在prepare_inputs_labels_for_multimodal中调用

    Args:
        pooled_features: [T, N, D] 池化后的特征
        image_newline: [D] 帧分隔符
        pooled_height: 池化后高度
        pooled_width: 池化后宽度
        compression_config: compression配置

    Returns:
        processed_features: 处理后的特征 [V_output, D]
        metadata: 元数据
    """
    wrapper = CompressionWrapper(compression_config)
    return wrapper.process_video_features(
        pooled_features,
        image_newline,
        pooled_height,
        pooled_width
    )
