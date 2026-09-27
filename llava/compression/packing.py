"""
Token Packing Module

Implements token ordering and packing with frame separators.
"""

import torch
from typing import List, Tuple, Optional, Dict, Any


def get_token_coordinates(
    selected_indices: List[torch.Tensor],
    pooled_height: int,
    pooled_width: int
) -> List[List[Tuple[int, int]]]:
    """
    将token索引转换为2D坐标

    Args:
        selected_indices: 每帧选中的token索引 [T个tensor]
        pooled_height: 池化后的高度
        pooled_width: 池化后的宽度

    Returns:
        coordinates: 每帧的(row, col)坐标列表
    """
    coordinates = []

    for indices in selected_indices:
        coords = []
        for idx in indices:
            idx_val = idx.item()
            row = idx_val // pooled_width
            col = idx_val % pooled_width
            coords.append((row, col))
        coordinates.append(coords)

    return coordinates


def pack_tokens(
    image_features: torch.Tensor,
    selected_indices: List[torch.Tensor],
    image_newline: torch.Tensor,
    pooled_height: int,
    pooled_width: int,
    order: str = "frame_row_col",
    add_separators: bool = False  # 新增参数：是否添加分隔符
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    将选中的token打包成LLM输入序列

    Args:
        image_features: 原始特征 [T, N, D]
        selected_indices: 每帧选中的token索引 [T个tensor]
        image_newline: 帧分隔符 [D]
        pooled_height: 池化后的高度
        pooled_width: 池化后的宽度
        order: 排序方式
            - "frame_row_col": 按帧、行、列排序（默认）
            - "frame_score": 按帧内分数排序（需要额外传入分数）
            - "frame_index": 按帧内原始索引排序

    Returns:
        packed_features: 打包后的特征 [B+T, D]
        metadata: 打包元数据
    """
    T, N, D = image_features.shape
    device = image_features.device
    dtype = image_features.dtype

    packed_tokens = []
    token_metadata = []

    for t in range(T):
        indices = selected_indices[t]  # [n_t]

        if indices.numel() == 0:
            # 空帧，只添加newline
            packed_tokens.append(image_newline.unsqueeze(0))
            token_metadata.append({
                "frame_id": t,
                "token_id": -1,
                "row": -1,
                "col": -1,
                "is_separator": True
            })
            continue

        # 提取特征
        frame_features = image_features[t]  # [N, D]
        selected_features = frame_features[indices]  # [n_t, D]

        # 计算坐标
        rows = indices // pooled_width
        cols = indices % pooled_width

        # 排序
        if order == "frame_row_col":
            # 按row、col排序
            sort_keys = rows * pooled_width + cols
            sorted_order = torch.argsort(sort_keys)
        elif order == "frame_index":
            # 按原始索引排序
            sorted_order = torch.argsort(indices)
        else:
            raise ValueError(f"Unknown packing order: {order}")

        # 应用排序
        selected_features = selected_features[sorted_order]
        indices = indices[sorted_order]
        rows = rows[sorted_order]
        cols = cols[sorted_order]

        # 添加到packed序列
        packed_tokens.append(selected_features)

        # 记录元数据
        for i in range(indices.numel()):
            token_metadata.append({
                "frame_id": t,
                "token_id": indices[i].item(),
                "row": rows[i].item(),
                "col": cols[i].item(),
                "is_separator": False
            })

        # 添加帧分隔符（可选）
        if add_separators:
            packed_tokens.append(image_newline.unsqueeze(0))
            token_metadata.append({
                "frame_id": t,
                "token_id": -1,
                "row": -1,
                "col": -1,
                "is_separator": True
            })

    # 拼接所有token
    packed_features = torch.cat(packed_tokens, dim=0)  # [B, D] or [B+T, D]

    metadata = {
        "total_tokens": packed_features.shape[0],
        "content_tokens": sum(idx.numel() for idx in selected_indices),
        "separator_tokens": T if add_separators else 0,
        "per_frame_count": [idx.numel() for idx in selected_indices],
        "token_metadata": token_metadata
    }

    return packed_features, metadata


def pack_tokens_with_scores(
    image_features: torch.Tensor,
    selected_indices: List[torch.Tensor],
    scores: List[torch.Tensor],
    image_newline: torch.Tensor,
    pooled_height: int,
    pooled_width: int
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    打包token并按分数排序（用于可视化和分析）

    Args:
        image_features: 原始特征 [T, N, D]
        selected_indices: 每帧选中的token索引
        scores: 每帧的token分数
        image_newline: 帧分隔符
        pooled_height: 池化后的高度
        pooled_width: 池化后的宽度

    Returns:
        packed_features: 打包后的特征
        metadata: 包含分数的元数据
    """
    T, N, D = image_features.shape
    device = image_features.device

    packed_tokens = []
    token_metadata = []

    for t in range(T):
        indices = selected_indices[t]
        if indices.numel() == 0:
            packed_tokens.append(image_newline.unsqueeze(0))
            token_metadata.append({
                "frame_id": t,
                "token_id": -1,
                "score": 0.0,
                "is_separator": True
            })
            continue

        frame_features = image_features[t]
        frame_scores = scores[t][indices]  # 选中token的分数
        selected_features = frame_features[indices]

        # 按分数降序排序
        sorted_order = torch.argsort(frame_scores, descending=True)
        selected_features = selected_features[sorted_order]
        indices = indices[sorted_order]
        frame_scores = frame_scores[sorted_order]

        # 计算坐标
        rows = indices // pooled_width
        cols = indices % pooled_width

        packed_tokens.append(selected_features)

        for i in range(indices.numel()):
            token_metadata.append({
                "frame_id": t,
                "token_id": indices[i].item(),
                "row": rows[i].item(),
                "col": cols[i].item(),
                "score": frame_scores[i].item(),
                "is_separator": False
            })

        packed_tokens.append(image_newline.unsqueeze(0))
        token_metadata.append({
            "frame_id": t,
            "token_id": -1,
            "score": 0.0,
            "is_separator": True
        })

    packed_features = torch.cat(packed_tokens, dim=0)

    metadata = {
        "total_tokens": packed_features.shape[0],
        "content_tokens": sum(idx.numel() for idx in selected_indices),
        "separator_tokens": T,
        "token_metadata": token_metadata
    }

    return packed_features, metadata


def unpack_tokens(
    packed_features: torch.Tensor,
    metadata: Dict[str, Any]
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    从打包序列中恢复每帧的token（用于调试）

    Args:
        packed_features: 打包后的特征 [B+T, D]
        metadata: 打包元数据

    Returns:
        frame_features: 每帧的特征列表
        frame_indices: 每帧的token索引列表
    """
    token_metadata = metadata["token_metadata"]
    T = metadata["separator_tokens"]

    frame_features = [[] for _ in range(T)]
    frame_indices = [[] for _ in range(T)]

    for i, meta in enumerate(token_metadata):
        if not meta["is_separator"]:
            frame_id = meta["frame_id"]
            frame_features[frame_id].append(packed_features[i])
            frame_indices[frame_id].append(meta["token_id"])

    # 转换为tensor
    frame_features = [torch.stack(f) if len(f) > 0 else torch.empty(0, packed_features.shape[1])
                      for f in frame_features]
    frame_indices = [torch.tensor(idx, dtype=torch.long) if len(idx) > 0 else torch.empty(0, dtype=torch.long)
                     for idx in frame_indices]

    return frame_features, frame_indices


def create_position_ids(
    packed_features: torch.Tensor,
    text_length: int
) -> torch.Tensor:
    """
    为视觉+文本序列创建position_ids

    Args:
        packed_features: 打包后的视觉特征 [B+T, D]
        text_length: 文本token长度

    Returns:
        position_ids: 位置ID [B+T+text_length]
    """
    visual_length = packed_features.shape[0]
    total_length = visual_length + text_length

    position_ids = torch.arange(total_length, dtype=torch.long, device=packed_features.device)

    return position_ids


def compute_packing_statistics(
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    计算打包统计信息

    Args:
        metadata: 打包元数据

    Returns:
        stats: 统计信息
    """
    token_metadata = metadata["token_metadata"]

    # 统计每帧的token分布
    frame_token_counts = metadata["per_frame_count"]
    content_tokens = metadata["content_tokens"]
    separator_tokens = metadata["separator_tokens"]

    stats = {
        "total_tokens": metadata["total_tokens"],
        "content_tokens": content_tokens,
        "separator_tokens": separator_tokens,
        "compression_ratio": content_tokens / (content_tokens + separator_tokens),
        "tokens_per_frame": {
            "mean": sum(frame_token_counts) / len(frame_token_counts),
            "min": min(frame_token_counts),
            "max": max(frame_token_counts),
            "std": torch.tensor(frame_token_counts, dtype=torch.float32).std().item()
        }
    }

    return stats
