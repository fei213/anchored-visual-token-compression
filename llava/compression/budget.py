"""
Budget Allocation and Token Selection Module

Implements budget calculation and token selection strategies.
"""

import torch
from typing import Dict, List, Tuple, Any, Optional


def compute_budget(
    V0: int,
    budget_ratio: float,
    T: int,
    rounding: str = "floor"
) -> Tuple[int, int]:
    """
    计算token预算

    Args:
        V0: 原始LLM视觉序列长度（包含separators）
        budget_ratio: 预算比例（如0.25表示25%）
        T: 帧数
        rounding: 舍入方式 ("floor" | "ceil" | "round")

    Returns:
        V_target: 目标总token数
        B: 内容预算（扣除T个separator后）
    """
    if rounding == "floor":
        V_target = int(budget_ratio * V0)
    elif rounding == "ceil":
        V_target = int(torch.ceil(torch.tensor(budget_ratio * V0)).item())
    elif rounding == "round":
        V_target = int(torch.round(torch.tensor(budget_ratio * V0)).item())
    else:
        raise ValueError(f"Unknown rounding mode: {rounding}")

    # 内容预算 = 总预算 - 每帧1个separator
    B = V_target - T

    return V_target, B


def check_budget_feasibility(
    B: int,
    T: int,
    N: int,
    min_tokens_per_frame: int = 1
) -> Tuple[bool, str]:
    """
    检查预算是否可行

    Args:
        B: 内容预算
        T: 帧数
        N: 每帧原始token数
        min_tokens_per_frame: 每帧最小token数

    Returns:
        feasible: 是否可行
        reason: 不可行的原因
    """
    if B < T * min_tokens_per_frame:
        return False, f"预算B={B}小于最小要求T*{min_tokens_per_frame}={T * min_tokens_per_frame}"

    if B > T * N:
        return False, f"预算B={B}超过原始token总数T*N={T * N}"

    return True, "预算可行"


def allocate_budget_per_frame(
    B: int,
    T: int,
    m: int = 4,
    allocation_mode: str = "balanced"
) -> torch.Tensor:
    """
    将内容预算B分配到T帧

    Args:
        B: 内容预算
        T: 帧数
        m: 每帧覆盖token数（最低配额）
        allocation_mode: 分配模式
            - "balanced": 均匀分配，前B%T帧多1个
            - "coverage_first": 先保证m个覆盖token，剩余全局竞争

    Returns:
        per_frame_budget: 每帧的token配额 [T]
    """
    if allocation_mode == "balanced":
        # 均匀分配
        base = B // T
        remainder = B % T

        per_frame = torch.full((T,), base, dtype=torch.long)
        per_frame[:remainder] += 1  # 前remainder帧多1个

        return per_frame

    elif allocation_mode == "coverage_first":
        # 这种模式下返回None，由select_tokens_global处理
        return None

    else:
        raise ValueError(f"Unknown allocation mode: {allocation_mode}")


def select_tokens_global(
    scores: List[torch.Tensor],
    B: int,
    T: int,
    m: int = 4,
    coverage_indices: Optional[List[torch.Tensor]] = None
) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
    """
    全局竞争选择token（用于coverage_first模式）

    Args:
        scores: 每帧的token分数 [T个tensor, 每个shape=[N]]
        B: 内容预算
        T: 帧数
        m: 每帧覆盖token数
        coverage_indices: 每帧的覆盖token索引（如果已预选）

    Returns:
        selected_indices: 每帧选中的token索引 [T个tensor]
        metadata: 选择元数据
    """
    device = scores[0].device
    N = scores[0].shape[0]

    # 1. 预留覆盖token
    if B >= T * m:
        # 预算充足：先保留覆盖token，剩余全局竞争
        reserved_per_frame = m
        global_budget = B - T * m
    else:
        # 预算紧张：按balanced模式分配
        base = B // T
        remainder = B % T
        per_frame_budget = torch.full((T,), base, dtype=torch.long)
        per_frame_budget[:remainder] += 1

        # 每帧独立选择
        selected_indices = []
        for t in range(T):
            budget_t = per_frame_budget[t].item()
            score_t = scores[t]

            if coverage_indices is not None and coverage_indices[t].numel() > 0:
                # 从覆盖token中选
                cov_idx = coverage_indices[t][:budget_t]
            else:
                # 按分数选
                _, top_indices = torch.topk(score_t, k=min(budget_t, N))
                cov_idx = top_indices

            selected_indices.append(cov_idx)

        metadata = {
            "mode": "balanced_allocation",
            "per_frame_count": [idx.numel() for idx in selected_indices]
        }
        return selected_indices, metadata

    # 2. 全局竞争模式
    selected_indices = []
    selected_set_per_frame = [set() for _ in range(T)]

    # 先添加覆盖token
    for t in range(T):
        if coverage_indices is not None and coverage_indices[t].numel() > 0:
            cov_idx = coverage_indices[t][:m]
            selected_set_per_frame[t].update(cov_idx.tolist())

    # 收集剩余候选token及其分数
    candidates = []  # (frame_id, token_id, score)
    for t in range(T):
        for i in range(N):
            if i not in selected_set_per_frame[t]:
                candidates.append((t, i, scores[t][i].item()))

    # 按分数降序排序，并列按(frame_id, token_id)升序
    candidates.sort(key=lambda x: (-x[2], x[0], x[1]))

    # 选择top-global_budget
    for i in range(min(global_budget, len(candidates))):
        t, token_id, _ = candidates[i]
        selected_set_per_frame[t].add(token_id)

    # 转换为tensor
    for t in range(T):
        indices = sorted(selected_set_per_frame[t])
        selected_indices.append(torch.tensor(indices, dtype=torch.long, device=device))

    metadata = {
        "mode": "coverage_first_global",
        "coverage_per_frame": m,
        "global_budget": global_budget,
        "per_frame_count": [idx.numel() for idx in selected_indices]
    }

    return selected_indices, metadata


def select_tokens_per_frame(
    scores: List[torch.Tensor],
    per_frame_budget: torch.Tensor,
    coverage_indices: Optional[List[torch.Tensor]] = None
) -> List[torch.Tensor]:
    """
    每帧独立选择token（用于balanced模式）

    Args:
        scores: 每帧的token分数 [T个tensor, 每个shape=[N]]
        per_frame_budget: 每帧的token配额 [T]
        coverage_indices: 每帧的覆盖token索引（优先选择）

    Returns:
        selected_indices: 每帧选中的token索引 [T个tensor]
    """
    T = len(scores)
    selected_indices = []

    for t in range(T):
        budget_t = per_frame_budget[t].item()
        score_t = scores[t]
        N = score_t.shape[0]

        if coverage_indices is not None and coverage_indices[t].numel() > 0:
            # 优先使用覆盖token
            cov_idx = coverage_indices[t]
            if cov_idx.numel() >= budget_t:
                # 覆盖token足够，按分数选前budget_t个
                cov_scores = score_t[cov_idx]
                _, top_k = torch.topk(cov_scores, k=budget_t)
                selected = cov_idx[top_k]
            else:
                # 覆盖token不足，补充其他高分token
                selected_set = set(cov_idx.tolist())
                remaining_budget = budget_t - cov_idx.numel()

                # 从非覆盖token中选
                non_cov_mask = torch.ones(N, dtype=torch.bool, device=score_t.device)
                non_cov_mask[cov_idx] = False
                non_cov_scores = score_t.clone()
                non_cov_scores[~non_cov_mask] = -float('inf')

                _, top_indices = torch.topk(non_cov_scores, k=min(remaining_budget, N - cov_idx.numel()))
                selected = torch.cat([cov_idx, top_indices])
        else:
            # 直接按分数选top-k
            _, top_indices = torch.topk(score_t, k=min(budget_t, N))
            selected = top_indices

        # 排序（按token_id升序）
        selected = torch.sort(selected)[0]
        selected_indices.append(selected)

    return selected_indices


def verify_selection(
    selected_indices: List[torch.Tensor],
    B: int,
    T: int,
    N: int
) -> Tuple[bool, Dict[str, Any]]:
    """
    验证选择结果

    Args:
        selected_indices: 每帧选中的token索引
        B: 内容预算
        T: 帧数
        N: 每帧原始token数

    Returns:
        valid: 是否有效
        report: 验证报告
    """
    total_selected = sum(idx.numel() for idx in selected_indices)
    report = {
        "total_selected": total_selected,
        "target_budget": B,
        "budget_match": total_selected == B,
        "per_frame_count": [idx.numel() for idx in selected_indices],
        "errors": []
    }

    # 检查总数
    if total_selected != B:
        report["errors"].append(f"总选择数{total_selected}不等于预算{B}")

    # 检查每帧
    for t, indices in enumerate(selected_indices):
        # 检查范围
        if indices.numel() > 0:
            if indices.min() < 0 or indices.max() >= N:
                report["errors"].append(f"帧{t}索引越界: min={indices.min()}, max={indices.max()}, N={N}")

            # 检查重复
            if indices.numel() != torch.unique(indices).numel():
                report["errors"].append(f"帧{t}存在重复索引")

    valid = len(report["errors"]) == 0
    return valid, report
