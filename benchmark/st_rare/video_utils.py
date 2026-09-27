"""Shared video-frame loading helper.

Extracted from the early batch-experiment script so the controlled protocol
(benchmark/st_rare/run_balanced_dev.py) no longer depends on legacy code paths.
"""
from __future__ import annotations

import torch
from PIL import Image
from decord import VideoReader, cpu


def load_video(video_path, max_frames_num: int = 16):
    """Load `max_frames_num` uniformly sampled frames from a video file."""
    try:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        frame_idx = torch.linspace(0, total_frame_num - 1, max_frames_num,
                                   dtype=torch.int64).tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return [Image.fromarray(frame) for frame in spare_frames]
    except Exception as e:                                  # noqa: BLE001
        print(f"video load failed: {video_path} ({type(e).__name__}: {e})")
        return None
