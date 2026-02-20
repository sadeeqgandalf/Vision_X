# Motion: frame-diff magnitude (optionally downscaled). Color: dominant RGB via histogram bins.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np


@dataclass
class FrameStats:
    frame_index: int
    timestamp_sec: float
    motion_magnitude: float  # scalar, 0 = no motion
    dominant_color_rgb: tuple[int, int, int]  # single dominant color
    palette_rgb: list[tuple[int, int, int]]  # up to N dominant colors


def _frame_diff_magnitude(prev: np.ndarray, curr: np.ndarray, max_size: int = 320) -> float:
    """Mean absolute difference (grayscale, optional downscale). Returns [0, 255] scalar."""
    if prev.shape != curr.shape:
        curr = cv2.resize(curr, (prev.shape[1], prev.shape[0]), interpolation=cv2.INTER_LINEAR)
    h, w = prev.shape[:2]
    if max_size and (h > max_size or w > max_size):
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        prev_small = cv2.resize(prev, (new_w, new_h))
        curr_small = cv2.resize(curr, (new_w, new_h))
    else:
        prev_small, curr_small = prev, curr

    if len(prev_small.shape) == 3:
        prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray, curr_gray = prev_small, curr_small

    diff = np.abs(prev_gray.astype(np.float32) - curr_gray.astype(np.float32))
    return float(np.mean(diff))


def _dominant_colors(
    frame: np.ndarray,
    n_colors: int = 5,
    downscale: int = 4,
) -> list[tuple[int, int, int]]:
    """Dominant RGB via 8³ histogram bins. Input BGR. Returns list of (R,G,B)."""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // downscale, h // downscale))
    pixels = small.reshape(-1, 3)
    # 8 bins per channel -> 512 bins; top n by count
    bins = 8
    r = (pixels[:, 2] * (bins - 1) / 256).astype(np.int32)
    g = (pixels[:, 1] * (bins - 1) / 256).astype(np.int32)
    b = (pixels[:, 0] * (bins - 1) / 256).astype(np.int32)
    bin_idx = r * bins * bins + g * bins + b
    counts = np.bincount(bin_idx, minlength=bins ** 3)
    top_bins = np.argsort(counts)[::-1][:n_colors]
    colors = []
    for idx in top_bins:
        rr = (idx // (bins * bins)) * (256 // bins) + (256 // bins) // 2
        gg = ((idx // bins) % bins) * (256 // bins) + (256 // bins) // 2
        bb = (idx % bins) * (256 // bins) + (256 // bins) // 2
        colors.append((int(rr), int(gg), int(bb)))
    return colors


def compute_motion_and_color(
    frames: list[Any],
    get_frame: Callable[[Any], np.ndarray] | None = None,
    get_index: Callable[[Any], int] | None = None,
    get_timestamp: Callable[[Any], float] | None = None,
    motion_downscale: int = 320,
    n_palette: int = 5,
) -> list[FrameStats]:
    """Per-frame motion and dominant colors. Objects need .frame, .index, .timestamp_sec or custom get_*."""
    if get_frame is None:
        get_frame = lambda o: o.frame
    if get_index is None:
        get_index = lambda o: o.index
    if get_timestamp is None:
        get_timestamp = lambda o: o.timestamp_sec

    prev = None
    result: list[FrameStats] = []
    for i, obj in enumerate(frames):
        frame = get_frame(obj)
        idx = get_index(obj)
        ts = get_timestamp(obj)
        motion = 0.0
        if prev is not None:
            motion = _frame_diff_magnitude(prev, frame, max_size=motion_downscale)
        palette = _dominant_colors(frame, n_colors=n_palette)
        dominant = palette[0] if palette else (0, 0, 0)
        result.append(
            FrameStats(
                frame_index=idx,
                timestamp_sec=ts,
                motion_magnitude=motion,
                dominant_color_rgb=dominant,
                palette_rgb=palette,
            )
        )
        prev = frame.copy()
    return result
