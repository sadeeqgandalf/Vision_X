# Video I/O: load from path or URL (yt-dlp), uniform frame sampling.

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import cv2
import numpy as np


@dataclass
class VideoMetadata:
    path: str
    frame_count: int
    fps: float
    width: int
    height: int
    duration_sec: float


@dataclass
class SampledFrame:
    index: int
    timestamp_sec: float
    frame: np.ndarray  # BGR, shape (H, W, 3)


def _is_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://") or path.startswith("www.")


def _download_from_url(url: str, out_dir: str | Path | None = None) -> str:
    """Download via yt-dlp. Returns path to saved file."""
    out_dir = Path(out_dir or tempfile.gettempdir())
    out_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(out_dir / "vision_x_video_%(id)s.%(ext)s")
    try:
        subprocess.run(
            [
                "yt-dlp",
                "--no-playlist",
                "-f", "best[ext=mp4]/best",
                "-o", out_template,
                url,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"yt-dlp failed to download {url}: {e.stderr}") from e
    except FileNotFoundError:
        raise RuntimeError(
            "yt-dlp not found. Install with: pip install yt-dlp (or brew install yt-dlp)"
        ) from None

    # Resolve output filename (yt-dlp template)
    candidates = sorted(
        Path(out_dir).glob("vision_x_video_*.mp4"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError(f"No file downloaded from {url} into {out_dir}")
    return str(candidates[0])


def load_video(path_or_url: str, download_dir: str | Path | None = None) -> tuple[cv2.VideoCapture, str]:
    """Open video (path or URL). Returns VideoCapture and resolved path."""
    path = path_or_url.strip()
    if _is_url(path):
        path = _download_from_url(path, download_dir)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Video file not found: {path}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    return cap, path


def get_metadata(cap: cv2.VideoCapture, path: str = "") -> VideoMetadata:
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = frame_count / fps if frame_count else 0.0
    return VideoMetadata(
        path=path,
        frame_count=frame_count,
        fps=fps,
        width=width,
        height=height,
        duration_sec=duration_sec,
    )


def sample_frames(
    cap: cv2.VideoCapture,
    max_frames: int = 120,
    strategy: str = "uniform",
) -> list[SampledFrame]:
    """Sample up to max_frames. strategy='uniform' → evenly spaced indices."""
    meta = get_metadata(cap)
    total = meta.frame_count
    if total == 0:
        return []

    if strategy == "uniform":
        if total <= max_frames:
            indices = list(range(total))
        else:
            indices = np.linspace(0, total - 1, max_frames, dtype=int).tolist()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    result: list[SampledFrame] = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        t_sec = i / meta.fps
        result.append(SampledFrame(index=i, timestamp_sec=t_sec, frame=frame))
    return result


def sample_frames_generator(
    cap: cv2.VideoCapture,
    max_frames: int = 120,
    strategy: str = "uniform",
) -> Generator[SampledFrame, None, None]:
    """Yield sampled frames one at a time (memory-efficient)."""
    meta = get_metadata(cap)
    total = meta.frame_count
    if total == 0:
        return
    if strategy == "uniform":
        indices = (
            np.linspace(0, total - 1, min(max_frames, total), dtype=int).tolist()
            if total > max_frames
            else list(range(total))
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        yield SampledFrame(index=i, timestamp_sec=i / meta.fps, frame=frame)
