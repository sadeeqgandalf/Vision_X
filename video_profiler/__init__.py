# video_profiler: load/sample -> motion/color -> detection -> aggregation -> report

from .video_io import load_video, sample_frames
from .motion import compute_motion_and_color
from .detection import run_detection
from .aggregation import aggregate
from .report import generate_report

__all__ = [
    "load_video",
    "sample_frames",
    "compute_motion_and_color",
    "run_detection",
    "aggregate",
    "generate_report",
]
