# Aggregate per-frame stats into fixed time segments and global summary.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .motion import FrameStats
from .detection import FrameDetections


@dataclass
class SegmentSummary:
    start_sec: float
    end_sec: float
    mean_motion: float
    max_motion: float
    dominant_colors_rgb: list[tuple[int, int, int]]
    object_counts: dict[str, int]  # class_name -> count
    person_count_max: int
    frame_count: int


@dataclass
class AggregatedProfile:
    duration_sec: float
    total_frames_analyzed: int
    segments: list[SegmentSummary]
    global_mean_motion: float
    global_max_motion: float
    global_object_counts: dict[str, int]
    global_dominant_colors_rgb: list[tuple[int, int, int]]
    timestamps_sec: list[float]
    motion_per_frame: list[float]
    person_count_per_frame: list[int]
    class_counts_per_frame: list[dict[str, int]] = field(default_factory=list)


def _segment_by_fixed_windows(
    n_frames: int,
    duration_sec: float,
    window_sec: float,
) -> list[tuple[float, float]]:
    """(start_sec, end_sec) for each fixed window. The last window is end-inclusive."""
    if n_frames <= 0:
        return []
    if duration_sec <= 0 or window_sec <= 0:
        # e.g. a single analyzed frame: one zero-length segment so it is not dropped
        return [(0.0, max(duration_sec, 0.0))]
    segments = []
    t = 0.0
    while t < duration_sec:
        end = min(t + window_sec, duration_sec)
        segments.append((t, end))
        t = end
    return segments


def _aggregate_colors(color_lists: list[list[tuple[int, int, int]]], top_k: int = 5) -> list[tuple[int, int, int]]:
    """Most frequent dominant color per frame, binned to 32 levels."""
    if not color_lists:
        return []
    firsts = [lst[0] if lst else (0, 0, 0) for lst in color_lists]
    from collections import Counter
    binned = [tuple(c // 32 for c in fc) for fc in firsts]
    cnt = Counter(binned)
    top = cnt.most_common(top_k)
    return [tuple((b * 32 + 16 for b in bc)) for bc, _ in top]


def aggregate(
    motion_stats: list[FrameStats],
    detections: list[FrameDetections],
    segment_window_sec: float = 10.0,
) -> AggregatedProfile:
    """Segment by fixed windows; per-segment and global motion, counts, palette. Inputs same order as frames."""
    if not motion_stats or not detections:
        return AggregatedProfile(
            duration_sec=0.0,
            total_frames_analyzed=0,
            segments=[],
            global_mean_motion=0.0,
            global_max_motion=0.0,
            global_object_counts={},
            global_dominant_colors_rgb=[],
            timestamps_sec=[],
            motion_per_frame=[],
            person_count_per_frame=[],
        )
    n = len(motion_stats)
    duration_sec = motion_stats[-1].timestamp_sec - motion_stats[0].timestamp_sec if n > 1 else 0.0
    if n > 1 and duration_sec <= 0:
        duration_sec = (motion_stats[-1].timestamp_sec + 1.0 / 30.0) - motion_stats[0].timestamp_sec
    timestamps = [m.timestamp_sec for m in motion_stats]
    motion_per_frame = [m.motion_magnitude for m in motion_stats]
    person_per_frame = []
    class_per_frame = []
    for fd in detections:
        obj: dict[str, int] = {}
        for d in fd.detections:
            obj[d.class_name] = obj.get(d.class_name, 0) + 1
        class_per_frame.append(obj)
        person_per_frame.append(obj.get("person", 0))
    global_obj = {}
    for d in detections:
        for det in d.detections:
            global_obj[det.class_name] = global_obj.get(det.class_name, 0) + 1

    global_mean_motion = sum(motion_per_frame) / n if n else 0.0
    global_max_motion = max(motion_per_frame) if motion_per_frame else 0.0
    global_colors = _aggregate_colors([m.palette_rgb for m in motion_stats], top_k=5)

    segments = _segment_by_fixed_windows(n, duration_sec, segment_window_sec)
    segment_summaries: list[SegmentSummary] = []
    for seg_i, (start_sec, end_sec) in enumerate(segments):
        is_last = seg_i == len(segments) - 1
        indices = [
            i for i in range(n)
            if start_sec <= timestamps[i] < end_sec or (is_last and timestamps[i] == end_sec)
        ]
        if not indices:
            continue
        seg_motions = [motion_per_frame[i] for i in indices]
        seg_colors = [motion_stats[i].palette_rgb for i in indices]
        seg_dets = [detections[i] for i in indices]
        obj_cnt = {}
        person_max = 0
        for fd in seg_dets:
            for d in fd.detections:
                obj_cnt[d.class_name] = obj_cnt.get(d.class_name, 0) + 1
            p = sum(1 for d in fd.detections if d.class_name == "person")
            person_max = max(person_max, p)
        segment_summaries.append(
            SegmentSummary(
                start_sec=start_sec,
                end_sec=end_sec,
                mean_motion=sum(seg_motions) / len(seg_motions) if seg_motions else 0.0,
                max_motion=max(seg_motions) if seg_motions else 0.0,
                dominant_colors_rgb=_aggregate_colors(seg_colors, top_k=3),
                object_counts=obj_cnt,
                person_count_max=person_max,
                frame_count=len(indices),
            )
        )

    return AggregatedProfile(
        duration_sec=duration_sec,
        total_frames_analyzed=n,
        segments=segment_summaries,
        global_mean_motion=global_mean_motion,
        global_max_motion=global_max_motion,
        global_object_counts=global_obj,
        global_dominant_colors_rgb=global_colors,
        timestamps_sec=timestamps,
        motion_per_frame=motion_per_frame,
        person_count_per_frame=person_per_frame,
        class_counts_per_frame=class_per_frame,
    )
