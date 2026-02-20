# Report: JSON, Markdown, motion/counts/palette figures, bbox samples, narrative summary.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .aggregation import AggregatedProfile
from .detection import FrameDetections


def _profile_to_dict(
    profile: AggregatedProfile,
    detections_with_bbox: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    out = {
        "duration_sec": profile.duration_sec,
        "total_frames_analyzed": profile.total_frames_analyzed,
        "global_mean_motion": round(profile.global_mean_motion, 4),
        "global_max_motion": round(profile.global_max_motion, 4),
        "global_object_counts": profile.global_object_counts,
        "global_dominant_colors_rgb": profile.global_dominant_colors_rgb,
        "n_segments": len(profile.segments),
        "segments": [
            {
                "start_sec": s.start_sec,
                "end_sec": s.end_sec,
                "mean_motion": round(s.mean_motion, 4),
                "max_motion": round(s.max_motion, 4),
                "object_counts": s.object_counts,
                "person_count_max": s.person_count_max,
                "frame_count": s.frame_count,
            }
            for s in profile.segments
        ],
    }
    if detections_with_bbox is not None:
        out["detections_with_bbox"] = detections_with_bbox
    return out


def _write_motion_chart(profile: AggregatedProfile, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(profile.timestamps_sec, profile.motion_per_frame, color="steelblue", alpha=0.8, linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Motion magnitude")
    ax.set_title("Motion intensity over time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _write_counts_chart(profile: AggregatedProfile, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(
        profile.timestamps_sec,
        profile.person_count_per_frame,
        color="coral",
        alpha=0.9,
        linewidth=1,
        label="persons",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Count")
    ax.set_title("Person count over time")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _write_color_palette(profile: AggregatedProfile, out_path: Path) -> None:
    colors = profile.global_dominant_colors_rgb
    if not colors:
        return
    fig, ax = plt.subplots(figsize=(8, 1.2))
    n = len(colors)
    for i, (r, g, b) in enumerate(colors):
        ax.axvspan(i / n, (i + 1) / n, color=(r / 255, g / 255, b / 255))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")
    ax.set_title("Dominant color palette (global)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _build_detections_with_bbox(
    detections: list[FrameDetections],
    max_frames: int = 5,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for fd in detections[:max_frames]:
        result.append({
            "frame_index": fd.frame_index,
            "timestamp_sec": round(fd.timestamp_sec, 2),
            "detections": [
                {
                    "class_name": d.class_name,
                    "score": round(d.score, 3),
                    "bbox_xyxy": [round(x, 1) for x in d.bbox],
                }
                for d in fd.detections
            ],
        })
    return result


def _draw_bboxes_on_frame(
    frame_bgr: np.ndarray,
    frame_detections: FrameDetections,
    thickness: int = 2,
) -> np.ndarray:
    out = frame_bgr.copy()
    color = (0, 255, 0)
    for d in frame_detections.detections:
        x1, y1, x2, y2 = [int(x) for x in d.bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = f"{d.class_name} {d.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(out, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return out


def _write_bbox_samples(
    frames: list[Any],
    detections: list[FrameDetections],
    figures_dir: Path,
    get_frame: Any = None,
    sample_indices: list[int] | None = None,
) -> None:
    if not frames or not detections or len(frames) != len(detections):
        return
    get_frame = get_frame or (lambda o: o.frame)
    n = len(frames)
    if sample_indices is None:
        if n <= 3:
            sample_indices = list(range(n))
        else:
            sample_indices = [0, n // 2, n - 1]
    for i, idx in enumerate(sample_indices):
        if idx >= n:
            continue
        frame = get_frame(frames[idx])
        fd = detections[idx]
        out_img = _draw_bboxes_on_frame(frame, fd)
        out_path = figures_dir / f"detection_bbox_sample_{i + 1:02d}.png"
        cv2.imwrite(str(out_path), out_img)


def _narrative_summary(profile: AggregatedProfile) -> str:
    duration = profile.duration_sec
    n_frames = profile.total_frames_analyzed
    mean_motion = profile.global_mean_motion
    max_motion = profile.global_max_motion
    objs = profile.global_object_counts
    persons = objs.get("person", 0)
    n_segments = len(profile.segments)

    motion_level = "low" if mean_motion < 5 else "moderate" if mean_motion < 15 else "high"
    segment_desc = f"{n_segments} segment(s)" if n_segments else "single segment"
    obj_desc = ", ".join(f"{k}: {v}" for k, v in sorted(objs.items())[:8])
    if len(objs) > 8:
        obj_desc += ", ..."

    return (
        f"This {duration:.1f}s video was analyzed over {n_frames} sampled frames. "
        f"Overall motion is {motion_level} (mean {mean_motion:.1f}, max {max_motion:.1f}). "
        f"Detected objects: {obj_desc or 'none'}. "
        f"Person count across frames: max {max(profile.person_count_per_frame) if profile.person_count_per_frame else 0}, "
        f"total detections {persons}. "
        f"Temporal segmentation: {segment_desc}."
    )


def generate_report(
    profile: AggregatedProfile,
    output_dir: str | Path,
    video_path: str = "",
    frames: list[Any] | None = None,
    detections: list[FrameDetections] | None = None,
    max_bbox_frames_in_json: int = 5,
) -> tuple[Path, Path, Path]:
    """Write report.json, report.md, figures/. If frames+detections given, add detections_with_bbox and bbox sample PNGs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detections_with_bbox: list[dict[str, Any]] | None = None
    if detections is not None:
        detections_with_bbox = _build_detections_with_bbox(detections, max_frames=max_bbox_frames_in_json)

    report_json = output_dir / "report.json"
    with open(report_json, "w") as f:
        json.dump(_profile_to_dict(profile, detections_with_bbox=detections_with_bbox), f, indent=2)

    # Markdown
    report_md = output_dir / "report.md"
    narrative = _narrative_summary(profile)
    md_lines = [
        "# Video Data Profiler Report",
        "",
        f"**Source:** {video_path or '(unknown)'}",
        "",
        "## Summary",
        "",
        narrative,
        "",
        "## Global stats",
        f"- Duration: {profile.duration_sec:.2f} s",
        f"- Frames analyzed: {profile.total_frames_analyzed}",
        f"- Mean motion: {profile.global_mean_motion:.2f}",
        f"- Max motion: {profile.global_max_motion:.2f}",
        f"- Object counts: {profile.global_object_counts}",
        "",
        "## Temporal segments (time windows)",
        "",
        "Segments are fixed time windows over the video. For each segment: mean/max motion, "
        "object counts, and max person count.",
        "",
        "| Start (s) | End (s) | Mean motion | Object counts | Person max |",
        "|-----------|--------|-------------|----------------|------------|",
    ]
    for s in profile.segments[:30]:
        md_lines.append(f"| {s.start_sec:.1f} | {s.end_sec:.1f} | {s.mean_motion:.2f} | {s.object_counts} | {s.person_count_max} |")
    if len(profile.segments) > 30:
        md_lines.append(f"| ... | ... | ... | ({len(profile.segments) - 30} more; see report.json) | ... |")
    md_lines.append("")
    if detections_with_bbox:
        md_lines.extend([
            "## Bounding boxes",
            "",
            "Per-frame detections with bounding boxes (xyxy) are in **report.json** under `detections_with_bbox`. "
            "Sample frames with boxes drawn are in **figures/detection_bbox_sample_*.png**.",
            "",
        ])
    with open(report_md, "w") as f:
        f.write("\n".join(md_lines))

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _write_motion_chart(profile, figures_dir / "motion_over_time.png")
    _write_counts_chart(profile, figures_dir / "person_count_over_time.png")
    _write_color_palette(profile, figures_dir / "color_palette.png")
    if frames is not None and detections is not None:
        _write_bbox_samples(frames, detections, figures_dir)

    return report_json, report_md, figures_dir
