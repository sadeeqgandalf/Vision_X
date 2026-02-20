#!/usr/bin/env python3
# Vision-X Video Data Profiler. Usage: run_profiler.py --video <path|url> [--output-dir ...]

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from video_profiler.video_io import load_video, get_metadata, sample_frames
from video_profiler.motion import compute_motion_and_color
from video_profiler.detection import run_detection
from video_profiler.aggregation import aggregate
from video_profiler.report import generate_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile a video: motion, colors, object counts, and scene summary.",
    )
    parser.add_argument(
        "--video",
        "-v",
        required=True,
        help="Path to video file or YouTube URL",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="./video_report",
        help="Output directory for report and figures (default: ./video_report)",
    )
    parser.add_argument(
        "--max-frames",
        "-n",
        type=int,
        default=120,
        help="Max number of frames to sample (default: 120)",
    )
    parser.add_argument(
        "--segment-window-sec",
        type=float,
        default=10.0,
        help="Temporal segment length in seconds (default: 10)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Device for detection (default: auto)",
    )
    parser.add_argument(
        "--download-dir",
        default=None,
        help="Directory for downloading video from URL (default: system temp)",
    )
    args = parser.parse_args()

    video_path = args.video.strip()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading video...")
    try:
        cap, resolved_path = load_video(video_path, args.download_dir)
    except Exception as e:
        print(f"Error loading video: {e}", file=sys.stderr)
        return 1

    try:
        meta = get_metadata(cap, resolved_path)
        print(f"  Duration: {meta.duration_sec:.1f}s, {meta.frame_count} frames, {meta.fps:.1f} fps")

        print("Sampling frames...")
        frames = sample_frames(cap, max_frames=args.max_frames, strategy="uniform")
        print(f"  Sampled {len(frames)} frames")

        if len(frames) < 2:
            print("Not enough frames to analyze.", file=sys.stderr)
            return 1

        print("Computing motion and color...")
        motion_stats = compute_motion_and_color(frames)

        print("Running object detection...")
        detections = run_detection(
            frames,
            device=args.device,
            min_score=0.5,
        )

        print("Aggregating...")
        profile = aggregate(
            motion_stats,
            detections,
            segment_window_sec=args.segment_window_sec,
        )

        print("Writing report and figures...")
        json_path, md_path, figures_dir = generate_report(
            profile,
            output_dir,
            video_path=resolved_path,
            frames=frames,
            detections=detections,
        )
        print(f"  Report (JSON): {json_path}")
        print(f"  Report (MD):   {md_path}")
        print(f"  Figures:       {figures_dir}")
        print("Done.")
        return 0
    finally:
        cap.release()


if __name__ == "__main__":
    sys.exit(main())
