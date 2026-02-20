# Vision-X Video Data Profiler

Pipeline for video analytics: frame sampling, motion and color stats, COCO object detection, temporal aggregation, and structured report generation. Input: local file or YouTube URL. No training or labels required.

## Pipeline

1. **Load** — Local path or URL (yt-dlp). OpenCV `VideoCapture`.
2. **Sample** — Uniform frame sampling (configurable max frames).
3. **Motion & color** — Per-frame mean absolute frame difference; dominant colors via 8³ histogram bins.
4. **Detection** — Faster R-CNN ResNet50 FPN (COCO, 80 classes). CPU or CUDA.
5. **Aggregation** — Fixed-length time segments; per-segment and global motion, counts, palette.
6. **Report** — JSON, Markdown, and figures.

## Outputs

| Output | Description |
|--------|-------------|
| `report.json` | Global stats, segments, optional `detections_with_bbox` (first N frames). |
| `report.md` | Summary, global stats, segment table, bbox reference. |
| `figures/motion_over_time.png` | Motion magnitude vs time. |
| `figures/person_count_over_time.png` | Person count vs time. |
| `figures/color_palette.png` | Global dominant color bar. |
| `figures/detection_bbox_sample_*.png` | Sample frames with drawn bounding boxes (first, middle, last). |

## Requirements

- Python 3.9+
- opencv-python-headless, numpy, torch, torchvision, matplotlib
- yt-dlp (for URL input): `pip install yt-dlp` or system install

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run_profiler.py --video /path/to/video.mp4
python run_profiler.py --video "https://www.youtube.com/watch?v=..."
python run_profiler.py --video video.mp4 --output-dir ./out --max-frames 60 --segment-window-sec 15
```

Output directory defaults to `./video_report`.

## Layout

```
video_profiler/
  video_io.py    # Load (file/URL), sample frames
  motion.py      # Frame-diff motion, dominant colors
  detection.py   # COCO Faster R-CNN
  aggregation.py # Time segments, global stats
  report.py      # JSON, MD, matplotlib figures, bbox samples
run_profiler.py  # CLI
requirements.txt
```

## Options

- `--video` — Path or URL (required).
- `--output-dir` — Report and figures directory.
- `--max-frames` — Max sampled frames (default 120).
- `--segment-window-sec` — Segment length in seconds (default 10).
- `--device` — `cpu` or `cuda` (default: auto).
- `--download-dir` — Where to save video when input is URL.
