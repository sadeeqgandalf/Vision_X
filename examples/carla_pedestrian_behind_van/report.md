# Video Data Profiler Report

**Source:** pedestrian_behind_van.mp4 (author's own CARLA simulator recording, Town05)

## Summary

This 16.0s video was analyzed over 40 sampled frames. Overall motion is moderate (mean 12.8, max 24.0). Detections summed over the sampled frames (an object seen in several frames is counted in each): bird: 1, bus: 3, car: 108, motorcycle: 1, person: 6, truck: 24. People visible in one frame: at most 2 (6 person detections in total). Temporal segmentation: 4 segment(s).

## Global stats
- Duration: 15.97 s
- Frames analyzed: 40
- Mean motion: 12.79
- Max motion: 23.95
- Detections summed over sampled frames: {'car': 108, 'truck': 24, 'person': 6, 'motorcycle': 1, 'bus': 3, 'bird': 1}

## Temporal segments (time windows)

Segments are fixed time windows over the video. For each segment: mean/max motion, detections summed over the sampled frames, and the most people seen in one frame.

| Start (s) | End (s) | Mean motion | Detections (summed over frames) | Most people in one frame |
|-----------|--------|-------------|----------------|------------|
| 0.0 | 5.0 | 11.06 | {'car': 52, 'truck': 5, 'person': 4} | 2 |
| 5.0 | 10.0 | 14.50 | {'car': 35, 'truck': 17, 'motorcycle': 1, 'bus': 3, 'person': 1} | 1 |
| 10.0 | 15.0 | 13.81 | {'person': 1, 'car': 18, 'truck': 2, 'bird': 1} | 1 |
| 15.0 | 16.0 | 9.41 | {'car': 3} | 0 |

## Bounding boxes

Per-frame detections with bounding boxes (xyxy) are in **report.json** under `detections_with_bbox`. Sample frames with boxes drawn are in **figures/detection_bbox_sample_*.png**.
