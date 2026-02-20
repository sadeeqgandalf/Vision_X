#!/usr/bin/env python3
# Synthetic 3s test video. Requires opencv, numpy.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2
import numpy as np

out = Path(__file__).resolve().parent.parent / "sample_video.mp4"
w, h, fps = 640, 360, 15
sec = 3
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(out), fourcc, fps, (w, h))
for i in range(fps * sec):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (i % 80 + 50, 100, 150)  # slow color change
    cv2.circle(img, (w//2 + int(50 * np.sin(i/10)), h//2), 30, (255, 255, 255), -1)
    writer.write(img)
writer.release()
print(f"Wrote {out}")
