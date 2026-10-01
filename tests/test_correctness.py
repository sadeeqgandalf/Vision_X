"""Regression tests for correctness fixes. Run: python -m pytest tests/ -q"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from video_profiler.aggregation import _aggregate_colors, aggregate
from video_profiler.detection import (
    COCO_CATEGORIES_91,
    COCO_CLASSES,
    Detection,
    FrameDetections,
    _label_to_name,
)
from video_profiler.motion import FrameStats, _dominant_colors, _frame_diff_magnitude
from video_profiler.report import _write_color_palette


def _solid_bgr(b, g, r, size=40):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = (b, g, r)
    return img


# --- colour -----------------------------------------------------------------

def test_dominant_color_white_is_top_bin():
    # Before the fix, uint8 * 7 wrapped around and white was reported as (16, 16, 16).
    assert _dominant_colors(_solid_bgr(255, 255, 255), n_colors=1) == [(240, 240, 240)]


def test_dominant_color_is_rgb_order():
    # Pure red in BGR input must come out as red in (R, G, B).
    assert _dominant_colors(_solid_bgr(0, 0, 255), n_colors=1) == [(240, 16, 16)]
    assert _dominant_colors(_solid_bgr(255, 0, 0), n_colors=1) == [(16, 16, 240)]


def test_dominant_color_bins_match_centres():
    # Each value v must land in bin v // 32, i.e. centre (v // 32) * 32 + 16.
    for v in (0, 31, 32, 100, 150, 224, 255):
        c = (v // 32) * 32 + 16
        assert _dominant_colors(_solid_bgr(v, v, v), n_colors=1) == [(c, c, c)]


def test_aggregate_colors_stays_in_range():
    # Before the fix, (240, 16, 16) became (7184, 16, 16).
    assert _aggregate_colors([[(240, 16, 16)], [(240, 16, 16)], [(16, 112, 208)]]) == [
        (240, 16, 16),
        (16, 112, 208),
    ]


def test_palette_figure_with_real_colors(tmp_path):
    colors = _aggregate_colors([[(240, 240, 240)], [(144, 112, 48)]])
    profile = aggregate(
        [FrameStats(0, 0.0, 0.0, colors[0], colors)],
        [FrameDetections(0, 0.0, [])],
    )
    out = tmp_path / "palette.png"
    _write_color_palette(profile, out)  # raised ValueError (RGBA out of range) before the fix
    assert out.exists()


# --- motion -----------------------------------------------------------------

def test_frame_diff_no_uint8_wraparound():
    a = _solid_bgr(10, 10, 10)
    b = _solid_bgr(20, 20, 20)
    assert _frame_diff_magnitude(a, b) == pytest.approx(10.0, abs=1.0)
    assert _frame_diff_magnitude(b, a) == pytest.approx(10.0, abs=1.0)


# --- detection labels -------------------------------------------------------

def test_label_list_matches_torchvision():
    torchvision = pytest.importorskip("torchvision")
    cats = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta["categories"]
    assert COCO_CATEGORIES_91 == list(cats)


def test_label_mapping_uses_91_id_space():
    assert _label_to_name(1) == "person"
    assert _label_to_name(13) == "stop sign"  # was "parking meter"
    assert _label_to_name(44) == "bottle"  # was "knife"
    assert _label_to_name(90) == "toothbrush"  # was dropped
    assert _label_to_name(0) is None
    assert _label_to_name(12) is None  # N/A
    assert _label_to_name(91) is None


def test_all_80_class_names_reachable():
    names = {n for n in COCO_CATEGORIES_91 if n not in ("__background__", "N/A")}
    assert names == set(COCO_CLASSES)


# --- aggregation ------------------------------------------------------------

def _stats(n, fps=15.0, persons=None):
    ms = [FrameStats(i, i / fps, 1.0, (0, 0, 0), [(16, 16, 16)]) for i in range(n)]
    persons = persons or [0] * n
    ds = [
        FrameDetections(i, i / fps, [Detection("person", 0.9, (0, 0, 1, 1))] * persons[i])
        for i in range(n)
    ]
    return ms, ds


def test_last_frame_included_in_segments():
    ms, ds = _stats(45)
    p = aggregate(ms, ds, segment_window_sec=10.0)
    assert sum(s.frame_count for s in p.segments) == 45


def test_last_frame_included_multi_segment():
    ms, ds = _stats(46, fps=1.0, persons=[0] * 45 + [3])  # duration 45s, 10s windows
    p = aggregate(ms, ds, segment_window_sec=10.0)
    assert sum(s.frame_count for s in p.segments) == 46
    assert p.segments[-1].person_count_max == 3


def test_single_frame_has_one_segment():
    ms, ds = _stats(1, persons=[2])
    p = aggregate(ms, ds, segment_window_sec=10.0)
    assert p.duration_sec == 0.0
    assert len(p.segments) == 1
    assert p.segments[0].frame_count == 1
    assert p.segments[0].person_count_max == 2


def test_empty_input():
    p = aggregate([], [], segment_window_sec=10.0)
    assert p.segments == [] and p.total_frames_analyzed == 0
