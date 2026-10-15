#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check that a RealSense capture is a valid stereo scene for the depth stage -- no camera, no GPU.

Writes a synthetic D455 capture through `tools/realsense/layout.py` (a textured plane at a known
distance, seen by both infrared imagers) and runs the pipeline's OWN geometry over it: camera
loading, partner selection, rectification and the `sensor` depth source. A capture
that the depth stage misreads still produces a plausible-looking depth map, so each property that
would fail silently is asserted here:

- camera 1 is chosen as the partner, and rectification is a no-op on an already-rectified pair;
- camera 0 is the LEFT camera, so no mirroring, and the rectified pair still differs by exactly
  the disparity the plane's distance predicts;
- the sensor source returns the camera's depth in metres and is not reusable as a prediction.

Usage:
    ./.venv/bin/python test/check_realsense_layout.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR / "src"))
sys.path.insert(0, str(PIPELINE_DIR / "tools"))

from foundationpose_perception_pipeline.config import CONFIG_ENV_VAR  # noqa: E402
from foundationpose_perception_pipeline.inference.depth import _is_predicted_cache  # noqa: E402
from foundationpose_perception_pipeline.inference.source import (  # noqa: E402
    SENSOR_SOURCE,
    registered_sources,
)
from foundationpose_perception_pipeline.inference.stereo.depth import load_cameras  # noqa: E402
from foundationpose_perception_pipeline.inference.stereo.rectify import (  # noqa: E402
    rectify_pair,
    select_partner_camera,
)
from realsense import layout  # noqa: E402

# A D455 at 1280x720, as its factory calibration reports it.
WIDTH, HEIGHT = 1280, 720
FOCAL_PX, CX, CY = 646.95, 636.66, 358.58
BASELINE_M = 0.0948
PLANE_M = 0.8


def synthetic_calibration() -> layout.StereoCalibration:
    k = layout.intrinsics_matrix(FOCAL_PX, FOCAL_PX, CX, CY)
    return layout.StereoCalibration(
        k_left=k, k_right=k.copy(), rotation=np.eye(3),
        translation_m=np.array([-BASELINE_M, 0.0, 0.0]),
        depth_scale_mm=1.0, width=WIDTH, height=HEIGHT,
    )


def synthetic_pair(disparity_px: float) -> tuple[np.ndarray, np.ndarray]:
    """A textured fronto-parallel plane: the right view is the left shifted by the disparity."""
    rng = np.random.default_rng(0)
    left = cv2.GaussianBlur(rng.integers(0, 256, (HEIGHT, WIDTH), dtype=np.uint8), (0, 0), 2.0)
    xs, ys = np.meshgrid(np.arange(WIDTH, dtype=np.float32), np.arange(HEIGHT, dtype=np.float32))
    right = cv2.remap(left, xs + np.float32(disparity_px), ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return left, right


def check_scene(root: Path) -> None:
    calibration = synthetic_calibration()
    disparity = FOCAL_PX * BASELINE_M / PLANE_M
    left, right = synthetic_pair(disparity)
    dataset_dir = root / "cap"
    scene_dir = layout.write_scene(out_split=dataset_dir / "test", scene_id=0, ir_left=left, ir_right=right,
                                   calibration=calibration)
    depth_raw = np.full((HEIGHT, WIDTH), int(PLANE_M * 1000), dtype=np.uint16)
    depth_raw[:, :40] = 0  # the D4xx's invalid left band
    layout.write_sensor_depth(depth_split=root / "cap" / "test", scene_id=0, depth_raw=depth_raw,
                              depth_scale_mm=calibration.depth_scale_mm)

    intrinsics, extrinsics, keys = load_cameras(scene_dir)
    assert keys == ["0", "1"], keys
    assert np.allclose(extrinsics[1][:, 3], [-BASELINE_M, 0, 0]), extrinsics[1]

    selection = select_partner_camera(base_camera=0, intrinsics=intrinsics, extrinsics=extrinsics,
                                      image_width=WIDTH, image_height=HEIGHT)
    assert selection is not None and selection[0] == 1, selection
    assert selection[2] < 1e-3, f"rectification rotation {selection[2]} deg on a rectified pair"

    image_left = cv2.imread(str(scene_dir / "rgb" / "000000.png"), cv2.IMREAD_COLOR)
    image_right = cv2.imread(str(scene_dir / "rgb" / "000001.png"), cv2.IMREAD_COLOR)
    rect = rectify_pair(image_left, image_right, intrinsics[0], intrinsics[1], extrinsics[0], extrinsics[1])
    assert rect.baseline_x_signed_m < 0, "camera 0 must be the LEFT camera, or the depth stage mirrors it"
    assert abs(abs(rect.baseline_x_signed_m) - BASELINE_M) < 1e-6, rect.baseline_x_signed_m
    assert np.allclose(rect.r1, np.eye(3), atol=1e-9), rect.r1

    # The rectified pair must still be the plane's disparity apart, at the rectified focal length.
    expected = rect.k_rect[0, 0] * BASELINE_M / PLANE_M
    crop = np.s_[200:520, 300:980]
    gray_l = cv2.cvtColor(rect.left, cv2.COLOR_BGR2GRAY)[crop].astype(np.float32)
    gray_r = cv2.cvtColor(rect.right, cv2.COLOR_BGR2GRAY)[crop].astype(np.float32)
    (shift_x, shift_y), _ = cv2.phaseCorrelate(gray_r, gray_l)
    assert abs(shift_x - expected) < 0.1 and abs(shift_y) < 0.1, (shift_x, shift_y, expected)
    print(f"  pair: partner 1, rotation {selection[2]:.1e} deg, rectified disparity {shift_x:.2f} px "
          f"(expected {expected:.2f})")

    args = SimpleNamespace(collected_root=root, split="test")
    depth_dir = root / "run" / "depth" / "000000"
    depth_m = registered_sources()[SENSOR_SOURCE].provide(
        scene_dir=scene_dir, depth_dir=depth_dir, args=args, dataset="cap", scene_id=0
    )
    assert depth_m.shape == (HEIGHT, WIDTH)
    assert np.isnan(depth_m[:, :40]).all(), "zero sensor depth must load as NaN"
    assert np.allclose(depth_m[:, 40:], PLANE_M), np.nanmedian(depth_m)
    metadata = json.loads((depth_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["depth_source"] == SENSOR_SOURCE
    assert not _is_predicted_cache(depth_dir), "a sensor copy must not be reused as predicted depth"
    print(f"  sensor source: median {np.nanmedian(depth_m):.3f} m, valid {metadata['depth_m_valid_fraction']:.3f}")

    assert layout.next_scene_id(dataset_dir / "test") == 1


def main() -> None:
    # The sensor source resolves the collected-depth filename from the active profile.
    os.environ.setdefault(CONFIG_ENV_VAR, "example_realsense")
    with tempfile.TemporaryDirectory() as tmp:
        check_scene(Path(tmp))
    print("RealSense layout OK: capture tree loads, pairs, rectifies and feeds the sensor source.")


if __name__ == "__main__":
    main()
