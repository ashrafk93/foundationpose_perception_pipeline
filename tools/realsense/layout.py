#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write a D4xx capture as the scene tree the pipeline reads. No camera, no `pyrealsense2`.

One captured frame becomes one scene:

    <root>/<name>/test/<scene:06d>/rgb/000000.png      left infrared  -- the BASE camera
                                  /rgb/000001.png      right infrared -- its stereo partner
                                  /scene_camera.json   factory intrinsics + left->right extrinsics
    <depth_root>/<name>/test/<scene:06d>/scene_cam0_depth.png   the camera's own depth, uint16 mm
    <root>/<name>/scene_index.json                               per-scene capture provenance

The object meshes (`models/`, `dataset_map.json`) are not written here; the user adds them.

The left imager is the base camera because the D4xx computes its depth in that imager's frame:
the sensor depth then lines up with `rgb/000000.png` pixel for pixel, with no reprojection.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from bop_adapt import emit

MM_PER_M = 1000.0
BASE_IM_ID = 0
PARTNER_IM_ID = 1


@dataclass(frozen=True)
class StereoCalibration:
    """A D4xx infrared pair's factory calibration, in this pipeline's conventions.

    `rotation`/`translation_m` map a point from the LEFT imager's frame into the RIGHT's, which
    is exactly BOP's world-to-camera transform for the right camera once the left one is the world.
    """

    k_left: np.ndarray
    k_right: np.ndarray
    rotation: np.ndarray
    translation_m: np.ndarray
    # Millimetres per raw depth unit: 1.0 for a D4xx at its default 1 mm depth units.
    depth_scale_mm: float
    width: int
    height: int

    @property
    def baseline_m(self) -> float:
        """Distance between the two imagers' optical centres."""
        return float(np.linalg.norm(self.translation_m))


def intrinsics_matrix(fx: float, fy: float, ppx: float, ppy: float) -> np.ndarray:
    """A pinhole camera matrix from `rs.intrinsics` fields."""
    return np.array([[fx, 0.0, ppx], [0.0, fy, ppy], [0.0, 0.0, 1.0]], dtype=np.float64)


def scene_camera(calibration: StereoCalibration) -> dict[str, dict[str, Any]]:
    """`scene_camera.json` for one capture: the left imager is the world origin."""

    def entry(k: np.ndarray, rotation: np.ndarray, translation_m: np.ndarray) -> dict[str, Any]:
        return {
            "cam_K": np.asarray(k, dtype=np.float64).reshape(-1).tolist(),
            "cam_R_w2c": np.asarray(rotation, dtype=np.float64).reshape(-1).tolist(),
            "cam_t_w2c": (np.asarray(translation_m, dtype=np.float64).reshape(-1) * MM_PER_M).tolist(),
            "depth_scale": float(calibration.depth_scale_mm),
        }

    return {
        str(BASE_IM_ID): entry(calibration.k_left, np.eye(3), np.zeros(3)),
        str(PARTNER_IM_ID): entry(calibration.k_right, calibration.rotation, calibration.translation_m),
    }


def as_three_channel(image: np.ndarray) -> np.ndarray:
    """Replicate an 8-bit infrared frame to three channels, the form the RGB readers expect."""
    if image.ndim == 3:
        return image
    return cv2.merge([image, image, image])


def next_scene_id(out_split: Path) -> int:
    """The first unused scene id, so a second capture session appends instead of overwriting."""
    existing = [int(p.name) for p in out_split.glob("*") if p.is_dir() and p.name.isdigit()]
    return max(existing, default=-1) + 1


def write_scene(
    *,
    out_split: Path,
    scene_id: int,
    ir_left: np.ndarray,
    ir_right: np.ndarray,
    calibration: StereoCalibration,
) -> Path:
    """Write one capture as a two-camera scene and return its directory."""
    expected = (calibration.height, calibration.width)
    for label, image in (("left", ir_left), ("right", ir_right)):
        if image.shape[:2] != expected:
            raise ValueError(f"{label} infrared frame is {image.shape[:2]}, calibration says {expected}")
    scene_dir = out_split / f"{scene_id:06d}"
    (scene_dir / "rgb").mkdir(parents=True, exist_ok=True)
    for im_id, image in ((BASE_IM_ID, ir_left), (PARTNER_IM_ID, ir_right)):
        if not cv2.imwrite(str(scene_dir / "rgb" / f"{im_id:06d}.png"), as_three_channel(image)):
            raise OSError(f"could not write {scene_dir / 'rgb' / f'{im_id:06d}.png'}")
    emit.write_json(scene_dir / "scene_camera.json", scene_camera(calibration))
    return scene_dir


def write_sensor_depth(*, depth_split: Path, scene_id: int, depth_raw: np.ndarray, depth_scale_mm: float) -> None:
    """Write the camera's own depth as the scene's collected depth, uint16 mm, 0 = invalid."""
    if depth_raw.dtype != np.uint16:
        raise ValueError(f"expected uint16 Z16 depth, got {depth_raw.dtype}")
    emit.write_collected_depth(depth_split=depth_split, out_id=scene_id, raw=depth_raw, depth_scale=depth_scale_mm)


def append_scene_index(dataset_dir: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Add capture provenance to `scene_index.json`, keeping what earlier sessions wrote."""
    path = dataset_dir / "scene_index.json"
    index = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    emit.write_scene_index(dataset_dir, [*index, *rows])

