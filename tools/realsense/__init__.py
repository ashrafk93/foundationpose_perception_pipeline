#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture from a RealSense D4xx camera into the layout this pipeline runs on.

A D4xx is already the rig the depth stage wants: its two infrared imagers are a factory-
calibrated stereo pair, rectified on the camera, with a horizontal baseline (50 mm on a D435,
95 mm on a D455). So nothing in the pipeline changes -- the capture is written as a two-camera
scene and the shipped FoundationStereo path runs on it as-is:

- `layout` turns frames and calibration into that tree. Pure numpy, no camera needed, so it is
  what `test/check_realsense_layout.py` exercises.
- `capture` is the CLI that reads a live camera or a recording (`.bag` / `.db3`) through `pyrealsense2`.
"""
