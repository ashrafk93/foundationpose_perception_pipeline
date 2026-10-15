#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture a RealSense D4xx (D415/D435/D455/...) into scenes the pipeline can run on.

    # live camera: 5 scenes, one every 15 frames
    python tools/realsense/capture.py --config realsense --name my_capture --frames 5

    # frame the shot first, SPACE saves a scene, Q quits
    python tools/realsense/capture.py --config realsense --name my_capture --interactive

    # replay a recording instead of a camera (needs infrared 1 + 2 and depth; .bag, or .db3 from SDK 2.59+)
    python tools/realsense/capture.py --config realsense --name my_capture --bag recording.bag

Each saved frame is one scene: the left and right infrared images as cameras 0 and 1, the
factory calibration as `scene_camera.json`, and the camera's own depth as the scene's collected
depth. See `tools/realsense/layout.py` for the tree. Output paths come from the profile, so the
capture lands exactly where `script/infer.py --config <same profile>` will look for it.
The object meshes are yours to add: see "Capturing with a RealSense D4xx camera" in README.md.

The infrared projector is OFF by default. Its dot pattern lands on the very image SAM3 segments,
and FoundationStereo does not need it; pass `--emitter on` to trade that for denser sensor depth.

Needs `pyrealsense2` (`uv sync --inexact --extra realsense`); nothing else in the pipeline does.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPELINE_ROOT / "src"))
sys.path.insert(0, str(PIPELINE_ROOT / "tools"))

from foundationpose_perception_pipeline.config import add_config_argument, settings_from_argv  # noqa: E402
from realsense import layout  # noqa: E402

FRAME_TIMEOUT_MS = 5000
PREVIEW_WINDOW = "RealSense capture (SPACE: save scene, Q: quit)"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments; output locations default to the active profile."""
    settings = settings_from_argv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_argument(parser)
    parser.add_argument("--name", default=settings.dataset.name,
                        help="Dataset folder to write under --out-root; pass it to infer.py as --dataset.")
    parser.add_argument("--out-root", type=Path, default=settings.dataset.root)
    parser.add_argument("--depth-root", type=Path, default=settings.dataset.collected_depth_root,
                        help="Where the camera's own depth goes. Defaults to the profile's "
                             "dataset.collected_depth_root.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--serial", default=None, help="Camera serial number, when more than one is connected.")
    source.add_argument("--bag", type=Path, default=None, help="Read a recording (.bag, or .db3 from SDK 2.59+) instead of a live camera.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=1,
                        help="Scenes to save; 0 with --bag saves the whole recording. Ignored with --interactive.")
    parser.add_argument("--stride", type=int, default=15, help="Frames between saved scenes.")
    parser.add_argument("--warmup", type=int, default=30,
                        help="Frames to discard first, so auto-exposure settles. Ignored for --bag.")
    parser.add_argument("--emitter", choices=("off", "on"), default="off")
    parser.add_argument("--interactive", action="store_true", help="Preview window; SPACE saves a scene.")
    return parser.parse_args()


def _require_pyrealsense() -> Any:
    try:
        import pyrealsense2 as rs
    except ImportError as error:
        raise SystemExit(
            "pyrealsense2 is not installed. Install it with `uv sync --inexact --extra realsense`."
        ) from error
    return rs


def start_pipeline(rs: Any, args: argparse.Namespace) -> tuple[Any, Any]:
    """Start streaming infrared 1 + 2 and depth, from a camera or a recording."""
    config = rs.config()
    if args.bag is not None:
        config.enable_device_from_file(str(args.bag), repeat_playback=False)
        # Resolution and rate come from the recording.
        config.enable_stream(rs.stream.infrared, 1)
        config.enable_stream(rs.stream.infrared, 2)
        config.enable_stream(rs.stream.depth)
    else:
        if args.serial:
            config.enable_device(args.serial)
        config.enable_stream(rs.stream.infrared, 1, args.width, args.height, rs.format.y8, args.fps)
        config.enable_stream(rs.stream.infrared, 2, args.width, args.height, rs.format.y8, args.fps)
        config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    pipeline = rs.pipeline()
    try:
        profile = pipeline.start(config)
    except RuntimeError as error:
        raise SystemExit(
            f"could not start the camera: {error}. Check it is a D4xx on USB 3 and that "
            f"{args.width}x{args.height}@{args.fps} is a supported infrared+depth mode "
            "(README.md section 6 lists the usual causes)."
        ) from error
    device = profile.get_device()
    if args.bag is not None:
        # Every recorded frame, not just those that keep up with the wall clock.
        device.as_playback().set_real_time(False)
    else:
        depth_sensor = device.first_depth_sensor()
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 1.0 if args.emitter == "on" else 0.0)
    return pipeline, profile


def read_calibration(rs: Any, profile: Any) -> layout.StereoCalibration:
    """The infrared pair's factory calibration, from the streams actually running."""
    left = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
    right = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
    k_left, k_right = left.get_intrinsics(), right.get_intrinsics()
    for label, intrinsics in (("left", k_left), ("right", k_right)):
        if any(abs(c) > 1e-6 for c in intrinsics.coeffs):
            # Y8 infrared on a D4xx is rectified on the camera; non-zero coefficients mean it is not.
            print(f"warning: {label} infrared reports distortion {list(intrinsics.coeffs)}; the "
                  "pipeline treats both imagers as undistorted pinhole cameras")
    extrinsics = left.get_extrinsics_to(right)
    depth_units_m = profile.get_device().first_depth_sensor().get_depth_scale()
    return layout.StereoCalibration(
        k_left=layout.intrinsics_matrix(k_left.fx, k_left.fy, k_left.ppx, k_left.ppy),
        k_right=layout.intrinsics_matrix(k_right.fx, k_right.fy, k_right.ppx, k_right.ppy),
        # librealsense stores rotation column-major.
        rotation=np.asarray(extrinsics.rotation, dtype=np.float64).reshape(3, 3).T,
        translation_m=np.asarray(extrinsics.translation, dtype=np.float64),
        depth_scale_mm=float(depth_units_m) * layout.MM_PER_M,
        width=int(k_left.width),
        height=int(k_left.height),
    )


def device_info(rs: Any, profile: Any) -> dict[str, Any]:
    """Provenance recorded per scene: which camera, firmware and SDK produced it."""
    device = profile.get_device()
    info = {}
    for key, field in (("name", rs.camera_info.name), ("serial", rs.camera_info.serial_number),
                       ("firmware", rs.camera_info.firmware_version),
                       ("usb", rs.camera_info.usb_type_descriptor)):
        info[key] = device.get_info(field) if device.supports(field) else None
    info["librealsense"] = getattr(rs, "__version__", None)
    return info


def frames_of(frameset: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """(left IR, right IR, depth) as arrays, or None if the frameset is incomplete."""
    left, right, depth = frameset.get_infrared_frame(1), frameset.get_infrared_frame(2), frameset.get_depth_frame()
    if not (left and right and depth):
        return None
    return (np.asanyarray(left.get_data()).copy(), np.asanyarray(right.get_data()).copy(),
            np.asanyarray(depth.get_data()).copy())


def emitter_state(rs: Any, frameset: Any) -> bool | None:
    """Whether the projector was on for this frame, when the camera reports it."""
    depth = frameset.get_depth_frame()
    key = rs.frame_metadata_value.frame_laser_power_mode
    return bool(depth.get_frame_metadata(key)) if depth and depth.supports_frame_metadata(key) else None


def preview(left: np.ndarray, depth: np.ndarray, saved: int) -> int:
    """Show left IR beside colourised depth; return the key pressed."""
    import cv2

    valid = depth > 0
    scaled = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        lo, hi = np.percentile(depth[valid], [2, 98])
        scaled[valid] = np.clip((depth[valid] - lo) / max(hi - lo, 1.0) * 255.0, 0, 255).astype(np.uint8)
    coloured = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    coloured[~valid] = 0
    canvas = np.hstack([cv2.cvtColor(left, cv2.COLOR_GRAY2BGR), coloured])
    cv2.putText(canvas, f"saved: {saved}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow(PREVIEW_WINDOW, canvas)
    return cv2.waitKey(1) & 0xFF


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    if args.frames < 0 or (args.frames == 0 and args.bag is None and not args.interactive):
        raise SystemExit("--frames must be positive (0 means the whole recording, and needs --bag)")
    if args.stride < 1:
        raise SystemExit("--stride must be at least 1")
    if args.depth_root is None:
        raise SystemExit("No collected-depth root: set dataset.collected_depth_root in the profile or pass --depth-root.")
    rs = _require_pyrealsense()

    dataset_dir = args.out_root / args.name
    out_split = dataset_dir / "test"
    depth_split = args.depth_root / args.name / "test"
    out_split.mkdir(parents=True, exist_ok=True)
    depth_split.mkdir(parents=True, exist_ok=True)

    pipeline, profile = start_pipeline(rs, args)
    calibration = read_calibration(rs, profile)
    camera = device_info(rs, profile)
    print(f"{camera['name']} {camera['serial']} (firmware {camera['firmware']}, USB {camera['usb']}): "
          f"{calibration.width}x{calibration.height}, baseline {calibration.baseline_m * 1000:.1f} mm")

    scene_id = next_id = layout.next_scene_id(out_split)
    index_rows: list[dict[str, Any]] = []
    frame_count = 0
    warmup = 0 if args.bag is not None else args.warmup
    try:
        while args.interactive or not args.frames or len(index_rows) < args.frames:
            ok, frameset = pipeline.try_wait_for_frames(FRAME_TIMEOUT_MS)
            if not ok:
                if args.bag is not None:
                    break  # end of the recording
                raise SystemExit(f"no frames from the camera for {FRAME_TIMEOUT_MS} ms")
            frame_count += 1
            arrays = frames_of(frameset)
            if arrays is None or frame_count <= warmup:
                continue
            left, right, depth = arrays
            emitter = emitter_state(rs, frameset)
            if args.bag is None and emitter is not None and emitter != (args.emitter == "on"):
                continue  # the projector has not switched yet
            if args.interactive:
                key = preview(left, depth, len(index_rows))
                if key in (ord("q"), 27):
                    break
                if key != ord(" "):
                    continue
            elif (frame_count - warmup - 1) % args.stride:
                continue

            layout.write_scene(out_split=out_split, scene_id=scene_id, ir_left=left, ir_right=right,
                               calibration=calibration)
            layout.write_sensor_depth(depth_split=depth_split, scene_id=scene_id, depth_raw=depth,
                                      depth_scale_mm=calibration.depth_scale_mm)
            index_rows.append({
                "out_scene": scene_id,
                "source": str(args.bag.resolve()) if args.bag is not None else "live",
                "camera": camera,
                "frame_number": int(frameset.get_frame_number()),
                "timestamp_ms": float(frameset.get_timestamp()),
                "captured_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "resolution_wh": [calibration.width, calibration.height],
                "baseline_m": calibration.baseline_m,
                "emitter_on": emitter,
                "depth_valid_fraction": float((depth > 0).mean()),
            })
            print(f"scene {scene_id:06d}: frame {frameset.get_frame_number()}, "
                  f"sensor depth valid {100 * (depth > 0).mean():.1f}%")
            scene_id += 1
    finally:
        pipeline.stop()
        if args.interactive:
            import cv2

            cv2.destroyAllWindows()

    layout.append_scene_index(dataset_dir, index_rows)
    if not index_rows:
        raise SystemExit("no scenes saved")
    print(f"\nwrote scenes {next_id:06d}-{scene_id - 1:06d} to {out_split}")
    print(f"sensor depth under {depth_split}")
    config_hint = f" --config {args.config}" if args.config else ""
    print("\nnext:\n"
          f"  add {dataset_dir / 'models' / 'obj_000001.ply'} (mm) and {dataset_dir / 'dataset_map.json'} "
          '{"<name>": 1}\n'
          f"  python tools/build_tao_engine.py --onnx <deployable>.onnx --shape-from-scene "
          f"{out_split / f'{next_id:06d}'}{config_hint}\n"
          f"  python script/infer.py{config_hint} --dataset {args.name} --objects <name> ... "
          f"--output-dir output/{args.name}")


if __name__ == "__main__":
    main()
