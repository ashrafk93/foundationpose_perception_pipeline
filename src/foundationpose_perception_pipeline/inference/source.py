#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where the pose input comes from: the depth-source registry, and the two sources that ship.

A *backend* (`inference/depth.py`) is a way of computing depth from a stereo pair. A *source* is
the answer to a different question -- where FoundationPose's depth comes from at all. The default
answer is "predict it" (`foundationstereo`); a capture that ships measured depth can feed that
instead (`sensor`), which is how a stereo camera's own depth is compared against the prediction.

Keeping them separate is what makes both extensible without either knowing about the other: a
source that reads depth from disk needs no backend, and a new backend needs no new source.

Registering one is the same shape as registering a backend, and for the same reason -- the CLI
choices are the registry keys, so an entry point never enumerates what exists:

    from foundationpose_perception_pipeline.inference.source import DepthSource, register_source

    register_source(DepthSource(name="mine", describe="…", provide=…))

See `foundationpose_perception_pipeline.extensions` for how a package outside this one gets loaded.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PREDICTED_SOURCE = "foundationstereo"
# Depth the capture device measured itself -- a RealSense D4xx, a structured-light sensor -- and
# shipped beside the images. The pose input it gives is what the sensor alone would achieve, so a
# run under each source isolates what predicted depth is worth on the same frames and masks.
SENSOR_SOURCE = "sensor"


@dataclass(frozen=True)
class DepthSource:
    """One way of obtaining the depth map FoundationPose registers against.

    `provide` writes `depth_m.npy` into `depth_dir` and returns the array. Returning it as well
    as writing it saves every caller a `np.load` of the file it just produced.

    `uses_depth_backend` says whether this source runs a model at all. It is what lets an entry
    point resolve (and assert) the depth backend once per run without asking what the source is.
    """

    name: str
    provide: Callable[..., np.ndarray]
    describe: str = ""
    uses_depth_backend: bool = True
    # Whether this source reads the collected-depth tree, and whether scoring predicted depth
    # against that tree means anything for it. Both are asked by entry points that have to decide
    # whether the tree is required for a run -- the alternative is every entry point knowing the
    # name of every source, which is exactly what the registry exists to avoid.
    needs_collected_depth: bool = False
    depth_metrics_meaningful: bool = True
    add_arguments: Callable[[Any], None] | None = None
    options: Callable[[Any], dict[str, Any]] = field(default=lambda _args: {})


_SOURCES: dict[str, DepthSource] = {}


def register_source(source: DepthSource) -> None:
    """Add a depth source to the registry, refusing a duplicate name."""
    if source.name in _SOURCES:
        raise ValueError(f"depth source {source.name!r} is already registered")
    _SOURCES[source.name] = source


def registered_sources() -> dict[str, DepthSource]:
    """Return the registry, extensions included."""
    from foundationpose_perception_pipeline.extensions import load_extensions

    load_extensions()
    return dict(_SOURCES)


def depth_source_choices() -> tuple[str, ...]:
    """Values `--depth-source` accepts, shipped source first so it reads as the default."""
    others = sorted(name for name in registered_sources() if name != PREDICTED_SOURCE)
    return (PREDICTED_SOURCE, *others)


def add_source_arguments(parser: Any) -> None:
    """Let every registered source add its own flags."""
    for source in registered_sources().values():
        if source.add_arguments is not None:
            source.add_arguments(parser)


def provide_predicted_depth(
    *,
    scene_dir: Path,
    depth_dir: Path,
    args: Any,
    **_ignored: Any,
) -> np.ndarray:
    """Predict depth from the scene's stereo pair, through the resolved depth backend."""
    from foundationpose_perception_pipeline.inference.depth import backend_options, generate_scene_depth

    generate_scene_depth(
        scene_dir=scene_dir,
        depth_dir=depth_dir,
        max_width=args.foundation_stereo_max_width,
        model=args.foundation_stereo_model,
        overwrite=args.overwrite_depth,
        backend=getattr(args, "depth_backend", "auto"),
        min_working_distance_m=args.min_working_distance_m,
        max_working_distance_m=args.max_working_distance_m,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_detail_boost=args.clahe_detail_boost,
        **backend_options(args),
    )
    return np.load(depth_dir / "depth_m.npy")


def provide_sensor_depth(
    *,
    scene_dir: Path,
    depth_dir: Path,
    args: Any,
    dataset: str,
    scene_id: int,
    **_ignored: Any,
) -> np.ndarray:
    """Read the scene's measured depth from the collected-depth tree, e.g. a RealSense D4xx's own.

    Written to `depth_dir` in the same files a backend writes, so nothing downstream can tell the
    sources apart -- except `metadata.json`, whose `depth_source` keeps a later predicted run from
    mistaking this directory for its own cache.
    """
    import cv2

    from foundationpose_perception_pipeline.config import active_settings
    from foundationpose_perception_pipeline.io.bop import load_collected_depth_m

    collected_root = getattr(args, "collected_root", None)
    if collected_root is None:
        raise SystemExit(
            f"--depth-source {SENSOR_SOURCE} reads the collected-depth tree, and none is configured. "
            "Set `dataset.collected_depth_root` in the config profile, or pass --collected-root."
        )
    filename = active_settings().dataset.depth_filename
    collected_scene_dir = Path(collected_root) / dataset / args.split / f"{scene_id:06d}"
    depth_m = load_collected_depth_m(collected_scene_dir, filename)

    # The pose stage pairs this map with `rgb/000000.png` and its K, so it must be on that grid.
    base_image = cv2.imread(str(scene_dir / "rgb" / "000000.png"), cv2.IMREAD_UNCHANGED)
    if base_image is not None and base_image.shape[:2] != depth_m.shape:
        raise SystemExit(
            f"{collected_scene_dir / filename} is {depth_m.shape[1]}x{depth_m.shape[0]} but "
            f"rgb/000000.png is {base_image.shape[1]}x{base_image.shape[0]}; measured depth must be "
            "registered to the base camera."
        )

    camera = json.loads((scene_dir / "scene_camera.json").read_text(encoding="utf-8"))["0"]
    finite = np.isfinite(depth_m)
    metadata = {
        "depth_source": SENSOR_SOURCE,
        "scene_dir": str(scene_dir),
        "collected_depth_path": str(collected_scene_dir / filename),
        "camera_matrix": np.asarray(camera["cam_K"], dtype=np.float64).reshape(3, 3).tolist(),
        "depth_m_valid_fraction": float(finite.mean()),
        "depth_m_valid_min": float(np.nanmin(depth_m)) if finite.any() else None,
        "depth_m_valid_max": float(np.nanmax(depth_m)) if finite.any() else None,
        "depth_m_valid_median": float(np.nanmedian(depth_m)) if finite.any() else None,
    }
    depth_dir.mkdir(parents=True, exist_ok=True)
    np.save(depth_dir / "depth_m.npy", depth_m)
    # Last, as for a backend: both files present is what "this scene is done" means.
    (depth_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return depth_m


register_source(
    DepthSource(
        name=PREDICTED_SOURCE,
        provide=provide_predicted_depth,
        describe="predicted from the scene's stereo pair",
        uses_depth_backend=True,
    )
)
register_source(
    DepthSource(
        name=SENSOR_SOURCE,
        provide=provide_sensor_depth,
        describe="measured by the camera itself, read from the collected-depth tree",
        uses_depth_backend=False,
        needs_collected_depth=True,
        # Scoring a measurement against itself is zero by construction.
        depth_metrics_meaningful=False,
    )
)
