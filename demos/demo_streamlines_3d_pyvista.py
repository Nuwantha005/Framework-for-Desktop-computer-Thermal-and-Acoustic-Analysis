#!/usr/bin/env python3
"""
Demo: 3D streamline visualization from saved volume fields.

Reads a saved VTK structured grid (volume_fields.vts), seeds streamlines from
case inlet regions, overlays the casing geometry transparently, and saves a
still image plus an animated GIF preview.

Usage:
    python demos/demo_streamlines_3d_pyvista.py cases/pc_casing

Outputs (default):
    <case>/out/panel_solver/streamlines/streamlines.png
    <case>/out/panel_solver/streamlines/streamlines.gif
"""

from __future__ import annotations

import argparse
import inspect
import shutil
import sys
from pathlib import Path

import numpy as np

# Add src to import path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.io.case_loader import CaseLoader
from solvers.actuator.disk_mesh import (
    generate_actuator_disk_mesh,
    generate_rectangular_boundary_mesh,
)


def _require_pyvista():
    try:
        import pyvista as pv

        return pv
    except ImportError as exc:
        raise ImportError(
            "PyVista is required for streamline visualization. Install with `pip install pyvista`."
        ) from exc


def _resolve_volume_path(case_dir: Path, override: Path | None) -> Path:
    if override is not None:
        return override
    return case_dir / "out" / "panel_solver" / "volume_fields.vts"


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        raise ValueError("Normal vector must be nonzero")
    return vector / norm


def _build_inlet_seed_mesh(config, seed_density: float):
    normal = _normalize(np.asarray(config.normal, dtype=np.float64))
    if config.shape == "circle":
        if config.radius is None:
            raise ValueError(f"Inlet '{config.name}' missing radius")
        n_r = max(1, int(round(config.n_r * seed_density)))
        n_theta = max(3, int(round(config.n_theta * seed_density)))
        return generate_actuator_disk_mesh(
            center=config.center,
            normal=normal,
            radius=float(config.radius),
            n_r=n_r,
            n_theta=n_theta,
        )
    if config.shape == "rectangle":
        if config.width is None or config.height is None:
            raise ValueError(f"Inlet '{config.name}' missing width/height")
        n_w = max(1, int(round(config.n_r * seed_density)))
        n_h = max(1, int(round(config.n_theta * seed_density)))
        return generate_rectangular_boundary_mesh(
            center=config.center,
            normal=normal,
            width=float(config.width),
            height=float(config.height),
            n_w=n_w,
            n_h=n_h,
        )
    raise NotImplementedError(f"Inlet shape '{config.shape}' not supported")


def _collect_inlet_seeds(inlets, seed_density: float, seed_offset: float) -> np.ndarray:
    seeds = []
    for inlet in inlets:
        mesh = _build_inlet_seed_mesh(inlet, seed_density)
        normal = _normalize(np.asarray(inlet.normal, dtype=np.float64))
        offset = normal * seed_offset
        seeds.append(mesh.centers + offset)
    if not seeds:
        return np.empty((0, 3), dtype=np.float64)
    return np.vstack(seeds)


def _filter_seeds_to_bounds(seeds: np.ndarray, bounds: tuple[float, ...]) -> np.ndarray:
    if seeds.size == 0:
        return seeds
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    mask = (
        (seeds[:, 0] >= xmin)
        & (seeds[:, 0] <= xmax)
        & (seeds[:, 1] >= ymin)
        & (seeds[:, 1] <= ymax)
        & (seeds[:, 2] >= zmin)
        & (seeds[:, 2] <= zmax)
    )
    return seeds[mask]


def _select_streamline_params(method, params: dict) -> dict:
    try:
        supported = set(inspect.signature(method).parameters.keys())
    except (TypeError, ValueError):
        return params

    filtered = {key: value for key, value in params.items() if key in supported}

    if "step_length" in params and "step_length" not in supported and "max_step_length" in supported:
        filtered["max_step_length"] = params["step_length"]

    return filtered


def _compute_streamlines(pv, dataset, seeds, vector_name: str, params: dict):
    source = pv.PolyData(seeds)
    if hasattr(dataset, "streamlines_from_source"):
        method = dataset.streamlines_from_source
        resolved = _select_streamline_params(method, params)
        return method(source, vectors=vector_name, **resolved)
    method = dataset.streamlines
    resolved = _select_streamline_params(method, params)
    return method(vectors=vector_name, source=source, **resolved)


def _ensure_speed_field(streamlines) -> str:
    if "speed" in streamlines.point_data:
        return "speed"
    if "velocity" in streamlines.point_data:
        velocity = np.asarray(streamlines.point_data["velocity"], dtype=np.float64)
        streamlines.point_data["speed"] = np.linalg.norm(velocity, axis=1)
        return "speed"
    return ""


def _build_plotter(pv, dataset, casing_mesh, streamlines, speed_name: str, window_size: tuple[int, int]):
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    if casing_mesh is not None:
        plotter.add_mesh(
            casing_mesh,
            color="#e6e6e6",
            opacity=0.18,
            show_edges=False,
            smooth_shading=True,
        )
    if streamlines.n_points > 0:
        plotter.add_mesh(
            streamlines,
            scalars=speed_name if speed_name else None,
            cmap="viridis",
            line_width=2.0,
            render_lines_as_tubes=True,
        )
    plotter.set_background("#0b0f14")
    plotter.add_axes()
    bounds = dataset.bounds
    center = (
        0.5 * (bounds[0] + bounds[1]),
        0.5 * (bounds[2] + bounds[3]),
        0.5 * (bounds[4] + bounds[5]),
    )
    plotter.camera.focal_point = center
    plotter.camera_position = "iso"
    return plotter


def _prepare_offscreen(pv):
    pv.OFF_SCREEN = True
    if hasattr(pv, "start_xvfb"):
        if shutil.which("Xvfb") is None:
            return
        try:
            pv.start_xvfb()
        except Exception:
            pass


def _rotate_camera(plotter, angle: float) -> None:
    camera = plotter.camera
    azimuth = getattr(camera, "azimuth", None)
    if callable(azimuth):
        azimuth(angle)
        return
    vtk_azimuth = getattr(camera, "Azimuth", None)
    if callable(vtk_azimuth):
        vtk_azimuth(angle)


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize 3D streamlines from volume fields")
    parser.add_argument("case_dir", type=Path, help="Path to case directory")
    parser.add_argument(
        "--volume-field",
        type=Path,
        default=None,
        help="Override volume field path (default: <case>/out/panel_solver/volume_fields.vts)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <case>/out/panel_solver/streamlines)",
    )
    parser.add_argument("--image-name", type=str, default="streamlines.png", help="Image filename")
    parser.add_argument("--gif-name", type=str, default="streamlines.gif", help="GIF filename")
    parser.add_argument("--no-image", action="store_true", help="Skip still image")
    parser.add_argument("--no-gif", action="store_true", help="Skip GIF animation")
    parser.add_argument("--frames", type=int, default=90, help="Number of frames in GIF")
    parser.add_argument("--fps", type=int, default=20, help="GIF frames per second")
    parser.add_argument("--seed-offset", type=float, default=-0.002, help="Offset along inlet normal [m]")
    parser.add_argument("--seed-density", type=float, default=1.0, help="Seed density multiplier")
    parser.add_argument("--seed-step", type=int, default=1, help="Subsample seeds (keep every Nth)")
    parser.add_argument(
        "--step-length",
        type=float,
        default=None,
        help="Streamline integration step length [m]",
    )
    parser.add_argument(
        "--initial-step",
        type=float,
        default=None,
        help="Streamline initial step length [m]",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=None,
        help="Streamline max integration time",
    )
    args = parser.parse_args()

    case_dir = args.case_dir.resolve()
    if not case_dir.exists():
        print(f"Error: case directory not found: {case_dir}")
        return 1

    volume_path = _resolve_volume_path(case_dir, args.volume_field)
    if not volume_path.exists():
        print(f"Error: volume field not found: {volume_path}")
        return 1

    pv = _require_pyvista()
    _prepare_offscreen(pv)

    case = CaseLoader.load_case(case_dir)
    if case.mesh.dimension != 3:
        print(f"Error: expected 3D mesh, got dimension={case.mesh.dimension}")
        return 1

    if not case.config.inlets:
        print("Error: no inlets defined in case.yaml")
        return 1

    output_dir = args.output_dir or (case_dir / "out" / "panel_solver" / "streamlines")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = pv.read(str(volume_path))
    if "velocity" not in dataset.point_data:
        print("Error: volume field missing 'velocity' vector data")
        return 1
    dataset.set_active_vectors("velocity")

    seeds = _collect_inlet_seeds(case.config.inlets, args.seed_density, args.seed_offset)
    seeds = _filter_seeds_to_bounds(seeds, dataset.bounds)
    if args.seed_step > 1 and seeds.shape[0] > 0:
        seeds = seeds[:: args.seed_step]

    if seeds.shape[0] == 0:
        print("Error: no valid seeds inside the volume bounds")
        return 1

    bounds = dataset.bounds
    diag = np.linalg.norm([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]])
    step_length = args.step_length if args.step_length is not None else diag / 200.0
    initial_step = args.initial_step if args.initial_step is not None else diag / 400.0
    max_time = args.max_time if args.max_time is not None else diag * 2.0

    streamline_params = {
        "integration_direction": "forward",
        "max_time": float(max_time),
        "initial_step_length": float(initial_step),
        "step_length": float(step_length),
    }

    streamlines = _compute_streamlines(pv, dataset, seeds, "velocity", streamline_params)
    speed_name = _ensure_speed_field(streamlines)

    casing_mesh = case.mesh.to_pyvista().clean()

    window_size = (1600, 1000)
    plotter = _build_plotter(pv, dataset, casing_mesh, streamlines, speed_name, window_size)

    image_path = output_dir / args.image_name
    gif_path = output_dir / args.gif_name

    if not args.no_gif:
        try:
            plotter.open_gif(str(gif_path), fps=args.fps, loop=0)
        except TypeError:
            plotter.open_gif(str(gif_path), fps=args.fps)

    plotter.show(auto_close=False)

    if not args.no_image:
        plotter.screenshot(str(image_path))

    if not args.no_gif:
        for _ in range(args.frames):
            _rotate_camera(plotter, 360.0 / args.frames)
            plotter.write_frame()

    plotter.close()

    print("Streamline visualization saved:")
    if not args.no_image:
        print(f"  - {image_path}")
    if not args.no_gif:
        print(f"  - {gif_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
