#!/usr/bin/env python3
"""Run a 3D case and export panel + volume fields for ParaView.

This is a generic 3D case runner. It solves the panel case and writes two VTK
files under ``<case>/out/panel_solver/``:

- ``surface_panels.vtp``: panel surface with surface quantities
- ``volume_fields.vts``: structured volume grid with velocity/pressure fields

Use ``validation/scripts/3d/compare_surface.py`` as the reference workflow for
3D case loading and solver setup.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Add src to import path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.io.case_loader import CaseLoader
from solvers.factory import SolverFactory
from core.geometry.io.vtk_export import export_solution_vtk


def _require_pyvista():
    try:
        import pyvista as pv

        return pv
    except ImportError as exc:
        raise ImportError(
            "PyVista is required for ParaView export. Install with `pip install pyvista`."
        ) from exc


def _surface_export(case_dir: Path, mesh, solver, config, surface_path: Path) -> None:
    """Export panel-surface results to VTP."""
    cp = np.asarray(solver.Cp, dtype=np.float64)
    v_surface = np.asarray(solver.surface_velocity, dtype=np.float64)
    vt = np.linalg.norm(v_surface, axis=1)

    rho = float(config.fluid.density)
    p_ref = float(config.fluid.reference_pressure)
    v_inf = float(np.linalg.norm(config.get_freestream_velocity()))
    q_inf = 0.5 * rho * v_inf * v_inf
    p_surface = p_ref + q_inf * cp

    mesh.cell_data["Vt"] = vt
    mesh.cell_data["Cp"] = cp
    mesh.cell_data["pressure"] = p_surface
    mesh.cell_data["pressure_gauge"] = p_surface - p_ref
    mesh.cell_data["velocity_vector"] = v_surface

    export_solution_vtk(mesh, surface_path)


def _volume_export(
    pv,
    solver,
    config,
    x_range,
    y_range,
    z_range,
    resolution,
    volume_path: Path,
) -> None:
    """Export structured 3D volume fields to VTS."""
    nx, ny, nz = resolution
    x = np.linspace(x_range[0], x_range[1], nx, dtype=np.float64)
    y = np.linspace(y_range[0], y_range[1], ny, dtype=np.float64)
    z = np.linspace(z_range[0], z_range[1], nz, dtype=np.float64)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    # Create the PyVista grid FIRST so we can use its correctly ordered points
    grid = pv.StructuredGrid(xx, yy, zz)
    points = grid.points  # This correctly uses VTK's Fortran ordering

    # Convert our mesh to PV PolyData for interior masking
    verts = solver._mesh.nodes
    faces = []
    for p in solver._mesh.panels:
        faces.append(len(p))
        faces.extend(p)
    surface_pd = pv.PolyData(verts, faces)
    
    print("Masking interior points...")
    enclosed = grid.select_enclosed_points(surface_pd, check_surface=False)
    mask = enclosed['SelectedPoints'] == 1
    
    vel = np.zeros((points.shape[0], 3), dtype=np.float64)
    vel[:] = np.nan
    
    exterior_mask = ~mask
    exterior_points = points[exterior_mask]
    
    print(f"Computing velocity for {len(exterior_points)} exterior points out of {len(points)}...")
    vel[exterior_mask] = solver.velocity_at(exterior_points)

    speed = np.linalg.norm(vel, axis=1)

    rho = float(config.fluid.density)
    p_ref = float(config.fluid.reference_pressure)
    v_inf = float(np.linalg.norm(config.get_freestream_velocity()))
    q_inf = 0.5 * rho * v_inf * v_inf

    if v_inf > 1e-14:
        cp = 1.0 - (speed / v_inf) ** 2
    else:
        cp = np.full_like(speed, np.nan, dtype=np.float64)
    pressure = p_ref + q_inf * cp

    grid.point_data["velocity"] = vel
    grid.point_data["velocity_magnitude"] = speed
    grid.point_data["Cp"] = cp
    grid.point_data["pressure"] = pressure
    grid.point_data["pressure_gauge"] = pressure - p_ref
    grid.save(str(volume_path))


def _run_panel_solver(case_dir: Path, mesh_level: int):
    """Load 3D case and solve panel solver using full freestream vector."""
    scene, config = CaseLoader.load(case_dir / "case.yaml", mesh_level_index=mesh_level)
    mesh = scene.assemble()
    if mesh.dimension != 3:
        raise ValueError(f"Expected a 3D case/mesh, got dimension={mesh.dimension}")

    freestream = np.asarray(config.get_freestream_velocity(), dtype=np.float64)
    solver = SolverFactory.create(
        config=config.solver,
        mesh=mesh,
        v_inf=float(np.linalg.norm(freestream)),
        aoa=0.0,
    )
    # Keep full 3D freestream direction (same pattern as compare_surface.py).
    solver._v_inf = freestream
    solver.solve()
    return mesh, solver, config


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 3D case and export ParaView fields")
    parser.add_argument("case_dir", type=Path, help="Path to case directory")
    parser.add_argument("--solver-type", type=str, default=None, help="Panel solver type override")
    parser.add_argument("--mesh-level", type=int, default=0, help="Mesh level index")
    parser.add_argument(
        "--resolution",
        nargs=3,
        type=int,
        metavar=("NX", "NY", "NZ"),
        default=None,
        help="Override volume grid resolution (default: case visualization resolution in case.yaml)",
    )
    parser.add_argument(
        "--x-range",
        nargs=2,
        type=float,
        metavar=("XMIN", "XMAX"),
        default=None,
        help="Override x-range (default: case visualization domain)",
    )
    parser.add_argument(
        "--y-range",
        nargs=2,
        type=float,
        metavar=("YMIN", "YMAX"),
        default=None,
        help="Override y-range (default: case visualization domain)",
    )
    parser.add_argument(
        "--z-range",
        nargs=2,
        type=float,
        metavar=("ZMIN", "ZMAX"),
        default=None,
        help="Override z-range (default: case visualization domain)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <case>/out/panel_solver)",
    )
    args = parser.parse_args()

    pv = _require_pyvista()

    case_dir = args.case_dir.resolve()
    if not case_dir.exists():
        print(f"Error: case directory not found: {case_dir}")
        return 1

    case_yaml = case_dir / "case.yaml"
    if not case_yaml.exists():
        print(f"Error: case.yaml not found at {case_yaml}")
        return 1

    try:
        mesh, solver, config = _run_panel_solver(case_dir, args.mesh_level)
    except Exception as exc:
        print(f"Error while running 3D panel solver: {exc}")
        return 1

    print(f"Loaded case: {config.name}")
    print(f"Panels: {mesh.num_panels}")
    print("Panel solver completed.")

    resolution = tuple(args.resolution) if args.resolution is not None else tuple(config.visualization.resolution)
    if len(resolution) != 3:
        print(f"Error: expected 3D resolution (nx, ny, nz), got: {resolution}")
        return 1

    x_range = tuple(args.x_range) if args.x_range is not None else tuple(config.visualization.get_x_range())
    y_range = tuple(args.y_range) if args.y_range is not None else tuple(config.visualization.get_y_range())
    z_range = tuple(args.z_range) if args.z_range is not None else tuple(config.visualization.get_z_range())

    output_dir = args.output_dir or (case_dir / "out" / "panel_solver")
    output_dir.mkdir(parents=True, exist_ok=True)

    surface_path = output_dir / "surface_panels.vtp"
    volume_path = output_dir / "volume_fields.vts"

    _surface_export(case_dir, mesh, solver, config, surface_path)
    _volume_export(pv, solver, config, x_range, y_range, z_range, resolution, volume_path)

    print("Export complete:")
    print(f"  - {surface_path}")
    print(f"  - {volume_path}")
    print("Open both files in ParaView (same scene) for surface + volume inspection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
