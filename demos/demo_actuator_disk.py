#!/usr/bin/env python3
"""Run a 3D case with optional actuator disk coupling.

Usage:
    python demos/demo_actuator_disk.py --case cases/cicular_vent
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.io.case_loader import CaseLoader
from core.geometry.io.vtk_export import export_solution_vtk
from solvers.actuator import ActuatorDiskCoupledSolver3D


def _surface_export(case_dir: Path, solver) -> Path:
    """Export body surface quantities to VTP."""
    mesh = solver.mesh
    output_dir = case_dir / "out" / "adm"
    output_dir.mkdir(parents=True, exist_ok=True)
    surface_path = output_dir / "body_surface_with_adm.vtp"

    velocity = np.asarray(solver.surface_velocity, dtype=np.float64)
    mesh.cell_data["velocity_vector"] = velocity
    mesh.cell_data["velocity_magnitude"] = np.linalg.norm(velocity, axis=1)
    if hasattr(solver, "Cp"):
        mesh.cell_data["Cp"] = np.asarray(solver.Cp, dtype=np.float64)

    export_solution_vtk(mesh, surface_path)
    return surface_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an actuator disk case")
    parser.add_argument(
        "--case",
        dest="case_dir",
        type=Path,
        required=True,
        help="Path to case directory containing case.yaml",
    )
    parser.add_argument("--mesh-level", type=int, default=0, help="Mesh level index")
    parser.add_argument(
        "--no-surface-export",
        action="store_true",
        help="Skip body surface VTP export",
    )
    args = parser.parse_args()

    case_dir = args.case_dir.resolve()
    if not (case_dir / "case.yaml").exists():
        print(f"Error: case.yaml not found in {case_dir}")
        return 1

    case = CaseLoader.load_case(case_dir, mesh_level_index=args.mesh_level)
    print("=" * 72)
    print(f"Case: {case.name}")
    print(f"Mesh level: {args.mesh_level}")
    print(f"Panels: {case.mesh.num_panels}")
    print(f"Freestream: {case.freestream.tolist()} m/s")
    print(f"Actuator disks: {len(case.config.actuator_disks)}")
    print("=" * 72)

    solver = case.create_solver()
    if not isinstance(solver, ActuatorDiskCoupledSolver3D):
        print("No actuator disks configured; running the normal case solver.")

    solver.solve()

    print("\nSolve summary")
    print("-" * 72)
    if isinstance(solver, ActuatorDiskCoupledSolver3D):
        for result in solver.actuator_results:
            status = "converged" if result.converged else "not converged"
            print(
                f"{result.name}: Q={result.flow_rate:.6e} m^3/s, "
                f"dp={result.pressure_rise:.6e} Pa, "
                f"iterations={result.iterations}, {status}"
            )
            if result.warning:
                print(f"  warning: {result.warning}")
        print(f"Convergence records: {len(solver.convergence_history)}")
        print(f"ADM outputs: {case_dir / 'out' / 'adm'}")
        print(f"Solver run bundle: {case_dir / 'out' / 'solverRuns' / 'adm_solution.npz'}")

    if not args.no_surface_export:
        surface_path = _surface_export(case_dir, solver)
        print(f"Body surface export: {surface_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
