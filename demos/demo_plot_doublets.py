#!/usr/bin/env python3
"""Plot doublet values throughout iterations for all actuator disks.

Usage:
    python demos/demo_plot_doublets.py --case cases/cicular_vent
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

# Add src folder to python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

# Set matplotlib config dir before importing matplotlib
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt

from core.io.case_loader import CaseLoader
from solvers.actuator import ActuatorDiskCoupledSolver3D
from solvers.actuator.doublet_influence import pressure_jump_to_doublet_strength


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot and export actuator disk doublet values throughout iterations"
    )
    parser.add_argument(
        "--case",
        dest="case_dir",
        type=Path,
        required=True,
        help="Path to case directory containing case.yaml",
    )
    parser.add_argument("--mesh-level", type=int, default=0, help="Mesh level index")
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
        print("Error: Case does not use Actuator Disk Coupling.")
        return 1

    # Solve the system
    solver.solve()

    if not solver.convergence_history:
        print("No convergence history found.")
        return 0

    # Build lookup dictionary for disk properties
    disk_lookup = {disk.config.name: disk for disk in solver._disks}

    # Extract history of doublets
    data = []
    for record in solver.convergence_history:
        disk = disk_lookup.get(record.disk_name)
        if disk is None:
            continue
        mu = pressure_jump_to_doublet_strength(
            pressure_rise=record.pressure_rise,
            density=solver._density,
            reference_velocity=disk.reference_velocity,
            characteristic_length=disk.config.radius,
        )
        data.append(
            {
                "iteration": record.iteration,
                "disk_name": record.disk_name,
                "doublet_value": mu,
                "flow_rate": record.flow_rate,
                "pressure_rise": record.pressure_rise,
                "residual": record.pressure_residual,
            }
        )

    # Output directory
    adm_dir = case_dir / "out" / "adm"
    adm_dir.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    csv_path = adm_dir / "doublet_iterations.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "iteration",
                "disk_name",
                "doublet_value",
                "flow_rate",
                "pressure_rise",
                "residual",
            ],
        )
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    print(f"Exported data to: {csv_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    disk_names = sorted({row["disk_name"] for row in data})
    for name in disk_names:
        disk_rows = [row for row in data if row["disk_name"] == name]
        iterations = [row["iteration"] for row in disk_rows]
        doublets = [row["doublet_value"] for row in disk_rows]
        ax.plot(
            iterations,
            doublets,
            marker="o",
            linestyle="-",
            linewidth=1.5,
            label=name,
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Doublet value $\mu$ [$m^2/s$]")
    ax.set_title("Actuator Disk Doublet Strength Convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    plot_path = adm_dir / "doublet_iterations.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot to: {plot_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
