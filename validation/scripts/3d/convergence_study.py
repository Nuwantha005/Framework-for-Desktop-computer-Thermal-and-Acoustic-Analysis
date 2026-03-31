#!/usr/bin/env python3
"""
3D Grid Convergence Study for Panel Method.

Runs a parametric case at all specified mesh levels, calculates metrics
such as the velocity at a specific test point and the $L_{\infty}$ norm
of the surface velocity, and plots the convergence behavior.

Usage:
    python validation/scripts/3d/convergence_study.py cases/sphere_flow
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.io.case_loader import CaseLoader
from solvers.factory import SolverFactory


def run_convergence(case_dir: Path, test_point: np.ndarray):
    print(f"Running grid convergence study for {case_dir.name}...")
    
    # Load just to find how many mesh levels exist
    _, config = CaseLoader.load(case_dir / "case.yaml")
    comp = config.components[0]
    num_levels = len(comp.mesh_levels) if comp.mesh_levels else 1
    
    num_panels_list = []
    test_pt_vel_list = []
    max_vt_list = []
    
    freestream = config.get_freestream_velocity()
    
    for level in range(num_levels):
        print(f"\nEvaluating Mesh Level {level}...")
        scene, _ = CaseLoader.load(case_dir / "case.yaml", mesh_level_index=level)
        mesh = scene.assemble()
        
        num_panels = mesh.num_panels
        num_panels_list.append(num_panels)
        print(f"  Panels: {num_panels}")
        
        solver = SolverFactory.create(
            config=config.solver,
            mesh=mesh,
            v_inf=freestream[0],
            aoa=0.0
        )
        solver._v_inf = np.asarray(freestream, dtype=np.float64)
        solver.solve()
        
        # Metric 1: Max surface velocity (L_inf)
        vt_max = np.max(np.linalg.norm(solver.surface_velocity, axis=1))
        max_vt_list.append(vt_max)
        print(f"  Max Vt: {vt_max:.6f}")
        
        # Metric 2: Velocity at test point
        vel_at_pt = solver.velocity_at(test_point)[0]
        vel_mag = np.linalg.norm(vel_at_pt)
        test_pt_vel_list.append(vel_mag)
        print(f"  Velocity at test point: {vel_mag:.6f}")

    return num_panels_list, test_pt_vel_list, max_vt_list


def plot_convergence(out_dir, num_panels, test_pt_vel, max_vt, test_point):
    plt.style.use('seaborn-v0_8-darkgrid')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Velocity at Test Point
    ax1.plot(num_panels, test_pt_vel, marker='o', linestyle='-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Panels')
    ax1.set_ylabel(f'Velocity Magnitude [m/s]')
    ax1.set_title(f'Convergence of Velocity at {test_point}')
    ax1.set_xscale('log')
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    
    # Right: Max Surface Velocity
    ax2.plot(num_panels, max_vt, marker='s', linestyle='-', color='orange', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Panels')
    ax2.set_ylabel(r'Maximum Surface Velocity ($V_t$) [m/s]')
    ax2.set_title(r'Convergence of $L_{\infty}$ norm of $V_t$')
    ax2.set_xscale('log')
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    
    out_file = out_dir / "grid_convergence.png"
    plt.savefig(out_file, dpi=150)
    print(f"\nSaved convergence plot to {out_file}")


def main():
    parser = argparse.ArgumentParser(description="3D Panel Method Grid Convergence Study")
    parser.add_argument("case_dir", type=Path, help="Path to case directory")
    parser.add_argument("--test-point", type=float, nargs=3, default=[0.0, 1.5, 0.0],
                       help="X Y Z coordinates of the test point (default: 0.0 1.5 0.0)")
    
    args = parser.parse_args()
    case_dir = args.case_dir.resolve()
    
    if not case_dir.exists():
        print(f"Error: Case directory not found: {case_dir}")
        return 1
        
    out_dir = case_dir / "out" / "panel_solver" / "convergence"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    test_pt = np.array(args.test_point)
    panels, pt_vels, max_vts = run_convergence(case_dir, test_pt)
    
    plot_convergence(out_dir, panels, pt_vels, max_vts, args.test_point)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
