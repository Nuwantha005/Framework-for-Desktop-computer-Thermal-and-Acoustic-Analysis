#!/usr/bin/env python3
"""
Calculate and plot velocities on a 2D cut plane through a 3D volume.
Useful for debugging 3D calculations independently of PyVista/VTK exports.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to import path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from core.io.case_loader import CaseLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case_dir", type=Path, help="Path to case directory")
    parser.add_argument("--mesh-level", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=100, help="Grid resolution")
    parser.add_argument("--axis", type=str, choices=['x', 'y', 'z'], default='z', help="Axis normal to the cut plane")
    parser.add_argument("--offset", type=float, default=0.0, help="Offset along the normal axis")
    args = parser.parse_args()

    case_dir = args.case_dir.resolve()
    case = CaseLoader.load_case(case_dir, mesh_level_index=args.mesh_level)
    config = case.config
    freestream = case.freestream
    solver = case.create_solver()
    if config.actuator_disks:
        print(f"Actuator disks detected: {len(config.actuator_disks)}. Using ADM-coupled solver.")
    print("Solving panel method...")
    solver.solve()

    print(f"Creating grid for {args.axis}={args.offset} cut...")
    res = args.resolution
    
    # Get domain from config
    x_range = config.visualization.get_x_range()
    y_range = config.visualization.get_y_range()
    z_range = config.visualization.get_z_range()

    if args.axis == 'z':
        x = np.linspace(x_range[0], x_range[1], res)
        y = np.linspace(y_range[0], y_range[1], res)
        xx, yy = np.meshgrid(x, y)
        zz = np.full_like(xx, args.offset)
        points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        xlabel, ylabel = 'X', 'Y'
    elif args.axis == 'y':
        x = np.linspace(x_range[0], x_range[1], res)
        z = np.linspace(z_range[0], z_range[1], res)
        xx, zz = np.meshgrid(x, z)
        yy = np.full_like(xx, args.offset)
        points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        xlabel, ylabel = 'X', 'Z'
    else:  # 'x'
        y = np.linspace(y_range[0], y_range[1], res)
        z = np.linspace(z_range[0], z_range[1], res)
        yy, zz = np.meshgrid(y, z)
        xx = np.full_like(yy, args.offset)
        points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        xlabel, ylabel = 'Y', 'Z'

    print("Computing velocities...")
    vel = solver.velocity_at(points)
    speed = np.linalg.norm(vel, axis=1)

    speed_2d = speed.reshape((res, res))

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Mask out extreme values (e.g. inside body or singularities)
    v_inf_mag = np.linalg.norm(freestream)
    if v_inf_mag > 1e-12:
        speed_2d_plot = np.where(speed_2d > 5 * v_inf_mag, np.nan, speed_2d)
    else:
        speed_2d_plot = speed_2d
    
    if args.axis == 'z':
        c = ax.contourf(xx, yy, speed_2d_plot, levels=50, cmap='viridis')
    elif args.axis == 'y':
        c = ax.contourf(xx, zz, speed_2d_plot, levels=50, cmap='viridis')
    else:
        c = ax.contourf(yy, zz, speed_2d_plot, levels=50, cmap='viridis')
        
    fig.colorbar(c, ax=ax, label='Velocity Magnitude')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Velocity Magnitude Cut Plane (Normal to {args.axis.upper()} at {args.offset})")
    ax.set_aspect('equal')

    out_dir = case_dir / "out" / "panel_solver" / "cuts"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # normal format e.g. nx0_ny0_nz1
    nx, ny, nz = 0, 0, 0
    if args.axis == 'x': nx = 1
    elif args.axis == 'y': ny = 1
    elif args.axis == 'z': nz = 1
    
    filename = f"cut_{args.axis}_{args.offset}_nx{nx}_ny{ny}_nz{nz}.png"
    out_path = out_dir / filename
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
