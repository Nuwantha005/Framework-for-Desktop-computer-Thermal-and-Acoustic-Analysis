#!/usr/bin/env python3
"""
Demo: Streamline Visualization

Loads a case, solves it, and plots streamlines.
Usage:
    python demo_streamlines.py <case_dir> [--show] [--save] [--cores N]
    
Example:
    python demo_streamlines.py ../cases/cylinder_flow --show
    python demo_streamlines.py ../cases/single_square --save --cores 6
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.io import CaseLoader
from solvers.panel2d.spm import SourcePanelSolver
from visualization import Visualizer
from visualization.field2d import VelocityField2D


def main():
    parser = argparse.ArgumentParser(description="Plot streamlines from case file")
    parser.add_argument("case_dir", type=str, help="Path to case directory")
    parser.add_argument("--show", action="store_true", help="Display plot interactively")
    parser.add_argument("--save", action="store_true", help="Save plot to case_dir/out/")
    parser.add_argument("--cores", type=int, default=6, help="Number of CPU cores (default: 6)")
    parser.add_argument("--protect", action="store_true", help="Save to timestamped subfolder")
    args = parser.parse_args()
    
    if not args.show and not args.save:
        args.save = True
    
    # Resolve path and load case
    case_dir = Path(args.case_dir).resolve()
    case = CaseLoader.load_case(case_dir)
    
    print(f"Loaded: {case.name}")
    print(f"  Panels: {case.num_panels}")
    print(f"  V_inf: {case.v_inf}, AoA: {case.aoa}°")
    
    # Solve
    print("Solving...")
    solver = SourcePanelSolver(case.mesh, v_inf=case.v_inf, aoa=case.aoa)
    solver.solve()
    
    if solver.Cp is None:
        print("Error: Solver failed")
        sys.exit(1)
    
    print(f"  Cp range: [{min(solver.Cp):.4f}, {max(solver.Cp):.4f}]")
    
    # Get visualization settings directly from case
    x_range = case.x_range
    y_range = case.y_range
    resolution = case.resolution
    
    print(f"Domain: x={x_range}, y={y_range}")
    
    # Compute velocity field
    print(f"Computing velocity field ({resolution[0]}x{resolution[1]}, {args.cores} cores)...")
    field = VelocityField2D(case.mesh, case.v_inf, case.aoa, solver.sigma)
    XX, YY, Vx, Vy = field.compute(x_range, y_range, resolution, num_cores=args.cores)
    
    # Plot
    output_dir = case.output_dir if args.save else None
    viz = Visualizer(output_dir=output_dir, protect_overwrite=args.protect)
    
    viz.create_figure(figsize=(12, 8))
    viz.plot_streamlines(XX, YY, Vx, Vy, case.mesh, 
                         title=f"{case.name} - Streamlines")
    
    save_name = "streamlines.png" if args.save else None
    viz.finalize(save=save_name, show=args.show)
    
    print("Done.")


if __name__ == "__main__":
    main()
