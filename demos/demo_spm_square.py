
"""
Demo: Source Panel Method Solver - Square.
Tests constant source panel method on a non-smooth body (square).
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.io import generate_rectangle
from solvers.panel2d.spm import SourcePanelSolver
from visualization import StreamlineVisualizer

def main():
    print("Running Source Panel Method Demo: Square...")
    
    # 1. Create Geometry (Square)
    mesh = generate_rectangle(width=1.0, height=1.0, center=(0.0, 0.0), 
                              num_panels_x=10, num_panels_y=10)
    
    print(f"Created square mesh with {mesh.num_panels} panels")
    
    # 2. Setup Solver
    v_inf = 5.0
    aoa = 0.0
    solver = SourcePanelSolver(mesh, v_inf=v_inf, aoa=aoa)
    
    # 3. Solve
    solver.solve()
    
    if solver.Cp is None:
        print("Solver failed.")
        return

    print(f"Solved. Cp range: [{min(solver.Cp):.4f}, {max(solver.Cp):.4f}]")
    
    # 4. Visualize with Streamlines
    print("Generating streamline visualization (parallel)...")
    viz = StreamlineVisualizer(mesh, v_inf, aoa, solver.sigma)
    
    # Contour plot for verification
    contour_path = Path(__file__).parent / "out/demo_spm_square_contours.png"
    viz.plot_velocity_contours(
        x_range=(-2.0, 2.0),
        y_range=(-1.5, 1.5),
        grid_resolution=(150, 150),
        levels=30,
        save_path=contour_path
    )
    
    # Streamline plot
    output_path = Path(__file__).parent / "out/demo_spm_square_streamlines.png"
    viz.plot_streamlines(
        x_range=(-2.0, 2.0),
        y_range=(-1.5, 1.5),
        grid_resolution=(200, 160), # Higher resolution
        streamline_density=1.2,
        streamline_start='left',
        show_body=True,
        show_cp=False,
        save_path=output_path,
        num_cores=6
    )
    
    print(f"Done.")
    print(f"  - Contours saved to {contour_path}")
    print(f"  - Streamlines saved to {output_path}")

if __name__ == "__main__":
    main()
