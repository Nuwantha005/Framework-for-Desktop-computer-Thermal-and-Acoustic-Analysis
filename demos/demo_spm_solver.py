
"""
Demo: Source Panel Method Solver.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.io import generate_circle
from solvers.panel2d.spm import SourcePanelSolver
from visualization import StreamlineVisualizer

def main():
    print("Running Source Panel Method Demo: Circle...")
    
    # 1. Create Geometry (Circle)
    mesh = generate_circle(radius=1.0, num_panels=50)
    
    # 2. Setup Solver
    v_inf = 10.0
    aoa = 0.0
    solver = SourcePanelSolver(mesh, v_inf=v_inf, aoa=aoa)
    
    # 3. Solve
    solver.solve()
    
    if solver.Cp is None:
        print("Solver failed to compute Cp.")
        return

    print(f"Solved. Cp range: [{min(solver.Cp):.4f}, {max(solver.Cp):.4f}]")
    
    # 4. Visualize with Streamlines
    print("Generating streamline visualization...")
    viz = StreamlineVisualizer(mesh, v_inf, aoa, solver.sigma)
    
    output_path = Path(__file__).parent / "out/demo_spm_circle_streamlines.png"
    viz.plot_streamlines(
        x_range=(-2.5, 2.5),
        y_range=(-2.5, 2.5),
        grid_resolution=(120, 120),
        streamline_density=1.2,
        streamline_start='left',
        show_body=True,
        show_cp=False,
        save_path=output_path
    )

    # Contour plot for verification
    contour_path = Path(__file__).parent / "out/demo_spm_circle_contours.png"
    viz.plot_velocity_contours(
        x_range=(-2.5, 2.5),
        y_range=(-2.5, 2.5),
        grid_resolution=(120, 120),
        levels=30,
        save_path=contour_path
    )
    
    print(f"Done. Streamline plot saved to {output_path}")

if __name__ == "__main__":
    main()
