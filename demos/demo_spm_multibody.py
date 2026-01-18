
"""
Demo: Source Panel Method Solver - Multi-body.
Two circles in tandem.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.io import generate_circle
from core.geometry import Scene, Component, Transform
from core.geometry.primitives import rotation_matrix_z
from solvers.panel2d.spm import SourcePanelSolver
from visualization import StreamlineVisualizer

def main():
    print("Running Source Panel Method Demo: Multi-body...")
    
    # 1. Create Meshes
    mesh1 = generate_circle(radius=0.5, num_panels=40)
    mesh2 = generate_circle(radius=0.5, num_panels=40)
    
    # 2. Create Components
    t1 = Transform(translation=np.array([-1.0, 0.0, 0.0]), 
                   rotation_matrix=np.eye(3))
    comp1 = Component(name="FrontCircle", local_mesh=mesh1, transform=t1)
    
    t2 = Transform(translation=np.array([1.0, 0.0, 0.0]), 
                   rotation_matrix=np.eye(3))
    comp2 = Component(name="RearCircle", local_mesh=mesh2, transform=t2)
    
    # 3. Create Scene and Assemble
    scene = Scene(name="TwoCircles", components=[comp1, comp2], freestream=np.array([10.0, 0.0, 0.0]))
    global_mesh = scene.assemble()
    
    print(f"Assembled global mesh with {global_mesh.num_panels} panels")
    
    # 4. Solve
    v_inf = 10.0
    aoa = 0.0
    solver = SourcePanelSolver(global_mesh, v_inf=v_inf, aoa=aoa)
    solver.solve()
    
    if solver.Cp is None:
        print("Solver failed.")
        return
    
    print(f"Solved. Cp range: [{min(solver.Cp):.4f}, {max(solver.Cp):.4f}]")
    
    # 5. Visualize with Streamlines
    print("Generating streamline visualization...")
    viz = StreamlineVisualizer(global_mesh, v_inf, aoa, solver.sigma)
    
    output_path = Path(__file__).parent / "out/demo_spm_multibody_streamlines.png"
    viz.plot_streamlines(
        x_range=(-3.0, 3.0),
        y_range=(-2.0, 2.0),
        grid_resolution=(150, 100),
        streamline_density=1.2,
        streamline_start='left',
        show_body=True,
        show_cp=False,
        save_path=output_path
    )
    
    print(f"Done. Streamline plot saved to {output_path}")

if __name__ == "__main__":
    main()
