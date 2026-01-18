
"""
Run a simulation case defined by a YAML config file.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.io import CaseLoader
from solvers.panel2d.spm import SourcePanelSolver

def main():
    parser = argparse.ArgumentParser(description="Run Panel Method Solver Case")
    parser.add_argument("case_file", type=str, help="Path to YAML case file")
    args = parser.parse_args()
    
    case_path = Path(args.case_file).resolve()
    if not case_path.exists():
        print(f"Error: Case file not found: {case_path}")
        sys.exit(1)
        
    print(f"Loading case: {case_path.name}")
    try:
        scene, config = CaseLoader.load(case_path)
    except Exception as e:
        print(f"Error loading case: {e}")
        # Debugging aid: Print cwd
        print(f"CWD: {Path.cwd()}")
        sys.exit(1)
    
    print(f"Case '{config.name}' loaded successfully.")
    
    # Check solver type
    solver_type = config.solver.type
    if solver_type != "constant_source":
        print(f"Warning: Requested solver '{solver_type}' but only 'constant_source' is implemented.")
        print("Proceeding with Constant Source Panel Method...")
    
    # Assembler Global Mesh
    print("Assembling geometry...")
    try:
        global_mesh = scene.assemble()
    except Exception as e:
        print(f"Error assembling mesh: {e}")
        sys.exit(1)
        
    print(f"Global mesh: {global_mesh.num_panels} panels, {global_mesh.dimension}D")
    
    # Extract Flow Conditions
    vel = np.array(config.get_freestream_velocity())
    v_mag = np.linalg.norm(vel)
    if v_mag < 1e-9:
        print("Warning: Zero freestream velocity.")
        aoa = 0.0
    else:
        # Assuming flow in xy plane.
        aoa = np.degrees(np.arctan2(vel[1], vel[0]))
    
    print(f"Flow: V_inf = {v_mag:.4f}, AoA = {aoa:.2f} deg")
    
    # Initialize and Solve
    print("Initializing solver...")
    solver = SourcePanelSolver(global_mesh, v_inf=v_mag, aoa=aoa)
    
    print("Solving system...")
    solver.solve()
    
    # Visualization
    if config.visualization.enabled:
        print("Generating visualization...")
        
        # Simple plot for now - can be expanded to use dedicated Visualizer class
        xc = global_mesh.centers[:, 0]
        yc = global_mesh.centers[:, 1]
        cp = solver.Cp
        
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Geometry and Cp color
        plt.subplot(2, 1, 1)
        plt.scatter(xc, yc, c=cp, cmap='jet', s=20)
        plt.colorbar(label='Cp')
        
        # Draw panels
        nodes = global_mesh.nodes
        panels = global_mesh.panels
        for p in panels:
            p_nodes = nodes[p]
            plt.plot(p_nodes[:, 0], p_nodes[:, 1], 'k-', lw=0.5, alpha=0.5)
            
        plt.axis('equal')
        plt.title(f"{config.name} - Cp Distribution")
        
        # Plot 2: Cp vs X (scatter)
        plt.subplot(2, 1, 2)
        plt.scatter(xc, cp, c='b', s=10, label='Cp')
        plt.gca().invert_yaxis() # Convention: -Cp up
        plt.xlabel('X Coordinate')
        plt.ylabel('Cp (Inverted)')
        plt.title('Cp vs X')
        plt.grid(True)
        
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"{case_path.stem}_result.png"
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"Result plot saved to {output_file}")
        
    print("Done.")

if __name__ == "__main__":
    main()
