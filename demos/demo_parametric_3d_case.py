"""
3D Parametric Case Execution Demo.

Loads the parametric 3D case from cases/sphere_flow/case.yaml,
solves it using the 3D source panel solver, and exports the results
to the case's output folder in VTK format.
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.io.case_loader import CaseLoader
from core.geometry.io.vtk_export import export_solution_vtk
from solvers.factory import SolverFactory

def main():
    parser = argparse.ArgumentParser(description="Run 3D parametric case")
    parser.add_argument(
        "case_file", 
        type=str, 
        nargs="?", 
        default="cases/sphere_flow/case.yaml",
        help="Path to the case.yaml file"
    )
    parser.add_argument(
        "--level", 
        type=int, 
        default=0,
        help="Mesh resolution level index (default: 0 for coarse)"
    )
    args = parser.parse_args()
    
    case_path = Path(args.case_file)
    if not case_path.exists():
        print(f"Error: Case file not found: {case_path}")
        sys.exit(1)
        
    print("=" * 60)
    print(f"Running 3D Case: {case_path}")
    print("=" * 60)
    
    # Load case
    print(f"Loading scene (mesh level: {args.level})...")
    scene, config = CaseLoader.load(case_path, mesh_level_index=args.level)
    
    print(f"Case name: {config.name}")
    print(f"Description: {config.description}")
    
    # Assemble global mesh
    mesh = scene.assemble()
    print(f"Global mesh created with {mesh.num_panels} panels.")
    
    # Initialize solver
    freestream = config.get_freestream_velocity()
    print(f"Freestream velocity: {freestream}")
    print(f"Solver type: {config.solver.singularity_type} ({config.solver.panel_order})")
    
    solver = SolverFactory.create(
        config=config.solver,
        mesh=mesh,
        v_inf=freestream[0], # magnitude 
        aoa=0.0 # for 3D solver it will construct the vector using magnitude and aoa
    )
    
    # Override freestream vector for 3D specifically (SolverFactory currently simplifies it to 2D magnitude + aoa if not careful)
    # The new SolverFactory automatically creates a [Vx, Vy, 0] vector, but we have [Vx, Vy, Vz] from config.
    # To be precise, we inject the actual 3D vector from config.
    import numpy as np
    solver._v_inf = np.asarray(freestream, dtype=np.float64)
    
    # Solve
    print("\nSolving...")
    solver.solve()
    print("Solve complete.")
    
    # Check boundary condition (Vn = 0)
    bc_check = solver.validate_boundary_condition()
    print(f"Boundary condition (Vn=0) Check:")
    print(f"  Max |Vn|: {bc_check['Vn_max_abs']:.2e}")
    print(f"  RMS Vn: {bc_check['Vn_rms']:.2e}")
    
    # Save output
    out_dir = case_path.parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"solution_level_{args.level}.vtp"
    
    print(f"\nExporting results to VTK...")
    # The solver adds 'Cp', 'Vt', 'sigma' to mesh.cell_data during solve()
    export_solution_vtk(mesh, out_file)
    print(f"Saved: {out_file}")
    print("Open this file in ParaView to visualize the 3D fields.")

if __name__ == "__main__":
    main()
