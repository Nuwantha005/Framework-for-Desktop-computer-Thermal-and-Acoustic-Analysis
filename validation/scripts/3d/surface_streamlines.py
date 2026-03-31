#!/usr/bin/env python3
"""
3D Surface Streamlines Generator.

Runs the panel solver, extracts the slip velocity vector on each panel,
and generates a surface vector field which can be used to plot
surface (limiting) streamlines in ParaView.

Usage:
    python validation/scripts/3d/surface_streamlines.py cases/sphere_flow --mesh-level 0
"""

import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.io.case_loader import CaseLoader
from solvers.factory import SolverFactory
from core.geometry.io.vtk_export import export_solution_vtk

def main():
    parser = argparse.ArgumentParser(description="Generate 3D Surface Streamlines Data")
    parser.add_argument("case_dir", type=Path, help="Path to case directory")
    parser.add_argument("--mesh-level", type=int, default=-1, help="Panel mesh level index")
    
    args = parser.parse_args()
    case_dir = args.case_dir.resolve()
    
    if not case_dir.exists():
        print(f"Error: Case directory not found: {case_dir}")
        return 1
        
    print(f"Loading case from {case_dir} at mesh level {args.mesh_level}...")
    scene, config = CaseLoader.load(case_dir / "case.yaml", mesh_level_index=args.mesh_level)
    
    mesh = scene.assemble()
    print(f"Panel Method Mesh: {mesh.num_panels} panels.")
    
    freestream = config.get_freestream_velocity()
    solver = SolverFactory.create(
        config=config.solver,
        mesh=mesh,
        v_inf=freestream[0],
        aoa=0.0
    )
    solver._v_inf = np.asarray(freestream, dtype=np.float64)
    
    print("Solving panel method...")
    solver.solve()
    
    # Store explicit 3D velocity vectors on the cell data
    mesh.cell_data["U_surface"] = solver.surface_velocity
    
    # Optional: We could compute streamlines here using scipy.integrate, but
    # ParaView has an excellent native SurfaceLIC (Line Integral Convolution)
    # and EvenlySpacedStreamlines2D tool that works directly on the VTK file if it
    # contains surface vector fields. We rely on ParaView for visualization.
    
    out_dir = case_dir / "out" / "panel_solver" / "streamlines"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"surface_vectors_level_{args.mesh_level}.vtp"
    
    print(f"Exporting to {out_file}...")
    export_solution_vtk(mesh, out_file)
    
    print("\nNext steps in ParaView:")
    print("  1. Open the .vtp file.")
    print("  2. Apply the 'Cell Data to Point Data' filter (since velocity is per-panel).")
    print("  3. Apply the 'Surface LIC' representation or 'Surface Flow' filter")
    print("     and select 'U_surface' as the vector array.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
