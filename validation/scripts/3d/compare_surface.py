#!/usr/bin/env python3
"""
Compare 3D panel method vs Fluent surface distributions.

Extracts surface data from the panel method and Fluent, calculates
differences, and exports a VTK file containing the comparative fields
for visualization in ParaView.

Usage:
    python validation/scripts/compare_surface_3d.py cases/sphere_flow --mesh-level 0
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.io.case_loader import CaseLoader
from solvers.factory import SolverFactory
from core.geometry.io.vtk_export import export_solution_vtk
from src.validation.adapters.fluent.ascii_reader_3d import find_fluent_3d_export, read_3d_surface_data


def run_panel_solver(case_dir, mesh_level):
    """Load case, assemble mesh, and run panel solver."""
    print(f"Loading case from {case_dir} at mesh level {mesh_level}...")
    scene, config = CaseLoader.load(case_dir / "case.yaml", mesh_level_index=mesh_level)
    
    mesh = scene.assemble()
    print(f"Panel Method Mesh: {mesh.num_panels} panels.")
    
    freestream = config.get_freestream_velocity()
    solver = SolverFactory.create(
        config=config.solver,
        mesh=mesh,
        v_inf=freestream[0],
        aoa=0.0
    )
    # Inject exact 3D freestream vector
    solver._v_inf = np.asarray(freestream, dtype=np.float64)
    
    print("Solving panel method...")
    solver.solve()
    print("Panel method solve complete.")
    
    return mesh, solver, config


def interpolate_fluent_to_panels(mesh, fluent_data):
    """
    Interpolate Fluent point data to panel centers using nearest neighbor.
    For more accuracy, one could implement an IDW or RBF interpolator.
    """
    fluent_pts = fluent_data.points
    panel_centers = mesh.centers
    
    print(f"Building KDTree for {len(fluent_pts)} Fluent points...")
    kdtree = cKDTree(fluent_pts)
    
    # Find the nearest fluent point for each panel center
    distances, indices = kdtree.query(panel_centers, k=1)
    
    print(f"Average nearest-neighbor distance: {np.mean(distances):.2e}")
    
    # Extract fields
    fluent_vmag = fluent_data.get_velocity_magnitude()
    
    interpolated_fields = {
        "Vt_fluent": fluent_vmag[indices],
    }
    
    # If pressure is available, compute Cp
    # Note: If Fluent pressure is relative to operating pressure, you may need
    # reference pressure and density from the case config.
    if fluent_data.pressure is not None:
        interpolated_fields["pressure_fluent"] = fluent_data.pressure[indices]
        
    return interpolated_fields


def main():
    parser = argparse.ArgumentParser(description="Compare 3D panel vs Fluent surface distributions")
    parser.add_argument("case_dir", type=Path, help="Path to case directory")
    parser.add_argument("--mesh-level", type=int, default=0, help="Panel mesh level index")
    
    args = parser.parse_args()
    case_dir = args.case_dir.resolve()
    
    if not case_dir.exists():
        print(f"Error: Case directory not found: {case_dir}")
        return 1
        
    # 1. Run Panel Solver
    mesh, solver, config = run_panel_solver(case_dir, args.mesh_level)
    Vt_panel = np.linalg.norm(solver.surface_velocity, axis=1)
    Cp_panel = solver.Cp
    
    # 2. Load Fluent Data
    fluent_file = find_fluent_3d_export(case_dir)
    if fluent_file is None:
        print(f"\nWarning: Fluent export data not found in {case_dir}/fluent/export/panel/surface_data")
        print("Exporting only Panel Method results to VTK.")
        
        out_dir = case_dir / "out" / "panel_solver"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"panel_results_level_{args.mesh_level}.vtp"
        
        mesh.cell_data["Vt_panel"] = Vt_panel
        mesh.cell_data["Cp_panel"] = Cp_panel
        export_solution_vtk(mesh, out_file)
        
        print(f"Saved: {out_file}")
        return 0
        
    print(f"\nLoading Fluent data from {fluent_file}...")
    fluent_data = read_3d_surface_data(fluent_file)
    print(f"Loaded {fluent_data.num_nodes} points from Fluent.")
    
    # 3. Interpolate Fluent Data onto Panel Mesh
    interp_data = interpolate_fluent_to_panels(mesh, fluent_data)
    Vt_fluent = interp_data["Vt_fluent"]
    
    # Compute metrics
    Vt_diff = Vt_panel - Vt_fluent
    Vt_diff_mag = np.abs(Vt_diff)
    
    print("\n" + "="*50)
    print("Slip Velocity Magnitude (Vt) Comparison")
    print("="*50)
    print(f"  Panel Max Vt : {np.max(Vt_panel):.4f}")
    print(f"  Fluent Max Vt: {np.max(Vt_fluent):.4f}")
    print(f"  Max Diff     : {np.max(Vt_diff_mag):.4f}")
    print(f"  Mean Diff    : {np.mean(Vt_diff_mag):.4f}")
    print(f"  RMS Error    : {np.sqrt(np.mean(Vt_diff**2)):.4f}")
    
    # 4. Save to VTK
    # Update cell data with everything needed for comparison
    mesh.cell_data["Vt_panel"] = Vt_panel
    mesh.cell_data["Vt_fluent"] = Vt_fluent
    mesh.cell_data["Vt_difference"] = Vt_diff
    mesh.cell_data["Cp_panel"] = Cp_panel
    
    # Attempt Cp computation for Fluent if pressure is available
    if "pressure_fluent" in interp_data:
        rho = config.fluid.density
        v_inf_mag = np.linalg.norm(config.get_freestream_velocity())
        q_inf = 0.5 * rho * v_inf_mag**2
        # Assuming Fluent pressure is gauge pressure and freestream P=0
        Cp_fluent = interp_data["pressure_fluent"] / q_inf
        Cp_diff = Cp_panel - Cp_fluent
        
        mesh.cell_data["Cp_fluent"] = Cp_fluent
        mesh.cell_data["Cp_difference"] = Cp_diff
        
        print("\n" + "="*50)
        print("Pressure Coefficient (Cp) Comparison")
        print("="*50)
        print(f"  Panel Max Cp : {np.max(Cp_panel):.4f}")
        print(f"  Fluent Max Cp: {np.max(Cp_fluent):.4f}")
        print(f"  RMS Error    : {np.sqrt(np.mean(Cp_diff**2)):.4f}")

    out_dir = case_dir / "out" / "panel_solver" / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"comparison_level_{args.mesh_level}.vtp"
    
    print(f"\nExporting comparison results to VTK...")
    export_solution_vtk(mesh, out_file)
    print(f"Saved: {out_file}")
    print("\nNext steps in ParaView:")
    print("  1. Open the .vtp file.")
    print("  2. Select 'Vt_difference' in the coloring array.")
    print("  3. Check the color map scale to visualize areas of high deviation.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
