#!/usr/bin/env python3
"""
3D Field Vector Glyphs Data Generator.

Evaluates the velocity field at scattered points around the body
(e.g., a 3D bounding box grid) and exports it as a VTK file so it
can be visualized using 3D vector arrows (Glyphs) in ParaView.

Usage:
    python validation/scripts/3d/vector_glyphs.py cases/sphere_flow --mesh-level 0
"""

import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.io.case_loader import CaseLoader
from solvers.factory import SolverFactory
import pyvista as pv

def generate_field_grid(bounds, resolution):
    """Generate a 3D grid of points."""
    x = np.linspace(bounds[0], bounds[1], resolution[0])
    y = np.linspace(bounds[2], bounds[3], resolution[1])
    z = np.linspace(bounds[4], bounds[5], resolution[2])
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    return points, (xx, yy, zz)

def main():
    parser = argparse.ArgumentParser(description="Generate 3D Vector Glyphs Field")
    parser.add_argument("case_dir", type=Path, help="Path to case directory")
    parser.add_argument("--mesh-level", type=int, default=0, help="Panel mesh level index")
    
    args = parser.parse_args()
    case_dir = args.case_dir.resolve()
    
    if not case_dir.exists():
        print(f"Error: Case directory not found: {case_dir}")
        return 1
        
    print(f"Loading case from {case_dir} at mesh level {args.mesh_level}...")
    scene, config = CaseLoader.load(case_dir / "case.yaml", mesh_level_index=args.mesh_level)
    
    mesh = scene.assemble()
    freestream = config.get_freestream_velocity()
    solver = SolverFactory.create(
        config=config.solver,
        mesh=mesh,
        v_inf=freestream[0],
        aoa=0.0
    )
    solver._v_inf = np.asarray(freestream, dtype=np.float64)
    solver.solve()
    
    # Setup visualization domain
    x_range = config.visualization.get_x_range()
    y_range = config.visualization.get_y_range()
    z_range = config.visualization.get_z_range()
    
    # Use a coarser resolution for 3D vector fields otherwise ParaView gets slow
    res = config.visualization.resolution
    # Use max 30 points in each dimension if not specified
    nx = min(res[0] if len(res) > 0 else 30, 40)
    ny = min(res[1] if len(res) > 1 else 30, 40)
    nz = min(res[2] if len(res) > 2 else 30, 40)
    
    bounds = (x_range[0], x_range[1], y_range[0], y_range[1], z_range[0], z_range[1])
    print(f"Generating field points over bounds: {bounds} with resolution {nx}x{ny}x{nz}")
    
    points, (xx, yy, zz) = generate_field_grid(bounds, (nx, ny, nz))
    
    print("Evaluating velocity at field points (this may take a moment)...")
    velocities = solver.velocity_at(points)
    
    # Exclude points inside the body
    # (For a sphere of r=0.5, simple distance check)
    if "sphere" in str(case_dir):
        # Quick hack for sphere flow, mark points inside sphere as NaN
        # A more general approach requires point-in-poly test
        r = np.linalg.norm(points, axis=1)
        inside = r < 0.49 # Slightly less than 0.5 to avoid surface singularity
        velocities[inside] = 0.0
    
    # Create PyVista structured grid
    grid = pv.StructuredGrid(xx, yy, zz)
    grid.point_data["U"] = velocities
    
    out_dir = case_dir / "out" / "panel_solver" / "glyphs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"field_vectors_level_{args.mesh_level}.vts"
    
    grid.save(str(out_file))
    print(f"Exported to {out_file}")
    
    print("\nNext steps in ParaView:")
    print("  1. Open the .vts file.")
    print("  2. Apply the 'Glyph' filter.")
    print("  3. Set 'Orientation Array' to 'U'.")
    print("  4. Optionally scale by magnitude.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
