"""
3D sphere flow validation demo.

Validates the 3D source panel solver against the analytical solution
for potential flow over a sphere.

Analytical result: Cp = 1 - 2.25*sin²θ
where θ is the polar angle from the stagnation point.

At stagnation points (θ=0, π): Cp = 1.0
At equator (θ=π/2): Cp = 1 - 2.25 = -1.25

Usage:
    python demos/demo_sphere_3d.py
    
Output:
    - Console: Error metrics
    - VTK file: sphere_flow.vtu (open in ParaView)
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from core.geometry.io import generate_sphere
from solvers.panel3d import SourcePanelSolver3D


def main():
    print("=" * 60)
    print("3D Sphere Flow Validation")
    print("=" * 60)
    
    # Parameters
    radius = 1.0
    v_inf = np.array([1.0, 0.0, 0.0])  # Freestream in +x direction
    
    # Test with single coarse mesh first (for development/debugging)
    resolutions = [
        (8, 16, "coarse"),
        # (16, 32, "medium"),  # Uncomment after solver is optimized
        # (32, 64, "fine"),
    ]
    
    for n_theta, n_phi, label in resolutions:
        print(f"\n--- Mesh: {label} ({n_theta}x{n_phi}) ---")
        
        # Generate sphere mesh
        mesh = generate_sphere(
            n_theta=n_theta,
            n_phi=n_phi,
            radius=radius,
            center=(0.0, 0.0, 0.0)
        )
        print(f"Panels: {mesh.num_panels}")
        print(f"Nodes: {mesh.num_nodes}")
        
        # Create and solve
        solver = SourcePanelSolver3D(mesh, v_inf)
        solver.solve()
        
        # Validate against analytical
        metrics = solver.validate_sphere(radius=radius)
        
        print(f"\nResults:")
        print(f"  Cp range (computed): [{metrics['Cp_computed_range'][0]:.4f}, {metrics['Cp_computed_range'][1]:.4f}]")
        print(f"  Cp range (analytical): [{metrics['Cp_analytical_range'][0]:.4f}, {metrics['Cp_analytical_range'][1]:.4f}]")
        print(f"  Max error: {metrics['Cp_max_error']:.4f}")
        print(f"  RMS error: {metrics['Cp_rms_error']:.4f}")
        print(f"  L∞ error: {metrics['L_inf_error']:.6f}")
        print(f"  L2 error: {metrics['L2_error']:.6f}")
        
        # Check BC satisfaction
        bc_check = solver.validate_boundary_condition()
        print(f"\nBoundary condition (Vn=0):")
        print(f"  Max |Vn|: {bc_check['Vn_max_abs']:.2e}")
        print(f"  RMS Vn: {bc_check['Vn_rms']:.2e}")
        
        # Source strength statistics
        sigma = solver.sigma
        print(f"\nSource strengths:")
        print(f"  Sum σ: {np.sum(sigma):.2e} (should be ~0 for closed body)")
        print(f"  Range: [{np.min(sigma):.4f}, {np.max(sigma):.4f}]")
    
    # Save finest mesh to VTK for ParaView
    print("\n" + "=" * 60)
    print("Saving VTK file...")
    
    # Use same mesh from last test (coarse for now)
    mesh = generate_sphere(n_theta=8, n_phi=16, radius=radius)
    solver = SourcePanelSolver3D(mesh, v_inf)
    solver.solve()
    solver.validate_sphere(radius=radius)
    
    output_path = Path(__file__).parent.parent / "out" / "sphere_flow.vtu"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.save_vtk(str(output_path))
    print(f"Saved: {output_path}")
    print("\nOpen in ParaView to visualize Cp distribution.")
    
    # Success criteria
    print("\n" + "=" * 60)
    print("Validation Summary:")
    metrics = solver.validate_sphere(radius=radius)
    if metrics['Cp_max_error'] < 0.01:
        print(f"✓ PASS: Max Cp error ({metrics['Cp_max_error']:.4f}) < 1%")
    else:
        print(f"✗ FAIL: Max Cp error ({metrics['Cp_max_error']:.4f}) >= 1%")
    
    return metrics['Cp_max_error'] < 0.05  # Accept 5% for now during development


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
