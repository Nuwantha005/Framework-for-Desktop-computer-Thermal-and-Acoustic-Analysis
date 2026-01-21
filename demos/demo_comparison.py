#!/usr/bin/env python3
"""
Demo: Comparison Visualization

Shows how to compare fields from different sources:
- Side-by-side contour plots with unified colorbar
- Difference plots with error metrics
- Cp distribution comparisons
- Mesh convergence studies

This is useful for:
- Validating panel method against OpenFOAM (potentialFoam, simpleFoam)
- Mesh convergence studies
- Comparing different solver configurations

Usage:
    python demo_comparison.py [--show] [--save]
"""

import sys
import argparse
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.io import CaseLoader
from core.io.geometry_io import generate_circle
from core.geometry import Component, Scene, Transform
from solvers.panel2d.spm import SourcePanelSolver
from visualization import Visualizer, ComparisonVisualizer, FieldSeries, LineSeries
from visualization.field2d import VelocityField2D
from postprocessing import FieldData, FluidState, ProcessorPipeline, PressureProcessor


def demo_side_by_side(show: bool = True, save: bool = False):
    """
    Demo 1: Side-by-side comparison of two different mesh resolutions.
    """
    print("=" * 60)
    print("Demo 1: Side-by-Side Comparison (Mesh Resolution)")
    print("=" * 60)
    
    # Create two meshes with different resolutions
    v_inf = 1.0
    aoa = 0.0
    
    results = []
    for n_panels in [16, 64]:
        print(f"\nSolving with {n_panels} panels...")
        
        mesh = generate_circle(center=(0, 0), radius=0.5, num_panels=n_panels)
        solver = SourcePanelSolver(mesh, v_inf=v_inf, aoa=aoa)
        solver.solve()
        
        # Compute velocity field
        field = VelocityField2D(mesh, v_inf, aoa, solver.sigma)
        XX, YY, Vx, Vy = field.compute((-2, 3), (-2, 2), (100, 80), num_cores=4)
        
        # Store results
        results.append({
            'mesh': mesh,
            'n_panels': n_panels,
            'XX': XX, 'YY': YY,
            'Vx': Vx, 'Vy': Vy,
            'Cp': solver.Cp
        })
    
    # Create FieldSeries for comparison
    V_mag_coarse = np.sqrt(results[0]['Vx']**2 + results[0]['Vy']**2)
    V_mag_fine = np.sqrt(results[1]['Vx']**2 + results[1]['Vy']**2)
    
    field1 = FieldSeries(
        name="|V|",
        data=V_mag_coarse,
        XX=results[0]['XX'],
        YY=results[0]['YY'],
        units="m/s",
        source=f"{results[0]['n_panels']} panels"
    )
    
    field2 = FieldSeries(
        name="|V|",
        data=V_mag_fine,
        XX=results[1]['XX'],
        YY=results[1]['YY'],
        units="m/s",
        source=f"{results[1]['n_panels']} panels"
    )
    
    # Compare!
    comp = ComparisonVisualizer()
    comp.compare_contours(
        [field1, field2],
        mesh=results[1]['mesh'],  # Use finer mesh for outline
        unified_colorbar=True,
        title="Velocity Magnitude: Coarse vs Fine Mesh"
    )
    
    if save:
        comp.save("comparison_side_by_side.png")
    if show:
        comp.show()
    else:
        import matplotlib.pyplot as plt
        plt.close()


def demo_difference_plot(show: bool = True, save: bool = False):
    """
    Demo 2: Difference plot showing error between two solutions.
    """
    print("\n" + "=" * 60)
    print("Demo 2: Difference Plot with Error Metrics")
    print("=" * 60)
    
    v_inf = 1.0
    aoa = 0.0
    
    # Reference: Fine mesh
    print("\nSolving fine mesh (reference)...")
    mesh_fine = generate_circle(center=(0, 0), radius=0.5, num_panels=64)
    solver_fine = SourcePanelSolver(mesh_fine, v_inf=v_inf, aoa=aoa)
    solver_fine.solve()
    
    field_fine = VelocityField2D(mesh_fine, v_inf, aoa, solver_fine.sigma)
    XX, YY, Vx_fine, Vy_fine = field_fine.compute((-2, 3), (-2, 2), (100, 80), num_cores=4)
    
    # Coarse mesh
    print("Solving coarse mesh...")
    mesh_coarse = generate_circle(center=(0, 0), radius=0.5, num_panels=16)
    solver_coarse = SourcePanelSolver(mesh_coarse, v_inf=v_inf, aoa=aoa)
    solver_coarse.solve()
    
    field_coarse = VelocityField2D(mesh_coarse, v_inf, aoa, solver_coarse.sigma)
    _, _, Vx_coarse, Vy_coarse = field_coarse.compute((-2, 3), (-2, 2), (100, 80), num_cores=4)
    
    # Create FieldSeries
    V_mag_fine = np.sqrt(Vx_fine**2 + Vy_fine**2)
    V_mag_coarse = np.sqrt(Vx_coarse**2 + Vy_coarse**2)
    
    field_ref = FieldSeries(
        name="|V|", data=V_mag_fine, XX=XX, YY=YY,
        units="m/s", source="64 panels (reference)"
    )
    
    field_test = FieldSeries(
        name="|V|", data=V_mag_coarse, XX=XX, YY=YY,
        units="m/s", source="16 panels"
    )
    
    # Plot difference
    comp = ComparisonVisualizer()
    fig, metrics = comp.plot_difference(
        field_ref, field_test,
        mesh=mesh_fine, show_originals=True,
        title="Velocity Magnitude: Fine vs Coarse Mesh"
    )
    
    print("\nError Metrics:")
    print(metrics.summary())
    
    if save:
        comp.save("comparison_difference.png")
    if show:
        comp.show()
    else:
        import matplotlib.pyplot as plt
        plt.close()


def demo_cp_comparison(show: bool = True, save: bool = False):
    """
    Demo 3: Compare Cp distributions (line plots).
    """
    print("\n" + "=" * 60)
    print("Demo 3: Cp Distribution Comparison")
    print("=" * 60)
    
    v_inf = 1.0
    aoa = 0.0
    
    # Analytical Cp for cylinder: Cp = 1 - 4*sin²(θ)
    theta_analytical = np.linspace(0, 2*np.pi, 100)
    Cp_analytical = 1 - 4 * np.sin(theta_analytical)**2
    
    # Panel method with different resolutions
    cp_series = [
        (theta_analytical, Cp_analytical, "Analytical")
    ]
    
    for n_panels in [8, 16, 32, 64]:
        mesh = generate_circle(center=(0, 0), radius=0.5, num_panels=n_panels)
        solver = SourcePanelSolver(mesh, v_inf=v_inf, aoa=aoa)
        solver.solve()
        
        # Panel angles
        theta_panels = np.arctan2(mesh.centers[:, 1], mesh.centers[:, 0])
        theta_panels = np.where(theta_panels < 0, theta_panels + 2*np.pi, theta_panels)
        
        # Sort by angle
        sort_idx = np.argsort(theta_panels)
        cp_series.append((theta_panels[sort_idx], solver.Cp[sort_idx], f"{n_panels} panels"))
    
    # Plot comparison
    comp = ComparisonVisualizer()
    comp.compare_cp_distributions(
        cp_series,
        title="Cp Distribution: Panel Method vs Analytical (Cylinder)"
    )
    
    if save:
        comp.save("comparison_cp.png")
    if show:
        comp.show()
    else:
        import matplotlib.pyplot as plt
        plt.close()


def demo_mesh_convergence(show: bool = True, save: bool = False):
    """
    Demo 4: Mesh convergence study.
    """
    print("\n" + "=" * 60)
    print("Demo 4: Mesh Convergence Study")
    print("=" * 60)
    
    v_inf = 1.0
    aoa = 0.0
    
    # Analytical drag coefficient for cylinder in potential flow = 0 (D'Alembert's paradox)
    # But we can track Cp at stagnation point (should be 1.0)
    
    mesh_sizes = [8, 16, 32, 64, 128, 256]
    cp_stag_values = []
    
    for n in mesh_sizes:
        mesh = generate_circle(center=(0, 0), radius=0.5, num_panels=n)
        solver = SourcePanelSolver(mesh, v_inf=v_inf, aoa=aoa)
        solver.solve()
        
        # Find stagnation point (max Cp, should be at θ=0 or π)
        cp_max = np.max(solver.Cp)
        cp_stag_values.append(cp_max)
        
        print(f"  {n:4d} panels: Cp_stagnation = {cp_max:.6f}")
    
    # Plot convergence
    comp = ComparisonVisualizer()
    comp.plot_convergence(
        mesh_sizes,
        cp_stag_values,
        reference=1.0,  # Analytical value
        xlabel="Number of Panels",
        ylabel="Cp at Stagnation Point",
        title="Mesh Convergence: Cp at Stagnation Point"
    )
    
    if save:
        comp.save("comparison_convergence.png")
    if show:
        comp.show()
    else:
        import matplotlib.pyplot as plt
        plt.close()


def demo_line_series(show: bool = True, save: bool = False):
    """
    Demo 5: General line series comparison.
    """
    print("\n" + "=" * 60)
    print("Demo 5: Line Series Comparison (Custom Data)")
    print("=" * 60)
    
    # Create some mock data (as if comparing with OpenFOAM)
    x = np.linspace(0, 1, 50)
    
    line1 = LineSeries(
        name="Velocity Profile",
        x=x,
        y=4 * x * (1 - x),  # Parabolic profile
        source="Panel Method",
        x_label="y/H",
        y_label="u/U_max",
        style="-",
        marker="o"
    )
    
    line2 = LineSeries(
        name="Velocity Profile",
        x=x,
        y=4 * x * (1 - x) + 0.02 * np.random.randn(len(x)),  # With noise
        source="OpenFOAM (mock)",
        x_label="y/H",
        y_label="u/U_max",
        style="--",
        marker="s"
    )
    
    line3 = LineSeries(
        name="Velocity Profile",
        x=x,
        y=4.1 * x * (1 - x) - 0.05,  # Slightly different
        source="Experiment (mock)",
        x_label="y/H",
        y_label="u/U_max",
        style=":",
        marker="^"
    )
    
    comp = ComparisonVisualizer()
    comp.compare_lines(
        [line1, line2, line3],
        title="Velocity Profile Comparison",
        xlabel="y/H",
        ylabel="u/U_max"
    )
    
    if save:
        comp.save("comparison_lines.png")
    if show:
        comp.show()
    else:
        import matplotlib.pyplot as plt
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Comparison visualization demo")
    parser.add_argument("--show", action="store_true", help="Display plots")
    parser.add_argument("--save", action="store_true", help="Save plots")
    parser.add_argument("--all", action="store_true", help="Run all demos")
    parser.add_argument("--demo", type=int, choices=[1,2,3,4,5], help="Run specific demo")
    args = parser.parse_args()
    
    if not args.show and not args.save:
        args.show = True
    
    if args.all or args.demo is None:
        demos = [1, 2, 3, 4, 5]
    else:
        demos = [args.demo]
    
    for d in demos:
        if d == 1:
            demo_side_by_side(args.show, args.save)
        elif d == 2:
            demo_difference_plot(args.show, args.save)
        elif d == 3:
            demo_cp_comparison(args.show, args.save)
        elif d == 4:
            demo_mesh_convergence(args.show, args.save)
        elif d == 5:
            demo_line_series(args.show, args.save)
    
    print("\nDone.")


if __name__ == "__main__":
    main()
