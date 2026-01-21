#!/usr/bin/env python3
"""
Validation Script: Panel Method vs OpenFOAM (potentialFoam)

This script demonstrates the complete validation workflow:
1. Load a panel method case
2. Run panel method solver
3. Generate and run OpenFOAM case
4. Compare results using ComparisonVisualizer

Usage:
    python run_validation.py <case_dir> [options]

Examples:
    # Full workflow
    python run_validation.py ../cases/two_rounded_rects --run-openfoam
    
    # Just compare (if OpenFOAM already run)
    python run_validation.py ../cases/two_rounded_rects --compare-only
    
    # Generate OpenFOAM case without running
    python run_validation.py ../cases/two_rounded_rects --generate-only
"""

import sys
import argparse
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.io import CaseLoader
from solvers.panel2d.spm import SourcePanelSolver
from visualization import Visualizer, ComparisonVisualizer
from visualization.field2d import VelocityField2D
from visualization.comparison import FieldSeries
from validation import OpenFOAMCaseGenerator, OpenFOAMRunner, FoamCase


def run_panel_method(case, num_cores: int = 6):
    """
    Run panel method solver and compute velocity field.
    
    Returns:
        dict with keys: XX, YY, Vx, Vy, V_mag, sigma, Cp
    """
    print("\n" + "="*60)
    print("Running Panel Method Solver")
    print("="*60)
    
    # Solve
    print(f"  Mesh: {case.num_panels} panels, {case.num_components} components")
    solver = SourcePanelSolver(case.mesh, v_inf=case.v_inf, aoa=case.aoa)
    solver.solve()
    
    print(f"  Cp range: [{min(solver.Cp):.4f}, {max(solver.Cp):.4f}]")
    
    # Compute velocity field
    print(f"  Computing velocity field ({case.resolution[0]}×{case.resolution[1]})...")
    vfield = VelocityField2D(case.mesh, case.v_inf, case.aoa, solver.sigma)
    XX, YY, Vx, Vy = vfield.compute(
        case.x_range, case.y_range, case.resolution, num_cores=num_cores
    )
    
    V_mag = np.sqrt(Vx**2 + Vy**2)
    
    print(f"  V_mag range: [{np.nanmin(V_mag):.4f}, {np.nanmax(V_mag):.4f}]")
    
    return {
        'XX': XX,
        'YY': YY,
        'Vx': Vx,
        'Vy': Vy,
        'V_mag': V_mag,
        'sigma': solver.sigma,
        'Cp': solver.Cp
    }


def generate_openfoam_case(case, output_dir: Path, mesh_density: float = 10.0):
    """
    Generate OpenFOAM case from panel method case.
    
    Returns:
        Path to generated case directory
    """
    print("\n" + "="*60)
    print("Generating OpenFOAM Case")
    print("="*60)
    
    from validation.adapters.openfoam.case_generator import MeshSettings
    
    mesh_settings = MeshSettings(
        background_cells_per_unit=mesh_density,
        refinement_level=2,
        z_thickness=0.1
    )
    
    generator = OpenFOAMCaseGenerator(
        case=case,
        output_dir=output_dir,
        solver_type="potentialFoam",
        mesh_settings=mesh_settings,
        domain_padding=2.0
    )
    
    of_case_dir = generator.generate()
    
    print(f"  Generated case at: {of_case_dir}")
    print(f"  Domain: x=[{generator.domain['x'][0]:.1f}, {generator.domain['x'][1]:.1f}]")
    print(f"          y=[{generator.domain['y'][0]:.1f}, {generator.domain['y'][1]:.1f}]")
    
    # List generated files
    print(f"\n  Files created:")
    for subdir in ["0", "constant/triSurface", "system"]:
        subpath = of_case_dir / subdir
        if subpath.exists():
            files = list(subpath.glob("*"))
            print(f"    {subdir}/: {', '.join(f.name for f in files if f.is_file())}")
    
    return of_case_dir


def run_openfoam(of_case_dir: Path, use_snappy: bool = True):
    """
    Run OpenFOAM solver.
    
    Returns:
        OpenFOAMRunner instance with results
    """
    print("\n" + "="*60)
    print("Running OpenFOAM (potentialFoam)")
    print("="*60)
    
    runner = OpenFOAMRunner(of_case_dir, verbose=True)
    success = runner.run_all(solver="potentialFoam", use_snappy=use_snappy)
    
    if not success:
        print("\nWARNING: OpenFOAM run had errors. Check logs.")
    
    print(f"\n{runner.summary()}")
    
    return runner


def read_openfoam_results(of_case_dir: Path, case):
    """
    Read OpenFOAM results using foamlib and interpolate to panel method grid.
    
    Returns:
        dict with keys: XX, YY, Vx, Vy, V_mag, p, time (gridded)
                   and: cell_centres, U_raw, p_raw (raw cell data)
    """
    print("\n" + "="*60)
    print("Reading OpenFOAM Results (via foamlib)")
    print("="*60)
    
    from scipy.interpolate import griddata
    
    try:
        from validation import OpenFOAMRunner
        
        runner = OpenFOAMRunner(of_case_dir, verbose=False)
        
        # Get cell centres and fields
        C = runner.get_cell_centres()
        U = runner.get_velocity_field()
        p = runner.get_pressure_field()
        time_val = runner.get_latest_time()
        
        print(f"  Time: {time_val}")
        print(f"  Cells: {len(C)}")
        print(f"  U range: [{U[:,0].min():.4f}, {U[:,0].max():.4f}] (Ux)")
        print(f"  U range: [{U[:,1].min():.4f}, {U[:,1].max():.4f}] (Uy)")
        
        # Filter to midplane (z ≈ 0.05 for our thin slab)
        z_mid = 0.05
        z_tol = 0.03
        mask = np.abs(C[:, 2] - z_mid) < z_tol
        
        if mask.sum() < 100:
            print(f"  WARNING: Only {mask.sum()} cells near midplane, using all cells")
            mask = np.ones(len(C), dtype=bool)
        else:
            print(f"  Filtered to midplane: {mask.sum()} cells")
        
        points_2d = C[mask, :2]
        U_2d = U[mask, :2]  # Only Ux, Uy
        p_filtered = p[mask]
        
        # Interpolate to panel method grid
        print(f"  Interpolating to {case.resolution[0]}×{case.resolution[1]} grid...")
        nx, ny = case.resolution
        x = np.linspace(case.x_range[0], case.x_range[1], nx)
        y = np.linspace(case.y_range[0], case.y_range[1], ny)
        XX, YY = np.meshgrid(x, y)
        
        Vx = griddata(points_2d, U_2d[:, 0], (XX, YY), method='linear')
        Vy = griddata(points_2d, U_2d[:, 1], (XX, YY), method='linear')
        p_grid = griddata(points_2d, p_filtered, (XX, YY), method='linear')
        V_mag = np.sqrt(Vx**2 + Vy**2)
        
        print(f"  V_mag range: [{np.nanmin(V_mag):.4f}, {np.nanmax(V_mag):.4f}]")
        
        return {
            # Gridded data for visualization
            'XX': XX,
            'YY': YY,
            'Vx': Vx,
            'Vy': Vy,
            'V_mag': V_mag,
            'p': p_grid,
            'time': time_val,
            # Raw cell data for direct comparison
            'cell_centres': C,
            'U_raw': U,
            'p_raw': p
        }
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_results(panel_results, openfoam_results, case, output_dir: Path, show: bool = True):
    """
    Compare panel method and OpenFOAM results.
    """
    print("\n" + "="*60)
    print("Comparing Results")
    print("="*60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create FieldSeries for comparison
    panel_V = FieldSeries(
        name="|V|",
        data=panel_results['V_mag'],
        XX=panel_results['XX'],
        YY=panel_results['YY'],
        units="m/s",
        source="Panel Method"
    )
    
    of_V = FieldSeries(
        name="|V|",
        data=openfoam_results['V_mag'],
        XX=openfoam_results['XX'],
        YY=openfoam_results['YY'],
        units="m/s",
        source="potentialFoam"
    )
    
    # Create comparison visualizer
    comp = ComparisonVisualizer(output_dir=output_dir)
    
    # 1. Side-by-side comparison
    print("\n  Creating side-by-side comparison...")
    comp.compare_contours(
        [panel_V, of_V],
        mesh=case.mesh,
        levels=30,
        unified_colorbar=True,
        title=f"Velocity Magnitude: {case.name}"
    )
    comp.save("comparison_sidebyside.png")
    
    # 2. Difference plot
    print("  Creating difference plot...")
    fig, metrics = comp.plot_difference(
        panel_V, of_V,
        mesh=case.mesh,
        show_originals=True,
        symmetric=True,
        title=f"Panel Method vs potentialFoam: {case.name}"
    )
    comp.save("comparison_difference.png")
    
    print(f"\n  Error Metrics:")
    print(f"    L2 norm:    {metrics.l2_norm:.4e}")
    print(f"    L∞ norm:    {metrics.linf_norm:.4e}")
    print(f"    RMS:        {metrics.rms:.4e}")
    print(f"    Mean error: {metrics.mean_error:.4e}")
    print(f"    Max error location: ({metrics.max_error_location[0]:.2f}, {metrics.max_error_location[1]:.2f})")
    
    # 3. Individual visualizations
    print("  Creating individual plots...")
    
    viz = Visualizer(output_dir=output_dir)
    viz.create_figure(subplots=(2, 2), figsize=(14, 12))
    
    # Flatten axes array for easier indexing
    axes = viz.axes.flatten() if hasattr(viz.axes, 'flatten') else viz.axes
    
    # Panel method streamlines
    viz.plot_streamlines(
        panel_results['XX'], panel_results['YY'],
        panel_results['Vx'], panel_results['Vy'],
        case.mesh, ax_index=0, density=2.0
    )
    axes[0].set_title("Panel Method - Streamlines")
    
    # Panel method contours
    viz.plot_contours(
        panel_results['XX'], panel_results['YY'],
        panel_results['Vx'], panel_results['Vy'],
        case.mesh, ax_index=1, levels=30
    )
    axes[1].set_title("Panel Method - |V|")
    
    # OpenFOAM streamlines
    viz.plot_streamlines(
        openfoam_results['XX'], openfoam_results['YY'],
        openfoam_results['Vx'], openfoam_results['Vy'],
        case.mesh, ax_index=2, density=2.0
    )
    axes[2].set_title("potentialFoam - Streamlines")
    
    # OpenFOAM contours
    viz.plot_contours(
        openfoam_results['XX'], openfoam_results['YY'],
        openfoam_results['Vx'], openfoam_results['Vy'],
        case.mesh, ax_index=3, levels=30
    )
    axes[3].set_title("potentialFoam - |V|")
    
    viz.finalize(save="comparison_combined.png", show=False)
    
    print(f"\n  Saved figures to: {output_dir}")
    
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Validate Panel Method against OpenFOAM (potentialFoam)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full workflow (generate, run OpenFOAM, compare)
    python run_validation.py ../cases/two_rounded_rects --run-openfoam
    
    # Just generate OpenFOAM case (for manual inspection/running)
    python run_validation.py ../cases/cylinder_flow --generate-only
    
    # Compare existing results (OpenFOAM already run)
    python run_validation.py ../cases/two_rounded_rects --compare-only
    
    # Skip snappyHexMesh (use blockMesh only)
    python run_validation.py ../cases/cylinder_flow --run-openfoam --no-snappy
"""
    )
    
    parser.add_argument("case_dir", type=str, help="Path to panel method case directory")
    
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--run-openfoam", action="store_true",
        help="Generate and run OpenFOAM case (full workflow)"
    )
    mode_group.add_argument(
        "--generate-only", action="store_true",
        help="Only generate OpenFOAM case (don't run)"
    )
    mode_group.add_argument(
        "--compare-only", action="store_true",
        help="Only compare results (OpenFOAM already run)"
    )
    
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for OpenFOAM case (default: validation_results/<case_name>)"
    )
    parser.add_argument(
        "--mesh-density", type=float, default=10.0,
        help="OpenFOAM mesh cells per unit length (default: 10)"
    )
    parser.add_argument(
        "--no-snappy", action="store_true",
        help="Skip snappyHexMesh (use blockMesh only)"
    )
    parser.add_argument(
        "--cores", type=int, default=6,
        help="Number of CPU cores for panel method (default: 6)"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display plots interactively"
    )
    
    args = parser.parse_args()
    
    # Default to compare-only if no mode specified
    if not any([args.run_openfoam, args.generate_only, args.compare_only]):
        args.compare_only = True
    
    # Load case
    case_dir = Path(args.case_dir).resolve()
    print(f"\nLoading case: {case_dir}")
    case = CaseLoader.load_case(case_dir)
    print(f"  Name: {case.name}")
    print(f"  Panels: {case.num_panels}")
    print(f"  V_inf: {case.v_inf} m/s, AoA: {case.aoa}°")
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("validation_results") / case_dir.name
    
    of_case_dir = output_dir / "openfoam"
    comparison_dir = output_dir / "comparison"
    
    # Run panel method
    panel_results = run_panel_method(case, num_cores=args.cores)
    
    if args.generate_only:
        # Just generate OpenFOAM case
        generate_openfoam_case(case, of_case_dir, args.mesh_density)
        print(f"\n✓ OpenFOAM case generated at: {of_case_dir}")
        print("  Run with: cd {of_case_dir} && ./Allrun")
        return
    
    if args.run_openfoam:
        # Generate and run OpenFOAM
        generate_openfoam_case(case, of_case_dir, args.mesh_density)
        run_openfoam(of_case_dir, use_snappy=not args.no_snappy)
    
    # Read OpenFOAM results
    openfoam_results = read_openfoam_results(of_case_dir, case)
    
    if openfoam_results is None:
        print("\nERROR: Could not read OpenFOAM results.")
        print("  Try running with --run-openfoam to execute OpenFOAM")
        sys.exit(1)
    
    # Compare
    metrics = compare_results(
        panel_results, openfoam_results, case,
        comparison_dir, show=args.show
    )
    
    print("\n" + "="*60)
    print("Validation Complete!")
    print("="*60)
    print(f"  Panel method:   {case.num_panels} panels")
    print(f"  OpenFOAM mesh:  potentialFoam")
    print(f"  RMS error:      {metrics.rms:.4e}")
    print(f"  Results saved:  {comparison_dir}")


if __name__ == "__main__":
    main()
