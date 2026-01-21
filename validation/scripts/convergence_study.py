#!/usr/bin/env python3
"""
Panel Method Mesh Convergence Study

Validates panel method accuracy against OpenFOAM using a STRUCTURED GRID
comparison in the far-field region (excluding near-body areas where panel
methods are known to be inaccurate due to singularity effects).

Key insight: Panel methods use singularities (sources) on the surface which
create large velocity gradients near the panels. This is a fundamental
limitation of low-order panel methods, not a mesh convergence issue.

The structured grid approach:
1. Defines a uniform grid in the far-field
2. Excludes regions within a buffer distance of the bodies
3. Interpolates OpenFOAM results to this grid
4. Computes panel method results on the same grid
5. Compares fairly in the region where panel methods are accurate

Usage:
    python convergence_study.py [--show] [--save]
    
Example:
    python convergence_study.py --show --save
"""

import sys
import argparse
from pathlib import Path
import numpy as np
from multiprocessing import Pool
import time
from scipy.spatial import cKDTree
from scipy.interpolate import griddata

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.io.geometry_io import generate_rounded_rectangle
from core.geometry import Component, Scene, Transform, Mesh
from solvers.panel2d.spm import SourcePanelSolver
from visualization.field2d import _compute_point_velocity
from visualization import ComparisonVisualizer
from validation import OpenFOAMRunner


def create_structured_grid(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    nx: int,
    ny: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a structured comparison grid.
    
    Returns:
        (XX, YY, points) where XX, YY are meshgrid arrays and 
        points is (N, 2) array of all grid points
    """
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    XX, YY = np.meshgrid(x, y)
    points = np.column_stack([XX.ravel(), YY.ravel()])
    return XX, YY, points


def filter_points_near_body(
    points: np.ndarray,
    mesh: Mesh,
    min_distance: float = 0.1
) -> np.ndarray:
    """
    Filter out points that are too close to any panel center.
    
    Panel methods are inaccurate very close to the body surface
    due to singularity effects. This filter removes points within
    min_distance of any panel to allow fair comparison.
    
    Args:
        points: (N, 2) array of points to filter
        mesh: Panel mesh (uses centers)
        min_distance: Minimum distance from body surface
    
    Returns:
        Boolean mask of points that are far enough from body
    """
    # Build KD-tree from panel centers (2D)
    panel_centers_2d = mesh.centers[:, :2]
    tree = cKDTree(panel_centers_2d)
    
    # Query for nearest panel to each point
    distances, _ = tree.query(points, k=1)
    
    # Keep points that are far enough away
    return distances > min_distance


def create_two_rect_scene(
    n_panels_side: int = 5,
    n_panels_arc: int = 4
) -> tuple[Scene, Mesh]:
    """
    Create the two rounded rectangles scene with specified panel resolution.
    
    Args:
        n_panels_side: Number of panels per straight side
        n_panels_arc: Number of panels per corner arc
    
    Returns:
        (scene, combined_mesh)
    """
    # Front rectangle (at origin)
    mesh1 = generate_rounded_rectangle(
        center=(0, 0),
        width=2.0,
        height=0.6,
        corner_radius=0.15,
        num_panels_per_side=n_panels_side,
        num_panels_per_arc=n_panels_arc
    )
    
    # Back rectangle (offset and rotated)
    mesh2 = generate_rounded_rectangle(
        center=(0, 0),
        width=2.0,
        height=0.6,
        corner_radius=0.15,
        num_panels_per_side=n_panels_side,
        num_panels_per_arc=n_panels_arc
    )
    
    # Create components with transforms matching the case file
    comp1 = Component(
        name="rect_front",
        local_mesh=mesh1,
        transform=Transform.from_2d(0, 0, 0),
        bc_type="wall"
    )
    
    comp2 = Component(
        name="rect_back",
        local_mesh=mesh2,
        transform=Transform.from_2d(-3.0, 0.5, 10.0),  # x, y, rotation_deg
        bc_type="wall"
    )
    
    # Create scene
    scene = Scene(
        name="two_rounded_rects",
        components=[comp1, comp2],
        freestream=np.array([10.0, 0.0, 0.0])
    )
    
    # Assemble into single mesh for solver
    combined_mesh = scene.assemble()
    
    return scene, combined_mesh


def compute_panel_velocity_at_points(
    mesh: Mesh,
    sigma: np.ndarray,
    v_inf: float,
    aoa: float,
    points: np.ndarray,
    num_cores: int = 6
) -> np.ndarray:
    """
    Compute panel method velocity at arbitrary points.
    
    Args:
        mesh: Panel method mesh
        sigma: Source strengths from solver
        v_inf: Freestream velocity magnitude
        aoa: Angle of attack (degrees)
        points: (N, 2) array of (x, y) coordinates
        num_cores: Number of parallel workers
    
    Returns:
        (N, 2) array of velocity vectors (Vx, Vy)
    """
    # Extract panel geometry
    panel_start_indices = mesh.panels[:, 0]
    nodes_X = mesh.nodes[panel_start_indices, 0]
    nodes_Y = mesh.nodes[panel_start_indices, 1]
    S = mesh.areas
    tx = mesh.tangents[:, 0]
    ty = mesh.tangents[:, 1]
    phi = np.arctan2(ty, tx)
    phi = np.where(phi < 0, phi + 2*np.pi, phi)
    aoa_rad = np.radians(aoa)
    
    # Build task list
    tasks = [
        (points[i, 0], points[i, 1], sigma, nodes_X, nodes_Y, phi, S, v_inf, aoa_rad)
        for i in range(len(points))
    ]
    
    # Parallel computation
    with Pool(processes=num_cores) as pool:
        results = pool.map(_compute_point_velocity, tasks)
    
    return np.array(results)


def run_convergence_study(
    of_case_dir: Path,
    panel_configs: list[tuple[int, int]],
    num_cores: int = 6,
    grid_resolution: tuple[int, int] = (100, 80),
    body_distance_filter: float = 0.5
) -> dict:
    """
    Run mesh convergence study using STRUCTURED GRID comparison.
    
    This approach compares panel method and OpenFOAM on a uniform grid
    in the far-field region, excluding points close to the bodies where
    panel methods are inherently inaccurate due to singularity effects.
    
    Args:
        of_case_dir: Path to OpenFOAM case
        panel_configs: List of (n_panels_side, n_panels_arc) tuples
        num_cores: Number of parallel workers
        grid_resolution: (nx, ny) for comparison grid
        body_distance_filter: Exclude points within this distance of body (m)
    
    Returns:
        dict with convergence data
    """
    print("=" * 60)
    print("Panel Method Mesh Convergence Study (Structured Grid)")
    print("=" * 60)
    
    # Load OpenFOAM results (reference)
    print("\nLoading OpenFOAM reference solution...")
    runner = OpenFOAMRunner(of_case_dir, verbose=False)
    
    C = runner.get_cell_centres()
    U_of_raw = runner.get_velocity_field()
    
    # Filter to midplane for interpolation source
    z_mid = 0.05
    mask = np.abs(C[:, 2] - z_mid) < 0.02
    of_points_2d = C[mask, :2]
    of_U_2d = U_of_raw[mask, :2]
    print(f"  OpenFOAM midplane cells: {len(of_points_2d)}")
    
    # Create structured comparison grid
    # Domain covers the interesting region around the bodies
    nx, ny = grid_resolution
    x_range = (-6.0, 6.0)  # Bodies are roughly at x=[-4, 1]
    y_range = (-4.0, 4.0)  # Bodies are roughly at y=[-0.5, 1.5]
    
    XX, YY, grid_points = create_structured_grid(x_range, y_range, nx, ny)
    print(f"  Structured grid: {nx}×{ny} = {len(grid_points)} points")
    
    # Create reference mesh for body distance filtering
    _, ref_mesh = create_two_rect_scene(30, 20)  # High-res for accurate body shape
    
    # Filter out points near bodies
    far_mask = filter_points_near_body(grid_points, ref_mesh, body_distance_filter)
    comparison_points = grid_points[far_mask]
    print(f"  After body filter ({body_distance_filter}m): {len(comparison_points)} points")
    
    # Interpolate OpenFOAM results to comparison points
    print("  Interpolating OpenFOAM to structured grid...")
    U_of_Vx = griddata(of_points_2d, of_U_2d[:, 0], comparison_points, method='linear')
    U_of_Vy = griddata(of_points_2d, of_U_2d[:, 1], comparison_points, method='linear')
    
    # Handle NaN from extrapolation (points outside OpenFOAM convex hull)
    valid_mask = ~(np.isnan(U_of_Vx) | np.isnan(U_of_Vy))
    comparison_points = comparison_points[valid_mask]
    U_of_interp = np.column_stack([U_of_Vx[valid_mask], U_of_Vy[valid_mask]])
    print(f"  Valid comparison points: {len(comparison_points)}")
    
    V_of = np.sqrt((U_of_interp**2).sum(axis=1))
    print(f"  OpenFOAM |V| range: [{V_of.min():.2f}, {V_of.max():.2f}] m/s")
    
    # Results storage
    results = {
        'n_panels_side': [],
        'n_panels_arc': [],
        'total_panels': [],
        'rms_error': [],
        'max_error': [],
        'rel_rms_error': [],
        'solve_time': [],
        'compare_time': [],
        'comparison_points': len(comparison_points)
    }
    
    v_inf = 10.0
    aoa = 0.0
    
    # Run for each configuration
    for n_side, n_arc in panel_configs:
        print(f"\n--- Configuration: n_side={n_side}, n_arc={n_arc} ---")
        
        # Create mesh
        scene, mesh = create_two_rect_scene(n_side, n_arc)
        n_total = mesh.num_panels
        print(f"  Total panels: {n_total}")
        
        # Solve
        t0 = time.time()
        solver = SourcePanelSolver(mesh, v_inf=v_inf, aoa=aoa)
        solver.solve()
        solve_time = time.time() - t0
        print(f"  Solve time: {solve_time:.2f}s")
        
        # Compute velocity at comparison points
        t0 = time.time()
        U_pm = compute_panel_velocity_at_points(
            mesh, solver.sigma, v_inf, aoa, comparison_points, num_cores
        )
        compare_time = time.time() - t0
        print(f"  Comparison time: {compare_time:.2f}s")
        
        # Compute errors
        diff = U_pm - U_of_interp
        V_pm = np.sqrt((U_pm**2).sum(axis=1))
        
        rms_error = np.sqrt(np.mean(diff**2))
        max_error = np.abs(diff).max()
        rel_rms = rms_error / V_of.mean() * 100
        
        print(f"  RMS error: {rms_error:.4f} m/s ({rel_rms:.2f}%)")
        print(f"  Max error: {max_error:.4f} m/s")
        
        # Store results
        results['n_panels_side'].append(n_side)
        results['n_panels_arc'].append(n_arc)
        results['total_panels'].append(n_total)
        results['rms_error'].append(rms_error)
        results['max_error'].append(max_error)
        results['rel_rms_error'].append(rel_rms)
        results['solve_time'].append(solve_time)
        results['compare_time'].append(compare_time)
    
    return results


def plot_convergence(results: dict, output_dir: Path, show: bool = True, save: bool = True):
    """Plot convergence results."""
    import matplotlib.pyplot as plt
    
    n_panels = np.array(results['total_panels'])
    rms_error = np.array(results['rms_error'])
    max_error = np.array(results['max_error'])
    rel_rms = np.array(results['rel_rms_error'])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: RMS Error vs Panel Count
    ax = axes[0]
    ax.loglog(n_panels, rms_error, 'bo-', linewidth=2, markersize=8, label='RMS Error')
    ax.loglog(n_panels, max_error, 'rs--', linewidth=2, markersize=8, label='Max Error')
    ax.set_xlabel('Number of Panels', fontsize=12)
    ax.set_ylabel('Velocity Error (m/s)', fontsize=12)
    ax.set_title('Error vs Mesh Resolution', fontsize=14)
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    
    # Add convergence rate annotation
    if len(n_panels) > 2:
        # Fit power law: error = C * n^(-p)
        log_n = np.log(n_panels)
        log_err = np.log(rms_error)
        p, log_C = np.polyfit(log_n, log_err, 1)
        ax.text(0.05, 0.05, f'Slope: {p:.2f}', transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Relative RMS Error
    ax = axes[1]
    ax.semilogx(n_panels, rel_rms, 'go-', linewidth=2, markersize=8)
    ax.axhline(y=5, color='r', linestyle='--', alpha=0.5, label='5% threshold')
    ax.axhline(y=2, color='orange', linestyle='--', alpha=0.5, label='2% threshold')
    ax.set_xlabel('Number of Panels', fontsize=12)
    ax.set_ylabel('Relative RMS Error (%)', fontsize=12)
    ax.set_title('Relative Error Convergence', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Computation Time
    ax = axes[2]
    solve_time = np.array(results['solve_time'])
    compare_time = np.array(results['compare_time'])
    
    ax.loglog(n_panels, solve_time, 'b^-', linewidth=2, markersize=8, label='Solve Time')
    ax.loglog(n_panels, compare_time, 'rv-', linewidth=2, markersize=8, label='Compare Time')
    ax.set_xlabel('Number of Panels', fontsize=12)
    ax.set_ylabel('Time (s)', fontsize=12)
    ax.set_title('Computation Time', fontsize=14)
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        output_path = output_dir / "convergence_study.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def print_convergence_table(results: dict):
    """Print results as a formatted table."""
    print("\n" + "=" * 80)
    print("CONVERGENCE RESULTS")
    print("=" * 80)
    print(f"{'n_side':>8} {'n_arc':>8} {'Total':>8} {'RMS (m/s)':>12} {'Max (m/s)':>12} {'Rel RMS %':>10}")
    print("-" * 80)
    
    for i in range(len(results['total_panels'])):
        print(f"{results['n_panels_side'][i]:>8} "
              f"{results['n_panels_arc'][i]:>8} "
              f"{results['total_panels'][i]:>8} "
              f"{results['rms_error'][i]:>12.4f} "
              f"{results['max_error'][i]:>12.4f} "
              f"{results['rel_rms_error'][i]:>10.2f}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Panel method mesh convergence study against OpenFOAM (structured grid)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--show", action="store_true", help="Display plots")
    parser.add_argument("--save", action="store_true", default=True, help="Save plots")
    parser.add_argument("--cores", type=int, default=6, help="Number of CPU cores")
    parser.add_argument("--grid-nx", type=int, default=100, 
                        help="Grid resolution in X direction")
    parser.add_argument("--grid-ny", type=int, default=80,
                        help="Grid resolution in Y direction")
    parser.add_argument("--body-distance", type=float, default=0.5,
                        help="Minimum distance from body surface (m)")
    parser.add_argument("--of-case", type=str, 
                        default="validation_results/two_rounded_rects/openfoam",
                        help="Path to OpenFOAM case")
    parser.add_argument("--output-dir", type=str,
                        default="validation_results/two_rounded_rects/convergence",
                        help="Output directory for plots")
    args = parser.parse_args()
    
    of_case_dir = Path(args.of_case)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not of_case_dir.exists():
        print(f"ERROR: OpenFOAM case not found: {of_case_dir}")
        print("Run the validation first: python run_validation.py --run-openfoam cases/two_rounded_rects")
        sys.exit(1)
    
    # Define mesh configurations: (n_panels_side, n_panels_arc)
    # Total panels = 2 components × (4 sides × n_side + 4 corners × n_arc)
    #              = 2 × (4 × n_side + 4 × n_arc) = 8 × (n_side + n_arc)
    panel_configs = [
        (2, 2),    # 32 panels (very coarse)
        (3, 3),    # 48 panels
        (4, 4),    # 64 panels
        (5, 5),    # 80 panels
        (6, 6),    # 96 panels
        (8, 8),    # 128 panels
        (10, 10),  # 160 panels
        (15, 12),  # 216 panels
        (20, 16),  # 288 panels
        (30, 20),  # 400 panels
    ]
    
    # Run study with structured grid comparison
    results = run_convergence_study(
        of_case_dir,
        panel_configs,
        num_cores=args.cores,
        grid_resolution=(args.grid_nx, args.grid_ny),
        body_distance_filter=args.body_distance
    )
    
    # Print table
    print_convergence_table(results)
    
    # Plot
    plot_convergence(results, output_dir, show=args.show, save=args.save)
    
    # Save results to CSV
    import csv
    csv_path = output_dir / "convergence_data.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['n_side', 'n_arc', 'total_panels', 'rms_error', 'max_error', 
                        'rel_rms_percent', 'solve_time_s', 'compare_time_s'])
        for i in range(len(results['total_panels'])):
            writer.writerow([
                results['n_panels_side'][i],
                results['n_panels_arc'][i],
                results['total_panels'][i],
                results['rms_error'][i],
                results['max_error'][i],
                results['rel_rms_error'][i],
                results['solve_time'][i],
                results['compare_time'][i]
            ])
    print(f"Saved: {csv_path}")
    
    print("\n✓ Convergence study complete!")


if __name__ == "__main__":
    main()
