#!/usr/bin/env python3
"""
Unified Validation Pipeline: Panel Method vs OpenFOAM

This is the main entry point for all validation workflows. It consolidates:
1. OpenFOAM case generation from panel method case
2. OpenFOAM mesh convergence study
3. Panel method convergence study
4. Final comparison between converged solutions

Workflow:
    Step 1: Generate OpenFOAM case (--generate)
    Step 2: Run OpenFOAM mesh convergence (--of-convergence)
    Step 3: Run panel method convergence against converged OF (--pm-convergence)
    Step 4: Compare final results (--compare)
    
    Or run all steps: --full

Directory Structure:
    validation_results/<case_name>/
    ├── openfoam/                    # Base OpenFOAM case
    ├── openfoam_convergence/        # OF mesh convergence study
    │   ├── config.yaml
    │   ├── cases/                   # Individual mesh levels
    │   ├── convergence_data.csv
    │   └── mesh_convergence.png
    ├── final_case/                  # Converged OF mesh (copied)
    ├── panel_convergence/           # PM convergence study
    │   ├── convergence_data.csv
    │   └── panel_convergence.png
    └── comparison/                  # Final comparison plots

Usage:
    # Full workflow (recommended for first run)
    python validate.py cases/two_rounded_rects --full
    
    # Individual steps
    python validate.py cases/two_rounded_rects --generate
    python validate.py cases/two_rounded_rects --of-convergence
    python validate.py cases/two_rounded_rects --pm-convergence
    python validate.py cases/two_rounded_rects --compare
    
    # Quick comparison (skip convergence studies)
    python validate.py cases/two_rounded_rects --quick
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import shutil
import yaml
from multiprocessing import Pool
from typing import Optional
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.io import CaseLoader
from solvers.panel2d.spm import SourcePanelSolver
from visualization.field2d import VelocityField2D, _compute_point_velocity
from visualization.comparison import FieldSeries


# =============================================================================
# Panel Method Functions
# =============================================================================

def run_panel_method(case, num_cores: int = 6) -> dict:
    """Run panel method solver and compute velocity field."""
    print("\n" + "="*60)
    print("Running Panel Method Solver")
    print("="*60)
    
    solver = SourcePanelSolver(case.mesh, v_inf=case.v_inf, aoa=case.aoa)
    solver.solve()
    
    print(f"  Panels: {case.num_panels}")
    print(f"  Cp range: [{min(solver.Cp):.4f}, {max(solver.Cp):.4f}]")
    
    # Compute velocity field
    vfield = VelocityField2D(case.mesh, case.v_inf, case.aoa, solver.sigma)
    XX, YY, Vx, Vy = vfield.compute(
        case.x_range, case.y_range, case.resolution, num_cores=num_cores
    )
    
    V_mag = np.sqrt(Vx**2 + Vy**2)
    print(f"  |V| range: [{np.nanmin(V_mag):.4f}, {np.nanmax(V_mag):.4f}]")
    
    return {
        'XX': XX, 'YY': YY, 'Vx': Vx, 'Vy': Vy, 'V_mag': V_mag,
        'sigma': solver.sigma, 'Cp': solver.Cp
    }


def compute_panel_at_points(
    mesh, sigma: np.ndarray, v_inf: float, aoa: float,
    points: np.ndarray, num_cores: int = 6
) -> np.ndarray:
    """Compute panel method velocity at arbitrary 2D points."""
    panel_start_indices = mesh.panels[:, 0]
    nodes_X = mesh.nodes[panel_start_indices, 0]
    nodes_Y = mesh.nodes[panel_start_indices, 1]
    S = mesh.areas
    tx = mesh.tangents[:, 0]
    ty = mesh.tangents[:, 1]
    phi = np.arctan2(ty, tx)
    phi = np.where(phi < 0, phi + 2*np.pi, phi)
    aoa_rad = np.radians(aoa)
    
    tasks = [
        (points[i, 0], points[i, 1], sigma, nodes_X, nodes_Y, phi, S, v_inf, aoa_rad)
        for i in range(len(points))
    ]
    
    with Pool(processes=num_cores) as pool:
        results = pool.map(_compute_point_velocity, tasks)
    
    return np.array(results)


# =============================================================================
# OpenFOAM Functions
# =============================================================================

def generate_openfoam_case(case, output_dir: Path, mesh_density: float = 10.0) -> Path:
    """Generate OpenFOAM case from panel method case."""
    print("\n" + "="*60)
    print("Generating OpenFOAM Case")
    print("="*60)
    
    from validation.adapters.openfoam.case_generator import OpenFOAMCaseGenerator, MeshSettings
    
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
    print(f"  Generated: {of_case_dir}")
    
    return of_case_dir


def run_openfoam_case(of_case_dir: Path, use_snappy: bool = True) -> bool:
    """Run OpenFOAM meshing and solver."""
    print("\n" + "="*60)
    print("Running OpenFOAM (potentialFoam)")
    print("="*60)
    
    from validation import OpenFOAMRunner
    
    runner = OpenFOAMRunner(of_case_dir, verbose=True)
    success = runner.run_all(solver="potentialFoam", use_snappy=use_snappy)
    
    if not success:
        print("\nWARNING: OpenFOAM run had errors.")
    
    print(f"\n{runner.summary()}")
    return success


def read_openfoam_results(of_case_dir: Path) -> Optional[dict]:
    """Read OpenFOAM velocity field from case directory."""
    from validation import OpenFOAMRunner
    
    runner = OpenFOAMRunner(of_case_dir, verbose=False)
    
    C = runner.get_cell_centres()
    U = runner.get_velocity_field()
    
    if C is None or U is None:
        return None
    
    # Filter to midplane
    z_mid = 0.05
    mask = np.abs(C[:, 2] - z_mid) < 0.03
    
    return {
        'points_2d': C[mask, :2],
        'U_2d': U[mask, :2],
        'cell_centres': C,
        'U_raw': U
    }


# =============================================================================
# Comparison Functions
# =============================================================================

def create_comparison_grid(
    x_range: tuple, y_range: tuple, nx: int, ny: int,
    mesh, min_distance: float = 0.5
) -> tuple:
    """
    Create structured comparison grid excluding near-body region.
    
    Panel methods are inaccurate near body surfaces due to singularities.
    This function creates a grid and masks out points too close to panels.
    """
    from scipy.spatial import cKDTree
    
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    XX, YY = np.meshgrid(x, y)
    points = np.column_stack([XX.ravel(), YY.ravel()])
    
    # Filter out points near body
    panel_centers_2d = mesh.centers[:, :2]
    tree = cKDTree(panel_centers_2d)
    distances, _ = tree.query(points, k=1)
    valid_mask = distances > min_distance
    
    return XX, YY, points, valid_mask


def compare_on_structured_grid(
    panel_results: dict,
    of_results: dict,
    mesh,
    x_range: tuple,
    y_range: tuple,
    grid_resolution: tuple = (100, 80),
    body_distance: float = 0.5,
    num_cores: int = 6
) -> dict:
    """
    Compare panel method and OpenFOAM on a structured grid.
    
    This is the fairest comparison because:
    1. Uses identical points for both methods
    2. Excludes near-body region where panel methods are inherently inaccurate
    3. Uses far-field where both methods should agree
    """
    from scipy.interpolate import griddata
    
    print("\n" + "="*60)
    print("Structured Grid Comparison")
    print("="*60)
    
    nx, ny = grid_resolution
    XX, YY, points, valid_mask = create_comparison_grid(
        x_range, y_range, nx, ny, mesh, body_distance
    )
    
    valid_points = points[valid_mask]
    print(f"  Grid: {nx}×{ny} = {len(points)} points")
    print(f"  Valid (far-field): {len(valid_points)} points ({100*len(valid_points)/len(points):.1f}%)")
    
    # Interpolate OpenFOAM to grid
    of_U_grid = griddata(
        of_results['points_2d'], 
        of_results['U_2d'], 
        valid_points,
        method='linear'
    )
    of_V_mag = np.sqrt(of_U_grid[:, 0]**2 + of_U_grid[:, 1]**2)
    
    # Compute panel method at same points
    # This uses the pre-computed sigma from panel_results
    print("  Computing panel method at grid points...")
    pm_U = compute_panel_at_points(
        mesh, panel_results['sigma'], 
        10.0, 0.0,  # v_inf, aoa  
        valid_points, num_cores
    )
    pm_V_mag = np.sqrt(pm_U[:, 0]**2 + pm_U[:, 1]**2)
    
    # Compute error metrics
    diff = pm_V_mag - of_V_mag
    valid = ~np.isnan(diff)
    
    rmse = np.sqrt(np.mean(diff[valid]**2))
    mae = np.mean(np.abs(diff[valid]))
    max_err = np.max(np.abs(diff[valid]))
    mean_of = np.mean(of_V_mag[valid])
    rel_rmse = 100 * rmse / mean_of
    
    print(f"\n  Error Metrics (far-field only):")
    print(f"    RMSE:     {rmse:.4f} m/s ({rel_rmse:.2f}%)")
    print(f"    MAE:      {mae:.4f} m/s")
    print(f"    Max:      {max_err:.4f} m/s")
    print(f"    OF mean:  {mean_of:.4f} m/s")
    
    return {
        'XX': XX, 'YY': YY,
        'valid_mask': valid_mask,
        'valid_points': valid_points,
        'of_V_mag': of_V_mag,
        'pm_V_mag': pm_V_mag,
        'diff': diff,
        'rmse': rmse,
        'mae': mae,
        'max_error': max_err,
        'rel_rmse': rel_rmse
    }


def plot_comparison(
    panel_results: dict,
    of_case_dir: Path,
    comparison: dict,
    mesh,
    output_dir: Path,
    show: bool = False
):
    """
    Create comparison plots using ComparisonVisualizer.
    
    Uses the proper visualization class that:
    - Draws body boundaries
    - Computes proper error metrics
    
    Note: No body-distance filtering is applied for visualization.
    The filtering in compare_on_structured_grid is only for error metrics.
    """
    from scipy.interpolate import griddata
    from visualization import ComparisonVisualizer
    from visualization.comparison import FieldSeries
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read OpenFOAM results and interpolate to panel method grid
    of_results = read_openfoam_results(of_case_dir)
    of_V_grid = griddata(
        of_results['points_2d'],
        np.sqrt(of_results['U_2d'][:, 0]**2 + of_results['U_2d'][:, 1]**2),
        (panel_results['XX'], panel_results['YY']),
        method='linear'
    )
    
    # NOTE: No body-distance filtering for visualization!
    # The ComparisonVisualizer._draw_body_outline() will draw body boundaries,
    # and OpenFOAM naturally has NaN inside the body (no mesh cells there).
    # Panel method may have high values near body but that's expected behavior.
    
    # Create FieldSeries for comparison (unfiltered data)
    panel_field = FieldSeries(
        name="|V|",
        data=panel_results['V_mag'],
        XX=panel_results['XX'],
        YY=panel_results['YY'],
        units="m/s",
        source="Panel Method"
    )
    
    of_field = FieldSeries(
        name="|V|",
        data=of_V_grid,
        XX=panel_results['XX'],
        YY=panel_results['YY'],
        units="m/s",
        source="OpenFOAM (potentialFoam)"
    )
    
    # Use ComparisonVisualizer for difference plot with body outline
    comp = ComparisonVisualizer(output_dir=output_dir)
    
    fig, metrics = comp.plot_difference(
        panel_field, of_field,
        mesh=mesh,
        levels=30,
        cmap='RdBu_r',
        symmetric=True,
        show_body=True,
        show_originals=True,
        title=f"Panel Method vs OpenFOAM Comparison\n"
              f"Far-field RMSE: {comparison['rmse']:.4f} m/s ({comparison['rel_rmse']:.2f}%)"
    )
    
    # Save
    plot_path = output_dir / "comparison.png"
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved: {plot_path}")
    
    # Print metrics
    print(f"\nError Metrics (from ComparisonVisualizer):")
    print(metrics.summary())
    
    import matplotlib.pyplot as plt
    if show:
        plt.show()
    plt.close()
    
    return plot_path


# =============================================================================
# Main Workflow Functions  
# =============================================================================

def step_generate(case_path: Path, output_dir: Path) -> Path:
    """Step 1: Generate OpenFOAM case from panel method case."""
    case = CaseLoader.load_case(case_path)
    of_dir = output_dir / "openfoam"
    return generate_openfoam_case(case, of_dir)


def step_of_convergence(output_dir: Path, show: bool = False) -> dict:
    """Step 2: Run OpenFOAM mesh convergence study."""
    from validation.scripts.openfoam_convergence import (
        run_convergence_study, load_config, create_default_config
    )
    
    base_case = output_dir / "openfoam"
    conv_dir = output_dir / "openfoam_convergence"
    config_path = conv_dir / "config.yaml"
    
    if not base_case.exists():
        print(f"ERROR: Base OpenFOAM case not found: {base_case}")
        print("Run --generate first")
        sys.exit(1)
    
    conv_dir.mkdir(parents=True, exist_ok=True)
    
    if not config_path.exists():
        create_default_config(config_path)
    
    config = load_config(config_path)
    
    return run_convergence_study(
        base_case, conv_dir, config, show_plots=show
    )


def step_pm_convergence(
    case_path: Path, 
    output_dir: Path, 
    of_case_dir: Path,
    num_cores: int = 6,
    show: bool = False
) -> dict:
    """
    Step 3: Run panel method convergence study against converged OpenFOAM.
    
    Plots ACTUAL VALUES (not just errors) against number of panels,
    with OpenFOAM converged values shown as horizontal reference lines.
    
    Measured quantities:
    - Mean velocity magnitude (far-field)
    - Max velocity magnitude
    - Mean Cp on body surface
    - Min Cp (suction peak)
    - Velocity at probe points (far upstream, far downstream)
    """
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree
    
    print("\n" + "="*60)
    print("Panel Method Convergence Study")
    print("="*60)
    
    # Load OpenFOAM reference
    of_results = read_openfoam_results(of_case_dir)
    if of_results is None:
        print("ERROR: Could not read OpenFOAM results")
        sys.exit(1)
    
    # Load case to get geometry info
    case = CaseLoader.load_case(case_path)
    
    # Create structured comparison grid
    nx, ny = 100, 80
    x = np.linspace(case.x_range[0], case.x_range[1], nx)
    y = np.linspace(case.y_range[0], case.y_range[1], ny)
    XX, YY = np.meshgrid(x, y)
    grid_points = np.column_stack([XX.ravel(), YY.ravel()])
    
    # Interpolate OpenFOAM to grid
    of_U_grid = griddata(of_results['points_2d'], of_results['U_2d'], grid_points, method='linear')
    of_V_mag_grid = np.sqrt(of_U_grid[:, 0]**2 + of_U_grid[:, 1]**2)
    
    # Define probe points for far-field measurements
    # These should be far from bodies to avoid singularity effects
    probe_points = {
        'upstream': np.array([case.x_range[0] + 1.0, 0.0]),      # Far upstream
        'downstream': np.array([case.x_range[1] - 1.0, 0.0]),    # Far downstream  
        'above': np.array([0.0, case.y_range[1] - 1.0]),         # Far above
    }
    
    # Get OpenFOAM values at probe points
    of_probe_values = {}
    for name, point in probe_points.items():
        # Find nearest OF point
        distances = np.sqrt((of_results['points_2d'][:, 0] - point[0])**2 + 
                           (of_results['points_2d'][:, 1] - point[1])**2)
        nearest_idx = np.argmin(distances)
        of_U_probe = of_results['U_2d'][nearest_idx]
        of_probe_values[name] = np.sqrt(of_U_probe[0]**2 + of_U_probe[1]**2)
    
    # OpenFOAM reference statistics (for horizontal lines)
    # Use far-field mask for mean velocity (exclude near-body)
    # First we need a reference mesh to build the mask - use finest panel mesh
    of_V_valid = of_V_mag_grid[~np.isnan(of_V_mag_grid)]
    of_ref = {
        'mean_velocity': float(np.mean(of_V_valid)),
        'max_velocity': float(np.max(of_V_valid)),
        'probe_upstream': of_probe_values['upstream'],
        'probe_downstream': of_probe_values['downstream'],
        'probe_above': of_probe_values['above'],
    }
    
    print(f"\n  OpenFOAM Reference Values:")
    print(f"    Mean |V|:        {of_ref['mean_velocity']:.4f} m/s")
    print(f"    Max |V|:         {of_ref['max_velocity']:.4f} m/s")
    print(f"    Probe upstream:  {of_ref['probe_upstream']:.4f} m/s")
    print(f"    Probe downstream:{of_ref['probe_downstream']:.4f} m/s")
    
    # Panel configurations to test (n_panels_side, n_panels_arc)
    panel_configs = [
        (3, 2),   # ~40 panels per body
        (5, 4),   # ~72 panels per body
        (8, 6),   # ~112 panels per body
        (12, 8),  # ~160 panels per body
        (16, 10), # ~208 panels per body
        (20, 12), # ~256 panels per b
        (25, 15),
        (30, 20),
        (40, 25),
    ]
    
    # Results storage - actual values, not errors
    results = {
        'panel_counts': [],
        # Velocity metrics
        'mean_velocity': [],
        'max_velocity': [],
        # Pressure metrics (on body surface)
        'mean_Cp': [],
        'min_Cp': [],  # Suction peak
        'max_Cp': [],  # Stagnation
        # Probe points
        'probe_upstream': [],
        'probe_downstream': [],
        'probe_above': [],
        # Error metrics (for reference)
        'rmse': [],
        'mae': [],
    }
    
    # Import geometry generator
    from core.io.geometry_io import generate_rounded_rectangle
    from core.geometry import Component, Scene, Transform
    
    for n_side, n_arc in panel_configs:
        print(f"\n  Testing panels: side={n_side}, arc={n_arc}")
        
        # Create scene with specified panel resolution
        mesh1 = generate_rounded_rectangle(
            center=(0, 0), width=2.0, height=0.6, corner_radius=0.15,
            num_panels_per_side=n_side, num_panels_per_arc=n_arc
        )
        mesh2 = generate_rounded_rectangle(
            center=(0, 0), width=2.0, height=0.6, corner_radius=0.15,
            num_panels_per_side=n_side, num_panels_per_arc=n_arc
        )
        
        comp1 = Component("rect_front", mesh1, Transform.from_2d(0, 0, 0), "wall")
        comp2 = Component("rect_back", mesh2, Transform.from_2d(-3.0, 0.5, 10.0), "wall")
        scene = Scene("test", [comp1, comp2], np.array([10.0, 0.0, 0.0]))
        combined_mesh = scene.assemble()
        
        n_panels = combined_mesh.num_panels
        print(f"    Total panels: {n_panels}")
        results['panel_counts'].append(n_panels)
        
        # Build body distance filter for far-field comparison
        tree = cKDTree(combined_mesh.centers[:, :2])
        distances, _ = tree.query(grid_points, k=1)
        far_field_mask = distances > 0.5
        far_field_points = grid_points[far_field_mask]
        
        # Solve panel method
        solver = SourcePanelSolver(combined_mesh, v_inf=10.0, aoa=0.0)
        solver.solve()
        
        # =====================================================================
        # Collect ACTUAL VALUES
        # =====================================================================
        
        # 1. Surface pressure metrics
        results['mean_Cp'].append(float(np.mean(solver.Cp)))
        results['min_Cp'].append(float(np.min(solver.Cp)))  # Suction peak
        results['max_Cp'].append(float(np.max(solver.Cp)))  # Stagnation
        
        # 2. Compute velocity at far-field grid points
        pm_U = compute_panel_at_points(
            combined_mesh, solver.sigma, 10.0, 0.0, far_field_points, num_cores
        )
        pm_V_mag = np.sqrt(pm_U[:, 0]**2 + pm_U[:, 1]**2)
        
        valid = ~np.isnan(pm_V_mag)
        results['mean_velocity'].append(float(np.mean(pm_V_mag[valid])))
        results['max_velocity'].append(float(np.max(pm_V_mag[valid])))
        
        # 3. Probe point velocities
        for probe_name, probe_point in probe_points.items():
            pm_U_probe = compute_panel_at_points(
                combined_mesh, solver.sigma, 10.0, 0.0, 
                probe_point.reshape(1, 2), num_cores=1
            )
            pm_V_probe = np.sqrt(pm_U_probe[0, 0]**2 + pm_U_probe[0, 1]**2)
            results[f'probe_{probe_name}'].append(float(pm_V_probe))
        
        # 4. Error metrics (vs OpenFOAM)
        of_V_far = of_V_mag_grid[far_field_mask]
        diff = pm_V_mag - of_V_far
        valid_diff = ~np.isnan(diff)
        
        rmse = np.sqrt(np.mean(diff[valid_diff]**2))
        mae = np.mean(np.abs(diff[valid_diff]))
        results['rmse'].append(rmse)
        results['mae'].append(mae)
        
        print(f"    Mean |V|: {results['mean_velocity'][-1]:.4f} m/s")
        print(f"    Min Cp: {results['min_Cp'][-1]:.4f}, Max Cp: {results['max_Cp'][-1]:.4f}")
        print(f"    RMSE vs OF: {rmse:.4f} m/s")
    
    # Save results
    conv_dir = output_dir / "panel_convergence"
    conv_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV with all data
    import csv
    csv_path = conv_dir / "convergence_data.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = ['panels', 'mean_velocity', 'max_velocity', 'mean_Cp', 'min_Cp', 'max_Cp',
                   'probe_upstream', 'probe_downstream', 'probe_above', 'rmse', 'mae']
        writer.writerow(headers)
        for i in range(len(results['panel_counts'])):
            writer.writerow([
                results['panel_counts'][i],
                results['mean_velocity'][i],
                results['max_velocity'][i],
                results['mean_Cp'][i],
                results['min_Cp'][i],
                results['max_Cp'][i],
                results['probe_upstream'][i],
                results['probe_downstream'][i],
                results['probe_above'][i],
                results['rmse'][i],
                results['mae'][i],
            ])
    print(f"\n✓ CSV saved: {csv_path}")
    
    # Save OpenFOAM reference values
    import yaml
    ref_path = conv_dir / "openfoam_reference.yaml"
    with open(ref_path, 'w') as f:
        yaml.dump(of_ref, f, default_flow_style=False)
    print(f"✓ Reference values saved: {ref_path}")
    
    # =========================================================================
    # PLOTTING - Actual values with OF reference lines
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    panels = np.array(results['panel_counts'])
    
    # Plot 1: Mean Velocity (far-field)
    ax = axes[0, 0]
    ax.semilogx(panels, results['mean_velocity'], 'o-', linewidth=2, markersize=8, label='Panel Method')
    ax.axhline(of_ref['mean_velocity'], color='r', linestyle='--', linewidth=2, label=f"OF: {of_ref['mean_velocity']:.3f}")
    ax.set_xlabel('Number of Panels')
    ax.set_ylabel('Mean |V| (m/s)')
    ax.set_title('Mean Velocity (Far-Field)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Max Velocity
    ax = axes[0, 1]
    ax.semilogx(panels, results['max_velocity'], 'o-', linewidth=2, markersize=8, label='Panel Method')
    ax.axhline(of_ref['max_velocity'], color='r', linestyle='--', linewidth=2, label=f"OF: {of_ref['max_velocity']:.3f}")
    ax.set_xlabel('Number of Panels')
    ax.set_ylabel('Max |V| (m/s)')
    ax.set_title('Maximum Velocity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Probe Points
    ax = axes[0, 2]
    ax.semilogx(panels, results['probe_upstream'], 'o-', linewidth=2, markersize=6, label='Upstream')
    ax.semilogx(panels, results['probe_downstream'], 's-', linewidth=2, markersize=6, label='Downstream')
    ax.semilogx(panels, results['probe_above'], '^-', linewidth=2, markersize=6, label='Above')
    ax.axhline(of_ref['probe_upstream'], color='C0', linestyle='--', alpha=0.7)
    ax.axhline(of_ref['probe_downstream'], color='C1', linestyle='--', alpha=0.7)
    ax.axhline(of_ref['probe_above'], color='C2', linestyle='--', alpha=0.7)
    ax.set_xlabel('Number of Panels')
    ax.set_ylabel('|V| at Probe (m/s)')
    ax.set_title('Far-Field Probe Points')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Surface Cp (min/max/mean)
    ax = axes[1, 0]
    ax.semilogx(panels, results['min_Cp'], 'v-', linewidth=2, markersize=8, label='Min Cp (suction)', color='blue')
    ax.semilogx(panels, results['max_Cp'], '^-', linewidth=2, markersize=8, label='Max Cp (stagnation)', color='red')
    ax.semilogx(panels, results['mean_Cp'], 'o-', linewidth=2, markersize=8, label='Mean Cp', color='green')
    ax.set_xlabel('Number of Panels')
    ax.set_ylabel('Cp')
    ax.set_title('Surface Pressure Coefficient')
    ax.legend()
    ax.grid(True, alpha=0.3)
    # Note: No OF reference for Cp since potentialFoam pressure isn't directly comparable
    
    # Plot 5: Error vs OF (RMSE, MAE)
    ax = axes[1, 1]
    ax.loglog(panels, results['rmse'], 'o-', linewidth=2, markersize=8, label='RMSE')
    ax.loglog(panels, results['mae'], 's-', linewidth=2, markersize=8, label='MAE')
    ax.set_xlabel('Number of Panels')
    ax.set_ylabel('Error (m/s)')
    ax.set_title('Error vs OpenFOAM')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    # Plot 6: Summary / Convergence Rate
    ax = axes[1, 2]
    ax.axis('off')
    
    # Compute convergence info
    if len(panels) >= 2:
        # Estimate convergence rate from last two points
        p1, p2 = panels[-2], panels[-1]
        v1, v2 = results['mean_velocity'][-2], results['mean_velocity'][-1]
        rel_change = abs(v2 - v1) / abs(v2) * 100 if v2 != 0 else 0
    else:
        rel_change = 0
    
    summary_text = f"""
Panel Method Convergence Summary
================================

Panel counts tested: {len(panels)}
Range: {panels[0]} to {panels[-1]} panels

Final Values (N={panels[-1]}):
  Mean |V|:    {results['mean_velocity'][-1]:.4f} m/s
  Max |V|:     {results['max_velocity'][-1]:.4f} m/s
  Min Cp:      {results['min_Cp'][-1]:.4f}
  Max Cp:      {results['max_Cp'][-1]:.4f}

OpenFOAM Reference:
  Mean |V|:    {of_ref['mean_velocity']:.4f} m/s
  Max |V|:     {of_ref['max_velocity']:.4f} m/s

Final Error vs OF:
  RMSE:        {results['rmse'][-1]:.4f} m/s
  MAE:         {results['mae'][-1]:.4f} m/s

Change from 2nd-finest to finest:
  Mean |V|:    {rel_change:.2f}%
"""
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('Panel Method Mesh Convergence Study', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_path = conv_dir / "panel_convergence.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved: {plot_path}")
    
    if show:
        plt.show()
    plt.close()
    
    # Add OF reference to results for return
    results['of_reference'] = of_ref
    
    return results


def step_compare(
    case_path: Path,
    output_dir: Path,
    of_case_dir: Path,
    num_cores: int = 6,
    show: bool = False
) -> dict:
    """Step 4: Final comparison between panel method and OpenFOAM."""
    print("\n" + "="*60)
    print("Final Comparison")
    print("="*60)
    
    case = CaseLoader.load_case(case_path)
    
    # Run panel method
    panel_results = run_panel_method(case, num_cores)
    
    # Read OpenFOAM results
    of_results = read_openfoam_results(of_case_dir)
    if of_results is None:
        print("ERROR: Could not read OpenFOAM results")
        sys.exit(1)
    
    # Compare on structured grid
    comparison = compare_on_structured_grid(
        panel_results, of_results, case.mesh,
        case.x_range, case.y_range,
        grid_resolution=(100, 80),
        body_distance=0.5,
        num_cores=num_cores
    )
    
    # Generate plots
    comparison_dir = output_dir / "comparison"
    plot_comparison(panel_results, of_case_dir, comparison, case.mesh, comparison_dir, show)
    
    return comparison


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified Validation Pipeline: Panel Method vs OpenFOAM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow Steps:
    --generate       Generate OpenFOAM case from panel method case
    --of-convergence Run OpenFOAM mesh convergence study
    --pm-convergence Run panel method convergence study
    --compare        Compare final results
    
Convenience Options:
    --full           Run all steps (generate + of-convergence + pm-convergence + compare)
    --quick          Skip convergence studies (generate + run OF once + compare)
    
Examples:
    python validate.py cases/two_rounded_rects --full
    python validate.py cases/two_rounded_rects --compare  # If OF already run
    python validate.py cases/two_rounded_rects --quick --show
        """
    )
    parser.add_argument("case_path", help="Path to panel method case directory")
    
    # Workflow options
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--full", action="store_true", help="Run complete workflow")
    group.add_argument("--quick", action="store_true", help="Quick comparison (no convergence studies)")
    group.add_argument("--generate", action="store_true", help="Generate OpenFOAM case only")
    group.add_argument("--of-convergence", action="store_true", help="Run OpenFOAM mesh convergence")
    group.add_argument("--pm-convergence", action="store_true", help="Run panel method convergence")
    group.add_argument("--compare", action="store_true", help="Compare results only")
    
    # Options
    parser.add_argument("--show", action="store_true", help="Display plots")
    parser.add_argument("--cores", type=int, default=6, help="CPU cores for panel method")
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    
    args = parser.parse_args()
    
    case_path = Path(args.case_path).resolve()
    if not case_path.exists():
        print(f"ERROR: Case not found: {case_path}")
        sys.exit(1)
    
    case_name = case_path.name
    output_dir = Path(args.output_dir) if args.output_dir else Path("validation_results") / case_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Validation Pipeline")
    print("="*60)
    print(f"  Case: {case_path}")
    print(f"  Output: {output_dir}")
    
    # Determine which steps to run
    if args.full:
        steps = ['generate', 'of_convergence', 'pm_convergence', 'compare']
    elif args.quick:
        steps = ['generate', 'run_of', 'compare']
    elif args.generate:
        steps = ['generate']
    elif args.of_convergence:
        steps = ['of_convergence']
    elif args.pm_convergence:
        steps = ['pm_convergence']
    elif args.compare:
        steps = ['compare']
    else:
        # Default: show status and suggest next step
        print("\nCurrent state:")
        of_case = output_dir / "openfoam"
        final_case = output_dir / "final_case"
        conv_dir = output_dir / "openfoam_convergence"
        
        print(f"  OpenFOAM base case: {'✓' if of_case.exists() else '✗'}")
        print(f"  OF convergence study: {'✓' if (conv_dir / 'convergence_data.csv').exists() else '✗'}")
        print(f"  Final converged case: {'✓' if final_case.exists() else '✗'}")
        print(f"  Comparison: {'✓' if (output_dir / 'comparison' / 'comparison.png').exists() else '✗'}")
        
        print("\nRun with --full for complete workflow, or specific step flags.")
        return
    
    # Execute steps
    for step in steps:
        if step == 'generate':
            step_generate(case_path, output_dir)
            
        elif step == 'run_of':
            of_case = output_dir / "openfoam"
            run_openfoam_case(of_case, use_snappy=True)
            
        elif step == 'of_convergence':
            step_of_convergence(output_dir, show=args.show)
            
        elif step == 'pm_convergence':
            # Use final converged case if available, otherwise base case
            final_case = output_dir / "final_case"
            of_case = final_case if final_case.exists() else output_dir / "openfoam"
            step_pm_convergence(case_path, output_dir, of_case, args.cores, args.show)
            
        elif step == 'compare':
            # Use final converged case if available
            final_case = output_dir / "final_case"
            of_case = final_case if final_case.exists() else output_dir / "openfoam"
            step_compare(case_path, output_dir, of_case, args.cores, args.show)
    
    print("\n" + "="*60)
    print("Validation Complete")
    print("="*60)


if __name__ == "__main__":
    main()
