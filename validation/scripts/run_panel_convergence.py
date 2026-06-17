#!/usr/bin/env python3
"""
Run panel method convergence study against OpenFOAM reference.

Runs panel solver at multiple mesh refinement levels, extracts values at
monitoring points, and compares against OpenFOAM reference solution.

Usage:
    python validation/scripts/run_panel_convergence.py cases/case_name \\
        --reference-of-case of_case/cases/level_2
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from validation.scripts.utils import (
    load_of_config,
    load_viz_config,
    extract_monitoring_point_data,
    compute_convergence_metrics,
    save_panel_convergence_data,
    save_convergence_metrics,
    plot_convergence_curves,
    plot_value_vs_level,
)


def run_panel_at_level(
    case_dir: Path,
    mesh_level: int
) -> Tuple[object, object, int]:
    """
    Run panel solver at specific mesh refinement level.
    
    Args:
        case_dir: Path to case directory
        mesh_level: Number of refinement iterations (0 = original)
    
    Returns:
        (case, solver, panel_count) tuple
    """
    from core.io import CaseLoader
    from solvers.panel2d import SourcePanelSolver
    
    # Load case at the specified mesh level (for parametric cases)
    case = CaseLoader.load_case(case_dir, mesh_level_index=mesh_level)
    
    # Solve
    solver = SourcePanelSolver(case.mesh, v_inf=case.v_inf, aoa=case.aoa)
    solver.solve()
    
    # Count total panels
    total_panels = case.num_panels
    
    return case, solver, total_panels


def extract_panel_values_at_points(
    case: object,
    solver: object,
    monitoring_points: List[Dict],
    v_inf: float
) -> Dict[str, Dict[str, float]]:
    """
    Extract velocity and pressure at monitoring points from panel solution.
    
    Args:
        case: Case object with mesh
        solver: Solved SourcePanelSolver
        monitoring_points: List of points with 'name' and 'coordinates'
        v_inf: Freestream velocity
    
    Returns:
        Dict[point_name] = {velocity, pressure}
    """
    from visualization.field2d import VelocityField2D
    
    # Create velocity field evaluator using solver
    field = VelocityField2D(solver)
    
    results = {}
    
    for point in monitoring_points:
        name = point['name']
        coords = np.array(point['coordinates'])
        
        # Create small grid around point for evaluation
        delta = 0.01
        x_range = (coords[0] - delta, coords[0] + delta)
        y_range = (coords[1] - delta, coords[1] + delta)
        
        # Compute field on small grid
        field.compute(x_range=x_range, y_range=y_range, resolution=(3, 3))
        
        # Get cached velocity components
        XX, YY, Vx, Vy = field.get_cached()
        
        # Get velocity at center point (index 1,1 in 3x3 grid)
        velocity = float(np.sqrt(Vx[1, 1]**2 + Vy[1, 1]**2))
        
        # Compute pressure coefficient from Bernoulli (inviscid)
        # Cp = 1 - (v/v_inf)^2
        cp = 1.0 - (velocity / v_inf) ** 2
        
        # Store Cp as pressure value for comparison
        pressure = float(cp)
        
        results[name] = {
            'velocity': velocity,
            'pressure': pressure,
        }
    
    return results


def compute_panel_errors(
    panel_values: Dict[str, Dict[str, float]],
    reference_values: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """
    Compute errors between panel and reference values.
    
    Returns:
        Dict[point_name][quantity] = {absolute_error, relative_error}
    """
    errors = {}
    
    for point_name, panel_vals in panel_values.items():
        if point_name not in reference_values:
            continue
        
        ref_vals = reference_values[point_name]
        errors[point_name] = {}
        
        for quantity in ['velocity', 'pressure']:
            panel_val = panel_vals[quantity]
            ref_val = ref_vals[quantity]
            
            abs_err = abs(panel_val - ref_val)
            rel_err = abs_err / abs(ref_val) if abs(ref_val) > 1e-10 else 0.0
            
            errors[point_name][quantity] = {
                'absolute': float(abs_err),
                'relative': float(rel_err),
            }
    
    return errors


def main():
    parser = argparse.ArgumentParser(description="Run panel method convergence study")
    parser.add_argument("case_dir", type=Path, help="Path to case directory")
    parser.add_argument("--reference-of-case", type=Path, required=True,
                       help="Path to reference OpenFOAM case (finest level)")
    parser.add_argument("--max-levels", type=int, default=4,
                       help="Maximum panel refinement levels to test")
    parser.add_argument("--start-level", type=int, default=0,
                       help="Starting refinement level")
    
    args = parser.parse_args()
    
    case_dir = args.case_dir.resolve()
    of_case_dir = (case_dir / args.reference_of_case).resolve() if not args.reference_of_case.is_absolute() else args.reference_of_case
    
    if not case_dir.exists():
        print(f"Error: Case directory not found: {case_dir}")
        return 1
    
    if not of_case_dir.exists():
        print(f"Error: Reference OpenFOAM case not found: {of_case_dir}")
        return 1
    
    # Load configurations
    of_config = load_of_config(case_dir / "of_case" / "config.yaml")
    viz_config = load_viz_config(case_dir / "out" / "viz_config.yaml")
    
    # Load case
    from core.io import CaseLoader
    panel_case = CaseLoader.load_case(case_dir)
    
    monitoring_points = of_config['monitoring_points']
    v_inf = panel_case.v_inf
    
    print(f"Running panel convergence study...")
    print(f"  Reference: {of_case_dir}")
    print(f"  Refinement levels: {args.start_level} to {args.max_levels}")
    
    # Extract reference values from OpenFOAM
    print(f"\nExtracting reference values from OpenFOAM...")
    reference_values = extract_monitoring_point_data(of_case_dir, monitoring_points)
    
    for point_name, values in reference_values.items():
        print(f"  {point_name}: U={values['velocity']:.4f}, p={values['pressure']:.4f}")
    
    # Run panel solver at multiple levels
    panel_data = {}  # Dict[panel_count][point_name][quantity] = value
    level_info = []  # List of (level, panel_count) tuples
    
    print(f"\n{'='*60}")
    print("Running panel method at multiple refinement levels...")
    print(f"{'='*60}")
    
    for level in range(args.start_level, args.max_levels + 1):
        print(f"\nLevel {level}:")
        
        # Run panel solver
        case, solver, panel_count = run_panel_at_level(case_dir, level)
        level_info.append((level, panel_count))
        
        print(f"  Total panels: {panel_count}")
        
        # Extract values at monitoring points
        point_values = extract_panel_values_at_points(
            case, solver, monitoring_points, v_inf
        )
        
        panel_data[panel_count] = point_values
        
        # Compute errors vs reference
        errors = compute_panel_errors(point_values, reference_values)
        
        # Print results
        for point_name, values in point_values.items():
            err = errors.get(point_name, {})
            print(f"  {point_name}:")
            print(f"    U = {values['velocity']:.4f} (ref: {reference_values[point_name]['velocity']:.4f}, " +
                  f"error: {err.get('velocity', {}).get('relative', 0.0):.2%})")
            print(f"    p = {values['pressure']:.4f} (ref: {reference_values[point_name]['pressure']:.4f}, " +
                  f"error: {err.get('pressure', {}).get('relative', 0.0):.2%})")
    
    # Save raw data
    output_dir = case_dir / "out"
    print(f"\n{'='*60}")
    print("Saving raw data...")
    save_panel_convergence_data(output_dir, panel_data, reference_values)
    
    # Compute convergence metrics
    print(f"Computing convergence metrics...")
    
    metrics = {}
    quantities = ['velocity', 'pressure']
    
    for point in monitoring_points:
        pname = point['name']
        metrics[pname] = {}
        
        for quantity in quantities:
            # Extract values for this point and quantity
            panel_counts = sorted(panel_data.keys())
            values = [panel_data[pc][pname][quantity] for pc in panel_counts]
            
            if len(values) >= 2:
                try:
                    point_metrics = compute_convergence_metrics(values)
                    
                    # Add reference comparison
                    ref_val = reference_values[pname][quantity]
                    final_val = values[-1]
                    
                    point_metrics['reference_value'] = float(ref_val)
                    point_metrics['final_value'] = float(final_val)
                    point_metrics['final_error_absolute'] = abs(final_val - ref_val)
                    point_metrics['final_error_relative'] = abs(final_val - ref_val) / abs(ref_val) if abs(ref_val) > 1e-10 else 0.0
                    
                    metrics[pname][quantity] = point_metrics
                    
                    # Print summary
                    if 'gci' in point_metrics:
                        print(f"  {pname} - {quantity}:")
                        print(f"    GCI: {point_metrics['gci']:.6f}")
                        print(f"    Order: {point_metrics['order_of_convergence']:.3f}")
                        print(f"    Final error: {point_metrics['final_error_relative']:.3%}")
                except Exception as e:
                    print(f"Warning: Could not compute metrics for {pname}/{quantity}: {e}")
    
    save_convergence_metrics(output_dir, metrics, study_name="panel_convergence")
    
    # Generate visualizations
    print(f"\n{'='*60}")
    print("Generating visualizations...")
    
    plots_dir = output_dir / "panel_convergence" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Convergence curves with reference lines
    conv_config = viz_config.get('panel_convergence', {}).get('convergence_curves', {})
    if conv_config.get('enabled', True):
        try:
            # Prepare data in format expected by plot_convergence_curves
            # Convert panel_count keys to string level names
            level_names = [f"{pc}_panels" for pc in sorted(panel_data.keys())]
            plot_data = {name: panel_data[int(name.split('_')[0])] 
                        for name in level_names}
            
            figs = plot_convergence_curves(
                plot_data,
                level_names,
                quantities=['velocity', 'pressure'],
                config=viz_config,
                output_dir=plots_dir,
                reference_values=reference_values,
                plot_type='panel_convergence'
            )
            print(f"  ✓ Convergence curves saved")
        except Exception as e:
            print(f"  Warning: Could not generate convergence curves: {e}")
    
    # 2. Error vs panel count
    error_config = viz_config.get('panel_convergence', {}).get('error_vs_panels', {})
    if error_config.get('enabled', True):
        import matplotlib.pyplot as plt
        
        for point in monitoring_points:
            pname = point['name']
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            for i, quantity in enumerate(['velocity', 'pressure']):
                ax = axes[i]
                
                panel_counts = sorted(panel_data.keys())
                values = [panel_data[pc][pname][quantity] for pc in panel_counts]
                ref_val = reference_values[pname][quantity]
                
                errors = [abs(v - ref_val) / abs(ref_val) for v in values]
                
                ax.loglog(panel_counts, errors, 'o-', linewidth=2, markersize=8)
                ax.set_xlabel('Panel Count')
                ax.set_ylabel(f'Relative Error ({quantity})')
                ax.set_title(f'{quantity.capitalize()} Error - {pname}')
                ax.grid(True, which='both', alpha=0.3)
                
                # Add convergence rate line
                if len(panel_counts) >= 2:
                    # Fit power law: error = C * N^(-p)
                    log_n = np.log(panel_counts)
                    log_err = np.log(errors)
                    p = np.polyfit(log_n, log_err, 1)[0]
                    
                    ax.text(0.05, 0.95, f'Slope: {p:.2f}',
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            
            # Save
            fig_config = viz_config.get('figure', {})
            for fmt in fig_config.get('format', ['png']):
                save_path = plots_dir / f"error_vs_panels_{pname}.{fmt}"
                fig.savefig(save_path, dpi=fig_config.get('dpi', 300), bbox_inches='tight')
            
            plt.close(fig)
        
        print(f"  ✓ Error vs panel count plots saved")
    
    # 3. Per-point value vs level
    for point in monitoring_points:
        pname = point['name']
        
        for quantity in quantities:
            try:
                panel_counts = sorted(panel_data.keys())
                values = [panel_data[pc][pname][quantity] for pc in panel_counts]
                level_names = [str(pc) for pc in panel_counts]
                
                plot_value_vs_level(
                    values,
                    level_names,
                    f"{pname} - {quantity}",
                    plots_dir / f"{pname}_{quantity}_vs_panels",
                    reference_value=reference_values[pname][quantity],
                    config=viz_config
                )
            except Exception as e:
                print(f"  Warning: Could not plot {pname}/{quantity}: {e}")
    
    print(f"  ✓ Per-point plots saved")
    
    print(f"\n{'='*60}")
    print("Panel convergence study complete!")
    print(f"Results saved to: {output_dir / 'panel_convergence'}")
    print(f"  - Raw data: {output_dir / 'panel_convergence' / 'raw'}")
    print(f"  - Plots: {plots_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
