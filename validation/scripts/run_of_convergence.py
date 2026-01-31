#!/usr/bin/env python3
"""
Run OpenFOAM mesh convergence study.

Steps:
1. Load configuration from of_case/config.yaml
2. For each refinement level:
   - Copy base_case to of_case/cases/level_X
   - Modify blockMeshDict and snappyHexMeshDict using foamlib
   - Optionally run OpenFOAM workflow
3. Extract monitoring point data from each case
4. Compute convergence metrics (GCI, order)
5. Generate visualizations
6. Save raw data and results

Usage:
    python validation/scripts/run_of_convergence.py cases/case_name [--run-openfoam]
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from validation.scripts.utils import (
    load_of_config,
    load_viz_config,
    extract_monitoring_point_data,
    compute_convergence_metrics,
    save_monitoring_point_data,
    save_convergence_metrics,
    save_field_data,
    plot_field_overview_with_components,
    plot_convergence_curves,
    plot_value_vs_level,
    plot_change_between_levels,
    # foamlib helpers
    set_blockmesh_cells,
    set_snappy_levels_per_component,
    run_openfoam_workflow,
)


def extract_field_data_from_openfoam(
    case_dir: Path,
    domain: Dict[str, List[float]],
    resolution: int,
    time_idx: int = -1
) -> Dict[str, np.ndarray]:
    """
    Extract velocity and pressure field on a regular grid from OpenFOAM case.
    
    Args:
        case_dir: OpenFOAM case directory
        domain: Dict with x_range and y_range
        resolution: Grid resolution (number of points per side)
        time_idx: Time index to extract
    
    Returns:
        Dict with XX, YY, Vx, Vy, velocity_magnitude, pressure
    """
    from validation.adapters.openfoam import OpenFOAMRunner
    from scipy.interpolate import griddata
    
    runner = OpenFOAMRunner(case_dir, verbose=False)
    
    # Get cell centers
    cell_centers = runner.get_cell_centres(time_idx=time_idx)
    
    # Get fields
    velocity = runner.get_velocity_field(time_idx=time_idx)
    pressure = runner.get_pressure_field(time_idx=time_idx)
    
    # Create regular grid
    x_range = domain['x_range']
    y_range = domain['y_range']
    
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    XX, YY = np.meshgrid(x, y)
    
    # Extract 2D coordinates
    points_2d = cell_centers[:, :2]
    
    # Interpolate velocity components
    Vx = griddata(points_2d, velocity[:, 0], (XX, YY), method='linear')
    Vy = griddata(points_2d, velocity[:, 1], (XX, YY), method='linear')
    
    # Velocity magnitude
    velocity_mag = np.sqrt(Vx**2 + Vy**2)
    
    # Interpolate pressure
    P = griddata(points_2d, pressure, (XX, YY), method='linear')
    
    return {
        'XX': XX,
        'YY': YY,
        'Vx': Vx,
        'Vy': Vy,
        'velocity_magnitude': velocity_mag,
        'pressure': P,
    }


def main():
    parser = argparse.ArgumentParser(description="Run OpenFOAM mesh convergence study")
    parser.add_argument("case_dir", type=Path, help="Path to case directory")
    parser.add_argument("--run-openfoam", action="store_true", 
                       help="Run OpenFOAM workflow (otherwise just analyze existing results)")
    parser.add_argument("--parallel-snappy", action="store_true",
                       help="Run snappyHexMesh in parallel (faster for fine meshes)")
    parser.add_argument("--parallel-from-level", type=int, default=None,
                       help="Use parallel snappy starting from this level index (0-based)")
    parser.add_argument("--n-procs", type=int, default=4,
                       help="Number of MPI processes for parallel snappy (default: 4)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    case_dir = args.case_dir.resolve()
    if not case_dir.exists():
        print(f"Error: Case directory not found: {case_dir}")
        return 1
    
    # Load configurations
    of_config = load_of_config(case_dir / "of_case" / "config.yaml")
    viz_config = load_viz_config(case_dir / "out" / "viz_config.yaml")
    
    # Load panel case for domain info
    from core.io import CaseLoader
    panel_case = CaseLoader.load_case(case_dir)
    
    base_case_dir = case_dir / "of_case" / "base_case"
    cases_dir = case_dir / "of_case" / "cases"
    output_dir = case_dir / "out"
    
    if not base_case_dir.exists():
        print(f"Error: Base case not found. Run generate_base_case.py first.")
        return 1
    
    cases_dir.mkdir(exist_ok=True)
    
    refinement_levels = of_config['refinement_levels']
    monitoring_points = of_config['monitoring_points']
    
    print(f"Running OpenFOAM convergence study with {len(refinement_levels)} levels...")
    
    # Store results
    point_data = {}  # Dict[level_name][point_name][quantity] = value
    level_names = []
    
    # Process each refinement level
    for i, level in enumerate(refinement_levels):
        level_name = level['name']
        level_names.append(level_name)
        
        level_dir = cases_dir / f"level_{i}"
        
        print(f"\n{'='*60}")
        print(f"Processing Level {i}: {level_name}")
        print(f"{'='*60}")
        
        # Copy base case if not exists
        if not level_dir.exists():
            print(f"Copying base case to {level_dir}...")
            shutil.copytree(base_case_dir, level_dir)
            
            # Modify mesh parameters using foamlib helpers
            cells = tuple(level['blockMesh_cells'])
            print(f"Setting blockMesh cells: {cells}")
            set_blockmesh_cells(level_dir, cells)
            
            print(f"Setting snappy refinement levels...")
            set_snappy_levels_per_component(level_dir, level['components'])
        
        # Run OpenFOAM if requested
        if args.run_openfoam:
            # Determine if parallel snappy should be used for this level
            use_parallel = args.parallel_snappy
            if args.parallel_from_level is not None:
                use_parallel = i >= args.parallel_from_level
            
            if use_parallel:
                print(f"Running OpenFOAM workflow (parallel snappyHexMesh)...")
            else:
                print(f"Running OpenFOAM workflow...")
            
            success = run_openfoam_workflow(
                level_dir,
                verbose=args.verbose,
                parallel_snappy=use_parallel,
                n_procs=args.n_procs
            )
            if not success:
                print(f"Warning: OpenFOAM workflow failed for level {level_name}")
                continue
        
        # Extract monitoring point data
        print(f"Extracting monitoring point data...")
        try:
            point_values = extract_monitoring_point_data(level_dir, monitoring_points)
            point_data[level_name] = point_values
            
            # Print values
            for point_name, values in point_values.items():
                print(f"  {point_name}: U={values['velocity']:.4f}, p={values['pressure']:.4f}")
        except Exception as e:
            print(f"Warning: Could not extract data from level {level_name}: {e}")
            continue
        
        # Extract field data (only for finest level to save storage)
        if i == len(refinement_levels) - 1:
            print(f"Extracting field data for visualization...")
            try:
                viz_config = panel_case.config.visualization
                
                # Get domain - use from config or create default based on case
                if viz_config.domain:
                    domain = viz_config.domain
                else:
                    # Create default domain based on case bounding box
                    scene = panel_case.scene
                    bounds = scene.get_bounds()
                    margin = 0.5
                    domain = {
                        'x_range': [bounds[0] - margin, bounds[1] + margin],
                        'y_range': [bounds[2] - margin, bounds[3] + margin]
                    }
                
                # Use first element of resolution tuple as grid resolution
                resolution = viz_config.get_resolution()[0]  # Use nx for square grid
                
                field_data = extract_field_data_from_openfoam(
                    level_dir,
                    domain=domain,
                    resolution=resolution
                )
                save_field_data(output_dir, field_data, level_name, study_name="of_convergence")
                print(f"  Field data saved.")
            except Exception as e:
                print(f"Warning: Could not extract field data: {e}")
    
    # Save raw monitoring point data
    print(f"\n{'='*60}")
    print("Saving raw data...")
    save_monitoring_point_data(output_dir, point_data, level_names, study_name="of_convergence")
    
    # Compute convergence metrics
    print(f"Computing convergence metrics...")
    
    metrics = {}
    quantities = ['velocity', 'pressure']
    
    for point_name in monitoring_points:
        pname = point_name['name']
        metrics[pname] = {}
        
        for quantity in quantities:
            # Extract values for this point and quantity
            values = [point_data[level][pname][quantity] for level in level_names if level in point_data]
            
            if len(values) >= 2:
                try:
                    point_metrics = compute_convergence_metrics(values)
                    metrics[pname][quantity] = point_metrics
                    
                    # Print summary
                    if 'gci' in point_metrics:
                        print(f"  {pname} - {quantity}:")
                        print(f"    GCI: {point_metrics['gci']:.6f}")
                        print(f"    Order: {point_metrics['order_of_convergence']:.3f}")
                        print(f"    Converged: {point_metrics['converged']}")
                except Exception as e:
                    print(f"Warning: Could not compute metrics for {pname}/{quantity}: {e}")
    
    save_convergence_metrics(output_dir, metrics, study_name="of_convergence")
    
    # Generate visualizations
    print(f"\n{'='*60}")
    print("Generating visualizations...")
    
    plots_dir = output_dir / "of_convergence" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Load default viz config and get dict representation for utility functions
    from validation.scripts.utils import load_viz_config as load_viz_config_util
    if (output_dir / "viz_config.yaml").exists():
        viz_config_dict = load_viz_config_util(output_dir / "viz_config.yaml")
    else:
        # Use default viz config
        default_config_path = Path(__file__).parent.parent / "config" / "default_viz_config.yaml"
        viz_config_dict = load_viz_config_util(default_config_path)
    
    # 1. Field overview with monitoring points using proper visualization utilities
    field_config = viz_config_dict.get('of_convergence', {}).get('field_overview', {})
    if field_config.get('enabled', True):
        try:
            finest_level = level_names[-1]
            field_data = load_field_data(output_dir, finest_level, study_name="of_convergence")
            
            print(f"  Generating field overview with components and monitoring points...")
            
            # Get assembled mesh for proper component rendering
            mesh = panel_case.scene.assemble()
            
            fig = plot_field_overview_with_components(
                field_data,
                mesh,
                monitoring_points,
                viz_config_dict,
                plots_dir / "field_overview",
                title=f"Flow Field Overview - {finest_level}"
            )
            
            import matplotlib.pyplot as plt
            plt.close(fig)
            print(f"  ✓ Field overview plot saved")
        except Exception as e:
            print(f"  Warning: Could not generate field overview: {e}")
            import traceback
            traceback.print_exc()
    
    # 2. Convergence curves
    conv_config = viz_config_dict.get('of_convergence', {}).get('convergence_curves', {})
    if conv_config.get('enabled', True):
        try:
            figs = plot_convergence_curves(
                point_data,
                level_names,
                quantities=['velocity', 'pressure'],
                config=viz_config_dict,
                output_dir=plots_dir,
                plot_type='of_convergence'
            )
            print(f"  ✓ Convergence curves saved")
        except Exception as e:
            print(f"  Warning: Could not generate convergence curves: {e}")
    
    # 3. Per-point plots
    per_point_config = viz_config_dict.get('of_convergence', {}).get('per_point_plots', {})
    if per_point_config.get('enabled', True):
        for point in monitoring_points:
            pname = point['name']
            for quantity in quantities:
                try:
                    values = [point_data[level][pname][quantity] for level in level_names if level in point_data]
                    plot_value_vs_level(
                        values,
                        level_names[:len(values)],
                        f"{pname} - {quantity}",
                        plots_dir / f"{pname}_{quantity}",
                        config=viz_config_dict
                    )
                except Exception as e:
                    print(f"  Warning: Could not plot {pname}/{quantity}: {e}")
        print(f"  ✓ Per-point plots saved")
    
    print(f"\n{'='*60}")
    print("OpenFOAM convergence study complete!")
    print(f"Results saved to: {output_dir / 'of_convergence'}")
    print(f"  - Raw data: {output_dir / 'of_convergence' / 'raw'}")
    print(f"  - Plots: {plots_dir}")
    
    return 0


# Import load_field_data for visualization
from validation.scripts.utils import load_field_data


if __name__ == "__main__":
    sys.exit(main())
