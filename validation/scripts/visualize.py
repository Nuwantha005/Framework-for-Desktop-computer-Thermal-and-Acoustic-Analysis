#!/usr/bin/env python3
"""
Regenerate visualization plots from saved raw data.

Allows tweaking visualization settings in viz_config.yaml without
rerunning expensive simulations.

Usage:
    python validation/scripts/visualize.py cases/case_name \\
        --study {of_convergence|panel_convergence|surface_comparison|all}
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from validation.scripts.utils import (
    load_viz_config,
    load_of_config,
    load_monitoring_point_data,
    load_metadata,
    load_field_data,
    load_convergence_metrics,
    load_surface_data,
    load_error_metrics,
    plot_field_overview_with_components,
    plot_convergence_curves,
    plot_value_vs_level,
    plot_change_between_levels,
)

from visualization.surface_envelope import (
    plot_surface_envelope,
    plot_dual_surface_envelope,
    compute_outward_normals,
)


def visualize_of_convergence(case_dir: Path, output_dir: Path, viz_config: Dict):
    """Regenerate OpenFOAM convergence study plots."""
    
    print(f"Regenerating OpenFOAM convergence plots...")
    
    # Load raw data
    point_data_dict = load_monitoring_point_data(output_dir, "of_convergence")
    metadata = load_metadata(output_dir, "of_convergence")
    metrics = load_convergence_metrics(output_dir, "of_convergence")
    
    level_names = metadata.get('level_names', [])
    quantities = metadata.get('quantities', ['velocity', 'pressure'])
    point_names = metadata.get('point_names', [])
    
    if not level_names:
        print("  Error: No level names found in metadata")
        return
    
    # Convert DataFrame format to dict format for plotting
    point_data = {}
    for level in level_names:
        point_data[level] = {}
        for quantity, df in point_data_dict.items():
            level_row = df[df['level'] == level]
            if not level_row.empty:
                for pname in point_names:
                    if pname not in point_data[level]:
                        point_data[level][pname] = {}
                    if pname in level_row.columns:
                        point_data[level][pname][quantity] = float(level_row[pname].values[0])
    
    # Create monitoring points list - load actual coordinates from of_config
    case_dir = output_dir.parent
    of_config_path = case_dir / "of_case" / "config.yaml"
    
    monitoring_points = []
    if of_config_path.exists():
        # Load actual monitoring points from OF config
        of_config = load_of_config(of_config_path)
        monitoring_points = of_config.get('monitoring_points', [])
    
    if not monitoring_points:
        # Fallback to dummy points at origin
        monitoring_points = [{'name': name, 'coordinates': [0, 0]} for name in point_names]
    
    plots_dir = output_dir / "of_convergence" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Field overview with monitoring points
    field_config = viz_config.get('of_convergence', {}).get('field_overview', {})
    if field_config.get('enabled', True):
        try:
            finest_level = level_names[-1]
            field_data = load_field_data(output_dir, finest_level, "of_convergence")
            
            # Load panel case to get mesh for proper component rendering
            from core.io import CaseLoader
            panel_case = CaseLoader.load_case(case_dir)
            mesh = panel_case.scene.assemble()
            
            fig = plot_field_overview_with_components(
                field_data,
                mesh,
                monitoring_points,
                viz_config,
                plots_dir / "field_overview",
                title=f"Flow Field - {finest_level}"
            )
            import matplotlib.pyplot as plt
            plt.close(fig)
            print(f"  ✓ Field overview")
        except Exception as e:
            print(f"  Warning: Could not generate field overview: {e}")
    
    # 2. Convergence curves
    conv_config = viz_config.get('of_convergence', {}).get('convergence_curves', {})
    if conv_config.get('enabled', True):
        try:
            figs = plot_convergence_curves(
                point_data,
                level_names,
                quantities=['velocity', 'pressure'],
                config=viz_config,
                output_dir=plots_dir,
                plot_type='of_convergence'
            )
            print(f"  ✓ Convergence curves")
        except Exception as e:
            print(f"  Warning: Could not generate convergence curves: {e}")
    
    # 3. Per-point plots
    per_point_config = viz_config.get('of_convergence', {}).get('per_point_plots', {})
    if per_point_config.get('enabled', True):
        for pname in point_names:
            for quantity in quantities:
                try:
                    values = [point_data[level][pname][quantity] 
                             for level in level_names if level in point_data and pname in point_data[level]]
                    
                    if values:
                        plot_value_vs_level(
                            values,
                            level_names[:len(values)],
                            f"{pname} - {quantity}",
                            plots_dir / f"{pname}_{quantity}",
                            config=viz_config
                        )
                except Exception as e:
                    print(f"  Warning: Could not plot {pname}/{quantity}: {e}")
        print(f"  ✓ Per-point plots")
    
    print(f"  Plots saved to: {plots_dir}")


def visualize_panel_convergence(case_dir: Path, output_dir: Path, viz_config: Dict):
    """Regenerate panel convergence study plots."""
    
    print(f"Regenerating panel convergence plots...")
    
    # Load raw data
    import pandas as pd
    
    raw_dir = output_dir / "panel_convergence" / "raw"
    
    # Load panel values
    velocity_df = pd.read_csv(raw_dir / "panel_values_velocity.csv")
    pressure_df = pd.read_csv(raw_dir / "panel_values_pressure.csv")
    
    # Load reference values
    reference_df = pd.read_csv(raw_dir / "reference_values.csv")
    
    # Load metrics
    metrics = load_convergence_metrics(output_dir, "panel_convergence")
    
    plots_dir = output_dir / "panel_convergence" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to format for plotting
    panel_counts = velocity_df['panel_count'].tolist()
    level_names = [f"{pc}_panels" for pc in panel_counts]
    
    point_names = [col for col in velocity_df.columns if col != 'panel_count']
    
    panel_data = {}
    for i, pc in enumerate(panel_counts):
        panel_data[level_names[i]] = {}
        for pname in point_names:
            panel_data[level_names[i]][pname] = {
                'velocity': float(velocity_df.loc[i, pname]),
                'pressure': float(pressure_df.loc[i, pname]),
            }
    
    reference_values = {}
    for _, row in reference_df.iterrows():
        pname = row['point']
        reference_values[pname] = {
            'velocity': float(row['velocity']),
            'pressure': float(row['pressure']),
        }
    
    # 1. Convergence curves with reference
    conv_config = viz_config.get('panel_convergence', {}).get('convergence_curves', {})
    if conv_config.get('enabled', True):
        try:
            figs = plot_convergence_curves(
                panel_data,
                level_names,
                quantities=['velocity', 'pressure'],
                config=viz_config,
                output_dir=plots_dir,
                reference_values=reference_values,
                plot_type='panel_convergence'
            )
            print(f"  ✓ Convergence curves")
        except Exception as e:
            print(f"  Warning: Could not generate convergence curves: {e}")
    
    # 2. Error vs panel count
    error_config = viz_config.get('panel_convergence', {}).get('error_vs_panels', {})
    if error_config.get('enabled', True):
        import matplotlib.pyplot as plt
        
        for pname in point_names:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            for i, quantity in enumerate(['velocity', 'pressure']):
                ax = axes[i]
                
                values = [panel_data[ln][pname][quantity] for ln in level_names]
                ref_val = reference_values[pname][quantity]
                
                errors = [abs(v - ref_val) / abs(ref_val) if abs(ref_val) > 1e-10 else 0 
                         for v in values]
                
                ax.loglog(panel_counts, errors, 'o-', linewidth=2, markersize=8)
                ax.set_xlabel('Panel Count')
                ax.set_ylabel(f'Relative Error ({quantity})')
                ax.set_title(f'{quantity.capitalize()} Error - {pname}')
                ax.grid(True, which='both', alpha=0.3)
                
                # Add convergence rate
                if len(panel_counts) >= 2:
                    log_n = np.log(panel_counts)
                    log_err = np.log([e if e > 0 else 1e-10 for e in errors])
                    p = np.polyfit(log_n, log_err, 1)[0]
                    
                    ax.text(0.05, 0.95, f'Slope: {p:.2f}',
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            
            fig_config = viz_config.get('figure', {})
            for fmt in fig_config.get('format', ['png']):
                save_path = plots_dir / f"error_vs_panels_{pname}.{fmt}"
                fig.savefig(save_path, dpi=fig_config.get('dpi', 300), bbox_inches='tight')
            
            plt.close(fig)
        
        print(f"  ✓ Error vs panel count")
    
    # 3. Per-point value vs level
    for pname in point_names:
        for quantity in ['velocity', 'pressure']:
            try:
                values = [panel_data[ln][pname][quantity] for ln in level_names]
                
                plot_value_vs_level(
                    values,
                    [str(pc) for pc in panel_counts],
                    f"{pname} - {quantity}",
                    plots_dir / f"{pname}_{quantity}_vs_panels",
                    reference_value=reference_values[pname][quantity],
                    config=viz_config
                )
            except Exception as e:
                print(f"  Warning: Could not plot {pname}/{quantity}: {e}")
    
    print(f"  ✓ Per-point plots")
    print(f"  Plots saved to: {plots_dir}")


def visualize_surface_comparison(case_dir: Path, output_dir: Path, viz_config: Dict, of_level: str = None):
    """Regenerate surface comparison plots."""
    
    print(f"Regenerating surface comparison plots...")
    
    # Find all OF levels if not specified
    surf_dir = output_dir / "surface_comparison"
    
    if of_level:
        of_levels = [of_level]
    else:
        of_levels = [d.name.replace("of_", "") for d in surf_dir.iterdir() 
                    if d.is_dir() and d.name.startswith("of_")]
    
    for of_lvl in of_levels:
        print(f"\n  Processing OF level: {of_lvl}")
        
        # Load surface data
        try:
            panel_data = load_surface_data(output_dir, "panel", of_level=of_lvl)
            of_data = load_surface_data(output_dir, "openfoam", of_level=of_lvl)
            error_metrics = load_error_metrics(output_dir, "surface_comparison", of_level=of_lvl)
        except Exception as e:
            print(f"    Warning: Could not load data for level {of_lvl}: {e}")
            continue
        
        plots_dir = surf_dir / f"of_{of_lvl}" / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate comparison plots
        surf_config = viz_config.get('surface_comparison', {})
        quantities = surf_config.get('quantities', ['Vt', 'Cp'])
        
        for comp_name in panel_data.keys():
            if comp_name not in of_data:
                continue
            
            panel_df = panel_data[comp_name]
            of_df = of_data[comp_name]
            
            # Ensure strictly increasing 's' by dropping duplicates and sorting
            panel_df = panel_df.drop_duplicates(subset=['s'], keep='first').sort_values('s')
            of_df = of_df.drop_duplicates(subset=['s'], keep='first').sort_values('s')
            
            # Update the dicts so envelope plots also get the cleaned data
            panel_data[comp_name] = panel_df
            of_data[comp_name] = of_df
            
            for quantity in quantities:
                if quantity not in panel_df.columns or quantity not in of_df.columns:
                    continue
                
                try:
                    import matplotlib.pyplot as plt
                    from scipy.interpolate import interp1d
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                    
                    # Plot distributions
                    ax1.plot(panel_df['s'], panel_df[quantity], 
                            'b-', label='Panel Method', linewidth=2)
                    ax1.plot(of_df['s'], of_df[quantity],
                            'r--', label='OpenFOAM', linewidth=2)
                    ax1.set_xlabel('Surface Position s')
                    ax1.set_ylabel(quantity)
                    ax1.set_title(f'{quantity} Distribution - {comp_name}')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # Plot difference
                    s_min = max(panel_df['s'].min(), of_df['s'].min())
                    s_max = min(panel_df['s'].max(), of_df['s'].max())
                    s_common = np.linspace(s_min, s_max, 200)
                    
                    # Use unique 's' values to prevent divide-by-zero warnings in interp1d
                    p_s, p_idx = np.unique(panel_df['s'], return_index=True)
                    o_s, o_idx = np.unique(of_df['s'], return_index=True)
                    
                    panel_interp = interp1d(p_s, panel_df[quantity].values[p_idx], 
                                           kind='linear', fill_value='extrapolate')
                    of_interp = interp1d(o_s, of_df[quantity].values[o_idx],
                                        kind='linear', fill_value='extrapolate')
                    
                    diff = panel_interp(s_common) - of_interp(s_common)
                    
                    ax2.plot(s_common, diff, 'g-', linewidth=2)
                    ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
                    ax2.set_xlabel('Surface Position s')
                    ax2.set_ylabel(f'{quantity} Difference (Panel - OF)')
                    ax2.set_title(f'{quantity} Error - {comp_name}')
                    ax2.grid(True, alpha=0.3)
                    
                    # Add metrics
                    if comp_name in error_metrics and quantity in error_metrics[comp_name]:
                        metrics = error_metrics[comp_name][quantity]
                        metrics_text = f"L2: {metrics['L2']:.4f}\nL∞: {metrics['Linf']:.4f}\nRMS: {metrics['RMS']:.4f}"
                        ax2.text(0.02, 0.98, metrics_text, transform=ax2.transAxes,
                                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    
                    plt.tight_layout()
                    
                    # Save
                    fig_config = viz_config.get('figure', {})
                    for fmt in fig_config.get('format', ['png']):
                        save_path = plots_dir / f"{comp_name}_{quantity}.{fmt}"
                        fig.savefig(save_path, dpi=fig_config.get('dpi', 300), bbox_inches='tight')
                    
                    plt.close(fig)
                    
                    print(f"    ✓ {comp_name} - {quantity}")
                except Exception as e:
                    print(f"    Warning: Could not plot {comp_name}/{quantity}: {e}")
        
        # Generate surface envelope plots (wrapped distribution around body)
        envelope_config = surf_config.get('envelope_plots', {})
        if envelope_config.get('enabled', True):
            try:
                _generate_surface_envelope_plots(
                    panel_data, of_data, plots_dir, viz_config, quantities
                )
            except Exception as e:
                print(f"    Warning: Could not generate envelope plots: {e}")
        
        print(f"    Plots saved to: {plots_dir}")


def _generate_surface_envelope_plots(
    panel_data: Dict,
    of_data: Dict,
    plots_dir: Path,
    viz_config: Dict,
    quantities: List[str]
):
    """
    Generate surface envelope plots showing distributions wrapped around the body.
    
    The quantity values (Vt, Cp) are visualized as displacements along the surface
    normal, creating a visual "envelope" that wraps around the body.
    
    Reads settings from viz_config['surface_comparison']['envelope_plots'].
    """
    import matplotlib.pyplot as plt
    
    # Get envelope configuration
    envelope_config = viz_config.get('surface_comparison', {}).get('envelope_plots', {})
    fig_config = viz_config.get('figure', {})
    
    # Base settings
    base_scale = envelope_config.get('scale', 0.3)
    show_whiskers = envelope_config.get('show_whiskers', True)
    whisker_density_setting = envelope_config.get('whisker_density', 1)
    envelope_alpha = envelope_config.get('envelope_alpha', 0.3)
    
    # Quantity-specific colormaps
    colormaps = {
        'Vt': envelope_config.get('colormap_Vt', 'viridis'),
        'Cp': envelope_config.get('colormap_Cp', 'RdBu_r'),
    }
    
    # Inversion settings (for Cp, negative = suction = peaks outward)
    invert_Cp = envelope_config.get('invert_Cp', True)
    
    # Comparison plot colors
    comparison_colors = envelope_config.get('comparison_colors', {})
    panel_color = comparison_colors.get('panel', 'blue')
    of_color = comparison_colors.get('openfoam', 'red')
    
    for comp_name in panel_data.keys():
        if comp_name not in of_data:
            continue
        
        panel_df = panel_data[comp_name]
        of_df = of_data[comp_name]
        
        # Get coordinates from panel data (use as the reference body)
        if 'x' not in panel_df.columns or 'y' not in panel_df.columns:
            print(f"    Warning: No x,y coordinates in panel data for {comp_name}, skipping envelope")
            continue
        
        x = panel_df['x'].values
        y = panel_df['y'].values
        
        for quantity in quantities:
            if quantity not in panel_df.columns or quantity not in of_df.columns:
                continue
            
            panel_values = panel_df[quantity].values
            of_values = of_df[quantity].values
            of_x = of_df['x'].values
            of_y = of_df['y'].values
            
            # Determine if we should invert values
            invert = invert_Cp if quantity == 'Cp' else False
            
            # Get quantity-specific scale (fall back to base scale)
            q_scale = envelope_config.get(f'scale_{quantity}', base_scale)
            
            # Get colormap for this quantity (fall back to viridis)
            colormap = colormaps.get(quantity, 'viridis')
            
            # Calculate whisker density based on point count and setting
            if whisker_density_setting == 1:
                # Auto: show ~40 whiskers
                panel_whisker_density = max(1, len(x) // 40)
                of_whisker_density = max(1, len(of_x) // 40)
            else:
                panel_whisker_density = whisker_density_setting
                of_whisker_density = whisker_density_setting
            
            try:
                # 1. Single envelope plot for panel method (with colormap)
                fig, ax = plot_surface_envelope(
                    x, y, panel_values,
                    scale=q_scale,
                    quantity_name=quantity,
                    colormap=colormap,
                    invert_values=invert,
                    title=f'Panel Method - {quantity} Surface Distribution',
                    show_whiskers=show_whiskers,
                    whisker_density=panel_whisker_density,
                    envelope_alpha=envelope_alpha,
                )
                
                for fmt in fig_config.get('format', ['png']):
                    save_path = plots_dir / f"{comp_name}_{quantity}_envelope_panel.{fmt}"
                    fig.savefig(save_path, dpi=fig_config.get('dpi', 300), bbox_inches='tight')
                plt.close(fig)
                
                # 2. Single envelope plot for OpenFOAM (with colormap)
                fig, ax = plot_surface_envelope(
                    of_x, of_y, of_values,
                    scale=q_scale,
                    quantity_name=quantity,
                    colormap=colormap,
                    invert_values=invert,
                    title=f'OpenFOAM - {quantity} Surface Distribution',
                    show_whiskers=show_whiskers,
                    whisker_density=of_whisker_density,
                    envelope_alpha=envelope_alpha,
                )
                
                for fmt in fig_config.get('format', ['png']):
                    save_path = plots_dir / f"{comp_name}_{quantity}_envelope_openfoam.{fmt}"
                    fig.savefig(save_path, dpi=fig_config.get('dpi', 300), bbox_inches='tight')
                plt.close(fig)
                
                # 3. Dual comparison envelope plot (both on same body)
                # Interpolate OpenFOAM values onto panel mesh points for comparison
                from scipy.interpolate import interp1d
                
                # Use arc length for interpolation
                panel_s = panel_df['s'].values
                of_s = of_df['s'].values
                
                # Use unique 's' values to prevent divide-by-zero warnings in interp1d
                o_s, o_idx = np.unique(of_s, return_index=True)
                
                # Interpolate OF values to panel arc length positions
                of_interp = interp1d(o_s, of_values[o_idx], kind='linear', 
                                    bounds_error=False, fill_value='extrapolate')
                of_values_interp = of_interp(panel_s)
                
                fig, ax = plot_dual_surface_envelope(
                    x, y, 
                    panel_values, of_values_interp,
                    label1='Panel Method',
                    label2='OpenFOAM',
                    scale=q_scale,
                    quantity_name=quantity,
                    color1=panel_color,
                    color2=of_color,
                    invert_values=invert,
                    title=f'{quantity} Distribution Comparison (Envelope)',
                    show_difference=True,
                )
                
                for fmt in fig_config.get('format', ['png']):
                    save_path = plots_dir / f"{comp_name}_{quantity}_envelope_comparison.{fmt}"
                    fig.savefig(save_path, dpi=fig_config.get('dpi', 300), bbox_inches='tight')
                plt.close(fig)
                
                print(f"    ✓ {comp_name} - {quantity} envelope plots")
                
            except Exception as e:
                print(f"    Warning: Could not generate envelope plot for {comp_name}/{quantity}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Regenerate validation plots from raw data")
    parser.add_argument("case_dir", type=Path, help="Path to case directory")
    parser.add_argument("--study", choices=['of_convergence', 'panel_convergence', 'surface_comparison', 'all'],
                       default='all', help="Which study to regenerate")
    parser.add_argument("--of-level", type=str, help="Specific OF level for surface comparison")
    
    args = parser.parse_args()
    
    case_dir = args.case_dir.resolve()
    if not case_dir.exists():
        print(f"Error: Case directory not found: {case_dir}")
        return 1
    
    output_dir = case_dir / "out"
    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        return 1
    
    # Load visualization config
    viz_config = load_viz_config(output_dir / "viz_config.yaml")
    
    print(f"Regenerating visualizations for: {case_dir.name}")
    print(f"{'='*60}")
    
    # Run requested visualizations
    if args.study == 'all' or args.study == 'of_convergence':
        if (output_dir / "of_convergence" / "raw").exists():
            visualize_of_convergence(case_dir, output_dir, viz_config)
        else:
            print("OpenFOAM convergence data not found, skipping...")
    
    if args.study == 'all' or args.study == 'panel_convergence':
        if (output_dir / "panel_convergence" / "raw").exists():
            visualize_panel_convergence(case_dir, output_dir, viz_config)
        else:
            print("Panel convergence data not found, skipping...")
    
    if args.study == 'all' or args.study == 'surface_comparison':
        if (output_dir / "surface_comparison").exists():
            visualize_surface_comparison(case_dir, output_dir, viz_config, of_level=args.of_level)
        else:
            print("Surface comparison data not found, skipping...")
    
    print(f"\n{'='*60}")
    print("Visualization complete!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
