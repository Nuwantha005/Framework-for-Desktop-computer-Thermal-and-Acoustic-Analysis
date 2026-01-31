"""
Plot generators for validation studies.

These are thin wrappers around existing visualization tools, configured
via viz_config.yaml for flexibility.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_field_overview_with_components(
    field_data: Dict[str, np.ndarray],
    mesh,  # Assembled mesh from scene
    monitoring_points: List[Dict[str, Any]],
    config: Dict[str, Any],
    output_path: Path,
    title: str = "Flow Field Overview"
) -> Figure:
    """
    Plot velocity/pressure field with proper component boundaries and monitoring points.
    
    Uses the Visualizer class for proper component rendering.
    
    Args:
        field_data: Dict with 'XX', 'YY', 'velocity_magnitude', 'Vx', 'Vy', etc.
        mesh: Assembled mesh from scene (with component information)
        monitoring_points: List of points with 'name' and 'coordinates' 
        config: Visualization config dict
        output_path: Path to save figure
        title: Plot title
    
    Returns:
        Figure object
    """
    from visualization import Visualizer
    
    fig_config = config.get('figure', {})
    
    # Create field overview using Visualizer
    viz = Visualizer(figsize=(12, 8))
    fig, ax = viz.create_figure()
    
    # Plot velocity magnitude contours
    XX, YY = field_data['XX'], field_data['YY']
    velocity_magnitude = field_data['velocity_magnitude']
    
    # Manual contour plot with colorbar
    levels = 20
    cf = ax.contourf(XX, YY, velocity_magnitude, levels=levels, cmap='viridis')
    plt.colorbar(cf, ax=ax, label='Velocity Magnitude [m/s]')
    
    # Draw body outline using proper component handling
    viz._draw_body_outline(ax, mesh, fill=True)
    
    # Add monitoring points with proper coordinates and labels
    for point in monitoring_points:
        coords = point['coordinates']
        x_coord, y_coord = coords[0], coords[1]
        
        # Plot point
        ax.scatter(x_coord, y_coord, s=100, c='red', 
                  marker='o', edgecolors='white', linewidths=2, zorder=10)
        
        # Add label
        ax.annotate(point['name'], xy=(x_coord, y_coord), xytext=(5, 5),
                   textcoords='offset points', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                   zorder=11)
    
    # Format plot
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Set reasonable limits
    ax.set_xlim(XX.min(), XX.max())
    ax.set_ylim(YY.min(), YY.max())
    
    plt.tight_layout()
    
    # Save in multiple formats
    for fmt in fig_config.get('format', ['png']):
        save_path = output_path.with_suffix(f'.{fmt}')
        fig.savefig(save_path, dpi=fig_config.get('dpi', 300), bbox_inches='tight')
    
    return fig


def plot_field_with_points(
    field_data: Dict[str, np.ndarray],
    monitoring_points: List[Dict[str, Any]],
    config: Dict[str, Any],
    output_path: Path,
    title: str = "Flow Field with Monitoring Points"
) -> Figure:
    """
    Plot velocity/pressure field with monitoring point locations.
    
    DEPRECATED: Use plot_field_overview_with_components for proper component rendering.
    
    Uses existing VelocityField2D visualization with overlaid points.
    
    Args:
        field_data: Dict with 'XX', 'YY', 'field' (velocity magnitude or pressure)
        monitoring_points: List of points with 'name' and 'coordinates'
        config: Visualization config dict
        output_path: Path to save figure
        title: Plot title
    
    Returns:
        Figure object
    """
    from visualization.field2d import VelocityField2D
    
    fig_config = config.get('figure', {})
    field_config = config.get('of_convergence', {}).get('field_overview', {})
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=fig_config.get('dpi', 300))
    
    # Plot field as contour
    XX, YY = field_data['XX'], field_data['YY']
    field = field_data['field']
    
    levels = 20
    contour = ax.contourf(XX, YY, field, levels=levels, cmap='viridis')
    
    if field_config.get('colorbar', True):
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label(field_data.get('label', 'Field Value'))
    
    # Plot streamlines if requested
    if field_config.get('show_streamlines', False) and 'Vx' in field_data and 'Vy' in field_data:
        ax.streamplot(XX, YY, field_data['Vx'], field_data['Vy'], 
                     color='white', linewidth=0.5, density=1.5, arrowsize=0.8)
    
    # Plot monitoring points
    if field_config.get('show_monitoring_points', True):
        point_size = field_config.get('point_marker_size', 100)
        for i, point in enumerate(monitoring_points):
            coords = point['coordinates']
            ax.scatter(coords[0], coords[1], s=point_size, c='red', 
                      marker='o', edgecolors='white', linewidths=2, zorder=10,
                      label=point['name'] if i == 0 else None)
            ax.annotate(point['name'], xy=coords, xytext=(5, 5),
                       textcoords='offset points', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save in requested formats
    for fmt in fig_config.get('format', ['png']):
        save_path = output_path.with_suffix(f'.{fmt}')
        fig.savefig(save_path, dpi=fig_config.get('dpi', 300), bbox_inches='tight')
    
    return fig


def plot_convergence_curves(
    data: Dict[str, Dict[str, float]],
    level_names: List[str],
    quantities: List[str],
    config: Dict[str, Any],
    output_dir: Path,
    reference_values: Optional[Dict[str, float]] = None,
    plot_type: str = 'of_convergence'
) -> List[Figure]:
    """
    Plot convergence curves for monitoring points.
    
    Args:
        data: Dict mapping level names to {point_name: {qty: value}}
        level_names: Ordered list of level names
        quantities: Quantities to plot ('velocity', 'pressure')
        config: Visualization config
        output_dir: Directory to save plots
        reference_values: Optional reference values to plot as horizontal line
        plot_type: 'of_convergence' or 'panel_convergence'
    
    Returns:
        List of Figure objects
    """
    conv_config = config.get(plot_type, {}).get('convergence_curves', {})
    fig_config = config.get('figure', {})
    
    if not conv_config.get('enabled', True):
        return []
    
    figures = []
    
    for qty in quantities:
        fig, axes = plt.subplots(1, 2 if conv_config.get('show_change_rate', False) else 1,
                                figsize=(12 if conv_config.get('show_change_rate', False) else 6, 5),
                                dpi=fig_config.get('dpi', 300))
        
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        
        # Get all point names
        point_names = set()
        for level_data in data.values():
            point_names.update(level_data.keys())
        point_names = sorted(point_names)
        
        # Plot value vs level
        ax = axes[0]
        for point_name in point_names:
            values = [data[level][point_name][qty] 
                     for level in level_names if level in data and point_name in data[level]]
            x = range(len(values))
            ax.plot(x, values, marker='o', label=point_name, linewidth=2)
        
        # Plot reference line if provided
        if reference_values and conv_config.get('show_of_reference_line', False):
            ref_val = reference_values.get(qty)
            if ref_val is not None:
                ax.axhline(ref_val, 
                          linestyle=conv_config.get('of_line_style', '--'),
                          color=conv_config.get('of_line_color', 'red'),
                          linewidth=2, label='OpenFOAM Reference')
        
        ax.set_xlabel('Refinement Level')
        ax.set_ylabel(qty.capitalize())
        ax.set_title(f'{qty.capitalize()} Convergence')
        ax.set_xticks(range(len(level_names)))
        ax.set_xticklabels(level_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot change between levels if requested
        if len(axes) > 1:
            ax = axes[1]
            for point_name in point_names:
                values = [data[level][point_name][qty] 
                         for level in level_names if level in data and point_name in data[level]]
                changes = [values[i] - values[i-1] for i in range(1, len(values))]
                x = range(1, len(values))
                ax.plot(x, changes, marker='s', label=point_name, linewidth=2)
            
            ax.set_xlabel('Refinement Level')
            ax.set_ylabel(f'Δ{qty.capitalize()}')
            ax.set_title(f'Change in {qty.capitalize()}')
            ax.set_xticks(range(1, len(level_names)))
            ax.set_xticklabels([f'{level_names[i-1]}→{level_names[i]}' 
                               for i in range(1, len(level_names))], rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color='black', linewidth=0.5)
        
        plt.tight_layout()
        
        # Save
        for fmt in fig_config.get('format', ['png']):
            save_path = output_dir / f'{qty}_convergence.{fmt}'
            fig.savefig(save_path, dpi=fig_config.get('dpi', 300), bbox_inches='tight')
        
        figures.append(fig)
    
    return figures


def plot_value_vs_level(
    values: List[float],
    level_names: List[str],
    quantity: str,
    output_path: Path,
    reference_value: Optional[float] = None,
    config: Optional[Dict] = None
) -> Figure:
    """Simple value vs level plot for a single monitoring point."""
    if config is None:
        config = {}
    
    fig_config = config.get('figure', {})
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=fig_config.get('dpi', 300))
    
    x = range(len(values))
    ax.plot(x, values, marker='o', linewidth=2, markersize=8, label=quantity.capitalize())
    
    if reference_value is not None:
        ax.axhline(reference_value, linestyle='--', color='red', 
                  linewidth=2, label='Reference')
    
    ax.set_xlabel('Refinement Level')
    ax.set_ylabel(quantity.capitalize())
    ax.set_title(f'{quantity.capitalize()} vs Mesh Level')
    ax.set_xticks(x)
    ax.set_xticklabels(level_names, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    for fmt in fig_config.get('format', ['png']):
        save_path = output_path.with_suffix(f'.{fmt}')
        fig.savefig(save_path, dpi=fig_config.get('dpi', 300), bbox_inches='tight')
    
    return fig


def plot_change_between_levels(
    values: List[float],
    level_names: List[str],
    quantity: str,
    output_path: Path,
    config: Optional[Dict] = None
) -> Figure:
    """Plot change in value between consecutive mesh levels."""
    if config is None:
        config = {}
    
    fig_config = config.get('figure', {})
    
    changes = [values[i] - values[i-1] for i in range(1, len(values))]
    rel_changes = [(values[i] - values[i-1]) / values[i-1] * 100 
                   for i in range(1, len(values))]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=fig_config.get('dpi', 300))
    
    # Absolute change
    x = range(len(changes))
    ax1.plot(x, changes, marker='s', linewidth=2, markersize=8)
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.set_xlabel('Level Transition')
    ax1.set_ylabel(f'Δ{quantity.capitalize()}')
    ax1.set_title(f'Absolute Change in {quantity.capitalize()}')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{level_names[i]}→{level_names[i+1]}' 
                        for i in range(len(changes))], rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Relative change
    ax2.plot(x, rel_changes, marker='s', linewidth=2, markersize=8, color='orange')
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_xlabel('Level Transition')
    ax2.set_ylabel(f'Relative Change (%)')
    ax2.set_title(f'Relative Change in {quantity.capitalize()}')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{level_names[i]}→{level_names[i+1]}' 
                        for i in range(len(changes))], rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    for fmt in fig_config.get('format', ['png']):
        save_path = output_path.with_suffix(f'.{fmt}')
        fig.savefig(save_path, dpi=fig_config.get('dpi', 300), bbox_inches='tight')
    
    return fig
