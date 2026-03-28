"""
Surface envelope plots for visualizing distributions wrapped around body geometry.

Instead of plotting quantity vs arc length on a standard XY plot, this module
plots the distribution directly on the body surface where the quantity value
is represented as a displacement along the surface normal.

This creates a visual "envelope" around the body showing the distribution,
similar to:
- Star/radar plots wrapped around arbitrary shapes
- Wind tunnel visualization of surface properties
- Classic airfoil Cp distribution diagrams
"""

from typing import Optional, Tuple, List, Union
from matplotlib import scale
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors


def compute_outward_normals(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    closed: bool = True
) -> NDArray[np.float64]:
    """
    Compute outward-pointing unit normal vectors for a 2D curve.
    
    Assumes the curve is oriented counter-clockwise (CCW) for closed bodies.
    
    Args:
        x: X coordinates of curve points (N,)
        y: Y coordinates of curve points (N,)
        closed: If True, treat as closed curve (wrap around)
    
    Returns:
        normals: (N, 2) array of unit normal vectors pointing outward
    """
    n = len(x)
    normals = np.zeros((n, 2), dtype=np.float64)
    
    for i in range(n):
        if closed:
            # Use central difference with wrapping
            i_prev = (i - 1) % n
            i_next = (i + 1) % n
        else:
            # Handle endpoints
            i_prev = max(0, i - 1)
            i_next = min(n - 1, i + 1)
        
        # Tangent vector (forward difference approximation)
        tx = x[i_next] - x[i_prev]
        ty = y[i_next] - y[i_prev]
        
        # Normalize
        length = np.sqrt(tx**2 + ty**2)
        if length > 1e-10:
            tx /= length
            ty /= length
        
        # Outward normal (rotate tangent 90° clockwise for CCW curve)
        # For CCW: normal = (ty, -tx) points outward
        normals[i, 0] = ty
        normals[i, 1] = -tx
    
    # Verify outward direction using centroid test
    cx, cy = np.mean(x), np.mean(y)
    test_idx = 0
    to_centroid = np.array([cx - x[test_idx], cy - y[test_idx]])
    if np.dot(normals[test_idx], to_centroid) > 0:
        # Normals point inward, flip them
        normals = -normals
    
    return normals


def plot_surface_envelope(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    values: NDArray[np.float64],
    normals: Optional[NDArray[np.float64]] = None,
    ax: Optional[Axes] = None,
    scale: float = 0.3,
    quantity_name: str = "Value",
    show_body: bool = True,
    show_envelope: bool = True,
    show_whiskers: bool = True,
    whisker_density: int = 1,
    body_color: str = 'black',
    body_linewidth: float = 2.0,
    envelope_color: str = 'blue',
    envelope_linewidth: float = 1.5,
    envelope_fill: bool = True,
    envelope_alpha: float = 0.3,
    whisker_color: str = 'gray',
    whisker_alpha: float = 0.5,
    whisker_linewidth: float = 0.5,
    colormap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    invert_values: bool = False,
    show_colorbar: bool = True,
    title: Optional[str] = None,
) -> Tuple[Figure, Axes]:
    """
    Plot a surface distribution as an envelope around the body geometry.
    
    The quantity values are visualized as displacements along the surface normal,
    creating a visual "envelope" that wraps around the body.
    
    Args:
        x: X coordinates of surface points (N,)
        y: Y coordinates of surface points (N,)
        values: Quantity values at each surface point (N,)
        normals: Optional (N, 2) array of outward normal vectors. 
                 If None, computed automatically assuming CCW orientation.
        ax: Matplotlib axes to plot on. If None, creates new figure.
        scale: Scaling factor for envelope displacement (in geometry units)
        quantity_name: Name of the quantity for labels
        show_body: If True, plot the body surface
        show_envelope: If True, plot the envelope curve
        show_whiskers: If True, draw lines from surface to envelope
        whisker_density: Plot every Nth whisker (1 = all, 2 = every other, etc.)
        body_color: Color for body outline
        body_linewidth: Line width for body outline
        envelope_color: Color for envelope (if no colormap)
        envelope_linewidth: Line width for envelope
        envelope_fill: If True, fill the region between body and envelope
        envelope_alpha: Alpha for envelope fill
        whisker_color: Color for whiskers (if no colormap)
        whisker_alpha: Alpha for whiskers
        whisker_linewidth: Line width for whiskers
        colormap: Optional colormap name for coloring by value
        vmin: Minimum value for colormap normalization
        vmax: Maximum value for colormap normalization
        invert_values: If True, invert values (useful for Cp where negative = suction)
        show_colorbar: If True and colormap is used, show colorbar
        title: Optional plot title
    
    Returns:
        (fig, ax) tuple
    
    Example:
        >>> # Plot Vt distribution
        >>> fig, ax = plot_surface_envelope(x, y, Vt, scale=0.2, 
        ...                                  quantity_name='Vt', colormap='viridis')
        
        >>> # Plot Cp distribution (inverted so suction peaks outward)
        >>> fig, ax = plot_surface_envelope(x, y, -Cp, scale=0.3,
        ...                                  quantity_name='Cp', colormap='RdBu_r')
    """
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()
    
    # Compute normals if not provided
    if normals is None:
        normals = compute_outward_normals(x, y, closed=True)
    
    # Process values
    plot_values = -values if invert_values else values
    
    # Normalize values for displacement
    if vmin is None:
        vmin = np.nanmin(plot_values)
    if vmax is None:
        vmax = np.nanmax(plot_values)
    
    # Scale values to [0, 1] range then multiply by scale factor
    value_range = vmax - vmin if vmax != vmin else 1.0
    normalized_values = (plot_values - vmin) / value_range
    displacements = normalized_values * scale
    
    # Compute envelope points
    envelope_x = x + displacements * normals[:, 0]
    envelope_y = y + displacements * normals[:, 1]
    
    # Setup colormap if requested
    if colormap is not None:
        cmap = plt.cm.get_cmap(colormap)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        colors = cmap(norm(plot_values))
    
    # Plot body surface
    if show_body:
        # Close the curve for plotting
        body_x = np.append(x, x[0])
        body_y = np.append(y, y[0])
        ax.plot(body_x, body_y, color=body_color, linewidth=body_linewidth, 
                zorder=10, label='Body')
    
    # Plot filled region between body and envelope
    if envelope_fill and show_envelope:
        # Create polygon vertices (body + reversed envelope)
        n = len(x)
        body_pts = np.column_stack([x, y])
        envelope_pts = np.column_stack([envelope_x, envelope_y])
        
        if colormap is not None:
            # Use segments with individual colors
            for i in range(n):
                i_next = (i + 1) % n
                if np.isnan(envelope_x[i]) or np.isnan(envelope_x[i_next]):
                    continue
                quad_x = [x[i], envelope_x[i], envelope_x[i_next], x[i_next]]
                quad_y = [y[i], envelope_y[i], envelope_y[i_next], y[i_next]]
                color = cmap(norm((plot_values[i] + plot_values[i_next]) / 2))
                ax.fill(quad_x, quad_y, color=color, alpha=envelope_alpha, 
                       edgecolor='none', zorder=1)
        else:
            # Single color fill with contiguous segments
            valid_mask = ~(np.isnan(envelope_x) | np.isnan(envelope_y) | np.isnan(x) | np.isnan(y))
            segments = np.ma.clump_unmasked(np.ma.masked_where(~valid_mask, envelope_x))
            for seg in segments:
                polygon_x = np.concatenate([x[seg], envelope_x[seg][::-1]])
                polygon_y = np.concatenate([y[seg], envelope_y[seg][::-1]])
                ax.fill(polygon_x, polygon_y, color=envelope_color, 
                       alpha=envelope_alpha, zorder=1)
    
    # Plot whiskers (lines from body to envelope)
    if show_whiskers:
        whisker_indices = range(0, len(x), whisker_density)
        
        if colormap is not None:
            # Colored whiskers
            segments = []
            whisker_colors = []
            for i in whisker_indices:
                segments.append([(x[i], y[i]), (envelope_x[i], envelope_y[i])])
                whisker_colors.append(cmap(norm(plot_values[i])))
            
            lc = LineCollection(segments, colors=whisker_colors, 
                               linewidths=whisker_linewidth, alpha=whisker_alpha,
                               zorder=2)
            ax.add_collection(lc)
        else:
            # Single color whiskers
            for i in whisker_indices:
                ax.plot([x[i], envelope_x[i]], [y[i], envelope_y[i]],
                       color=whisker_color, alpha=whisker_alpha,
                       linewidth=whisker_linewidth, zorder=2)
    
    # Plot envelope curve
    if show_envelope:
        valid_mask = ~(np.isnan(envelope_x) | np.isnan(envelope_y))
        segments = np.ma.clump_unmasked(np.ma.masked_where(~valid_mask, envelope_x))
        is_fully_closed = len(segments) == 1 and segments[0].start == 0 and segments[0].stop == len(envelope_x)

        if colormap is not None:
            # Colored envelope segments
            # Apply mask to segment values
            segment_values = (plot_values + np.roll(plot_values, -1)) / 2
            
            for seg in segments:
                seg_x = envelope_x[seg]
                seg_y = envelope_y[seg]
                seg_vals = segment_values[seg]
                if is_fully_closed:
                    seg_x = np.append(seg_x, seg_x[0])
                    seg_y = np.append(seg_y, seg_y[0])
                    seg_vals = np.append(seg_vals, seg_vals[0])
                
                points = np.column_stack([seg_x, seg_y]).reshape(-1, 1, 2)
                if len(points) < 2:
                    continue
                line_segments = np.concatenate([points[:-1], points[1:]], axis=1)
                
                lc = LineCollection(line_segments, cmap=cmap, norm=norm,
                                   linewidths=envelope_linewidth, zorder=5)
                lc.set_array(seg_vals[:-1])
                ax.add_collection(lc)
        else:
            for i, seg in enumerate(segments):
                seg_x = envelope_x[seg]
                seg_y = envelope_y[seg]
                if is_fully_closed:
                    seg_x = np.append(seg_x, seg_x[0])
                    seg_y = np.append(seg_y, seg_y[0])
                ax.plot(seg_x, seg_y, color=envelope_color, 
                       linewidth=envelope_linewidth, zorder=5, label='Envelope' if i == 0 else "")
    
    # Add colorbar
    if colormap is not None and show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
        cbar_label = f'-{quantity_name}' if invert_values else quantity_name
        cbar.set_label(cbar_label)
    
    # Formatting
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    if title:
        ax.set_title(title)
    
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_surface_envelope_comparison(
    x_list: List[NDArray[np.float64]],
    y_list: List[NDArray[np.float64]],
    values_list: List[NDArray[np.float64]],
    labels: List[str],
    normals_list: Optional[List[NDArray[np.float64]]] = None,
    ax: Optional[Axes] = None,
    scale: float = 0.3,
    quantity_name: str = "Value",
    colors: Optional[List[str]] = None,
    show_body: bool = True,
    envelope_linewidth: float = 1.5,
    envelope_alpha: float = 0.4,
    title: Optional[str] = None,
    invert_values: bool = False,
) -> Tuple[Figure, Axes]:
    """
    Plot multiple surface envelopes for comparison (e.g., panel vs OpenFOAM).
    
    Args:
        x_list: List of X coordinate arrays
        y_list: List of Y coordinate arrays  
        values_list: List of value arrays
        labels: Labels for each dataset
        normals_list: Optional list of normal arrays
        ax: Matplotlib axes
        scale: Scaling factor for envelopes
        quantity_name: Name of quantity
        colors: Colors for each dataset
        show_body: Show body outline (uses first dataset)
        envelope_linewidth: Line width for envelopes
        envelope_alpha: Fill alpha for envelopes
        title: Plot title
        invert_values: Invert values for display
    
    Returns:
        (fig, ax) tuple
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()
    
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(x_list)))
    
    # Find global value range for consistent scaling
    all_values = np.concatenate(values_list)
    if invert_values:
        all_values = -all_values
    vmin, vmax = np.nanmin(all_values), np.nanmax(all_values)
    value_range = vmax - vmin if vmax != vmin else 1.0
    
    # Plot body (from first dataset)
    if show_body:
        body_x = np.append(x_list[0], x_list[0][0])
        body_y = np.append(y_list[0], y_list[0][0])
        ax.plot(body_x, body_y, 'k-', linewidth=2, zorder=10, label='Body')
    
    # Plot each envelope
    for i, (x, y, values, label) in enumerate(zip(x_list, y_list, values_list, labels)):
        # Compute normals
        if normals_list is not None and normals_list[i] is not None:
            normals = normals_list[i]
        else:
            normals = compute_outward_normals(x, y, closed=True)
        
        # Process values
        plot_values = -values if invert_values else values
        
        # Compute displacements
        normalized_values = (plot_values - vmin) / value_range
        displacements = normalized_values * scale
        
        # Envelope points
        envelope_x = x + displacements * normals[:, 0]
        envelope_y = y + displacements * normals[:, 1]
        
        # Find contiguous valid segments
        valid_mask = ~(np.isnan(envelope_x) | np.isnan(envelope_y) | np.isnan(x) | np.isnan(y))
        segments = np.ma.clump_unmasked(np.ma.masked_where(~valid_mask, envelope_x))
        is_fully_closed = len(segments) == 1 and segments[0].start == 0 and segments[0].stop == len(envelope_x)
        
        color = colors[i] if isinstance(colors[i], str) else colors[i]
        
        # Plot filled envelope in segments
        for seg in segments:
            polygon_x = np.concatenate([x[seg], envelope_x[seg][::-1]])
            polygon_y = np.concatenate([y[seg], envelope_y[seg][::-1]])
            ax.fill(polygon_x, polygon_y, color=color, alpha=envelope_alpha, 
                   edgecolor='none', zorder=1+i)
            
            # Plot envelope line
            seg_x = envelope_x[seg]
            seg_y = envelope_y[seg]
            if is_fully_closed:
                seg_x = np.append(seg_x, seg_x[0])
                seg_y = np.append(seg_y, seg_y[0])
                
            ax.plot(seg_x, seg_y, color=color, linewidth=envelope_linewidth, 
                   zorder=5+i, label=label if seg == segments[0] else "")
    
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    if title:
        ax.set_title(title)
    
    return fig, ax


def plot_dual_surface_envelope(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    values1: NDArray[np.float64],
    values2: NDArray[np.float64],
    label1: str = "Dataset 1",
    label2: str = "Dataset 2",
    normals: Optional[NDArray[np.float64]] = None,
    ax: Optional[Axes] = None,
    scale: float = 0.3,
    quantity_name: str = "Value",
    color1: str = 'blue',
    color2: str = 'red',
    show_difference: bool = False,
    title: Optional[str] = None,
    invert_values: bool = False,
) -> Tuple[Figure, Axes]:
    """
    Plot two distributions on the same body for direct comparison.
    Useful for comparing panel method vs OpenFOAM on the same geometry.
    
    Args:
        x, y: Surface coordinates (same for both)
        values1, values2: Two value arrays to compare
        label1, label2: Labels for each dataset
        normals: Surface normals
        ax: Matplotlib axes
        scale: Scaling factor
        quantity_name: Name of quantity
        color1, color2: Colors for each dataset
        show_difference: If True, highlight regions where values differ
        title: Plot title
        invert_values: Invert values for display
    
    Returns:
        (fig, ax) tuple
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.get_figure()
    
    # Compute normals
    if normals is None:
        normals = compute_outward_normals(x, y, closed=True)
    
    # Process values
    v1 = -values1 if invert_values else values1
    v2 = -values2 if invert_values else values2
    
    # Normalize each dataset independently so they both touch the body at their minima
    vmin1, vmax1 = np.nanmin(v1), np.nanmax(v1)
    vmin2, vmax2 = np.nanmin(v2), np.nanmax(v2)
    
    value_range1 = vmax1 - vmin1 if vmax1 != vmin1 else 1.0
    value_range2 = vmax2 - vmin2 if vmax2 != vmin2 else 1.0
    
    # Compute envelopes - each touches body at its minimum
    disp1 = (v1 - vmin1) / value_range1 * scale
    disp2 = (v2 - vmin2) / value_range2 * scale
    
    env1_x = x + disp1 * normals[:, 0]
    env1_y = y + disp1 * normals[:, 1]
    
    env2_x = x + disp2 * normals[:, 0]
    env2_y = y + disp2 * normals[:, 1]
    
    # Plot body
    body_x = np.append(x, x[0])
    body_y = np.append(y, y[0])
    ax.plot(body_x, body_y, 'k-', linewidth=2.5, zorder=10, label='Body')
    
    for env_x, env_y, color, label, alpha in [
        (env1_x, env1_y, color1, label1, 0.25),
        (env2_x, env2_y, color2, label2, 0.25),
    ]:
        # Filter out NaN values and break into contiguous segments
        valid_mask = ~(np.isnan(env_x) | np.isnan(env_y) | np.isnan(x) | np.isnan(y))
        segments = np.ma.clump_unmasked(np.ma.masked_where(~valid_mask, env_x))
        is_fully_closed = len(segments) == 1 and segments[0].start == 0 and segments[0].stop == len(env_x)
    
        for i, seg in enumerate(segments):
            # Build polygon from contiguous segment
            polygon_x = np.concatenate([x[seg], env_x[seg][::-1]])
            polygon_y = np.concatenate([y[seg], env_y[seg][::-1]])
            ax.fill(polygon_x, polygon_y, color=color, alpha=alpha, edgecolor='none')
        
            seg_x = env_x[seg]
            seg_y = env_y[seg]
            if is_fully_closed:
                seg_x = np.append(seg_x, seg_x[0])
                seg_y = np.append(seg_y, seg_y[0])
            ax.plot(seg_x, seg_y, color=color, linewidth=1.5, label=label if i == 0 else "")
    
    # Show difference regions
    if show_difference:
        diff = np.abs(v1 - v2)
        max_diff = np.max(diff)
        if max_diff > 0:
            # Highlight points with large differences
            threshold = max_diff * 0.5
            high_diff_mask = diff > threshold
            ax.scatter(x[high_diff_mask], y[high_diff_mask], 
                      c='orange', s=20, zorder=15, label='High difference', alpha=0.7)
    
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    if title:
        ax.set_title(title)
    
    return fig, ax