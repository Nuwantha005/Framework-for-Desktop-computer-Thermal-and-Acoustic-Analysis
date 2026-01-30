"""
Parametric 2D geometry generators.

All generators return Mesh objects with CCW-ordered nodes for outward normals.
Resolution parameters are passed as tuples matching function signatures, enabling
clean integration with case.yaml mesh_levels definitions.

Available primitives:
- circle: generate_circle(num_panels) - circular geometry
- rectangle: generate_rectangle(num_panels_x, num_panels_y) - sharp-cornered rectangle  
- rounded_rectangle: generate_rounded_rectangle(num_panels_per_side, num_panels_per_arc) - rectangle with circular arcs
"""

import numpy as np
from numpy.typing import NDArray

from .mesh import Mesh


def generate_circle(
    num_panels: int,
    radius: float = 1.0,
    center: tuple[float, float] = (0.0, 0.0)
) -> Mesh:
    """
    Generate circular mesh in 2D.
    
    Points are generated CCW from +x axis for outward normals.
    
    Args:
        num_panels: Number of panels around circumference
        radius: Circle radius
        center: Center point (x, y)
    
    Returns:
        2D Mesh of circle with `num_panels` panels
    
    Example:
        >>> mesh = generate_circle(num_panels=32, radius=0.5, center=(0, 0))
        >>> mesh.num_panels
        32
    """
    cx, cy = center
    
    # Generate points around circle (CCW from +x axis)
    theta = np.linspace(0, 2 * np.pi, num_panels + 1)[:-1]  # Exclude duplicate at 2π
    x = cx + radius * np.cos(theta)
    y = cy + radius * np.sin(theta)
    z = np.zeros_like(x)
    
    nodes = np.column_stack([x, y, z])
    
    # Generate panels
    panels = np.array([[i, (i + 1) % num_panels] for i in range(num_panels)], dtype=np.int32)
    
    component_ids = np.zeros(panels.shape[0], dtype=np.int32)
    
    return Mesh(nodes=nodes, panels=panels, dimension=2, component_ids=component_ids)


def generate_rectangle(
    num_panels_x: int,
    num_panels_y: int,
    width: float = 2.0,
    height: float = 1.0,
    center: tuple[float, float] = (0.0, 0.0)
) -> Mesh:
    """
    Generate rectangular mesh in 2D with sharp corners.
    
    Nodes are ordered CCW: bottom → right → top → left.
    
    Args:
        num_panels_x: Number of panels on horizontal sides (top and bottom)
        num_panels_y: Number of panels on vertical sides (left and right)
        width: Rectangle width
        height: Rectangle height
        center: Center point (x, y)
    
    Returns:
        2D Mesh of rectangle with (2*num_panels_x + 2*num_panels_y) panels
    
    Example:
        >>> mesh = generate_rectangle(4, 4, width=1.0, height=1.0, center=(0, 0))
        >>> mesh.num_panels
        16
    """
    cx, cy = center
    x0, x1 = cx - width / 2, cx + width / 2
    y0, y1 = cy - height / 2, cy + height / 2
    
    # Generate nodes around perimeter (CCW)
    # Bottom edge (y0, left to right)
    bottom_x = np.linspace(x0, x1, num_panels_x + 1)
    bottom_nodes = np.column_stack([bottom_x, np.full_like(bottom_x, y0), np.zeros_like(bottom_x)])
    
    # Right edge (x1, bottom to top, exclude first point to avoid duplicate)
    right_y = np.linspace(y0, y1, num_panels_y + 1)[1:]
    right_nodes = np.column_stack([np.full_like(right_y, x1), right_y, np.zeros_like(right_y)])
    
    # Top edge (y1, right to left, exclude first)
    top_x = np.linspace(x1, x0, num_panels_x + 1)[1:]
    top_nodes = np.column_stack([top_x, np.full_like(top_x, y1), np.zeros_like(top_x)])
    
    # Left edge (x0, top to bottom, exclude first and last to close loop)
    left_y = np.linspace(y1, y0, num_panels_y + 1)[1:-1]
    left_nodes = np.column_stack([np.full_like(left_y, x0), left_y, np.zeros_like(left_y)])
    
    # Concatenate all nodes
    nodes = np.vstack([bottom_nodes, right_nodes, top_nodes, left_nodes])
    
    # Generate panels
    num_nodes = nodes.shape[0]
    panels = np.array([[i, (i + 1) % num_nodes] for i in range(num_nodes)], dtype=np.int32)
    
    component_ids = np.zeros(panels.shape[0], dtype=np.int32)
    
    return Mesh(nodes=nodes, panels=panels, dimension=2, component_ids=component_ids)


def generate_rounded_rectangle(
    num_panels_per_side: int,
    num_panels_per_arc: int,
    width: float = 4.0,
    height: float = 2.0,
    corner_radius: float = 0.3,
    center: tuple[float, float] = (0.0, 0.0)
) -> Mesh:
    """
    Generate rounded rectangle mesh in 2D.
    
    Points are ordered CCW starting from right edge to ensure outward normals.
    Special case: if corner_radius=0 or num_panels_per_arc=0, generates sharp corners.
    
    Args:
        num_panels_per_side: Number of panels on each straight edge (4 edges total)
        num_panels_per_arc: Number of panels on each corner arc (4 corners total)
        width: Rectangle width (including corner radii)
        height: Rectangle height (including corner radii)
        corner_radius: Corner arc radius (must be <= min(width/2, height/2))
        center: Center point (x, y)
    
    Returns:
        2D Mesh with (4*num_panels_per_side + 4*num_panels_per_arc) total panels
    
    Example:
        >>> mesh = generate_rounded_rectangle(8, 4, width=2.0, height=1.0, corner_radius=0.2)
        >>> mesh.num_panels
        48
    
    Raises:
        ValueError: If corner_radius is too large for dimensions
    """
    cx, cy = center
    a = width / 2
    b = height / 2
    r = corner_radius

    # Special case: sharp corners
    if r <= 0 or num_panels_per_arc <= 0:
        return generate_rectangle(num_panels_per_side, num_panels_per_side, width, height, center)

    if r > min(a, b):
        raise ValueError(
            f"Corner radius {r} too large for dimensions (width={width}, height={height}). "
            f"Maximum allowed: {min(a, b)}"
        )

    pts = []

    # Generate points in CCW order starting from bottom-right straight edge
    # Note: linspace endpoint=False avoids duplicate points where segments join
    
    # 1. Right edge (vertical, going up)
    y_right = np.linspace(-b + r, b - r, num_panels_per_side + 1)[:-1]
    pts.extend([(a, y) for y in y_right])
    
    # 2. Top-right arc (0 to 90 degrees)
    theta_tr = np.linspace(0, np.pi/2, num_panels_per_arc + 1)[:-1]
    x0, y0 = a - r, b - r
    pts.extend([(x0 + r*np.cos(t), y0 + r*np.sin(t)) for t in theta_tr])
    
    # 3. Top edge (horizontal, going left)
    x_top = np.linspace(a - r, -a + r, num_panels_per_side + 1)[:-1]
    pts.extend([(x, b) for x in x_top])
    
    # 4. Top-left arc (90 to 180 degrees)
    theta_tl = np.linspace(np.pi/2, np.pi, num_panels_per_arc + 1)[:-1]
    x0, y0 = -a + r, b - r
    pts.extend([(x0 + r*np.cos(t), y0 + r*np.sin(t)) for t in theta_tl])
    
    # 5. Left edge (vertical, going down)
    y_left = np.linspace(b - r, -b + r, num_panels_per_side + 1)[:-1]
    pts.extend([(-a, y) for y in y_left])
    
    # 6. Bottom-left arc (180 to 270 degrees)
    theta_bl = np.linspace(np.pi, 3*np.pi/2, num_panels_per_arc + 1)[:-1]
    x0, y0 = -a + r, -b + r
    pts.extend([(x0 + r*np.cos(t), y0 + r*np.sin(t)) for t in theta_bl])
    
    # 7. Bottom edge (horizontal, going right)
    x_bottom = np.linspace(-a + r, a - r, num_panels_per_side + 1)[:-1]
    pts.extend([(x, -b) for x in x_bottom])
    
    # 8. Bottom-right arc (270 to 360 degrees)
    theta_br = np.linspace(3*np.pi/2, 2*np.pi, num_panels_per_arc + 1)[:-1]
    x0, y0 = a - r, -b + r
    pts.extend([(x0 + r*np.cos(t), y0 + r*np.sin(t)) for t in theta_br])

    # Shift to center
    pts = [(x + cx, y + cy) for x, y in pts]
    
    # Convert to 3D nodes
    nodes = np.array([[x, y, 0.0] for x, y in pts], dtype=np.float64)
    
    # Generate panels (connect consecutive points)
    num_nodes = len(nodes)
    panels = np.array([[i, (i + 1) % num_nodes] for i in range(num_nodes)], dtype=np.int32)
    
    component_ids = np.zeros(panels.shape[0], dtype=np.int32)
    
    return Mesh(nodes=nodes, panels=panels, dimension=2, component_ids=component_ids)
