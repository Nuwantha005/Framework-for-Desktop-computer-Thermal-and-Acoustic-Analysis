"""
Parametric 3D mesh generators.

Currently supports:
- UV sphere with quad panels
- Open cylinder shell with quad panels
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
from numpy.typing import NDArray

from ..mesh3d import Mesh3D


def generate_sphere(
    n_theta: int = 16,
    n_phi: int = 32,
    radius: float = 1.0,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> Mesh3D:
    """
    Generate UV sphere mesh with quadrilateral panels.
    
    Creates a sphere using UV parameterization (latitude/longitude grid).
    The mesh consists of quad panels, with degenerate quads at the poles.
    
    Args:
        n_theta: Number of divisions in polar angle (latitude bands, from pole to pole)
        n_phi: Number of divisions in azimuthal angle (longitude, around equator)
        radius: Sphere radius
        center: Center coordinates (cx, cy, cz)
    
    Returns:
        Mesh3D with approximately (n_theta - 1) * n_phi quad panels
        
    Example:
        >>> mesh = generate_sphere(n_theta=16, n_phi=32, radius=0.5)
        >>> print(mesh)
        Mesh3D(nodes=..., panels=..., components=1)
    
    Note:
        - n_theta=16, n_phi=32 gives ~480 panels (good for testing)
        - n_theta=32, n_phi=64 gives ~2016 panels (production quality)
        - Poles use degenerate quads (two coincident vertices)
    """
    cx, cy, cz = center
    
    # Generate node grid
    # theta: polar angle from 0 (north pole) to pi (south pole)
    # phi: azimuthal angle from 0 to 2*pi
    theta = np.linspace(0, np.pi, n_theta + 1)
    phi = np.linspace(0, 2 * np.pi, n_phi + 1)[:-1]  # Exclude duplicate at 2π
    
    # Create meshgrid
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
    
    # Spherical to Cartesian
    X = cx + radius * np.sin(THETA) * np.cos(PHI)
    Y = cy + radius * np.sin(THETA) * np.sin(PHI)
    Z = cz + radius * np.cos(THETA)
    
    # Flatten to node array (n_theta+1) * n_phi nodes
    nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # Generate quad panels
    # Node indexing: node[i, j] = i * n_phi + j
    panels = []
    
    for i in range(n_theta):
        for j in range(n_phi):
            # Current row indices
            n00 = i * n_phi + j
            n01 = i * n_phi + (j + 1) % n_phi
            # Next row indices
            n10 = (i + 1) * n_phi + j
            n11 = (i + 1) * n_phi + (j + 1) % n_phi
            
            # Quad vertices in CCW order (viewed from outside)
            # For outward-pointing normals: go around CCW
            panels.append([n00, n01, n11, n10])
    
    panels = np.array(panels, dtype=np.int32)
    component_ids = np.zeros(len(panels), dtype=np.int32)
    
    return Mesh3D(
        nodes=nodes,
        panels=panels,
        component_ids=component_ids
    )


def generate_cylinder(
    n_theta: int = 32,
    n_length: int = 16,
    radius: float = 1.0,
    length: float = 1.0,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    axis: str = "z",
) -> Mesh3D:
    """
    Generate an open cylinder shell mesh with quadrilateral panels.

    The cylinder is aligned to the specified axis and centered at `center`.
    This creates only the lateral surface (no end caps).

    Args:
        n_theta: Number of divisions around circumference
        n_length: Number of divisions along cylinder axis
        radius: Cylinder radius
        length: Cylinder length
        center: Center coordinates (cx, cy, cz)
        axis: Axis of cylinder ("x", "y", or "z")

    Returns:
        Mesh3D with n_theta * n_length quad panels
    """
    if n_theta < 3:
        raise ValueError("n_theta must be >= 3")
    if n_length < 1:
        raise ValueError("n_length must be >= 1")
    axis = axis.lower()
    if axis not in {"x", "y", "z"}:
        raise ValueError(f"Unsupported axis '{axis}'. Use 'x', 'y', or 'z'.")

    cx, cy, cz = center
    phi = np.linspace(0.0, 2.0 * np.pi, n_theta + 1)[:-1]
    s = np.linspace(-0.5 * length, 0.5 * length, n_length + 1)

    S, PHI = np.meshgrid(s, phi, indexing="ij")

    if axis == "z":
        X = cx + radius * np.cos(PHI)
        Y = cy + radius * np.sin(PHI)
        Z = cz + S
    elif axis == "x":
        X = cx + S
        Y = cy + radius * np.cos(PHI)
        Z = cz + radius * np.sin(PHI)
    else:  # axis == "y"
        X = cx + radius * np.cos(PHI)
        Y = cy + S
        Z = cz + radius * np.sin(PHI)

    nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    panels = []
    for i in range(n_length):
        for j in range(n_theta):
            n00 = i * n_theta + j
            n01 = i * n_theta + (j + 1) % n_theta
            n10 = (i + 1) * n_theta + j
            n11 = (i + 1) * n_theta + (j + 1) % n_theta
            panels.append([n00, n10, n11, n01])

    panels = np.array(panels, dtype=np.int32)
    component_ids = np.zeros(len(panels), dtype=np.int32)

    return Mesh3D(
        nodes=nodes,
        panels=panels,
        component_ids=component_ids
    )



def generate_thick_cylinder(
    n_theta: int = 32,
    n_length: int = 16,
    radius_inner: float = 0.95,
    radius_outer: float = 1.0,
    length: float = 2.0,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    axis: str = "z",
) -> Mesh3D:
    """
    Generate a thick cylinder (pipe) mesh with quadrilateral panels.

    The cylinder has an inner wall, an outer wall, and is capped at both ends
    to form a closed solid volume.

    Args:
        n_theta: Number of divisions around circumference
        n_length: Number of divisions along cylinder axis
        radius_inner: Inner radius
        radius_outer: Outer radius
        length: Cylinder length
        center: Center coordinates (x, y, z)
        axis: Axis of alignment ('x', 'y', or 'z')

    Returns:
        Mesh3D with quad panels
    """
    if n_theta < 3:
        raise ValueError("n_theta must be >= 3")
    if n_length < 1:
        raise ValueError("n_length must be >= 1")
    if radius_inner >= radius_outer:
        raise ValueError("radius_outer must be greater than radius_inner")
    
    axis = axis.lower()
    if axis not in {"x", "y", "z"}:
        raise ValueError(f"Unsupported axis '{axis}'. Use 'x', 'y', or 'z'.")

    cx, cy, cz = center
    
    theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    z = np.linspace(-length/2, length/2, n_length + 1)
    
    nodes = []
    
    # Helper to create points depending on axis
    def make_pt(r, t, zi):
        X, Y, Z = r*np.cos(t), r*np.sin(t), zi
        if axis == "z":
            return [cx + X, cy + Y, cz + Z]
        elif axis == "x":
            return [cx + Z, cy + X, cz + Y]
        else: # y
            return [cx + X, cy + Z, cz + Y]
            
    # Generate all inner nodes
    for zi in z:
        for t in theta:
            nodes.append(make_pt(radius_inner, t, zi))
    inner_offset = 0
    
    # Generate all outer nodes
    for zi in z:
        for t in theta:
            nodes.append(make_pt(radius_outer, t, zi))
    outer_offset = len(z) * n_theta
    
    nodes = np.array(nodes, dtype=np.float64)
    panels = []
    
    # Inner wall panels (normals point inward towards r=0)
    for i in range(n_length):
        for j in range(n_theta):
            n00 = inner_offset + i * n_theta + j
            n10 = inner_offset + i * n_theta + (j + 1) % n_theta
            n01 = inner_offset + (i + 1) * n_theta + j
            n11 = inner_offset + (i + 1) * n_theta + (j + 1) % n_theta
            
            # To face INWARD, viewing from origin, CCW: n00 -> n10 -> n11 -> n01
            panels.append([n00, n10, n11, n01])
            
    # Outer wall panels (normals point outward)
    for i in range(n_length):
        for j in range(n_theta):
            n00 = outer_offset + i * n_theta + j
            n10 = outer_offset + i * n_theta + (j + 1) % n_theta
            n01 = outer_offset + (i + 1) * n_theta + j
            n11 = outer_offset + (i + 1) * n_theta + (j + 1) % n_theta
            
            # To face OUTWARD, viewing from outside, CCW: n00 -> n01 -> n11 -> n10
            panels.append([n00, n01, n11, n10])
            
    # Bottom lip (z = -length/2) (normals point -z)
    for j in range(n_theta):
        in_0 = inner_offset + j
        in_1 = inner_offset + (j + 1) % n_theta
        out_0 = outer_offset + j
        out_1 = outer_offset + (j + 1) % n_theta
        
        # To face -z, viewing from bottom, CCW: in_0 -> out_0 -> out_1 -> in_1
        panels.append([in_0, out_0, out_1, in_1])
        
    # Top lip (z = length/2) (normals point +z)
    inner_top = inner_offset + n_length * n_theta
    outer_top = outer_offset + n_length * n_theta
    for j in range(n_theta):
        in_0 = inner_top + j
        in_1 = inner_top + (j + 1) % n_theta
        out_0 = outer_top + j
        out_1 = outer_top + (j + 1) % n_theta
        
        # To face +z, viewing from top, CCW: in_0 -> in_1 -> out_1 -> out_0
        panels.append([in_0, in_1, out_1, out_0])
        
    panels = np.array(panels, dtype=np.int32)
    component_ids = np.zeros(len(panels), dtype=np.int32)
    
    return Mesh3D(
        nodes=nodes,
        panels=panels,
        component_ids=component_ids
    )


def generate_box(
    nx: int = 4,
    ny: int = 4,
    nz: int = 4,
    width: float = 1.0,
    height: float = 1.0,
    depth: float = 1.0,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> Mesh3D:
    """
    Generate box mesh with quadrilateral panels.
    
    Creates a rectangular box with quad panels on each face.
    
    Args:
        nx: Number of panels in x-direction per face
        ny: Number of panels in y-direction per face
        nz: Number of panels in z-direction per face
        width: Box width (x-extent)
        height: Box height (y-extent)
        depth: Box depth (z-extent)
        center: Center coordinates (cx, cy, cz)
    
    Returns:
        Mesh3D with 2*(nx*ny + ny*nz + nx*nz) quad panels
    
    Note:
        This is a stub for future implementation.
    """
    raise NotImplementedError("Box generation not yet implemented. Use sphere for now.")
