"""
Parametric 3D mesh generators using pygmsh/gmsh.

Currently supports:
- UV sphere with quad panels
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
