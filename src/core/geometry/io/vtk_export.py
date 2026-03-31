"""
VTK export utilities for 3D meshes and solutions.

Exports Mesh3D and solution data to VTK formats for ParaView visualization.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..mesh3d import Mesh3D


def export_solution_vtk(
    mesh: "Mesh3D",
    path: str | Path,
    point_data: Optional[Dict[str, NDArray]] = None,
    cell_data: Optional[Dict[str, NDArray]] = None,
) -> None:
    """
    Export a 3D mesh with solution data to VTK file.
    
    Combines mesh geometry with cell_data already stored in the mesh
    plus any additional point/cell data provided.
    
    Args:
        mesh: Mesh3D object to export
        path: Output file path (.vtu, .vtk, or .vtp recommended)
        point_data: Optional dict of point (node) data arrays
        cell_data: Optional dict of additional cell (panel) data arrays
    
    Raises:
        ValueError: If mesh is not 3D
    
    Example:
        >>> from core.geometry import Mesh3D
        >>> from core.geometry.io import export_solution_vtk
        >>> mesh = generate_sphere(1.0, 16, 8)
        >>> mesh.cell_data['Cp'] = computed_cp
        >>> export_solution_vtk(mesh, 'sphere_solution.vtu')
    """
    if mesh.dimension != 3:
        raise ValueError(f"export_solution_vtk requires 3D mesh, got dimension={mesh.dimension}")
    
    # Use Mesh3D's built-in PyVista conversion
    polydata = mesh.to_pyvista()
    
    # Add any extra point data
    if point_data:
        for name, data in point_data.items():
            polydata.point_data[name] = data
    
    # Add any extra cell data (mesh.cell_data already added by to_pyvista)
    if cell_data:
        for name, data in cell_data.items():
            polydata.cell_data[name] = data
    
    # Save to file
    polydata.save(str(path))


def export_mesh_vtk(mesh: "Mesh3D", path: str | Path) -> None:
    """
    Export a 3D mesh geometry (without solution data) to VTK file.
    
    Args:
        mesh: Mesh3D object to export
        path: Output file path
    """
    mesh.save_vtk(str(path))
