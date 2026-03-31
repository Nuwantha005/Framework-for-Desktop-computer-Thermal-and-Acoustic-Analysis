"""
Mesh I/O utilities for 3D geometry.

Provides functions for:
- Generating parametric 3D meshes (sphere, box) via pygmsh
- Reading external mesh files (STL, MSH, UNV) via meshio
- Exporting meshes and solutions to VTK for ParaView
"""

from .sphere_generator import generate_sphere
from .stl_reader import read_stl, read_mesh
from .vtk_export import export_solution_vtk, export_mesh_vtk

__all__ = [
    "generate_sphere",
    "read_stl",
    "read_mesh",
    "export_solution_vtk",
    "export_mesh_vtk",
]
