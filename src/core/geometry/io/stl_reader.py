"""
STL and mesh file reader using meshio.

Reads external mesh files (STL, MSH, UNV, etc.) and converts to Mesh3D.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
from numpy.typing import NDArray


def read_stl(path: str | Path, component_id: int = 0):
    """
    Read an STL file and convert to Mesh3D.
    
    Note: STL files contain triangles. This function will attempt to
    merge adjacent triangles into quads where possible. If quad 
    conversion is not possible, it raises NotImplementedError.
    
    Args:
        path: Path to STL file
        component_id: Component ID to assign to all panels
    
    Returns:
        Mesh3D object
    
    Raises:
        ImportError: If meshio is not installed
        NotImplementedError: Quad-only meshes required, STL has triangles
    """
    try:
        import meshio
    except ImportError:
        raise ImportError("meshio is required: pip install meshio")
    
    mesh_data = meshio.read(str(path))
    
    # For now, only support quad meshes
    # TODO: Implement triangle-to-quad merging or triangle support
    if "quad" not in mesh_data.cells_dict:
        raise NotImplementedError(
            "STL files contain triangles. Quad-only meshes are currently required. "
            "Use a mesh converter or generate parametric meshes instead."
        )
    
    return _meshio_to_mesh3d(mesh_data, component_id)


def read_mesh(path: str | Path, component_id: int = 0):
    """
    Read a mesh file (any format supported by meshio) and convert to Mesh3D.
    
    Supports formats: MSH (Gmsh), VTK, VTU, UNV, PLY, OBJ, and more.
    See meshio documentation for full format list.
    
    Args:
        path: Path to mesh file
        component_id: Component ID to assign to all panels
    
    Returns:
        Mesh3D object
    
    Raises:
        ImportError: If meshio is not installed
        ValueError: If mesh does not contain quad cells
    """
    try:
        import meshio
    except ImportError:
        raise ImportError("meshio is required: pip install meshio")
    
    mesh_data = meshio.read(str(path))
    return _meshio_to_mesh3d(mesh_data, component_id)


def _meshio_to_mesh3d(mesh_data, component_id: int):
    """
    Convert meshio mesh to Mesh3D.
    
    Args:
        mesh_data: meshio.Mesh object
        component_id: Component ID to assign to all panels
    
    Returns:
        Mesh3D object
    
    Raises:
        ValueError: If mesh does not contain quad cells
    """
    from ..mesh3d import Mesh3D
    
    # Extract nodes
    nodes = np.asarray(mesh_data.points, dtype=np.float64)
    if nodes.shape[1] == 2:
        # Promote 2D points to 3D
        nodes = np.column_stack([nodes, np.zeros(len(nodes))])
    
    # Extract quad cells
    if "quad" not in mesh_data.cells_dict:
        available = list(mesh_data.cells_dict.keys())
        raise ValueError(
            f"Mesh must contain quad cells. Found: {available}. "
            "Consider using a mesh generator with quad output."
        )
    
    panels = np.asarray(mesh_data.cells_dict["quad"], dtype=np.int32)
    component_ids = np.full(len(panels), component_id, dtype=np.int32)
    
    return Mesh3D(
        nodes=nodes,
        panels=panels,
        component_ids=component_ids,
    )
