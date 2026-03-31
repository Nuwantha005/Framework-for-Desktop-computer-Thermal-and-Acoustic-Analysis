"""Geometry primitives, mesh, component, and scene."""

from .primitives import Point3D, Vector3D, rotation_matrix_z, rotation_matrix_xyz
from .mesh_base import MeshBase
from .mesh2d import Mesh2D
from .mesh3d import Mesh3D
from .mesh import Mesh  # Backward compatibility alias for Mesh2D
from .component import Transform, Component
from .scene import Scene

__all__ = [
    # Primitives
    "Point3D",
    "Vector3D",
    "rotation_matrix_z",
    "rotation_matrix_xyz",
    # Mesh classes
    "MeshBase",
    "Mesh2D",
    "Mesh3D",
    "Mesh",  # Alias for Mesh2D
    # Scene components
    "Transform",
    "Component",
    "Scene",
]
