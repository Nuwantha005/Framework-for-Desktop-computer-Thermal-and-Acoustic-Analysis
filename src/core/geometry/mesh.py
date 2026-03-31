"""
Backward compatibility shim for mesh module.

This module re-exports Mesh2D as Mesh for backward compatibility with
existing code that imports from `core.geometry.mesh`.

New code should import directly from:
- `core.geometry.mesh2d` for Mesh2D
- `core.geometry.mesh3d` for Mesh3D
- `core.geometry.mesh_base` for MeshBase
"""

from .mesh2d import Mesh2D

# Backward compatibility alias
Mesh = Mesh2D

__all__ = ["Mesh", "Mesh2D"]
