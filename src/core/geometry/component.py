"""
Component and Transform classes for scene assembly.

A Component is a distinct body with a local mesh and global placement transform.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from numpy.typing import NDArray

from .mesh import Mesh
from .primitives import rotation_matrix_z, rotation_matrix_xyz


@dataclass
class Transform:
    """
    3D affine transformation (works for 2D with z=0).
    
    Attributes:
        translation: Translation vector (3,)
        rotation_matrix: 3x3 rotation matrix
    """
    
    translation: NDArray[np.float64]      # (3,)
    rotation_matrix: NDArray[np.float64]  # (3, 3)
    
    def __post_init__(self):
        """Validate transform data."""
        if self.translation.shape != (3,):
            raise ValueError(f"translation must have shape (3,), got {self.translation.shape}")
        
        if self.rotation_matrix.shape != (3, 3):
            raise ValueError(
                f"rotation_matrix must have shape (3, 3), got {self.rotation_matrix.shape}"
            )
    
    @classmethod
    def identity(cls) -> Transform:
        """Create identity transform (no translation/rotation)."""
        return cls(
            translation=np.zeros(3, dtype=np.float64),
            rotation_matrix=np.eye(3, dtype=np.float64)
        )
    
    @classmethod
    def from_2d(cls, tx: float, ty: float, angle_deg: float) -> Transform:
        """
        Create 2D transform (rotation about z-axis).
        
        Args:
            tx, ty: Translation in x and y
            angle_deg: Rotation angle in degrees (positive = CCW)
        
        Returns:
            Transform object
        """
        translation = np.array([tx, ty, 0.0], dtype=np.float64)
        angle_rad = np.deg2rad(angle_deg)
        rotation_matrix = rotation_matrix_z(angle_rad)
        
        return cls(translation=translation, rotation_matrix=rotation_matrix)
    
    @classmethod
    def from_3d(cls, tx: float, ty: float, tz: float,
                rx_deg: float = 0.0, ry_deg: float = 0.0, rz_deg: float = 0.0) -> Transform:
        """
        Create 3D transform with Euler angles (XYZ convention).
        
        Args:
            tx, ty, tz: Translation
            rx_deg, ry_deg, rz_deg: Rotation angles about x, y, z axes (degrees)
        
        Returns:
            Transform object
        """
        translation = np.array([tx, ty, tz], dtype=np.float64)
        rx = np.deg2rad(rx_deg)
        ry = np.deg2rad(ry_deg)
        rz = np.deg2rad(rz_deg)
        rotation_matrix = rotation_matrix_xyz(rx, ry, rz)
        
        return cls(translation=translation, rotation_matrix=rotation_matrix)
    
    def apply_to_point(self, point: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Apply transform to a point.
        
        Args:
            point: Point coordinates (3,) or (N, 3)
        
        Returns:
            Transformed point(s)
        """
        if point.ndim == 1:
            # Single point
            return self.rotation_matrix @ point + self.translation
        else:
            # Multiple points (N, 3)
            return (self.rotation_matrix @ point.T).T + self.translation
    
    def apply_to_vector(self, vector: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Apply rotation to a vector (no translation).
        
        Args:
            vector: Vector(s) (3,) or (N, 3)
        
        Returns:
            Rotated vector(s)
        """
        if vector.ndim == 1:
            return self.rotation_matrix @ vector
        else:
            return (self.rotation_matrix @ vector.T).T
    
    def to_matrix_4x4(self) -> NDArray[np.float64]:
        """
        Convert to 4x4 homogeneous transformation matrix.
        
        Returns:
            4x4 matrix
        """
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = self.rotation_matrix
        mat[:3, 3] = self.translation
        return mat
    
    def __repr__(self) -> str:
        return (
            f"Transform(translation={self.translation}, "
            f"rotation=\n{self.rotation_matrix})"
        )


@dataclass
class Component:
    """
    A distinct body with local mesh and global placement.
    
    Attributes:
        name: Component identifier
        local_mesh: Mesh in local coordinates
        transform: Placement transform (local → global)
        bc_type: Boundary condition type ('wall', 'inlet', 'outlet', etc.)
        bc_value: BC value (e.g., normal velocity for inlet)
        metadata: Arbitrary additional data
    """
    
    name: str
    local_mesh: Mesh
    transform: Transform
    bc_type: str = "wall"
    bc_value: Optional[float] = None
    metadata: dict = None
    
    def __post_init__(self):
        """Initialize metadata if None."""
        if self.metadata is None:
            self.metadata = {}
    
    def get_global_mesh(self, component_id: int) -> Mesh:
        """
        Apply transform to local mesh and return global mesh.
        
        Args:
            component_id: ID to assign to all panels in this component
        
        Returns:
            Mesh in global coordinates with updated component_ids
        """
        # Transform nodes
        global_nodes = self.transform.apply_to_point(self.local_mesh.nodes)
        
        # Transform normals and tangents (rotation only)
        global_normals = None
        global_tangents = None
        
        if self.local_mesh.normals is not None:
            global_normals = self.transform.apply_to_vector(self.local_mesh.normals)
        
        if self.local_mesh.tangents is not None:
            global_tangents = self.transform.apply_to_vector(self.local_mesh.tangents)
        
        # Transform centers
        global_centers = None
        if self.local_mesh.centers is not None:
            global_centers = self.transform.apply_to_point(self.local_mesh.centers)
        
        # Create new mesh with global coordinates
        # Areas don't change under rigid transform
        global_mesh = Mesh(
            nodes=global_nodes,
            panels=self.local_mesh.panels.copy(),
            dimension=self.local_mesh.dimension,
            component_ids=np.full(self.local_mesh.num_panels, component_id, dtype=np.int32)
        )
        
        # Override computed geometry with transformed values
        global_mesh.centers = global_centers
        global_mesh.normals = global_normals
        global_mesh.tangents = global_tangents
        global_mesh.areas = self.local_mesh.areas.copy() if self.local_mesh.areas is not None else None
        
        return global_mesh
    
    def __repr__(self) -> str:
        return (
            f"Component(name='{self.name}', "
            f"panels={self.local_mesh.num_panels}, "
            f"bc_type='{self.bc_type}')"
        )
