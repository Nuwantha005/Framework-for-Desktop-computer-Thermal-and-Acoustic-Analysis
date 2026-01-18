"""
Geometric primitives: Point3D and Vector3D.

All geometry uses 3D arrays (z=0 for 2D problems) for seamless 2D→3D transition.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from typing import Union


@dataclass
class Point3D:
    """3D point (use z=0 for 2D problems)."""
    x: float
    y: float
    z: float = 0.0
    
    def to_array(self) -> NDArray[np.float64]:
        """Convert to NumPy array (3,)."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)
    
    @classmethod
    def from_array(cls, arr: NDArray[np.float64]) -> Point3D:
        """Create from NumPy array."""
        if arr.shape != (3,):
            raise ValueError(f"Expected array of shape (3,), got {arr.shape}")
        return cls(x=float(arr[0]), y=float(arr[1]), z=float(arr[2]))
    
    def distance_to(self, other: Point3D) -> float:
        """Euclidean distance to another point."""
        return np.linalg.norm(self.to_array() - other.to_array())
    
    def __add__(self, vector: Vector3D) -> Point3D:
        """Point + Vector = Point."""
        arr = self.to_array() + vector.to_array()
        return Point3D.from_array(arr)
    
    def __sub__(self, other: Point3D) -> Vector3D:
        """Point - Point = Vector."""
        arr = self.to_array() - other.to_array()
        return Vector3D.from_array(arr)
    
    def __repr__(self) -> str:
        return f"Point3D({self.x:.6f}, {self.y:.6f}, {self.z:.6f})"


@dataclass
class Vector3D:
    """3D vector (use z=0 for 2D problems)."""
    x: float
    y: float
    z: float = 0.0
    
    def to_array(self) -> NDArray[np.float64]:
        """Convert to NumPy array (3,)."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)
    
    @classmethod
    def from_array(cls, arr: NDArray[np.float64]) -> Vector3D:
        """Create from NumPy array."""
        if arr.shape != (3,):
            raise ValueError(f"Expected array of shape (3,), got {arr.shape}")
        return cls(x=float(arr[0]), y=float(arr[1]), z=float(arr[2]))
    
    def magnitude(self) -> float:
        """Vector magnitude (L2 norm)."""
        return float(np.linalg.norm(self.to_array()))
    
    def normalize(self) -> Vector3D:
        """Return unit vector in same direction."""
        mag = self.magnitude()
        if mag < 1e-14:
            raise ValueError("Cannot normalize zero vector")
        arr = self.to_array() / mag
        return Vector3D.from_array(arr)
    
    def dot(self, other: Vector3D) -> float:
        """Dot product."""
        return float(np.dot(self.to_array(), other.to_array()))
    
    def cross(self, other: Vector3D) -> Vector3D:
        """Cross product."""
        arr = np.cross(self.to_array(), other.to_array())
        return Vector3D.from_array(arr)
    
    def angle_to(self, other: Vector3D) -> float:
        """Angle to another vector in radians [0, π]."""
        cos_angle = self.dot(other) / (self.magnitude() * other.magnitude())
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Numerical safety
        return float(np.arccos(cos_angle))
    
    def __add__(self, other: Vector3D) -> Vector3D:
        """Vector addition."""
        arr = self.to_array() + other.to_array()
        return Vector3D.from_array(arr)
    
    def __sub__(self, other: Vector3D) -> Vector3D:
        """Vector subtraction."""
        arr = self.to_array() - other.to_array()
        return Vector3D.from_array(arr)
    
    def __mul__(self, scalar: float) -> Vector3D:
        """Scalar multiplication."""
        arr = self.to_array() * scalar
        return Vector3D.from_array(arr)
    
    def __rmul__(self, scalar: float) -> Vector3D:
        """Scalar multiplication (reversed)."""
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar: float) -> Vector3D:
        """Scalar division."""
        if abs(scalar) < 1e-14:
            raise ValueError("Division by zero")
        arr = self.to_array() / scalar
        return Vector3D.from_array(arr)
    
    def __neg__(self) -> Vector3D:
        """Negation."""
        return Vector3D.from_array(-self.to_array())
    
    def __repr__(self) -> str:
        return f"Vector3D({self.x:.6f}, {self.y:.6f}, {self.z:.6f})"


def rotation_matrix_z(angle_rad: float) -> NDArray[np.float64]:
    """
    3D rotation matrix about z-axis (for 2D transformations).
    
    Args:
        angle_rad: Rotation angle in radians (positive = CCW when viewed from +z)
    
    Returns:
        3x3 rotation matrix
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)


def rotation_matrix_xyz(rx: float, ry: float, rz: float) -> NDArray[np.float64]:
    """
    3D rotation matrix from Euler angles (XYZ convention).
    
    Args:
        rx, ry, rz: Rotation angles about x, y, z axes (radians)
    
    Returns:
        3x3 rotation matrix (R = Rz * Ry * Rx)
    """
    # Rotation about x-axis
    cx, sx = np.cos(rx), np.sin(rx)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    
    # Rotation about y-axis
    cy, sy = np.cos(ry), np.sin(ry)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    
    # Rotation about z-axis
    cz, sz = np.cos(rz), np.sin(rz)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    
    return Rz @ Ry @ Rx
