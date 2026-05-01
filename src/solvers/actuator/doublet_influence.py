"""Constant-strength doublet approximation for actuator disk influence."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from core.geometry.mesh3d import Mesh3D
from solvers.panel3d.influences.doublet3d import compute_all_doublet_velocities


def pressure_jump_to_doublet_strength(
    pressure_rise: float,
    density: float,
    reference_velocity: float,
    characteristic_length: float = 1.0,
) -> float:
    """Map pressure rise to a potential jump/doublet strength.

    The simple ADM represents the fan as a prescribed potential jump. For the
    first coupled implementation, the jump is scaled by the local dynamic
    velocity scale and a disk-scale length. For stationary fan-driven cases
    the velocity scale is derived from the fan curve instead of freestream.
    """
    u_ref = max(abs(float(reference_velocity)), 1e-8)
    length = max(abs(float(characteristic_length)), 1e-8)
    return float(pressure_rise) * length / (float(density) * u_ref)


def compute_point_doublet_velocity(
    points: NDArray[np.float64],
    disk_mesh: Mesh3D,
    doublet_strength: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute velocity induced by disk panels using constant-strength doublets.
    
    A constant-strength doublet panel is mathematically equivalent to a vortex
    ring around its perimeter, avoiding the leakage issues of point-doublets.

    Args:
        points: Evaluation points, shape ``(M, 3)``.
        disk_mesh: Disk panel mesh.
        doublet_strength: Potential jump on each disk panel.

    Returns:
        Induced velocity at points, shape ``(M, 3)``.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, 3)

    return compute_all_doublet_velocities(
        points=points,
        vertices=disk_mesh.nodes,
        panels=disk_mesh.panels,
        mu=doublet_strength,
    )


def compute_disk_normal_velocity(
    velocity: NDArray[np.float64],
    disk_mesh: Mesh3D,
) -> NDArray[np.float64]:
    """Project velocity vectors onto disk panel normals."""
    return np.einsum("ij,ij->i", velocity, disk_mesh.normals)


def integrate_flow_rate(
    normal_velocity: NDArray[np.float64],
    disk_mesh: Mesh3D,
) -> float:
    """Integrate volumetric flow rate through the disk."""
    return float(np.sum(normal_velocity * disk_mesh.areas))
