"""Point-doublet approximation for actuator disk influence."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from core.geometry.mesh3d import Mesh3D


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
    exclude_self: bool = False,
) -> NDArray[np.float64]:
    """Compute velocity induced by disk panels using point doublets.

    Args:
        points: Evaluation points, shape ``(M, 3)``.
        disk_mesh: Disk panel mesh.
        doublet_strength: Potential jump on each disk panel.
        exclude_self: Skip panel self-influence when points align with disk
            centers and counts match.

    Returns:
        Induced velocity at points, shape ``(M, 3)``.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, 3)

    centers = np.asarray(disk_mesh.centers, dtype=np.float64)
    normals = np.asarray(disk_mesh.normals, dtype=np.float64)
    areas = np.asarray(disk_mesh.areas, dtype=np.float64)
    mu = np.asarray(doublet_strength, dtype=np.float64)

    velocities = np.zeros((points.shape[0], 3), dtype=np.float64)
    eps2 = 1e-16
    for j, center in enumerate(centers):
        moment = mu[j] * areas[j] * normals[j]
        r = points - center
        r2 = np.einsum("ij,ij->i", r, r)
        if exclude_self and points.shape[0] == centers.shape[0]:
            r2[j] = np.inf
        r2 = np.maximum(r2, eps2)
        r_mag = np.sqrt(r2)
        mdotr = r @ moment
        coeff = 1.0 / (4.0 * np.pi)
        velocities += coeff * (
            3.0 * r * mdotr[:, None] / (r_mag[:, None] ** 5)
            - moment / (r_mag[:, None] ** 3)
        )
    return velocities


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
