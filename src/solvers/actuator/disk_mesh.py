"""Actuator disk mesh generation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from core.geometry.mesh3d import Mesh3D


def _orthonormal_basis(normal: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    n = np.asarray(normal, dtype=np.float64)
    n_norm = np.linalg.norm(n)
    if n_norm <= 1e-14:
        raise ValueError("Disk normal must be nonzero")
    n = n / n_norm

    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(ref, n))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    t1 = np.cross(ref, n)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    t2 /= np.linalg.norm(t2)
    return t1, t2, n


def generate_actuator_disk_mesh(
    center: tuple[float, float, float] | NDArray[np.float64],
    normal: tuple[float, float, float] | NDArray[np.float64],
    radius: float,
    n_r: int,
    n_theta: int,
) -> Mesh3D:
    """Generate a polar quadrilateral mesh for a thin actuator disk.

    Args:
        center: Disk center coordinates.
        normal: Disk normal and positive flow direction.
        radius: Disk radius [m].
        n_r: Number of radial subdivisions.
        n_theta: Number of azimuthal subdivisions.

    Returns:
        Mesh3D containing quad panels.
    """
    if radius <= 0:
        raise ValueError("Disk radius must be positive")
    if n_r < 1:
        raise ValueError("n_r must be at least 1")
    if n_theta < 3:
        raise ValueError("n_theta must be at least 3")

    center_arr = np.asarray(center, dtype=np.float64)
    t1, t2, n = _orthonormal_basis(np.asarray(normal, dtype=np.float64))

    radii = np.linspace(0.0, radius, n_r + 1)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)

    nodes = [center_arr]
    ring_indices: list[list[int]] = []
    for i in range(1, n_r + 1):
        ring = []
        for angle in theta:
            point = center_arr + radii[i] * (np.cos(angle) * t1 + np.sin(angle) * t2)
            nodes.append(point)
            ring.append(len(nodes) - 1)
        ring_indices.append(ring)

    panels = []
    first_ring = ring_indices[0]
    for j in range(n_theta):
        panels.append([0, first_ring[j], first_ring[(j + 1) % n_theta], 0])

    for i in range(1, n_r):
        inner = ring_indices[i - 1]
        outer = ring_indices[i]
        for j in range(n_theta):
            panels.append([
                inner[j],
                outer[j],
                outer[(j + 1) % n_theta],
                inner[(j + 1) % n_theta],
            ])

    mesh = Mesh3D(
        nodes=np.asarray(nodes, dtype=np.float64),
        panels=np.asarray(panels, dtype=np.int32),
        component_ids=np.zeros(len(panels), dtype=np.int32),
    )

    # Mesh3D normal orientation follows node ordering; force the configured normal.
    if np.mean(mesh.normals @ n) < 0:
        mesh.panels = mesh.panels[:, [0, 3, 2, 1]].astype(np.int32)
        mesh.compute_geometry()
    mesh.normals[:] = n
    mesh.tangent2 = np.cross(mesh.normals, mesh.tangent1)
    return mesh


def generate_rectangular_boundary_mesh(
    center: tuple[float, float, float] | NDArray[np.float64],
    normal: tuple[float, float, float] | NDArray[np.float64],
    width: float,
    height: float,
    n_w: int,
    n_h: int,
) -> Mesh3D:
    """Generate a Cartesian quadrilateral mesh for a rectangular boundary.

    Args:
        center: Boundary center coordinates.
        normal: Boundary normal direction.
        width: Width [m].
        height: Height [m].
        n_w: Number of width subdivisions.
        n_h: Number of height subdivisions.

    Returns:
        Mesh3D containing quad panels.
    """
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive")
    if n_w < 1 or n_h < 1:
        raise ValueError("Subdivisions must be at least 1")

    center_arr = np.asarray(center, dtype=np.float64)
    t1, t2, n = _orthonormal_basis(np.asarray(normal, dtype=np.float64))

    xs = np.linspace(-width / 2, width / 2, n_w + 1)
    ys = np.linspace(-height / 2, height / 2, n_h + 1)

    nodes = []
    for y in ys:
        for x in xs:
            nodes.append(center_arr + x * t1 + y * t2)

    panels = []
    for j in range(n_h):
        for i in range(n_w):
            p0 = j * (n_w + 1) + i
            p1 = p0 + 1
            p2 = (j + 1) * (n_w + 1) + i + 1
            p3 = (j + 1) * (n_w + 1) + i
            panels.append([p0, p1, p2, p3])

    mesh = Mesh3D(
        nodes=np.asarray(nodes, dtype=np.float64),
        panels=np.asarray(panels, dtype=np.int32),
        component_ids=np.zeros(len(panels), dtype=np.int32),
    )

    if np.mean(mesh.normals @ n) < 0:
        mesh.panels = mesh.panels[:, [0, 3, 2, 1]].astype(np.int32)
        mesh.compute_geometry()
    mesh.normals[:] = n
    mesh.tangent2 = np.cross(mesh.normals, mesh.tangent1)
    return mesh
    """Generate a polar quadrilateral mesh for a thin actuator disk.

    Args:
        center: Disk center coordinates.
        normal: Disk normal and positive flow direction.
        radius: Disk radius [m].
        n_r: Number of radial subdivisions.
        n_theta: Number of azimuthal subdivisions.

    Returns:
        Mesh3D containing quad panels.
    """
    if radius <= 0:
        raise ValueError("Disk radius must be positive")
    if n_r < 1:
        raise ValueError("n_r must be at least 1")
    if n_theta < 3:
        raise ValueError("n_theta must be at least 3")

    center_arr = np.asarray(center, dtype=np.float64)
    t1, t2, n = _orthonormal_basis(np.asarray(normal, dtype=np.float64))

    radii = np.linspace(0.0, radius, n_r + 1)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)

    nodes = [center_arr]
    ring_indices: list[list[int]] = []
    for i in range(1, n_r + 1):
        ring = []
        for angle in theta:
            point = center_arr + radii[i] * (np.cos(angle) * t1 + np.sin(angle) * t2)
            nodes.append(point)
            ring.append(len(nodes) - 1)
        ring_indices.append(ring)

    panels = []
    first_ring = ring_indices[0]
    for j in range(n_theta):
        panels.append([0, first_ring[j], first_ring[(j + 1) % n_theta], 0])

    for i in range(1, n_r):
        inner = ring_indices[i - 1]
        outer = ring_indices[i]
        for j in range(n_theta):
            panels.append([
                inner[j],
                outer[j],
                outer[(j + 1) % n_theta],
                inner[(j + 1) % n_theta],
            ])

    mesh = Mesh3D(
        nodes=np.asarray(nodes, dtype=np.float64),
        panels=np.asarray(panels, dtype=np.int32),
        component_ids=np.zeros(len(panels), dtype=np.int32),
    )

    # Mesh3D normal orientation follows node ordering; force the configured normal.
    if np.mean(mesh.normals @ n) < 0:
        mesh.panels = mesh.panels[:, [0, 3, 2, 1]].astype(np.int32)
        mesh.compute_geometry()
    mesh.normals[:] = n
    mesh.tangent2 = np.cross(mesh.normals, mesh.tangent1)
    return mesh
