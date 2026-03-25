"""
Coordinate transformation and interpolation for BL comparison.

This module handles the transformation from BL solver's (s, y) coordinate
system to global (x, y) coordinates, and interpolates Fluent velocity
data onto the BL solver grid.

The key operation is:
    For each (s[i], y[j]) in the BL solver grid:
    1. Find the panel centre at arc-length s[i]
    2. Compute the outward normal at that panel
    3. Global coordinates: (X, Y) = (x_panel, y_panel) + y[j] * normal

This allows direct comparison of velocity fields between the BL solver
(which works in boundary-layer coordinates) and Fluent (which provides
velocity in global Cartesian coordinates).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

if TYPE_CHECKING:
    from core.geometry import Mesh
    from solvers.boundary_layer.field import BLFieldData
    from solvers.boundary_layer.runner import BoundaryLayerPathResult
    from validation.adapters.fluent.data_types import FluentFieldData


@dataclass
class InterpolatedBLField:
    """Interpolated Fluent velocity field on BL solver grid.

    This matches the structure of :class:`BLFieldData` but contains
    Fluent-derived velocities instead of reconstructed BL profiles.

    Attributes:
        s: Arc-length stations [m], shape (M,).
        y: Wall-normal coordinates per station [m], shape (M, Ny).
        u: Tangential velocity from Fluent [m/s], shape (M, Ny).
        delta: BL thickness per station [m], shape (M,).
            (Copied from panel-method result for reference)
        Ue: Edge velocity per station [m/s], shape (M,).
            (Interpolated from Fluent)
        source: Identifier string ("fluent").
    """

    s: NDArray[np.float64]
    y: NDArray[np.float64]
    u: NDArray[np.float64]
    delta: NDArray[np.float64]
    Ue: NDArray[np.float64]
    source: str = "fluent"

    @property
    def shape(self) -> Tuple[int, int]:
        """(num_stations, num_y_points)."""
        return self.u.shape


class BLFieldInterpolator:
    """Transform BL solver grid to global coordinates and interpolate Fluent.

    This class performs the coordinate transformation from the BL solver's
    (s, y) coordinate system to global Cartesian coordinates, then
    interpolates Fluent velocity data onto these points.

    Args:
        bl_path: BL solver path result containing arc-length stations
            and panel geometry.
        mesh: Panel method mesh with normals.

    Example::

        interpolator = BLFieldInterpolator(bl_result.upper, case.mesh)

        # Transform BL grid to global coordinates
        X, Y = interpolator.transform_to_global(bl_field)

        # Interpolate Fluent velocity
        fluent_u = interpolator.interpolate_fluent_velocity(
            X, Y, fluent_field_data, bl_path
        )
    """

    def __init__(
        self,
        bl_path: "BoundaryLayerPathResult",
        surface_x: NDArray[np.float64],
        surface_y: NDArray[np.float64],
        panel_normals: NDArray[np.float64],
    ) -> None:
        """
        Args:
            bl_path: BL path result with panel_indices, s, x, y arrays.
            surface_x: Full body x-coordinates (all panels).
            surface_y: Full body y-coordinates (all panels).
            panel_normals: Outward normals for all panels, shape (M, 2) or (M, 3).
        """
        self.bl_path = bl_path
        self.surface_x = surface_x
        self.surface_y = surface_y
        self.panel_normals = panel_normals[:, :2]  # Keep only 2D

        # Extract panel data for this path
        self.panel_indices = bl_path.panel_indices
        self.path_x = bl_path.x
        self.path_y = bl_path.y
        self.path_s = bl_path.s

        # Compute path tangents and normals
        self._compute_path_geometry()

    def _compute_path_geometry(self) -> None:
        """Compute tangent and normal vectors along the path."""
        M = len(self.path_x)
        self.tangents = np.zeros((M, 2), dtype=np.float64)
        self.normals = np.zeros((M, 2), dtype=np.float64)

        for i in range(M):
            # Compute tangent from neighboring points
            if i == 0:
                tx = self.path_x[1] - self.path_x[0]
                ty = self.path_y[1] - self.path_y[0]
            elif i == M - 1:
                tx = self.path_x[M - 1] - self.path_x[M - 2]
                ty = self.path_y[M - 1] - self.path_y[M - 2]
            else:
                tx = self.path_x[i + 1] - self.path_x[i - 1]
                ty = self.path_y[i + 1] - self.path_y[i - 1]

            t_len = np.sqrt(tx * tx + ty * ty)
            if t_len > 1e-12:
                tx /= t_len
                ty /= t_len

            self.tangents[i] = [tx, ty]

            # Use mesh normals directly if available
            panel_idx = self.panel_indices[i]
            self.normals[i] = self.panel_normals[panel_idx]

    def transform_to_global(
        self,
        field: "BLFieldData",
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Transform BL solver (s, y) grid to global (X, Y) coordinates.

        For each grid point (s[i], y[i,j]):
            X[i,j] = x_panel[i] + y[i,j] * normal_x[i]
            Y[i,j] = y_panel[i] + y[i,j] * normal_y[i]

        Args:
            field: BL field data with s and y arrays.

        Returns:
            Tuple of (X, Y) global coordinate arrays, same shape as field.u.
        """
        M, Ny = field.u.shape
        X = np.zeros((M, Ny), dtype=np.float64)
        Y = np.zeros((M, Ny), dtype=np.float64)

        # Map field.s to path indices
        # field.s may not match exactly with self.path_s due to filtering
        # Use interpolation to find corresponding path positions

        for i in range(M):
            s_i = field.s[i]

            # Find interpolated position along path
            idx = np.searchsorted(self.path_s, s_i)
            if idx == 0:
                x_base = self.path_x[0]
                y_base = self.path_y[0]
                nx = self.normals[0, 0]
                ny = self.normals[0, 1]
            elif idx >= len(self.path_s):
                x_base = self.path_x[-1]
                y_base = self.path_y[-1]
                nx = self.normals[-1, 0]
                ny = self.normals[-1, 1]
            else:
                # Linear interpolation
                t = (s_i - self.path_s[idx - 1]) / (
                    self.path_s[idx] - self.path_s[idx - 1] + 1e-30
                )
                t = np.clip(t, 0.0, 1.0)

                x_base = (1 - t) * self.path_x[idx - 1] + t * self.path_x[idx]
                y_base = (1 - t) * self.path_y[idx - 1] + t * self.path_y[idx]
                nx = (1 - t) * self.normals[idx - 1, 0] + t * self.normals[idx, 0]
                ny = (1 - t) * self.normals[idx - 1, 1] + t * self.normals[idx, 1]

                # Renormalize
                n_len = np.sqrt(nx * nx + ny * ny)
                if n_len > 1e-12:
                    nx /= n_len
                    ny /= n_len

            # Transform each y point
            for j in range(Ny):
                y_j = field.y[i, j]
                X[i, j] = x_base + y_j * nx
                Y[i, j] = y_base + y_j * ny

        return X, Y

    def interpolate_fluent_velocity(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        fluent_field: "FluentFieldData",
    ) -> NDArray[np.float64]:
        """Interpolate Fluent velocity and project to tangential component.

        Args:
            X: Global x-coordinates, shape (M, Ny).
            Y: Global y-coordinates, shape (M, Ny).
            fluent_field: Fluent field data with scattered velocity.

        Returns:
            Tangential velocity Vt at each grid point, shape (M, Ny).
        """
        M, Ny = X.shape

        # Build interpolator from Fluent scattered data
        fluent_points = fluent_field.points  # (N, 2)
        vx_interp = LinearNDInterpolator(fluent_points, fluent_field.vx)
        vy_interp = LinearNDInterpolator(fluent_points, fluent_field.vy)

        # Fallback interpolators
        vx_nearest = NearestNDInterpolator(fluent_points, fluent_field.vx)
        vy_nearest = NearestNDInterpolator(fluent_points, fluent_field.vy)

        # Flatten for interpolation
        X_flat = X.ravel()
        Y_flat = Y.ravel()
        query_points = np.column_stack([X_flat, Y_flat])

        # Interpolate velocity components
        vx = vx_interp(query_points)
        vy = vy_interp(query_points)

        # Fill NaN with nearest neighbor
        nan_mask = np.isnan(vx)
        if np.any(nan_mask):
            vx[nan_mask] = vx_nearest(query_points[nan_mask])
            vy[nan_mask] = vy_nearest(query_points[nan_mask])

        vx = vx.reshape(M, Ny)
        vy = vy.reshape(M, Ny)

        # Project onto tangent direction for each station
        Vt = np.zeros((M, Ny), dtype=np.float64)

        for i in range(M):
            # Get tangent for this station (interpolated)
            s_i = self.bl_path.s[min(i, len(self.bl_path.s) - 1)]
            idx = np.searchsorted(self.path_s, s_i)
            idx = np.clip(idx, 0, len(self.tangents) - 1)

            if idx > 0 and idx < len(self.path_s):
                t = (s_i - self.path_s[idx - 1]) / (
                    self.path_s[idx] - self.path_s[idx - 1] + 1e-30
                )
                t = np.clip(t, 0.0, 1.0)
                tx = (1 - t) * self.tangents[idx - 1, 0] + t * self.tangents[idx, 0]
                ty = (1 - t) * self.tangents[idx - 1, 1] + t * self.tangents[idx, 1]
            else:
                tx = self.tangents[idx, 0]
                ty = self.tangents[idx, 1]

            # Project velocity onto tangent
            for j in range(Ny):
                Vt[i, j] = abs(vx[i, j] * tx + vy[i, j] * ty)

        return Vt


def create_interpolated_field(
    bl_field: "BLFieldData",
    fluent_u: NDArray[np.float64],
    fluent_Ue: NDArray[np.float64],
) -> InterpolatedBLField:
    """Create an interpolated field from BL field structure and Fluent data.

    Args:
        bl_field: Original BL solver field (provides s, y, delta structure).
        fluent_u: Interpolated Fluent velocity, shape matching bl_field.u.
        fluent_Ue: Interpolated Fluent edge velocity at each station.

    Returns:
        :class:`InterpolatedBLField` with Fluent-derived velocities.
    """
    return InterpolatedBLField(
        s=bl_field.s.copy(),
        y=bl_field.y.copy(),
        u=fluent_u,
        delta=bl_field.delta.copy(),
        Ue=fluent_Ue,
        source="fluent",
    )
