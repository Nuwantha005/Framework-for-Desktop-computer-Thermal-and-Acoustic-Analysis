"""
Extract boundary layer quantities from Fluent data.

This module extracts BL quantities (Ue, δ, Cf, separation) from Fluent
export data, aligned to the panel-method mesh. The extraction uses:

1. GeometryMapper to project Fluent wall points onto the body surface
   and compute arc-length coordinates.
2. Stagnation point detection from the panel-method BL result to split
   into upper/lower paths.
3. Bernoulli equation for edge velocity from wall pressure.
4. Normal marching for BL thickness (0.99 Ue criterion).
5. Wall shear stress for skin friction coefficient.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional, Tuple, List

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from .data_types import (
    FluentFieldData,
    FluentWallData,
    FluentBLPathResult,
    FluentBLResult,
)

if TYPE_CHECKING:
    from solvers.boundary_layer.runner import BoundaryLayerCaseResult


class FluentBLExtractor:
    """Extract boundary layer quantities from Fluent data.

    Uses the panel-method BL result to define the surface geometry and
    stagnation points, then extracts Fluent-derived BL quantities at
    matching arc-length stations.

    Args:
        bl_result: Panel-method BL result with surface geometry and
            stagnation point indices.
        field_data: Fluent field data (velocity, pressure in domain).
        wall_data: Fluent wall data (pressure, shear at wall).
        rho: Fluid density [kg/m³].
        U_inf: Freestream velocity magnitude [m/s].
        P0_inf: Freestream total (stagnation) pressure [Pa].
            Default: computed from freestream conditions.

    Example::

        from validation.adapters.fluent import (
            load_fluent_bl_data, FluentBLExtractor
        )
        from solvers.boundary_layer.runner import BoundaryLayerRunner

        # Run panel-method BL
        runner = BoundaryLayerRunner(case, solver)
        bl_result = runner.run()

        # Load Fluent data
        field_data, wall_data = load_fluent_bl_data(case_dir)

        # Extract Fluent BL quantities
        extractor = FluentBLExtractor(
            bl_result, field_data, wall_data,
            rho=1.225, U_inf=1.0, P0_inf=101325.0
        )
        fluent_bl = extractor.extract()
    """

    def __init__(
        self,
        bl_result: "BoundaryLayerCaseResult",
        field_data: FluentFieldData,
        wall_data: FluentWallData,
        rho: float,
        U_inf: float,
        P0_inf: Optional[float] = None,
    ) -> None:
        self.bl_result = bl_result
        self.field_data = field_data
        self.wall_data = wall_data
        self.rho = rho
        self.U_inf = U_inf
        # Default: stagnation pressure from Bernoulli
        self.P0_inf = P0_inf if P0_inf is not None else 0.5 * rho * U_inf**2

        # Build field interpolators (scattered data → regular queries)
        self._build_interpolators()

    def _build_interpolators(self) -> None:
        """Build scipy interpolators for Fluent field data."""
        points = self.field_data.points  # (N, 2)

        # Use linear interpolation with nearest-neighbor fallback
        self._vx_interp = LinearNDInterpolator(points, self.field_data.vx)
        self._vy_interp = LinearNDInterpolator(points, self.field_data.vy)
        self._p_interp = LinearNDInterpolator(points, self.field_data.pressure)

        # Fallback for points outside convex hull
        self._vx_nearest = NearestNDInterpolator(points, self.field_data.vx)
        self._vy_nearest = NearestNDInterpolator(points, self.field_data.vy)
        self._p_nearest = NearestNDInterpolator(points, self.field_data.pressure)

    def _interp_velocity(
        self, x: NDArray[np.float64], y: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Interpolate velocity at given coordinates."""
        points = np.column_stack([x, y])
        vx = self._vx_interp(points)
        vy = self._vy_interp(points)

        # Fill NaN with nearest neighbor
        nan_mask = np.isnan(vx)
        if np.any(nan_mask):
            vx[nan_mask] = self._vx_nearest(points[nan_mask])
            vy[nan_mask] = self._vy_nearest(points[nan_mask])

        return vx, vy

    def extract(self) -> FluentBLResult:
        """Extract BL quantities from Fluent data for both paths.

        Returns:
            :class:`FluentBLResult` with upper and lower path results.
        """
        upper = self._extract_path("upper")
        lower = self._extract_path("lower")

        return FluentBLResult(
            upper=upper,
            lower=lower,
            rho=self.rho,
            U_inf=self.U_inf,
            P0_inf=self.P0_inf,
        )

    def _extract_path(self, side: str) -> FluentBLPathResult:
        """Extract BL quantities for one path (upper or lower)."""
        # Get path from panel-method result
        bl_path = self.bl_result.sides[side]
        panel_indices = bl_path.panel_indices
        M = len(panel_indices)

        # Surface coordinates and arc-length from BL solver
        s = bl_path.s.copy()
        x = bl_path.x.copy()
        y = bl_path.y.copy()

        # Allocate output arrays
        Ue = np.full(M, np.nan, dtype=np.float64)
        delta = np.full(M, np.nan, dtype=np.float64)
        Cf = np.full(M, np.nan, dtype=np.float64)
        tau_w = np.full(M, np.nan, dtype=np.float64)

        # --- Extract wall quantities at each station ---
        # Match Fluent wall points to BL path stations
        wall_s, wall_Ue, wall_tau = self._match_wall_data_to_path(
            x, y, s, side
        )

        # Interpolate wall quantities to BL stations
        if len(wall_s) > 2:
            # Sort by arc-length for interpolation
            sort_idx = np.argsort(wall_s)
            ws_sorted = wall_s[sort_idx]
            wUe_sorted = wall_Ue[sort_idx]
            wtau_sorted = wall_tau[sort_idx]

            # Interpolate to BL stations
            Ue = np.interp(s, ws_sorted, wUe_sorted)
            tau_w = np.interp(s, ws_sorted, wtau_sorted)

            # Compute Cf from wall shear
            q_inf = 0.5 * self.rho * self.U_inf**2
            Cf = tau_w / q_inf

        # --- Extract BL thickness by marching along normals ---
        delta = self._compute_bl_thickness(x, y, Ue, side)

        # --- Detect separation point ---
        separation_s = self._detect_separation(s, tau_w)

        return FluentBLPathResult(
            side=side,
            s=s,
            x=x,
            y=y,
            Ue=Ue,
            delta=delta,
            Cf=Cf,
            tau_w=tau_w,
            separation_s=separation_s,
        )

    def _match_wall_data_to_path(
        self,
        path_x: NDArray[np.float64],
        path_y: NDArray[np.float64],
        path_s: NDArray[np.float64],
        side: str,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Match Fluent wall data points to the given BL path.

        For each Fluent wall point:
        1. Find nearest panel on the path
        2. Compute arc-length
        3. Compute Ue from wall pressure via Bernoulli
        4. Keep wall shear stress

        Returns:
            (arc_length, Ue, tau_w) arrays for matched points.
        """
        wall_x = self.wall_data.x
        wall_y = self.wall_data.y
        wall_p = self.wall_data.pressure
        wall_tau = self.wall_data.wall_shear

        # Build path as polyline for projection
        path_points = np.column_stack([path_x, path_y])

        # For each wall point, find closest path segment and compute s
        matched_s: List[float] = []
        matched_Ue: List[float] = []
        matched_tau: List[float] = []

        for i in range(len(wall_x)):
            wx, wy = wall_x[i], wall_y[i]

            # Find closest point on path
            s_proj, dist = self._project_to_path(
                wx, wy, path_x, path_y, path_s
            )

            # Skip points too far from path (likely on wrong side)
            # Use a threshold based on typical BL thickness
            if dist > 0.05:  # 5cm threshold (adjustable)
                continue

            # Compute Ue from wall pressure via Bernoulli:
            # P0_inf = P_wall + 0.5 * rho * Ue^2
            # Ue = sqrt(2 * (P0_inf - P_wall) / rho)
            p_wall = wall_p[i]
            dp = self.P0_inf - p_wall
            if dp > 0:
                Ue_point = np.sqrt(2.0 * dp / self.rho)
            else:
                # Adverse pressure gradient beyond stagnation
                Ue_point = 0.0

            matched_s.append(s_proj)
            matched_Ue.append(Ue_point)
            matched_tau.append(wall_tau[i])

        return (
            np.array(matched_s),
            np.array(matched_Ue),
            np.array(matched_tau),
        )

    def _project_to_path(
        self,
        px: float,
        py: float,
        path_x: NDArray[np.float64],
        path_y: NDArray[np.float64],
        path_s: NDArray[np.float64],
    ) -> Tuple[float, float]:
        """Project a point onto the path and return (arc_length, distance)."""
        # Simple approach: find closest segment and interpolate
        M = len(path_x)

        best_s = path_s[0]
        best_dist = float("inf")

        for i in range(M - 1):
            # Segment from i to i+1
            ax, ay = path_x[i], path_y[i]
            bx, by = path_x[i + 1], path_y[i + 1]

            # Vector from a to b
            dx, dy = bx - ax, by - ay
            seg_len_sq = dx * dx + dy * dy

            if seg_len_sq < 1e-20:
                continue

            # Project point onto segment
            t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
            t = max(0.0, min(1.0, t))

            # Closest point on segment
            cx = ax + t * dx
            cy = ay + t * dy

            # Distance to closest point
            dist = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)

            if dist < best_dist:
                best_dist = dist
                # Arc-length at projected point
                seg_len = np.sqrt(seg_len_sq)
                best_s = path_s[i] + t * seg_len

        return best_s, best_dist

    def _compute_bl_thickness(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        Ue: NDArray[np.float64],
        side: str,
    ) -> NDArray[np.float64]:
        """Compute BL thickness by marching along panel normals.

        For each station, march outward along the panel normal until
        the tangential velocity reaches 0.99 * Ue.

        Returns:
            BL thickness δ at each station [m].
        """
        from core.geometry import Mesh

        mesh = self.bl_result.upper  # Access mesh via parent
        # Actually we need the mesh from somewhere - let's get normals differently

        # Get panel indices for this path
        bl_path = self.bl_result.sides[side]
        panel_indices = bl_path.panel_indices

        M = len(x)
        delta = np.full(M, np.nan, dtype=np.float64)

        # We need panel normals - compute from consecutive points
        # (outward normal for CCW ordering)
        for i in range(M):
            if np.isnan(Ue[i]) or Ue[i] < 1e-10:
                continue

            # Compute local tangent and normal from path
            if i == 0:
                tx = x[1] - x[0]
                ty = y[1] - y[0]
            elif i == M - 1:
                tx = x[M - 1] - x[M - 2]
                ty = y[M - 1] - y[M - 2]
            else:
                tx = x[i + 1] - x[i - 1]
                ty = y[i + 1] - y[i - 1]

            t_len = np.sqrt(tx * tx + ty * ty)
            if t_len < 1e-12:
                continue

            tx /= t_len
            ty /= t_len

            # Outward normal (90° CCW from tangent for upper, CW for lower)
            if side == "upper":
                nx, ny = -ty, tx
            else:
                nx, ny = ty, -tx

            # March along normal to find δ
            delta[i] = self._march_for_delta(
                x[i], y[i], nx, ny, tx, ty, Ue[i]
            )

        return delta

    def _march_for_delta(
        self,
        x0: float,
        y0: float,
        nx: float,
        ny: float,
        tx: float,
        ty: float,
        Ue: float,
        max_dist: float = 0.5,
        n_steps: int = 100,
    ) -> float:
        """March along normal until Vt reaches 0.99 * Ue.

        Args:
            x0, y0: Starting point (wall).
            nx, ny: Outward normal direction.
            tx, ty: Tangent direction.
            Ue: Edge velocity target.
            max_dist: Maximum marching distance [m].
            n_steps: Number of sampling points.

        Returns:
            BL thickness δ [m], or NaN if not found.
        """
        target = 0.99 * Ue
        ds = max_dist / n_steps

        for step in range(1, n_steps + 1):
            dist = step * ds
            xp = x0 + dist * nx
            yp = y0 + dist * ny

            # Interpolate velocity at this point
            vx, vy = self._interp_velocity(
                np.array([xp]), np.array([yp])
            )

            # Project onto tangent to get Vt
            Vt = abs(vx[0] * tx + vy[0] * ty)

            if Vt >= target:
                # Linear interpolation between steps
                if step == 1:
                    return dist

                prev_dist = (step - 1) * ds
                xp_prev = x0 + prev_dist * nx
                yp_prev = y0 + prev_dist * ny
                vx_prev, vy_prev = self._interp_velocity(
                    np.array([xp_prev]), np.array([yp_prev])
                )
                Vt_prev = abs(vx_prev[0] * tx + vy_prev[0] * ty)

                if Vt - Vt_prev > 1e-12:
                    t = (target - Vt_prev) / (Vt - Vt_prev)
                    return prev_dist + t * ds
                else:
                    return dist

        return np.nan  # BL thickness not found within max_dist

    def _detect_separation(
        self,
        s: NDArray[np.float64],
        tau_w: NDArray[np.float64],
    ) -> Optional[float]:
        """Detect separation point where τw crosses zero.

        Returns:
            Arc-length at separation, or None if no separation detected.
        """
        # Look for sign change in tau_w (positive → negative or zero crossing)
        valid = ~np.isnan(tau_w)
        if not np.any(valid):
            return None

        s_valid = s[valid]
        tau_valid = tau_w[valid]

        # Find where tau crosses zero from positive to negative
        for i in range(len(tau_valid) - 1):
            if tau_valid[i] > 0 and tau_valid[i + 1] <= 0:
                # Linear interpolation to find crossing
                if tau_valid[i] - tau_valid[i + 1] > 1e-12:
                    t = tau_valid[i] / (tau_valid[i] - tau_valid[i + 1])
                    return float(s_valid[i] + t * (s_valid[i + 1] - s_valid[i]))
                else:
                    return float(s_valid[i + 1])

        return None
