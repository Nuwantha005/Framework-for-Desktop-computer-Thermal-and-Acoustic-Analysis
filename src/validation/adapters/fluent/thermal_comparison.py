"""Thermal BL comparison runner against Fluent exports."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from .comparison import BLComparisonMetrics
from .thermal_ascii_reader import load_fluent_thermal_data
from .thermal_data_types import (
    FluentThermalFieldData,
    FluentThermalPathResult,
    FluentThermalResult,
    FluentThermalWallData,
    InterpolatedThermalField,
    ThermalComparisonResult,
)


class ThermalComparisonRunner:
    """Build Fluent-vs-thermal solver comparison products for plotting."""

    def __init__(self, case, bl_result, upper_thermal_result, lower_thermal_result) -> None:
        self.case = case
        self.bl_result = bl_result
        self.upper_thermal_result = upper_thermal_result
        self.lower_thermal_result = lower_thermal_result

    def run(self) -> ThermalComparisonResult:
        case_dir = Path(self.case.case_dir)
        field_data, wall_data = load_fluent_thermal_data(case_dir)

        upper_idx = self._valid_panel_indices("upper", self.upper_thermal_result)
        lower_idx = self._valid_panel_indices("lower", self.lower_thermal_result)

        if wall_data is None:
            warnings.warn(
                "Fluent thermal wall_data unavailable; returning comparison without Fluent data.",
                stacklevel=2,
            )
            return ThermalComparisonResult(
                bl_result=self.bl_result,
                upper_thermal_result=self.upper_thermal_result,
                lower_thermal_result=self.lower_thermal_result,
                fluent_wall_result=None,
                upper_panel_indices=upper_idx,
                lower_panel_indices=lower_idx,
            )

        fluent_wall = self._extract_wall_result(wall_data)
        wall_metrics = self._compute_wall_metrics(fluent_wall)

        upper_field = self._interpolate_field(self.upper_thermal_result, field_data)
        lower_field = self._interpolate_field(self.lower_thermal_result, field_data)
        field_metrics = self._compute_field_metrics(upper_field, lower_field)

        return ThermalComparisonResult(
            bl_result=self.bl_result,
            upper_thermal_result=self.upper_thermal_result,
            lower_thermal_result=self.lower_thermal_result,
            fluent_wall_result=fluent_wall,
            upper_panel_indices=upper_idx,
            lower_panel_indices=lower_idx,
            upper_fluent_field=upper_field,
            lower_fluent_field=lower_field,
            wall_metrics=wall_metrics,
            field_metrics=field_metrics,
        )

    def _valid_panel_indices(self, side: str, thermal_result) -> list[int]:
        path = self.bl_result.sides[side]
        return list(path.panel_indices[: thermal_result.num_stations])

    def _extract_wall_result(self, wall_data: FluentThermalWallData) -> FluentThermalResult:
        upper = self._extract_path_wall("upper", self.upper_thermal_result, wall_data)
        lower = self._extract_path_wall("lower", self.lower_thermal_result, wall_data)
        return FluentThermalResult(upper=upper, lower=lower)

    def _extract_path_wall(self, side: str, thermal_result, wall_data: FluentThermalWallData) -> FluentThermalPathResult:
        s = np.asarray(thermal_result.arc_length)
        x = np.asarray(thermal_result.x)
        y = np.asarray(thermal_result.y)

        if len(s) == 0:
            return FluentThermalPathResult(
                side=side,
                s=s,
                x=x,
                y=y,
                wall_temperature=np.array([], dtype=np.float64),
                heat_transfer_coeff=np.array([], dtype=np.float64),
            )

        matched_s, matched_t, matched_h = self._match_wall_points(x, y, s, wall_data)
        if len(matched_s) < 2:
            t_interp = np.full_like(s, np.nan, dtype=np.float64)
            h_interp = np.full_like(s, np.nan, dtype=np.float64)
        else:
            order = np.argsort(matched_s)
            ms = matched_s[order]
            mt = matched_t[order]
            mh = matched_h[order]
            t_interp = np.interp(s, ms, mt, left=np.nan, right=np.nan)
            h_interp = np.interp(s, ms, mh, left=np.nan, right=np.nan)

        return FluentThermalPathResult(
            side=side,
            s=s,
            x=x,
            y=y,
            wall_temperature=t_interp,
            heat_transfer_coeff=h_interp,
        )

    def _match_wall_points(
        self,
        path_x: NDArray[np.float64],
        path_y: NDArray[np.float64],
        path_s: NDArray[np.float64],
        wall_data: FluentThermalWallData,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        matched_s = []
        matched_t = []
        matched_h = []
        for i in range(len(wall_data.x)):
            s_proj, dist = self._project_to_path(wall_data.x[i], wall_data.y[i], path_x, path_y, path_s)
            if dist > 0.05:
                continue
            matched_s.append(s_proj)
            matched_t.append(float(wall_data.temperature[i]))
            matched_h.append(float(wall_data.heat_transfer_coeff[i]))

        return (
            np.asarray(matched_s, dtype=np.float64),
            np.asarray(matched_t, dtype=np.float64),
            np.asarray(matched_h, dtype=np.float64),
        )

    @staticmethod
    def _project_to_path(
        px: float,
        py: float,
        path_x: NDArray[np.float64],
        path_y: NDArray[np.float64],
        path_s: NDArray[np.float64],
    ) -> Tuple[float, float]:
        best_s = float(path_s[0])
        best_dist = float("inf")
        for i in range(len(path_x) - 1):
            ax = path_x[i]
            ay = path_y[i]
            bx = path_x[i + 1]
            by = path_y[i + 1]
            dx = bx - ax
            dy = by - ay
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq < 1e-20:
                continue
            t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
            t = max(0.0, min(1.0, t))
            cx = ax + t * dx
            cy = ay + t * dy
            dist = float(np.hypot(px - cx, py - cy))
            if dist < best_dist:
                best_dist = dist
                best_s = float(path_s[i] + t * np.sqrt(seg_len_sq))
        return best_s, best_dist

    def _interpolate_field(self, thermal_result, field_data: Optional[FluentThermalFieldData]) -> Optional[InterpolatedThermalField]:
        if field_data is None or not thermal_result.has_field:
            return None

        field = thermal_result.field
        if field is None:
            return None

        points = field_data.points
        lin = LinearNDInterpolator(points, field_data.temperature)
        near = NearestNDInterpolator(points, field_data.temperature)

        q = np.column_stack([field.x.ravel(), field.y.ravel()])
        T = lin(q)
        nan_mask = np.isnan(T)
        if np.any(nan_mask):
            T[nan_mask] = near(q[nan_mask])
        T = T.reshape(field.T.shape)

        return InterpolatedThermalField(
            s=field.s.copy(),
            y=field.y_normal.copy(),
            T=T,
            delta=thermal_result.thermal_bl_thickness.copy(),
            T_inf=float(field.T_inf),
            source="fluent",
        )

    def _compute_wall_metrics(self, fluent_wall: FluentThermalResult) -> Dict[str, Dict[str, BLComparisonMetrics]]:
        metrics: Dict[str, Dict[str, BLComparisonMetrics]] = {}
        for side, thermal_result in [("upper", self.upper_thermal_result), ("lower", self.lower_thermal_result)]:
            fl = fluent_wall.sides[side]
            s_bl = np.asarray(thermal_result.arc_length)
            if len(s_bl) < 2:
                continue

            s_min = max(float(np.min(s_bl)), float(np.min(fl.s)))
            s_max = min(float(np.max(s_bl)), float(np.max(fl.s)))
            if s_max <= s_min:
                continue

            s_common = np.linspace(s_min, s_max, 100)
            bl_tw = np.interp(s_common, s_bl, np.asarray(thermal_result.wall_temperature))
            bl_h = np.interp(s_common, s_bl, np.asarray(thermal_result.heat_transfer_coeff))
            fl_tw = np.interp(s_common, fl.s, fl.wall_temperature)
            fl_h = np.interp(s_common, fl.s, fl.heat_transfer_coeff)

            metrics[side] = {
                "wall_temperature": BLComparisonMetrics.compute(fl_tw, bl_tw),
                "heat_transfer_coeff": BLComparisonMetrics.compute(fl_h, bl_h),
            }
        return metrics

    def _compute_field_metrics(
        self,
        upper_field: Optional[InterpolatedThermalField],
        lower_field: Optional[InterpolatedThermalField],
    ) -> Dict[str, Dict[str, BLComparisonMetrics]]:
        metrics: Dict[str, Dict[str, BLComparisonMetrics]] = {}
        if upper_field is not None and self.upper_thermal_result.has_field:
            metrics["upper"] = {
                "T": BLComparisonMetrics.compute(upper_field.T, self.upper_thermal_result.field.T),
            }
        if lower_field is not None and self.lower_thermal_result.has_field:
            metrics["lower"] = {
                "T": BLComparisonMetrics.compute(lower_field.T, self.lower_thermal_result.field.T),
            }
        return metrics
