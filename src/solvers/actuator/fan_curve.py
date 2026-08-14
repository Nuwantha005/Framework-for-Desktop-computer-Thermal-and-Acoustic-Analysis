"""Fan P-Q curve loading and interpolation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class FanCurve:
    """Interpolated fan static-pressure curve."""

    flow_rate: NDArray[np.float64]
    pressure: NDArray[np.float64]
    interpolation: str = "linear"

    @classmethod
    def from_csv(cls, path: str | Path, interpolation: str = "linear") -> "FanCurve":
        """Load a fan curve from CSV.

        Args:
            path: CSV containing flow rate and static pressure columns.
            interpolation: ``"linear"`` or ``"cubic"``.

        Returns:
            FanCurve instance.
        """
        path = Path(path)
        flow_values = []
        pressure_values = []
        with open(path, newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for row in reader:
                parts = []
                for item in row:
                    parts.extend(token.strip() for token in item.split("\t") if token.strip())
                if len(parts) < 2:
                    continue
                try:
                    pressure_value = float(parts[-1])
                    flow_value = float("".join(parts[:-1]))
                except ValueError:
                    continue
                if np.isfinite(flow_value) and np.isfinite(pressure_value):
                    flow_values.append(flow_value)
                    pressure_values.append(pressure_value)

        if not flow_values:
            raise ValueError(f"Fan curve has no numeric data rows: {path}")

        flow_rate = np.asarray(flow_values, dtype=np.float64)
        pressure = np.asarray(pressure_values, dtype=np.float64)
        return cls(flow_rate=flow_rate, pressure=pressure, interpolation=interpolation).validated()

    def validated(self) -> "FanCurve":
        """Return a sorted, validated curve."""
        if self.interpolation not in {"linear", "cubic"}:
            raise ValueError(f"Unsupported fan curve interpolation: {self.interpolation}")
        if self.flow_rate.shape != self.pressure.shape:
            raise ValueError("Fan curve flow_rate and pressure arrays must have matching shapes")
        if self.flow_rate.size < 2:
            raise ValueError("Fan curve requires at least two points")

        order = np.argsort(self.flow_rate)
        q = np.asarray(self.flow_rate[order], dtype=np.float64)
        dp = np.asarray(self.pressure[order], dtype=np.float64)

        unique_q, unique_idx = np.unique(q, return_index=True)
        if unique_q.size != q.size:
            q = unique_q
            dp = dp[unique_idx]
        if q.size < 2:
            raise ValueError("Fan curve requires at least two unique flow-rate points")

        return FanCurve(flow_rate=q, pressure=dp, interpolation=self.interpolation)

    @property
    def midpoint_pressure(self) -> float:
        """Pressure at the midpoint of the tabulated flow range."""
        q_mid = 0.5 * (float(self.flow_rate[0]) + float(self.flow_rate[-1]))
        return self.pressure_at(q_mid)

    @property
    def q_min(self) -> float:
        """Minimum tabulated flow rate."""
        return float(self.flow_rate[0])

    @property
    def q_max(self) -> float:
        """Maximum tabulated flow rate."""
        return float(self.flow_rate[-1])

    def contains_flow_rate(self, flow_rate: float) -> bool:
        """Return whether flow_rate is inside the tabulated fan-curve range."""
        return self.q_min <= float(flow_rate) <= self.q_max

    def pressure_at(self, flow_rate: float) -> float:
        """Interpolate static pressure at the requested flow rate."""
        q = float(np.clip(flow_rate, self.flow_rate[0], self.flow_rate[-1]))
        if self.interpolation == "cubic" and self.flow_rate.size >= 4:
            try:
                from scipy.interpolate import CubicSpline

                spline = CubicSpline(self.flow_rate, self.pressure, extrapolate=False)
                return float(spline(q))
            except Exception:
                return float(np.interp(q, self.flow_rate, self.pressure))
        return float(np.interp(q, self.flow_rate, self.pressure))
