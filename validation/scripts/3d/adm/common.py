#!/usr/bin/env python3
"""Shared helpers for 3D ADM validation against direct CFD exports."""

from __future__ import annotations

import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
from numpy.typing import NDArray


REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
for _path in (REPO_ROOT, SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from core.io.case_loader import CaseLoader
from validation.convergence.metrics import ErrorMetrics, compute_error_metrics


AXIS_NAMES = ("x", "y", "z")

REQUEST_ALIASES = {
    "velocity_magnitude": "velocity_magnitude",
    "velocitymagnitude": "velocity_magnitude",
    "velocitymag": "velocity_magnitude",
    "velmag": "velocity_magnitude",
    "speed": "velocity_magnitude",
    "temperature": "temperature",
    "temp": "temperature",
    "pressure": "pressure",
    "pressuregauge": "pressure_gauge",
    "gaugepressure": "pressure_gauge",
}

HEADER_ALIASES = {
    "x": ["x-coordinate", "xcoord", "x"],
    "y": ["y-coordinate", "ycoord", "y"],
    "z": ["z-coordinate", "zcoord", "z"],
    "x_velocity": ["x-velocity", "u-velocity", "xvelocity", "u"],
    "y_velocity": ["y-velocity", "v-velocity", "yvelocity", "v"],
    "z_velocity": ["z-velocity", "w-velocity", "zvelocity", "w"],
    "velocity_magnitude": ["velocity-magnitude", "vel-mag", "velocitymag"],
    "pressure": ["pressure", "static-pressure"],
    "temperature": ["temperature", "static-temperature", "temp", "wall-temperature", "walltemperature", "t"],
}

QUANTITY_META = {
    "velocity_magnitude": {
        "label": "Velocity Magnitude",
        "units": "m/s",
        "cmap": "viridis",
        "difference_cmap": "coolwarm",
    },
    "temperature": {
        "label": "Temperature",
        "units": "K",
        "cmap": "inferno",
        "difference_cmap": "coolwarm",
    },
    "pressure": {
        "label": "Gauge Pressure",
        "units": "Pa",
        "cmap": "cividis",
        "difference_cmap": "coolwarm",
    },
    "pressure_gauge": {
        "label": "Gauge Pressure",
        "units": "Pa",
        "cmap": "cividis",
        "difference_cmap": "coolwarm",
    },
}


@dataclass
class DirectCSVData:
    """Parsed direct CFD export data."""

    path: Path
    points: NDArray[np.float64]
    fields: dict[str, NDArray[np.float64]]


@dataclass
class PanelFieldSample:
    """Sampled panel-side fields at reference points."""

    fields: dict[str, NDArray[np.float64]]
    source: str
    warnings: list[str]


def normalize_quantity_name(name: str) -> str:
    """Normalize a user-provided quantity string."""
    key = re.sub(r"[^a-z0-9]+", "", name.lower())
    return REQUEST_ALIASES.get(key, name.lower().strip())


def canonicalize_quantities(quantities: list[str]) -> list[str]:
    """Normalize and deduplicate requested quantities."""
    normalized: list[str] = []
    for quantity in quantities:
        canonical = normalize_quantity_name(quantity)
        if canonical not in normalized:
            normalized.append(canonical)
    return normalized


def quantity_meta(quantity: str) -> dict[str, str]:
    """Return plotting metadata for a quantity."""
    return QUANTITY_META.get(
        quantity,
        {
            "label": quantity.replace("_", " ").title(),
            "units": "",
            "cmap": "viridis",
            "difference_cmap": "coolwarm",
        },
    )


def quantity_label(quantity: str) -> str:
    """Return a human-readable quantity label with units."""
    meta = quantity_meta(quantity)
    units = meta["units"]
    return f"{meta['label']} [{units}]" if units else meta["label"]


def safe_quantity_name(quantity: str) -> str:
    """Create a filesystem-safe quantity stem."""
    return quantity.lower().replace(" ", "_").replace("-", "_")


def format_metrics(metrics: ErrorMetrics) -> str:
    """Format error metrics for concise terminal output."""
    return (
        f"n={metrics.num_points}, "
        f"RMS={metrics.rms_error:.6g}, "
        f"MAE={metrics.mae:.6g}, "
        f"Linf={metrics.linf_error:.6g}, "
        f"RelRMS={metrics.relative_rms:.3f}%"
    )


def metrics_to_dict(metrics: ErrorMetrics) -> dict[str, float | int]:
    """Convert metrics to JSON-serializable primitives."""
    return {
        "l2_error": float(metrics.l2_error),
        "linf_error": float(metrics.linf_error),
        "rms_error": float(metrics.rms_error),
        "mae": float(metrics.mae),
        "relative_l2_percent": float(metrics.relative_l2),
        "relative_rms_percent": float(metrics.relative_rms),
        "num_points": int(metrics.num_points),
    }


def compare_quantity(reference: NDArray[np.float64], test: NDArray[np.float64]) -> ErrorMetrics:
    """Compute error metrics between reference and test arrays."""
    return compute_error_metrics(np.asarray(test), np.asarray(reference))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON file with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_csv(path: Path, columns: dict[str, NDArray[np.float64]]) -> None:
    """Write aligned array columns to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(columns.keys())
    lengths = {len(np.asarray(columns[key])) for key in keys}
    if len(lengths) != 1:
        raise ValueError(f"CSV columns must have equal length, got lengths: {sorted(lengths)}")

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(keys)
        for row in zip(*(np.asarray(columns[key]).tolist() for key in keys), strict=True):
            writer.writerow(row)


def load_direct_csv(path: Path) -> DirectCSVData:
    """Load a Fluent-style direct export CSV.

    Args:
        path: CSV file path.

    Returns:
        Parsed point coordinates and standardized fields.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Direct CFD export not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        header_line = handle.readline().strip()

    headers = [item.strip() for item in header_line.split(",")]
    data = np.genfromtxt(
        path,
        delimiter=",",
        skip_header=1,
        dtype=np.float64,
        encoding="utf-8",
    )
    if data.ndim == 1:
        data = data.reshape(1, -1)

    x = _extract_column(headers, data, HEADER_ALIASES["x"])
    y = _extract_column(headers, data, HEADER_ALIASES["y"])
    z = _extract_column(headers, data, HEADER_ALIASES["z"])
    if x is None or y is None or z is None:
        raise ValueError(f"Could not identify x/y/z columns in direct export: {path}")

    fields: dict[str, NDArray[np.float64]] = {}
    for quantity in ("pressure", "temperature", "x_velocity", "y_velocity", "z_velocity"):
        values = _extract_column(headers, data, HEADER_ALIASES[quantity])
        if values is not None:
            fields[quantity] = values.astype(np.float64)

    direct_speed = _extract_column(headers, data, HEADER_ALIASES["velocity_magnitude"])
    if direct_speed is not None:
        fields["velocity_magnitude"] = direct_speed.astype(np.float64)
    elif {"x_velocity", "y_velocity", "z_velocity"} <= set(fields):
        velocity = np.column_stack(
            [fields["x_velocity"], fields["y_velocity"], fields["z_velocity"]]
        )
        fields["velocity_magnitude"] = np.linalg.norm(velocity, axis=1)

    points = np.column_stack([x, y, z]).astype(np.float64)
    return DirectCSVData(path=path, points=points, fields=fields)


def detect_plane(points: NDArray[np.float64]) -> tuple[str, tuple[str, str], float]:
    """Infer the constant axis and in-plane axes for a cut plane."""
    spans = np.ptp(points, axis=0)
    tol = _span_tolerance(spans)
    varying = np.flatnonzero(spans > tol)
    fixed = np.flatnonzero(spans <= tol)
    if len(varying) != 2 or len(fixed) != 1:
        raise ValueError(
            "Could not infer a cut plane from the reference points. "
            f"spans={spans.tolist()}, tol={tol:.3e}"
        )
    normal_axis = AXIS_NAMES[int(fixed[0])]
    plane_axes = (AXIS_NAMES[int(varying[0])], AXIS_NAMES[int(varying[1])])
    plane_value = float(np.mean(points[:, fixed[0]]))
    return normal_axis, plane_axes, plane_value


def detect_line(points: NDArray[np.float64]) -> tuple[str, tuple[tuple[str, float], tuple[str, float]]]:
    """Infer the varying axis and fixed coordinates for an axis line."""
    spans = np.ptp(points, axis=0)
    tol = _span_tolerance(spans)
    varying = np.flatnonzero(spans > tol)
    fixed = np.flatnonzero(spans <= tol)
    if len(varying) != 1 or len(fixed) != 2:
        raise ValueError(
            "Could not infer an axis line from the reference points. "
            f"spans={spans.tolist()}, tol={tol:.3e}"
        )
    line_axis = AXIS_NAMES[int(varying[0])]
    fixed_values = tuple(
        (AXIS_NAMES[int(index)], float(np.mean(points[:, index]))) for index in fixed
    )
    return line_axis, fixed_values  # type: ignore[return-value]


def sample_panel_fields(
    case_dir: Path,
    points: NDArray[np.float64],
    quantities: list[str],
    mesh_level: int = 0,
    volume_field_path: Path | None = None,
) -> PanelFieldSample:
    """Sample panel-side fields at arbitrary points.

    Tries the saved VTK volume field first. If a velocity-derived quantity is
    still missing, falls back to a fresh solve in the case-configured ADM path.

    Args:
        case_dir: Case directory.
        points: Query points of shape ``(N, 3)``.
        quantities: Canonical quantity names to sample.
        mesh_level: Mesh refinement index for solver fallback.
        volume_field_path: Optional explicit VTK field path.

    Returns:
        Sampled panel fields plus source information.
    """
    case_dir = Path(case_dir).resolve()
    requested = canonicalize_quantities(quantities)
    fields: dict[str, NDArray[np.float64]] = {}
    warnings: list[str] = []
    source_parts: list[str] = []

    resolved_volume = resolve_volume_field_path(case_dir, volume_field_path)
    if resolved_volume is not None:
        vtk_fields, vtk_warnings = _sample_volume_field(resolved_volume, points, requested)
        fields.update(vtk_fields)
        warnings.extend(vtk_warnings)
        source_parts.append(f"saved volume field ({resolved_volume.relative_to(case_dir)})")

    missing_solver_fields = [
        quantity
        for quantity in requested
        if quantity in {
            "velocity_magnitude",
            "x_velocity",
            "y_velocity",
            "z_velocity",
            "pressure",
            "pressure_gauge",
        }
        and _field_missing(fields.get(quantity))
    ]
    if missing_solver_fields:
        solver_fields = _solve_case_fields(case_dir, points, mesh_level=mesh_level)
        for quantity, values in solver_fields.items():
            if quantity in requested and _field_missing(fields.get(quantity)):
                fields[quantity] = values
        source_parts.append("direct ADM solve fallback")

    source = " + ".join(source_parts) if source_parts else "unavailable"
    return PanelFieldSample(fields=fields, source=source, warnings=warnings)


def resolve_volume_field_path(case_dir: Path, explicit_path: Path | None = None) -> Path | None:
    """Resolve a saved panel volume-field path for ADM validation."""
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path).expanduser())
    candidates.extend(
        [
            case_dir / "out" / "panel_solver" / "volume_fields.vts",
            case_dir / "out" / "panel_solver" / "adm_smoke" / "volume_fields.vts",
        ]
    )
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate
    return None


def default_output_dir(case_dir: Path, leaf: str) -> Path:
    """Build the default ADM validation output directory."""
    return Path(case_dir).resolve() / "out" / "validation" / "adm" / leaf


def _sample_volume_field(
    volume_path: Path,
    points: NDArray[np.float64],
    quantities: list[str],
) -> tuple[dict[str, NDArray[np.float64]], list[str]]:
    """Sample requested quantities from a saved VTK volume field."""
    dataset = pv.read(volume_path)
    sampled = pv.PolyData(points).sample(dataset)
    fields: dict[str, NDArray[np.float64]] = {}
    warnings: list[str] = []

    if "vtkValidPointMask" in sampled.point_data:
        valid_mask = np.asarray(sampled.point_data["vtkValidPointMask"]).astype(bool)
    else:
        valid_mask = np.ones(len(points), dtype=bool)

    available_names = set(sampled.point_data.keys())
    for quantity in quantities:
        values: NDArray[np.float64] | None = None
        if quantity == "velocity_magnitude":
            if "velocity_magnitude" in available_names:
                values = np.asarray(sampled.point_data["velocity_magnitude"], dtype=np.float64)
            elif "velocity" in available_names:
                velocity = np.asarray(sampled.point_data["velocity"], dtype=np.float64)
                values = np.linalg.norm(velocity, axis=1)
        elif quantity in {"x_velocity", "y_velocity", "z_velocity"} and "velocity" in available_names:
            component = {"x_velocity": 0, "y_velocity": 1, "z_velocity": 2}[quantity]
            velocity = np.asarray(sampled.point_data["velocity"], dtype=np.float64)
            values = velocity[:, component]
        elif quantity == "pressure" and "pressure_gauge" in available_names:
            values = np.asarray(sampled.point_data["pressure_gauge"], dtype=np.float64)
        elif quantity in available_names:
            values = np.asarray(sampled.point_data[quantity], dtype=np.float64)

        if values is None:
            continue

        values = values.astype(np.float64, copy=True)
        values[~valid_mask] = np.nan
        if np.all(np.isnan(values)):
            warnings.append(
                f"Saved volume field contains only invalid samples for '{quantity}' at the requested points."
            )
            continue
        fields[quantity] = values

    return fields, warnings


def _solve_case_fields(
    case_dir: Path,
    points: NDArray[np.float64],
    mesh_level: int,
) -> dict[str, NDArray[np.float64]]:
    """Run the case-configured 3D solver and evaluate fields at points."""
    case = CaseLoader.load_case(case_dir, mesh_level_index=mesh_level)
    solver = case.create_solver()
    solver.solve()
    velocity = np.asarray(solver.velocity_at(points), dtype=np.float64)
    fields = {
        "x_velocity": velocity[:, 0],
        "y_velocity": velocity[:, 1],
        "z_velocity": velocity[:, 2],
        "velocity_magnitude": np.linalg.norm(velocity, axis=1),
    }
    if hasattr(solver, "pressure_at"):
        fields["pressure_absolute"] = np.asarray(solver.pressure_at(points), dtype=np.float64)
    if hasattr(solver, "pressure_gauge_at"):
        gauge = np.asarray(solver.pressure_gauge_at(points), dtype=np.float64)
        fields["pressure_gauge"] = gauge
        fields["pressure"] = gauge
    elif "pressure_absolute" in fields:
        fields["pressure"] = fields["pressure_absolute"]
    return fields


def _field_missing(values: NDArray[np.float64] | None) -> bool:
    """Return True when a sampled field is unavailable."""
    if values is None:
        return True
    return bool(np.all(np.isnan(np.asarray(values, dtype=np.float64))))


def _normalize_header(name: str) -> str:
    """Normalize a CSV header for relaxed matching."""
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _extract_column(
    headers: list[str],
    data: NDArray[np.float64],
    aliases: list[str],
) -> NDArray[np.float64] | None:
    """Extract one CSV column using relaxed alias matching."""
    normalized_headers = [_normalize_header(header) for header in headers]
    for alias in aliases:
        normalized_alias = _normalize_header(alias)
        for index, header in enumerate(normalized_headers):
            if header == normalized_alias:
                return data[:, index].astype(np.float64)
    for alias in aliases:
        normalized_alias = _normalize_header(alias)
        if len(normalized_alias) < 4:
            continue
        for index, header in enumerate(normalized_headers):
            if normalized_alias in header:
                return data[:, index].astype(np.float64)
    return None


def _span_tolerance(spans: NDArray[np.float64]) -> float:
    """Compute an absolute tolerance for fixed-axis detection."""
    scale = max(float(np.max(np.abs(spans))), 1.0)
    return scale * 1e-6
