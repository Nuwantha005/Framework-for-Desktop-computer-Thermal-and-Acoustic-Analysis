"""ASCII readers for Fluent thermal BL exports."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .thermal_data_types import FluentThermalFieldData, FluentThermalWallData


def _normalize_header(name: str) -> str:
    return name.strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def _column_index(headers: list[str], aliases: list[str]) -> int:
    norm = [_normalize_header(h) for h in headers]
    for alias in aliases:
        if alias in norm:
            return norm.index(alias)
    raise ValueError(f"Missing required Fluent thermal column. Tried aliases: {aliases}")


def _read_numeric_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        header_line = f.readline().strip()
    headers = [h.strip() for h in header_line.split(",")]
    data = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=np.float64, encoding="utf-8")
    if data.ndim == 1:
        data = data[np.newaxis, :]
    return headers, data


def read_thermal_wall_data(path: Path) -> FluentThermalWallData:
    """Parse Fluent thermal wall_data export."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fluent thermal wall_data not found: {path}")

    headers, data = _read_numeric_csv(path)
    if data.shape[1] < 5:
        raise ValueError(f"Invalid thermal wall_data shape: {data.shape}")

    i_node = _column_index(headers, ["nodenumber", "nodeid", "id"])
    i_x = _column_index(headers, ["xcoordinate", "x"])
    i_y = _column_index(headers, ["ycoordinate", "y"])
    i_t = _column_index(headers, ["temperature", "t", "walltemperature"])
    i_h = _column_index(headers, ["heattransfercoef", "heattransfercoefficient", "h"])

    return FluentThermalWallData(
        node_id=data[:, i_node].astype(np.int32),
        x=data[:, i_x],
        y=data[:, i_y],
        temperature=data[:, i_t],
        heat_transfer_coeff=data[:, i_h],
    )


def read_thermal_field_data(path: Path) -> FluentThermalFieldData:
    """Parse Fluent thermal filed_data (or field_data) export."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fluent thermal field data not found: {path}")

    headers, data = _read_numeric_csv(path)
    if data.shape[1] < 4:
        raise ValueError(f"Invalid thermal filed_data shape: {data.shape}")

    i_node = _column_index(headers, ["nodenumber", "nodeid", "id"])
    i_x = _column_index(headers, ["xcoordinate", "x"])
    i_y = _column_index(headers, ["ycoordinate", "y"])
    i_t = _column_index(headers, ["temperature", "t"])

    return FluentThermalFieldData(
        node_id=data[:, i_node].astype(np.int32),
        x=data[:, i_x],
        y=data[:, i_y],
        temperature=data[:, i_t],
    )


def find_fluent_thermal_export_dir(case_dir: Path) -> Optional[Path]:
    export_dir = case_dir / "fluent_case" / "export" / "thermal_bl"
    if export_dir.is_dir():
        return export_dir
    return None


def load_fluent_thermal_data(
    case_dir: Path,
) -> Tuple[Optional[FluentThermalFieldData], Optional[FluentThermalWallData]]:
    """Load Fluent thermal field/wall data from standard case structure."""
    export_dir = find_fluent_thermal_export_dir(case_dir)
    if export_dir is None:
        warnings.warn(
            f"Fluent thermal export directory not found in {case_dir}. "
            "Expected: fluent_case/export/thermal_bl/",
            stacklevel=2,
        )
        return None, None

    field_data = None
    wall_data = None

    field_path = export_dir / "filed_data"
    if not field_path.exists():
        alt = export_dir / "field_data"
        if alt.exists():
            field_path = alt

    if field_path.exists():
        try:
            field_data = read_thermal_field_data(field_path)
        except Exception as exc:
            warnings.warn(f"Failed to load Fluent thermal field data: {exc}", stacklevel=2)
    else:
        warnings.warn(f"Fluent thermal filed_data not found: {field_path}", stacklevel=2)

    wall_path = export_dir / "wall_data"
    if wall_path.exists():
        try:
            wall_data = read_thermal_wall_data(wall_path)
        except Exception as exc:
            warnings.warn(f"Failed to load Fluent thermal wall_data: {exc}", stacklevel=2)
    else:
        warnings.warn(f"Fluent thermal wall_data not found: {wall_path}", stacklevel=2)

    return field_data, wall_data
