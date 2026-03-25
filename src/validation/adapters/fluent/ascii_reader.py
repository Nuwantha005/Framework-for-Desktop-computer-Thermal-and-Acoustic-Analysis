"""
ASCII reader for Fluent export files.

Parses comma-separated ASCII exports from Ansys Fluent:
- filed_data: field quantities (velocity, pressure) at scattered nodes
- wall_data: wall quantities (pressure, wall shear) at wall nodes

Future extension: add readers for VTK/binary formats when moving to 3D.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .data_types import FluentFieldData, FluentWallData


def read_field_data(path: Path) -> FluentFieldData:
    """Parse Fluent ASCII filed_data export.

    Expected format (comma-separated, header row):
    ::

        nodenumber,    x-coordinate,    y-coordinate,        pressure,      x-velocity,      y-velocity
                 1, 8.427639012E-17,-7.500000000E-01,-3.011805813E-01, 1.228020079E+00,-2.323784826E-01
                 ...

    Args:
        path: Path to the filed_data file.

    Returns:
        :class:`FluentFieldData` with parsed arrays.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file format is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fluent field data not found: {path}")

    try:
        # Read CSV with flexible whitespace handling
        data = np.genfromtxt(
            path,
            delimiter=",",
            skip_header=1,
            dtype=np.float64,
            encoding="utf-8",
        )
    except Exception as e:
        raise ValueError(f"Failed to parse Fluent field data: {e}") from e

    if data.ndim != 2 or data.shape[1] < 6:
        raise ValueError(
            f"Invalid filed_data format: expected 6 columns, got shape {data.shape}"
        )

    return FluentFieldData(
        node_id=data[:, 0].astype(np.int32),
        x=data[:, 1],
        y=data[:, 2],
        pressure=data[:, 3],
        vx=data[:, 4],
        vy=data[:, 5],
    )


def read_wall_data(path: Path) -> FluentWallData:
    """Parse Fluent ASCII wall_data export.

    Expected format (comma-separated, header row):
    ::

        nodenumber,    x-coordinate,    y-coordinate,        pressure,      wall-shear
                 1, 2.220446049E-16,-5.000000000E-01,-4.862246001E-01, 1.522855383E-03
                 ...

    Args:
        path: Path to the wall_data file.

    Returns:
        :class:`FluentWallData` with parsed arrays.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file format is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fluent wall data not found: {path}")

    try:
        data = np.genfromtxt(
            path,
            delimiter=",",
            skip_header=1,
            dtype=np.float64,
            encoding="utf-8",
        )
    except Exception as e:
        raise ValueError(f"Failed to parse Fluent wall data: {e}") from e

    if data.ndim != 2 or data.shape[1] < 5:
        raise ValueError(
            f"Invalid wall_data format: expected 5 columns, got shape {data.shape}"
        )

    return FluentWallData(
        node_id=data[:, 0].astype(np.int32),
        x=data[:, 1],
        y=data[:, 2],
        pressure=data[:, 3],
        wall_shear=data[:, 4],
    )


def find_fluent_export_dir(case_dir: Path) -> Optional[Path]:
    """Locate the Fluent viscous BL export directory for a case.

    Standard path: ``<case_dir>/fluent_case/export/viscous_bl/``

    Args:
        case_dir: Root directory of the panel-method case.

    Returns:
        Path to the export directory, or None if not found.
    """
    export_dir = case_dir / "fluent_case" / "export" / "viscous_bl"
    if export_dir.is_dir():
        return export_dir
    return None


def load_fluent_bl_data(
    case_dir: Path,
) -> tuple[Optional[FluentFieldData], Optional[FluentWallData]]:
    """Load Fluent BL data from standard case directory structure.

    Looks for:
    - ``<case_dir>/fluent_case/export/viscous_bl/filed_data``
    - ``<case_dir>/fluent_case/export/viscous_bl/wall_data``

    Args:
        case_dir: Root directory of the panel-method case.

    Returns:
        Tuple of (field_data, wall_data). Either may be None if the
        corresponding file is missing (with a warning).
    """
    export_dir = find_fluent_export_dir(case_dir)
    if export_dir is None:
        warnings.warn(
            f"Fluent export directory not found in {case_dir}. "
            "Expected: fluent_case/export/viscous_bl/",
            stacklevel=2,
        )
        return None, None

    field_data: Optional[FluentFieldData] = None
    wall_data: Optional[FluentWallData] = None

    field_path = export_dir / "filed_data"
    if field_path.exists():
        try:
            field_data = read_field_data(field_path)
        except Exception as e:
            warnings.warn(f"Failed to load Fluent field data: {e}", stacklevel=2)
    else:
        warnings.warn(f"Fluent filed_data not found: {field_path}", stacklevel=2)

    wall_path = export_dir / "wall_data"
    if wall_path.exists():
        try:
            wall_data = read_wall_data(wall_path)
        except Exception as e:
            warnings.warn(f"Failed to load Fluent wall data: {e}", stacklevel=2)
    else:
        warnings.warn(f"Fluent wall_data not found: {wall_path}", stacklevel=2)

    return field_data, wall_data
