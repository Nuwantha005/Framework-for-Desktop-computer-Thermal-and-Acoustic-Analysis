"""
ASCII reader for 3D Fluent export files.

Parses comma-separated ASCII exports from Ansys Fluent for 3D cases.
Expects data at scattered nodes with headers.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

@dataclass
class Fluent3DSurfaceData:
    """Raw 3D surface data from Fluent ASCII export."""
    node_id: NDArray[np.int32]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    z: NDArray[np.float64]
    pressure: Optional[NDArray[np.float64]] = None
    vx: Optional[NDArray[np.float64]] = None
    vy: Optional[NDArray[np.float64]] = None
    vz: Optional[NDArray[np.float64]] = None
    velocity_magnitude: Optional[NDArray[np.float64]] = None

    @property
    def num_nodes(self) -> int:
        return len(self.node_id)

    @property
    def points(self) -> NDArray[np.float64]:
        return np.column_stack([self.x, self.y, self.z])

    def get_velocity_magnitude(self) -> NDArray[np.float64]:
        """Get velocity magnitude, calculating it if necessary."""
        if self.velocity_magnitude is not None:
            return self.velocity_magnitude
        if self.vx is not None and self.vy is not None and self.vz is not None:
            return np.sqrt(self.vx**2 + self.vy**2 + self.vz**2)
        raise ValueError("Velocity data is not complete in the Fluent export.")


def read_3d_surface_data(path: Path) -> Fluent3DSurfaceData:
    """Parse 3D Fluent ASCII export file.

    Flexibly reads headers to find columns for x, y, z, pressure, and velocity.

    Args:
        path: Path to the ASCII file.

    Returns:
        Fluent3DSurfaceData object.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fluent 3D surface data not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        header_line = f.readline().strip()
    
    headers = [h.strip().lower() for h in header_line.split(",")]
    
    try:
        data = np.genfromtxt(
            path,
            delimiter=",",
            skip_header=1,
            dtype=np.float64,
            encoding="utf-8",
        )
    except Exception as e:
        raise ValueError(f"Failed to parse Fluent 3D data: {e}") from e

    if data.ndim == 1:
        data = data.reshape(1, -1)

    # Find indices mapping
    def get_col(names):
        for name in names:
            for i, h in enumerate(headers):
                if name in h:
                    return data[:, i]
        return None

    node_id = get_col(["nodenumber", "node"])
    if node_id is None:
        node_id = np.arange(1, len(data) + 1, dtype=np.int32)
    else:
        node_id = node_id.astype(np.int32)

    x = get_col(["x-coordinate", "x-coord", "x"])
    y = get_col(["y-coordinate", "y-coord", "y"])
    z = get_col(["z-coordinate", "z-coord", "z"])

    if x is None or y is None or z is None:
        raise ValueError(f"Could not find x, y, or z coordinates in headers: {headers}")

    pressure = get_col(["pressure"])
    vx = get_col(["x-velocity", "u-velocity", "u"])
    vy = get_col(["y-velocity", "v-velocity", "v"])
    vz = get_col(["z-velocity", "w-velocity", "w"])
    vmag = get_col(["velocity-magnitude", "vel-mag"])

    return Fluent3DSurfaceData(
        node_id=node_id,
        x=x,
        y=y,
        z=z,
        pressure=pressure,
        vx=vx,
        vy=vy,
        vz=vz,
        velocity_magnitude=vmag
    )

def find_fluent_3d_export(case_dir: Path) -> Optional[Path]:
    """Locate the 3D Fluent export surface_data file."""
    candidates = [
        case_dir / "fluent" / "export" / "panel" / "surface_data",
        case_dir / "fluent_case" / "export" / "panel" / "surface_data",
        case_dir / "fluent" / "export" / "panel" / "surface_data.csv",
        case_dir / "fluent_case" / "export" / "panel" / "surface_data.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None
