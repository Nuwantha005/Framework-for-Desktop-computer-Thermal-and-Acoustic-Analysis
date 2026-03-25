"""
Data types for Fluent export data and extracted BL quantities.

This module defines the dataclasses used throughout the Fluent comparison
pipeline:

- :class:`FluentFieldData` — raw field data from filed_data export
- :class:`FluentWallData` — raw wall data from wall_data export
- :class:`FluentBLPathResult` — extracted BL quantities for one path
- :class:`FluentBLResult` — two-sided BL result from Fluent
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class FluentFieldData:
    """Raw field data from Fluent filed_data ASCII export.

    Contains velocity and pressure at scattered nodes in the flow domain
    near the body surface (used for BL thickness extraction).

    Attributes:
        node_id: Node identifiers from Fluent, shape (N,).
        x: X-coordinates of nodes [m], shape (N,).
        y: Y-coordinates of nodes [m], shape (N,).
        pressure: Static pressure at nodes [Pa], shape (N,).
        vx: X-component of velocity [m/s], shape (N,).
        vy: Y-component of velocity [m/s], shape (N,).
    """

    node_id: NDArray[np.int32]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    pressure: NDArray[np.float64]
    vx: NDArray[np.float64]
    vy: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate array shapes."""
        n = len(self.node_id)
        for name, arr in [
            ("x", self.x),
            ("y", self.y),
            ("pressure", self.pressure),
            ("vx", self.vx),
            ("vy", self.vy),
        ]:
            if len(arr) != n:
                raise ValueError(
                    f"Array '{name}' has length {len(arr)}, expected {n}"
                )

    @property
    def num_nodes(self) -> int:
        """Number of data nodes."""
        return len(self.node_id)

    @property
    def velocity_magnitude(self) -> NDArray[np.float64]:
        """Velocity magnitude |V| [m/s], shape (N,)."""
        return np.sqrt(self.vx**2 + self.vy**2)

    @property
    def points(self) -> NDArray[np.float64]:
        """Node coordinates as (N, 2) array."""
        return np.column_stack([self.x, self.y])


@dataclass
class FluentWallData:
    """Raw wall data from Fluent wall_data ASCII export.

    Contains pressure and wall shear stress at wall boundary nodes.

    Attributes:
        node_id: Node identifiers from Fluent, shape (N,).
        x: X-coordinates of wall nodes [m], shape (N,).
        y: Y-coordinates of wall nodes [m], shape (N,).
        pressure: Static pressure at wall [Pa], shape (N,).
        wall_shear: Wall shear stress magnitude [Pa], shape (N,).
    """

    node_id: NDArray[np.int32]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    pressure: NDArray[np.float64]
    wall_shear: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate array shapes."""
        n = len(self.node_id)
        for name, arr in [
            ("x", self.x),
            ("y", self.y),
            ("pressure", self.pressure),
            ("wall_shear", self.wall_shear),
        ]:
            if len(arr) != n:
                raise ValueError(
                    f"Array '{name}' has length {len(arr)}, expected {n}"
                )

    @property
    def num_nodes(self) -> int:
        """Number of wall nodes."""
        return len(self.node_id)

    @property
    def points(self) -> NDArray[np.float64]:
        """Wall node coordinates as (N, 2) array."""
        return np.column_stack([self.x, self.y])


@dataclass
class FluentBLPathResult:
    """Extracted boundary layer quantities for one path (upper or lower).

    All arrays share the same length M (number of arc-length stations
    along the path).

    Attributes:
        side: Path identifier ("upper" or "lower").
        s: Arc-length from forward stagnation [m], shape (M,).
        x: Surface x-coordinates [m], shape (M,).
        y: Surface y-coordinates [m], shape (M,).
        Ue: Edge velocity from Bernoulli [m/s], shape (M,).
        delta: Boundary layer thickness [m], shape (M,).
            Computed by marching along normal until Vt ≈ 0.99 Ue.
        Cf: Skin friction coefficient [-], shape (M,).
        tau_w: Wall shear stress [Pa], shape (M,).
        separation_s: Arc-length at separation point [m], or None.
    """

    side: str
    s: NDArray[np.float64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    Ue: NDArray[np.float64]
    delta: NDArray[np.float64]
    Cf: NDArray[np.float64]
    tau_w: NDArray[np.float64]
    separation_s: Optional[float] = None

    @property
    def num_stations(self) -> int:
        """Number of arc-length stations."""
        return len(self.s)


@dataclass
class FluentBLResult:
    """Two-sided boundary layer result extracted from Fluent data.

    Attributes:
        upper: BL quantities for upper surface path.
        lower: BL quantities for lower surface path.
        rho: Fluid density [kg/m³] used in calculations.
        U_inf: Freestream velocity magnitude [m/s].
        P0_inf: Freestream total (stagnation) pressure [Pa].
    """

    upper: FluentBLPathResult
    lower: FluentBLPathResult
    rho: float
    U_inf: float
    P0_inf: float

    @property
    def sides(self) -> dict[str, FluentBLPathResult]:
        """Convenience mapping {"upper": ..., "lower": ...}."""
        return {"upper": self.upper, "lower": self.lower}
