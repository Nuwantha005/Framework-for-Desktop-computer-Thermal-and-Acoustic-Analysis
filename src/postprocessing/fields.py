"""
Field data containers for post-processing.

These provide a unified way to store and access computed fields,
regardless of what solver produced them.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Tuple, List
import numpy as np
from numpy.typing import NDArray


@dataclass
class ScalarField:
    """
    A 2D scalar field (e.g., pressure, temperature, velocity magnitude).
    
    Attributes:
        data: 2D array of values (ny, nx)
        name: Field name (e.g., "pressure", "temperature")
        units: Physical units (e.g., "Pa", "K")
        XX: X-coordinate meshgrid
        YY: Y-coordinate meshgrid
    """
    data: NDArray
    name: str
    units: str = ""
    XX: Optional[NDArray] = None
    YY: Optional[NDArray] = None
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape
    
    @property
    def min(self) -> float:
        return float(np.nanmin(self.data))
    
    @property
    def max(self) -> float:
        return float(np.nanmax(self.data))
    
    def __repr__(self) -> str:
        return f"ScalarField({self.name}, shape={self.shape}, range=[{self.min:.4g}, {self.max:.4g}] {self.units})"


@dataclass
class VectorField:
    """
    A 2D vector field (e.g., velocity).
    
    Attributes:
        u: X-component (ny, nx)
        v: Y-component (ny, nx)
        name: Field name
        units: Physical units
        XX: X-coordinate meshgrid
        YY: Y-coordinate meshgrid
    """
    u: NDArray
    v: NDArray
    name: str
    units: str = ""
    XX: Optional[NDArray] = None
    YY: Optional[NDArray] = None
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self.u.shape
    
    @property
    def magnitude(self) -> NDArray:
        """Compute magnitude |V| = sqrt(u² + v²)."""
        return np.sqrt(self.u**2 + self.v**2)
    
    def to_scalar(self, component: str = "magnitude") -> ScalarField:
        """
        Extract a scalar field from this vector field.
        
        Args:
            component: "u", "v", "magnitude", "x", or "y"
        """
        if component in ("u", "x"):
            data = self.u
            name = f"{self.name}_x"
        elif component in ("v", "y"):
            data = self.v
            name = f"{self.name}_y"
        elif component == "magnitude":
            data = self.magnitude
            name = f"|{self.name}|"
        else:
            raise ValueError(f"Unknown component: {component}")
        
        return ScalarField(
            data=data,
            name=name,
            units=self.units,
            XX=self.XX,
            YY=self.YY
        )
    
    def __repr__(self) -> str:
        mag = self.magnitude
        return f"VectorField({self.name}, shape={self.shape}, |V| range=[{np.nanmin(mag):.4g}, {np.nanmax(mag):.4g}] {self.units})"


@dataclass
class FieldData:
    """
    Container for all computed fields in a simulation.
    
    Provides a registry-like interface where fields can be added dynamically.
    This allows the post-processing pipeline to be extensible without
    modifying existing code.
    
    Usage:
        fields = FieldData(XX, YY)
        fields.add_vector("velocity", Vx, Vy, units="m/s")
        fields.add_scalar("pressure", P, units="Pa")
        
        # Access fields
        vel = fields.velocity  # VectorField
        p = fields["pressure"]  # ScalarField
        
        # List available fields
        print(fields.available)
    """
    
    XX: NDArray
    YY: NDArray
    _scalars: Dict[str, ScalarField] = field(default_factory=dict)
    _vectors: Dict[str, VectorField] = field(default_factory=dict)
    _metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self.XX.shape
    
    @property
    def resolution(self) -> Tuple[int, int]:
        ny, nx = self.XX.shape
        return (nx, ny)
    
    @property
    def x_range(self) -> Tuple[float, float]:
        return (float(self.XX.min()), float(self.XX.max()))
    
    @property
    def y_range(self) -> Tuple[float, float]:
        return (float(self.YY.min()), float(self.YY.max()))
    
    # -------------------------------------------------------------------------
    # Add fields
    # -------------------------------------------------------------------------
    
    def add_scalar(self, name: str, data: NDArray, units: str = "") -> ScalarField:
        """Add a scalar field."""
        sf = ScalarField(data=data, name=name, units=units, XX=self.XX, YY=self.YY)
        self._scalars[name] = sf
        return sf
    
    def add_vector(self, name: str, u: NDArray, v: NDArray, units: str = "") -> VectorField:
        """Add a vector field."""
        vf = VectorField(u=u, v=v, name=name, units=units, XX=self.XX, YY=self.YY)
        self._vectors[name] = vf
        return vf
    
    # -------------------------------------------------------------------------
    # Access fields
    # -------------------------------------------------------------------------
    
    def get_scalar(self, name: str) -> Optional[ScalarField]:
        """Get a scalar field by name."""
        return self._scalars.get(name)
    
    def get_vector(self, name: str) -> Optional[VectorField]:
        """Get a vector field by name."""
        return self._vectors.get(name)
    
    def __getitem__(self, name: str) -> ScalarField | VectorField:
        """Get any field by name."""
        if name in self._scalars:
            return self._scalars[name]
        if name in self._vectors:
            return self._vectors[name]
        raise KeyError(f"Field '{name}' not found. Available: {self.available}")
    
    def __getattr__(self, name: str) -> ScalarField | VectorField:
        """Allow attribute-style access: fields.velocity"""
        if name.startswith('_'):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"No field named '{name}'")
    
    def __contains__(self, name: str) -> bool:
        return name in self._scalars or name in self._vectors
    
    # -------------------------------------------------------------------------
    # Query available fields
    # -------------------------------------------------------------------------
    
    @property
    def available(self) -> List[str]:
        """List all available field names."""
        return list(self._scalars.keys()) + list(self._vectors.keys())
    
    @property
    def scalars(self) -> List[str]:
        """List scalar field names."""
        return list(self._scalars.keys())
    
    @property
    def vectors(self) -> List[str]:
        """List vector field names."""
        return list(self._vectors.keys())
    
    def has(self, name: str) -> bool:
        """Check if a field exists."""
        return name in self
    
    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------
    
    def set_metadata(self, key: str, value: Any):
        """Store metadata (solver info, timestamps, etc.)."""
        self._metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Retrieve metadata."""
        return self._metadata.get(key, default)
    
    # -------------------------------------------------------------------------
    # Convenience
    # -------------------------------------------------------------------------
    
    def summary(self) -> str:
        """Print summary of all fields."""
        lines = [f"FieldData: {self.resolution[0]}×{self.resolution[1]} grid"]
        lines.append(f"  Domain: x={self.x_range}, y={self.y_range}")
        lines.append("  Scalars:")
        for name, sf in self._scalars.items():
            lines.append(f"    {name}: [{sf.min:.4g}, {sf.max:.4g}] {sf.units}")
        lines.append("  Vectors:")
        for name, vf in self._vectors.items():
            mag = vf.magnitude
            lines.append(f"    {name}: |V|=[{np.nanmin(mag):.4g}, {np.nanmax(mag):.4g}] {vf.units}")
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"FieldData(scalars={self.scalars}, vectors={self.vectors})"
