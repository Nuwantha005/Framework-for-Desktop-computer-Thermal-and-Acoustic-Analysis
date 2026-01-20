"""
Pydantic schemas for configuration validation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Tuple, Optional, Literal
from enum import Enum


class BoundaryConditionType(str, Enum):
    """Valid boundary condition types."""
    WALL = "wall"
    INLET = "inlet"
    OUTLET = "outlet"
    SYMMETRY = "symmetry"
    FREESTREAM = "freestream"


class TransformConfig(BaseModel):
    """Transform configuration (translation + rotation)."""
    translation: Tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0),
        description="Translation vector (x, y, z)"
    )
    rotation_deg: float = Field(
        default=0.0,
        description="Rotation angle in degrees (2D: about z-axis, 3D: use rotation_xyz)"
    )
    rotation_xyz_deg: Optional[Tuple[float, float, float]] = Field(
        default=None,
        description="3D rotation angles (rx, ry, rz) in degrees (overrides rotation_deg if set)"
    )
    
    @field_validator('rotation_deg')
    @classmethod
    def normalize_angle(cls, v):
        """Normalize angle to [0, 360)."""
        return v % 360.0


class ComponentConfig(BaseModel):
    """Configuration for a single component."""
    name: str = Field(..., description="Unique component identifier")
    geometry_file: str = Field(..., description="Path to geometry file (JSON/XY)")
    transform: TransformConfig = Field(
        default_factory=TransformConfig,
        description="Placement transform"
    )
    boundary_condition: dict = Field(
        default={"type": "wall"},
        description="Boundary condition specification"
    )
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        """Check name is valid identifier."""
        if not v or not v.strip():
            raise ValueError("Component name cannot be empty")
        return v.strip()


class SolverConfig(BaseModel):
    """Panel solver configuration."""
    type: Literal["constant_source", "constant_doublet", "linear_source"] = Field(
        default="constant_source",
        description="Singularity type"
    )
    tolerance: float = Field(
        default=1e-10,
        gt=0,
        description="Linear solver tolerance"
    )
    max_iterations: Optional[int] = Field(
        default=None,
        gt=0,
        description="Max iterations for iterative solvers (None = direct solver)"
    )


class OutputConfig(BaseModel):
    """Output configuration."""
    directory: str = Field(
        default="./results",
        description="Output directory path"
    )
    formats: List[Literal["vtk", "csv", "hdf5"]] = Field(
        default=["vtk"],
        description="Output formats"
    )
    save_mesh: bool = Field(
        default=True,
        description="Save mesh geometry"
    )
    save_fields: bool = Field(
        default=True,
        description="Save solution fields"
    )


class FluidConfig(BaseModel):
    """Fluid properties configuration."""
    density: float = Field(
        default=1.225,
        gt=0,
        description="Fluid density [kg/m³]"
    )
    viscosity: Optional[float] = Field(
        default=None,
        gt=0,
        description="Dynamic viscosity [Pa·s] (optional, for BL calculations)"
    )
    thermal_conductivity: Optional[float] = Field(
        default=None,
        gt=0,
        description="Thermal conductivity [W/(m·K)] (optional)"
    )
    specific_heat_cp: Optional[float] = Field(
        default=None,
        gt=0,
        description="Specific heat at constant pressure [J/(kg·K)]"
    )
    gravity: float = Field(
        default=0.0,
        description="Gravitational acceleration in -y direction [m/s²]"
    )
    reference_pressure: float = Field(
        default=101325.0,
        description="Reference pressure for Bernoulli [Pa]"
    )
    reference_type: Literal["freestream", "outlet"] = Field(
        default="freestream",
        description="Reference condition type"
    )


class VisualizationConfig(BaseModel):
    """Visualization settings."""
    enabled: bool = Field(default=True, description="Enable visualization")
    show_mesh: bool = Field(default=True, description="Plot mesh geometry")
    show_normals: bool = Field(default=False, description="Show panel normals")
    
    # Domain settings
    domain: Optional[dict] = Field(
        default=None,
        description="Visualization domain {x_range: [min, max], y_range: [min, max]}"
    )
    resolution: Tuple[int, int] = Field(
        default=(150, 120),
        description="Grid resolution for field plots (nx, ny)"
    )
    
    # Legacy field (deprecated, use resolution instead)
    contour_resolution: Optional[Tuple[int, int]] = Field(
        default=None,
        description="[Deprecated] Use 'resolution' instead"
    )
    
    streamline_seeds: Optional[List[Tuple[float, float, float]]] = Field(
        default=None,
        description="Seed points for streamlines"
    )
    
    def get_resolution(self) -> Tuple[int, int]:
        """Get resolution, handling legacy field."""
        if self.contour_resolution is not None:
            return self.contour_resolution
        return self.resolution
    
    def get_x_range(self, default: Tuple[float, float] = (-2.0, 3.0)) -> Tuple[float, float]:
        """Get x_range from domain or return default."""
        if self.domain and 'x_range' in self.domain:
            return tuple(self.domain['x_range'])
        return default
    
    def get_y_range(self, default: Tuple[float, float] = (-2.0, 2.0)) -> Tuple[float, float]:
        """Get y_range from domain or return default."""
        if self.domain and 'y_range' in self.domain:
            return tuple(self.domain['y_range'])
        return default


class SimulationConfig(BaseModel):
    """Top-level simulation configuration."""
    name: str = Field(..., description="Simulation name")
    case_type: Literal[
        "hardcoded_panels_2d",
        "primitive_2d",
        "gmsh_2d",
        "gmsh_3d",
        "step_import"
    ] = Field(..., description="Case type")
    description: str = Field(default="", description="Case description")
    
    freestream: dict = Field(
        default={"velocity": [1.0, 0.0, 0.0]},
        description="Freestream conditions"
    )
    
    components: List[ComponentConfig] = Field(
        ...,
        min_length=1,
        description="List of components"
    )
    
    solver: SolverConfig = Field(
        default_factory=SolverConfig,
        description="Solver settings"
    )
    
    fluid: FluidConfig = Field(
        default_factory=FluidConfig,
        description="Fluid properties"
    )
    
    output: OutputConfig = Field(
        default_factory=OutputConfig,
        description="Output settings"
    )
    
    visualization: VisualizationConfig = Field(
        default_factory=VisualizationConfig,
        description="Visualization settings"
    )
    
    class Config:
        """Pydantic config."""
        extra = "forbid"  # Catch typos in YAML
        validate_assignment = True
    
    @field_validator('components')
    @classmethod
    def check_unique_names(cls, v):
        """Ensure component names are unique."""
        names = [comp.name for comp in v]
        if len(names) != len(set(names)):
            duplicates = [name for name in names if names.count(name) > 1]
            raise ValueError(f"Duplicate component names: {set(duplicates)}")
        return v
    
    def get_freestream_velocity(self) -> Tuple[float, float, float]:
        """Extract freestream velocity vector."""
        vel = self.freestream.get("velocity", [0.0, 0.0, 0.0])
        if len(vel) != 3:
            raise ValueError(f"freestream velocity must have 3 components, got {len(vel)}")
        return tuple(vel)
