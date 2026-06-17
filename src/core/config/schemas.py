"""
Pydantic schemas for configuration validation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Tuple, Optional, Literal, Any
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


class GeometryConfig(BaseModel):
    """Parametric geometry configuration."""
    type: str = Field(..., description="Geometry type (circle, rectangle, rounded_rectangle, sphere, cylinder, box, external)")
    parameters: dict = Field(default_factory=dict, description="Shape parameters (width, height, radius, etc.)")
    file: Optional[str] = Field(None, description="Path to geometry file for external meshes")
    
    @field_validator('type')
    @classmethod
    def validate_type(cls, v):
        """Check type is non-empty string."""
        if not v or not v.strip():
            raise ValueError("Geometry type cannot be empty")
        return v.strip()


class ComponentConfig(BaseModel):
    """Configuration for a single component."""
    name: str = Field(..., description="Unique component identifier")
    geometry_file: Optional[str] = Field(None, description="Path to geometry file (JSON/XY) - for legacy cases")
    geometry: Optional[GeometryConfig] = Field(None, description="Parametric geometry definition")
    mesh_levels: Optional[List[Any]] = Field(
        None,
        description="Resolution levels for parametric geometry (each level can be a list of ints or a float for characteristic length)"
    )
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
    
    def model_post_init(self, __context):
        """Validate that exactly one of geometry_file or geometry is set."""
        if self.geometry_file is None and self.geometry is None:
            raise ValueError(f"Component '{self.name}': must specify either 'geometry_file' or 'geometry'")
        if self.geometry_file is not None and self.geometry is not None:
            raise ValueError(f"Component '{self.name}': cannot specify both 'geometry_file' and 'geometry'")
        
        # Validate mesh_levels is provided for parametric geometry
        if self.geometry is not None and self.geometry.type != "external" and self.mesh_levels is None:
            raise ValueError(f"Component '{self.name}': parametric geometry requires 'mesh_levels'")
        if self.geometry_file is not None and self.mesh_levels is not None:
            raise ValueError(f"Component '{self.name}': 'mesh_levels' only valid for parametric geometry")


class SolverConfig(BaseModel):
    """Panel solver configuration."""
    
    # Panel method dimensions (new format)
    singularity_type: Literal[
        "source", "doublet", "vortex",
        "source_doublet", "source_vortex",
        "doublet_vortex", "source_doublet_vortex"
    ] = Field(
        default="source",
        description="Singularity element type(s)"
    )
    panel_order: Literal["constant", "linear", "quadratic"] = Field(
        default="constant",
        description="Panel order (constant=0th, linear=1st, quadratic=2nd)"
    )
    panel_geometry: Literal["flat", "curved"] = Field(
        default="flat",
        description="Panel geometry type"
    )
    
    # Solver parameters
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
    
    # Legacy field for backward compatibility
    type: Optional[Literal["constant_source", "constant_doublet", "linear_source", "linear_vortex", "source_doublet", "linear_source_doublet"]] = Field(
        default=None,
        description="[Deprecated] Use singularity_type, panel_order, panel_geometry instead"
    )
    
    def model_post_init(self, __context):
        """Handle legacy 'type' field by mapping to new fields."""
        if self.type is not None:
            # Map old format to new three-dimensional format
            type_mapping = {
                "constant_source": ("source", "constant", "flat"),
                "constant_doublet": ("doublet", "constant", "flat"),
                "linear_source": ("source", "linear", "flat"),
                "linear_vortex": ("vortex", "linear", "flat"),
                "source_doublet": ("source_doublet", "constant", "flat"),
                "linear_source_doublet": ("source_doublet", "linear", "flat"),
            }
            if self.type in type_mapping:
                s, o, g = type_mapping[self.type]
                object.__setattr__(self, 'singularity_type', s)
                object.__setattr__(self, 'panel_order', o)
                object.__setattr__(self, 'panel_geometry', g)


class BoundaryLayerConfig(BaseModel):
    """Boundary layer solver configuration (optional section in case.yaml)."""
    enabled: bool = Field(
        default=False,
        description="Enable boundary layer computation"
    )
    profiles: List[Literal[
        "blasius", "pohlhausen", "falkner_skan",
        "power_law", "thwaites"
    ]] = Field(
        default=["thwaites"],
        description="Velocity profile(s) to run. Use multiple to compare."
    )
    power_law_n: int = Field(
        default=7,
        ge=3,
        le=20,
        description="Exponent for power-law profile (1/n law)"
    )
    transition_model: Optional[Literal["michel", "en"]] = Field(
        default=None,
        description="Transition prediction model (None = skip)"
    )
    n_crit: float = Field(
        default=9.0,
        gt=0,
        description="Critical amplification factor for e^N method"
    )
    envelope_scale: float = Field(
        default=0.15,
        gt=0,
        description="Scale factor for cf envelope plots"
    )
    thwaites_reconstruction: Literal["falkner_skan", "pohlhausen"] = Field(
        default="falkner_skan",
        description="Reconstruction pairing for Thwaites profile"
    )
    reconstruction_n_y: int = Field(
        default=80,
        ge=10,
        le=500,
        description="Number of wall-normal grid points for velocity reconstruction"
    )


class ThermalConfig(BaseModel):
    """Thermal boundary layer solver configuration (optional section in case.yaml)."""
    enabled: bool = Field(
        default=False,
        description="Enable thermal boundary layer computation"
    )
    solver: Literal["reynolds_analogy", "bdim"] = Field(
        default="reynolds_analogy",
        description="Thermal solver type: reynolds_analogy (simple, fast) or bdim (full integral method)"
    )
    envelope_scale: float = Field(
        default=0.15,
        gt=0,
        description="Scale factor for thermal envelope plots"
    )



class BoundaryRegionConfig(BaseModel):
    name: str = Field(..., description="Boundary region name")
    shape: Literal["circle", "rectangle"] = Field(default="circle")
    center: Tuple[float, float, float]
    normal: Tuple[float, float, float]
    radius: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None
    n_r: int = 6
    n_theta: int = 24


class ActuatorDiskConfig(BaseModel):
    """Configuration for one actuator disk / fan."""
    name: str = Field(..., description="Unique actuator disk identifier")
    center: Tuple[float, float, float] = Field(
        ...,
        description="Disk center coordinates [m]"
    )
    normal: Tuple[float, float, float] = Field(
        ...,
        description="Disk normal; defines positive flow and pressure-rise direction"
    )
    radius: float = Field(..., gt=0, description="Disk radius [m]")
    n_r: int = Field(default=6, ge=1, description="Number of radial disk subdivisions")
    n_theta: int = Field(default=48, ge=3, description="Number of azimuthal disk subdivisions")
    curve_file: str = Field(..., description="Fan P-Q curve CSV path relative to case directory")
    interpolation: Literal["linear", "cubic"] = Field(
        default="linear",
        description="Fan curve interpolation method"
    )
    dp_initial: Optional[float] = Field(
        default=None,
        description="Initial pressure rise [Pa]. If omitted, use curve midpoint."
    )
    relaxation: float = Field(
        default=0.4,
        gt=0,
        le=1,
        description="Under-relaxation factor for P-Q iteration"
    )
    tolerance: float = Field(
        default=1e-3,
        gt=0,
        description="Pressure-rise convergence tolerance [Pa]"
    )
    max_iterations: int = Field(
        default=50,
        ge=1,
        description="Maximum P-Q iterations"
    )

    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        """Check name is valid."""
        if not v or not v.strip():
            raise ValueError("Actuator disk name cannot be empty")
        return v.strip()

    @field_validator('normal')
    @classmethod
    def validate_normal(cls, v):
        """Check disk normal is nonzero."""
        norm_sq = sum(component * component for component in v)
        if norm_sq <= 1e-24:
            raise ValueError("Actuator disk normal must be nonzero")
        return v


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
    freestream_temperature: Optional[float] = Field(
        default=None,
        gt=0,
        description="Freestream temperature [K] (optional, for thermal calculations)"
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
        description="Visualization domain {x_range: [min, max], y_range: [min, max], z_range: [min, max]}"
    )
    resolution: Tuple[int, ...] = Field(
        default=(150, 120),
        description="Grid resolution for field plots (nx, ny, [nz])"
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

    def get_z_range(self, default: Tuple[float, float] = (-2.0, 2.0)) -> Tuple[float, float]:
        """Get z_range from domain or return default."""
        if self.domain and 'z_range' in self.domain:
            return tuple(self.domain['z_range'])
        return default


class SimulationConfig(BaseModel):
    """Top-level simulation configuration."""
    name: str = Field(..., description="Simulation name")
    case_type: Literal[
        "hardcoded_panels_2d",
        "primitive_2d",
        "parametric_2d",
        "gmsh_2d",
        "gmsh_3d",
        "step_import",
        "parametric_3d",
        "external_3d"
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
    
    boundary_layer: BoundaryLayerConfig = Field(
        default_factory=BoundaryLayerConfig,
        description="Boundary layer solver settings (optional)"
    )
    
    thermal: ThermalConfig = Field(
        default_factory=ThermalConfig,
        description="Thermal boundary layer solver settings (optional)"
    )

    actuator_disks: List[ActuatorDiskConfig] = Field(
        default_factory=list,
        description="Actuator disk / fan definitions (optional)"
    )
    
    inlets: List[BoundaryRegionConfig] = Field(
        default_factory=list,
        description="Inlet boundary conditions (optional)"
    )
    
    outlets: List[BoundaryRegionConfig] = Field(
        default_factory=list,
        description="Outlet boundary conditions (optional)"
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

    @field_validator('actuator_disks')
    @classmethod
    def check_unique_actuator_disk_names(cls, v):
        """Ensure actuator disk names are unique."""
        names = [disk.name for disk in v]
        if len(names) != len(set(names)):
            duplicates = [name for name in names if names.count(name) > 1]
            raise ValueError(f"Duplicate actuator disk names: {set(duplicates)}")
        return v
    
    def get_freestream_velocity(self) -> Tuple[float, float, float]:
        """Extract freestream velocity vector."""
        vel = self.freestream.get("velocity", [0.0, 0.0, 0.0])
        if len(vel) != 3:
            raise ValueError(f"freestream velocity must have 3 components, got {len(vel)}")
        return tuple(vel)
