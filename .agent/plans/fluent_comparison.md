
Comprehensive Plan for Fluent Comparison Pipeline


Based on my analysis, here's the proposed implementation plan:
Architecture Overview
```
src/validation/adapters/fluent/
├── __init__.py           # Public exports
├── data_types.py         # FluentFieldData, FluentWallData dataclasses
├── ascii_reader.py       # Parse ASCII export files (filed_data, wall_data)
├── bl_extractor.py       # Extract BL quantities from Fluent data:
│                         #   - Split into upper/lower paths (using GeometryMapper)
│                         #   - Compute Ue from wall pressure (Bernoulli)
│                         #   - Compute δ(s) by marching along normal
│                         #   - Compute Cf from wall shear
│                         #   - Detect separation point
├── interpolator.py       # Interpolate Fluent data onto BL solver grid
└── comparison.py         # BLComparisonResult, comparison metrics
```
Data Flow
```
Fluent Export (ASCII)
        │
        ▼
 ascii_reader.py ──► FluentFieldData / FluentWallData
        │
        ▼
 bl_extractor.py ──► FluentBLResult (per-side: Ue, δ, Cf, separation)
        │               uses GeometryMapper for arc-length
        │               uses case.yaml for reference conditions
        ▼
interpolator.py ──► Interpolated data at BL solver grid points
        │
        ▼
 comparison.py ──► BLComparisonResult with error metrics
        │
        ▼
  bl_plots.py    ──► plot_bl_of_comparison() (existing placeholder)
```

Implementation Phases

Phase 1: Data Loading (ASCII Reader)
- Parse comma-separated ASCII format from Fluent
- Return FluentFieldData (nodes with x, y, p, vx, vy)
- Return FluentWallData (wall nodes with x, y, p, τw)
- Future-proof: Abstract reader interface for later binary/VTK formats

Phase 2: BL Quantity Extraction
- Use GeometryMapper to project Fluent points to body surface and compute arc-length
- Split wall data into upper/lower paths using stagnation detection (same as BL runner)
- Compute Ue from wall pressure: Ue = sqrt(2(P0∞ - Pwall)/ρ)
- Compute δ(s) by marching along panel normal until Vt ≈ 0.99 Ue
- Compute Cf from wall shear: Cf = τw / (0.5 ρ U∞²)
- Detect separation (τw = 0 crossover)

Phase 3: Interpolation
- Use BL solver grid (s, y coordinates) as reference
- Transform (s, y) to global (x, y) using panel geometry
- Interpolate Fluent data (scattered) onto these global coordinates
- Convert Fluent velocity vector to tangential component using panel normal

Phase 4: Comparison & Metrics
- BLComparisonResult dataclass holding:
  - BL solver results
  - Fluent extracted results (interpolated)
  - Error metrics per quantity (L2, L∞, RMS, MAE)
- Wire up to existing plot_bl_of_comparison() function

Phase 5: Demo Integration
- Update demos/demo_boundary_layer.py to optionally load Fluent data
- Pass of_field (from Fluent) to plot_bl_of_comparison()
Config Integration
Extend viz_config.yaml with new section:
fluent_comparison:
  enabled: true
  quantities: ['velocity', 'Ue', 'delta', 'Cf']
  error_metrics: true
Case-specific paths derived automatically:
- cases/<name>/fluent_case/export/viscous_bl/filed_data
- cases/<name>/fluent_case/export/viscous_bl/wall_data
Reference conditions from case.yaml:
- fluid.density for ρ
- fluid.reference_pressure for P0∞
- freestream.velocity for U∞


Final Implementation Plan

Module Structure
```
src/validation/adapters/fluent/
├── __init__.py               # Public API exports
├── data_types.py             # FluentFieldData, FluentWallData, FluentBLResult
├── ascii_reader.py           # Parse Fluent ASCII exports
├── bl_extractor.py           # Extract BL quantities (Ue, δ, Cf, separation)
├── interpolator.py           # Coordinate transforms & interpolation
└── comparison.py             # BLComparisonRunner, BLComparisonResult
```

Detailed File Responsibilities
1. data_types.py
```
@dataclass
class FluentFieldData:
    """Raw field data from Fluent filed_data export."""
    node_id: NDArray[np.int32]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    pressure: NDArray[np.float64]
    vx: NDArray[np.float64]
    vy: NDArray[np.float64]
@dataclass
class FluentWallData:
    """Raw wall data from Fluent wall_data export."""
    node_id: NDArray[np.int32]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    pressure: NDArray[np.float64]
    wall_shear: NDArray[np.float64]
@dataclass
class FluentBLPathResult:
    """Extracted BL quantities for one path (upper or lower)."""
    side: str
    s: NDArray[np.float64]          # Arc-length stations
    x: NDArray[np.float64]          # Surface coordinates
    y: NDArray[np.float64]
    Ue: NDArray[np.float64]         # Edge velocity from Bernoulli
    delta: NDArray[np.float64]      # BL thickness (0.99 Ue criterion)
    Cf: NDArray[np.float64]         # Skin friction coefficient
    tau_w: NDArray[np.float64]      # Wall shear stress
    separation_s: Optional[float]   # Arc-length at separation
@dataclass
class FluentBLResult:
    """Two-sided BL result from Fluent."""
    upper: FluentBLPathResult
    lower: FluentBLPathResult
    rho: float
    U_inf: float
    P0_inf: float
```

2. ascii_reader.py

```
def read_field_data(path: Path) -> FluentFieldData:
    """Parse Fluent ASCII filed_data export."""
    
def read_wall_data(path: Path) -> FluentWallData:
    """Parse Fluent ASCII wall_data export."""
```
3. bl_extractor.py
```
class FluentBLExtractor:
    """Extract BL quantities from Fluent data aligned to panel mesh."""
    
    def __init__(
        self,
        bl_result: BoundaryLayerCaseResult,  # From panel method
        field_data: FluentFieldData,
        wall_data: FluentWallData,
        rho: float,
        U_inf: float,
        P0_inf: float,
    ): ...
    
    def extract(self) -> FluentBLResult:
        """
        1. Project wall data points to surface using GeometryMapper
        2. Split into upper/lower using BL result stagnation points
        3. Compute Ue = sqrt(2(P0-Pw)/ρ)
        4. Compute δ by marching along panel normals in field_data
        5. Compute Cf = τw / (0.5 ρ U∞²)
        6. Detect separation (τw sign change)
        """
```
4. interpolator.py
```
class BLFieldInterpolator:
    """Transform BL solver grid to global coords & interpolate Fluent data."""
    
    def __init__(
        self,
        bl_path: BoundaryLayerPathResult,
        mesh: Mesh,  # For panel normals
    ): ...
    
    def transform_to_global(
        self,
        field: BLFieldData,  # (s, y) grid
    ) -> Tuple[NDArray, NDArray]:
        """
        For each (s[i], y[j]) in BLFieldData:
        1. Find panel index for arc-length s[i]
        2. Get panel centre (x, y) and outward normal n
        3. global_x = x + y[j] * n[0]
        4. global_y = y + y[j] * n[1]
        Returns (X, Y) global coordinate grids
        """
    
    def interpolate_fluent_velocity(
        self,
        global_X: NDArray,
        global_Y: NDArray,
        fluent_field: FluentFieldData,
        panel_tangents: NDArray,  # For projection
    ) -> NDArray:
        """
        5. Build scipy interpolator from Fluent scattered data
        6. Interpolate (vx, vy) at global coordinates
        7. Project onto tangent: Vt = vx*tx + vy*ty
        Returns tangential velocity grid matching BLFieldData shape
        """
```
5. comparison.py
```
@dataclass
class BLComparisonMetrics:
    """Error metrics for one quantity."""
    L2: float
    L_inf: float
    RMS: float
    MAE: float
@dataclass
class BLComparisonResult:
    """Complete comparison result for visualization."""
    bl_result: BoundaryLayerCaseResult
    fluent_result: FluentBLResult
    
    # Interpolated Fluent data on BL solver grid
    upper_fluent_field: Optional[BLFieldData]
    lower_fluent_field: Optional[BLFieldData]
    
    # Error metrics per quantity
    metrics: Dict[str, Dict[str, BLComparisonMetrics]]  # {side: {qty: metrics}}
class BLComparisonRunner:
    """Orchestrate BL comparison pipeline."""
    
    def __init__(self, case: Case, bl_result: BoundaryLayerCaseResult): ...
    
    def run(
        self,
        profile_name: str = "thwaites",
    ) -> BLComparisonResult:
        """
        1. Check for Fluent exports; warn and return None if missing
        2. Load Fluent ASCII data
        3. Extract BL quantities from Fluent
        4. Interpolate Fluent onto BL solver grid
        5. Compute comparison metrics
        6. Return BLComparisonResult for visualization
        """
```

Integration Points
1. Update plot_bl_of_comparison() in bl_plots.py
- Replace of_field parameter with fluent_field: Optional[BLFieldData]
- Implement actual difference contour when fluent_field is provided
1. Update demos/demo_boundary_layer.py
- Add --compare-fluent flag
- Load Fluent data via BLComparisonRunner
- Pass to plot_bl_of_comparison()
1. Config in viz_config.yaml
fluent_comparison:
  enabled: true
  quantities:
    - velocity_magnitude
    - Ue
    - delta
    - Cf
  wall_plots: true      # Ue, δ, Cf line plots
  field_plots: true     # s-y velocity contour
  error_metrics: true
Error Handling
- Missing Fluent exports → warning + skip comparison (graceful)
- Malformed ASCII → warning with context + skip
- Interpolation failure → warning + NaN in affected cell