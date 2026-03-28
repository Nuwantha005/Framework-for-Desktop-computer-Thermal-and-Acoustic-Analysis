---
Implementation Plan: Connecting Viscous BL to Thermal BL Solver
Overview
This task involves:
1. Config/IO updates — Add thermal properties to case YAML schema
2. Thermal solver integration — Fix the stub BoundaryLayerResult usage, connect real BL results
3. Demo script — Full pipeline: panel → viscous BL → thermal BL → visualizations
4. Visualization — Temperature/heat transfer coefficient envelope and s-y plots
5. Documentation — Update .agent/modules/ and docs/
---
Phase 1: Case Configuration Updates
Files to modify:
- src/core/config/schemas.py
- src/core/geometry/component.py
- src/core/io/case_loader.py
- src/core/io/case.py
Changes:
1. Add to FluidConfig:
      freestream_temperature: Optional[float] = Field(
       default=None,
       gt=0,
       description="Freestream temperature [K] for thermal calculations"
   )
   
2. Add to ComponentConfig.boundary_condition handling:
   - Support heat_flux key (scalar per component, W/m²)
   - Example YAML:
          boundary_condition:
       type: "wall"
       heat_flux: 1000.0  # W/m²
     
3. Update Component class:
   - Add bc_heat_flux: Optional[float] = None
4. Update CaseLoader._build_scene():
   - Extract bc_heat_flux from boundary_condition dict
5. Update Case class:
   - Add convenience method get_thermal_config() returning freestream temp + per-component heat fluxes
---
Phase 2: Thermal Solver Interface Fixes
Files to modify:
- src/solvers/thermal/base.py
- src/solvers/thermal/reynolds_analogy.py
Changes:
1. Remove duplicate BoundaryLayerResult stub from thermal/base.py:
   - Instead, thermal solver will accept the fields it needs directly (not the full BL result object)
   - This avoids coupling issues since the thermal solver only needs: arc_length, Ue, cf, and optionally delta
2. Create adapter function to extract thermal-relevant data from real BoundaryLayerResult:
      @dataclass
   class ThermalBLInput:
       """Input data for thermal solver, extracted from viscous BL."""
       arc_length: NDArray    # s coordinates [m]
       Ue: NDArray            # Edge velocity [m/s]
       cf: NDArray            # Skin friction coefficient
       delta: Optional[NDArray] = None  # BL thickness for delta_T estimation
   
   def extract_thermal_input(bl_result: BoundaryLayerResult) -> ThermalBLInput:
       """Convert viscous BL result to thermal solver input."""
       # Handle NaN masking for separation region
       ...
   
3. Update ReynoldsAnalogyThermal:
   - Accept ThermalBLInput or raw arrays instead of the stub
   - Compute thermal_bl_thickness properly: δ_T ≈ δ / Pr^(1/3)
   - Handle NaN regions (separation) gracefully
4. Key consideration: The thermal solver should only run on the valid region before separation:
   - Filter out NaN values from the BL result
   - The separation point detection is already done in viscous BL (H > 3.5 or Ue < 5% of peak)
---
Phase 3: Thermal Result Container
Files to modify:
- src/solvers/thermal/base.py
Add per-side result container:
@dataclass
class ThermalPathResult:
    """Thermal BL results for one surface streamline (upper or lower)."""
    side: str                      # "upper" or "lower"
    arc_length: NDArray           # Valid s coordinates [m]
    temperature: NDArray          # Wall temperature T_w(s) [K]
    heat_transfer_coeff: NDArray  # h(s) [W/m²K]
    nusselt: NDArray              # Nu(s)
    heat_flux: NDArray            # q_w(s) [W/m²]
    thermal_bl_thickness: NDArray # δ_T(s) [m]
    total_heat_rate: float        # Q = ∫q ds [W/m]
---
Phase 4: Demo Script
New file: demos/demo_thermal_bl.py
Workflow:
1. Parse arguments (case path, profiles, options)
2. Load case with CaseLoader.load_case()
3. Create and solve panel method
4. Run viscous BL with BoundaryLayerRunner.run()
5. For each side (upper/lower):
   - Extract valid (non-NaN) BL data
   - Get heat flux from case config for the component
   - Run ReynoldsAnalogyThermal solver
   - Store results
6. Generate visualizations (saved to cases/<name>/out/thermal/)
7. Print summary metrics
Command line interface:
python demos/demo_thermal_bl.py cases/rounded_square
python demos/demo_thermal_bl.py cases/rounded_square --profile thwaites
python demos/demo_thermal_bl.py cases/rounded_square --heat-flux 500.0
---
### Phase 5: Visualization
**New file:** `src/visualization/thermal_plots.py`
**Functions to implement:**
1. **`plot_thermal_envelope()`** — Temperature or h(s) wrapped around body geometry
   - Reuse `plot_surface_envelope()` from `surface_envelope.py`
   - Similar pattern to `plot_bl_envelope()`
2. **`plot_thermal_envelope_two_sides()`** — Upper and lower thermal envelopes combined
3. **`plot_thermal_contour()`** — s-y contour plot (placeholder for future BDIM results)
   - For Reynolds analogy, this is simpler since we only have surface values
   - Could show T(s) as a line plot overlaid on BL velocity contour
4. **`plot_thermal_line()`** — Line plot of T_w, h, Nu vs arc-length
   - Similar to `plot_bl_line()` pattern
   - Mark separation region
5. **`plot_thermal_two_sides()`** — Side-by-side plots for upper/lower
**Reuse existing utilities:**
- `_cell_edges()` from `bl_plot_common.py`
- `plot_surface_envelope()` from `surface_envelope.py`
- Color/style patterns from `bl_line_plots.py`
---
Phase 6: Documentation
Files to update:
1. .agent/modules/solver.md:
   - Add thermal solver section under solvers
   - Document ReynoldsAnalogyThermal API
   - Document integration with viscous BL
2. docs/theory/thermal_boundary_layer.md:
   - Already exists with BDIM theory
   - Add Reynolds analogy section at the beginning
   - Document connection to viscous BL
3. docs/user_guide/ (new or existing):
   - Add thermal analysis workflow section
   - Show example case YAML with thermal config
---
Summary of Files
Phase	Action
1	Modify
1	Modify
1	Modify
1	Modify
2-3	Modify
2	Modify
4	Create
5	Create
5	Modify
6	Modify
6	Modify
---
