# Prompt: Implement Thermal Boundary Layer Solver (BDIM)

## Context
Implement a thermal boundary layer solver using the Boundary Domain Integral Method (BDIM)
based on the research paper by Gao et al. (2013). This solver computes the thermal boundary
layer and heat transfer coefficients given the velocity BL solution as input — the final
piece needed to calculate heat transfer rates from computer components.

## Reference Materials
- **Research paper (PDF → markdown)**: `notes_archived/bl_solvers/gao2013.pdf`
  - Convert to markdown using `marker-pdf` and place in same folder before starting
  - Key sections: BDIM formulation, fundamental solution, discretization, validation cases
- **Coupled solver design**: `notes_archived/solver_implementation/06_coupled_solvers.md`
  - "Thermal Solvers" section: Reynolds analogy approach as baseline
- **Von Kármán BL solver**: `src/solvers/boundary_layer/` (must be implemented first)
- **Theory doc (placeholder)**: `docs/theory/boundary_layers.md` → "Thermal BL" section

## Theory Overview
The energy equation for 2D steady incompressible flow with constant properties:

$$u\frac{\partial T}{\partial x} + v\frac{\partial T}{\partial y} = \alpha \nabla^2 T$$

BDIM transforms this PDE into a boundary integral equation using the fundamental solution
of the diffusion operator, avoiding the need for a volume mesh. The temperature field is
expressed as boundary integrals + domain integrals of the convective term.

### Two-Phase Approach
1. **Reynolds analogy baseline** (simple, fast):
   - $St = \frac{c_f}{2} \cdot Pr^{-2/3}$ (Chilton-Colburn analogy)
   - $Nu_x = St \cdot Re_x \cdot Pr$
   - Uses cf from the velocity BL solver directly
   - Good for validation of more complex methods

2. **Full BDIM** (from Gao 2013 paper):
   - Boundary integral formulation of energy equation
   - Handles non-uniform wall temperature, conjugate heat transfer
   - More accurate for complex geometries and separated regions

## Architecture

### Data Flow
```
Panel Solver → Ue(s) → BL Solver → δ, θ, cf(s) → Thermal Solver → Nu(s), h(s), q(s)
                                                          ↓
                                                   Heat transfer rate Q = ∫ h(Tw - T∞) ds
```

### File Structure
```
src/solvers/thermal/
├── __init__.py
├── base.py                  # ThermalSolver ABC, ThermalResult dataclass
├── reynolds_analogy.py      # ReynoldsAnalogyThermal — simple St-based
├── bdim/
│   ├── __init__.py
│   ├── solver.py            # BDIMThermalSolver — full BDIM formulation
│   ├── kernels.py           # Fundamental solutions, Green's functions
│   └── discretization.py    # Boundary element discretization
└── utils.py                 # Prandtl number correlations, fluid properties
```

### Key Classes
```python
@dataclass
class ThermalResult:
    """Output of thermal BL computation."""
    arc_length: NDArray          # s coordinates
    nusselt: NDArray             # Local Nusselt number Nu(s)
    heat_transfer_coeff: NDArray # Local h(s) [W/m²K]
    wall_heat_flux: NDArray      # q(s) = h(Tw - T∞)
    thermal_bl_thickness: NDArray # δ_T(s)
    total_heat_rate: float       # Q = ∫ q ds [W/m]

@dataclass
class ThermalSolver(ABC):
    """Base thermal BL solver."""
    bl_result: BoundaryLayerResult  # From velocity BL solver
    T_wall: float | NDArray         # Wall temperature [K] (uniform or distribution)
    T_inf: float                    # Freestream temperature [K]
    Pr: float                       # Prandtl number
    k: float                        # Thermal conductivity [W/mK]
    
    @abstractmethod
    def solve(self) -> ThermalResult: ...
```

## Implementation Order
1. **Reynolds analogy first** — simple, validates the pipeline, gives baseline Nu(s)
2. **BDIM formulation** — implement after reading/converting the Gao 2013 paper
3. **Validation** — compare both against analytical and OF results

## Validation Approach
1. **Flat plate**: Analytical solution $Nu_x = 0.332 \cdot Re_x^{1/2} \cdot Pr^{1/3}$ (laminar)
2. **Cylinder in crossflow**: Correlations (Churchill-Bernstein) for average Nu
3. **OpenFOAM comparison**: Run OF with energy equation enabled, extract Nu from wallHeatFlux
4. **Cross-validate**: Reynolds analogy vs BDIM on same geometry

## Integration with Case System
Add thermal properties to case YAML:
```yaml
fluid:
  density: 1.225
  viscosity: 1.81e-5
  thermal_conductivity: 0.0262
  specific_heat: 1005.0
  prandtl: 0.71
boundary_conditions:
  wall_temperature: 350.0   # K (or per-component)
  freestream_temperature: 300.0
```
Update `src/core/config/schemas.py` to include thermal config fields.

## Success Criteria
- Reynolds analogy Nu within 5% of flat-plate analytical
- BDIM matches OF heat flux within 15% for cylinder
- Can compute total heat transfer rate Q [W/m] for any case geometry
- Clean separation: thermal solver only depends on BL result + thermal properties

## The End Goal
Once this works:
```python
# Full pipeline: geometry → panel solve → BL → thermal → heat transfer rate
case = CaseLoader.load_case("cases/computer_component")
solver = case.create_solver()
solver.solve()
Ue = SurfaceDataExtractor(solver).extract().Vt
bl = BoundaryLayerSolver(Ue, arc_length, nu, profile=Pohlhausen()).solve()
thermal = ReynoldsAnalogyThermal(bl, T_wall=350, T_inf=300, Pr=0.71, k=0.026).solve()
print(f"Total heat transfer: {thermal.total_heat_rate:.1f} W/m")
```

## Post-Implementation
- Register in `SolverFactory` under category `"thermal"`
- Update `.agent/modules/solver.md`
- Update `docs/theory/boundary_layers.md` with thermal BL equations
- Record BDIM vs Reynolds analogy comparison in `.agent/decisions/`
