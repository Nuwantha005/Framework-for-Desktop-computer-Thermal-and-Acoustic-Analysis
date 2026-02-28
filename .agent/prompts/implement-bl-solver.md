# Prompt: Implement Viscous Boundary Layer Solver

## Context
Implement a viscous boundary layer solver using the Von Kármán momentum integral method.
This is the first step toward viscous-inviscid coupling: the panel method provides the
edge velocity Ue(s), the BL solver computes δ*(s), θ(s), cf(s), and eventually the
transpiration velocity feeds back to modify the panel method BC.

**Key constraint**: The solver must be **versatile** — it accepts different velocity profiles
as input (Blasius, Pohlhausen, Falkner-Skan, power-law, etc.) and the goal is to experiment
with all of them to find the best fit against OpenFOAM results.

## Reference Materials
- **Von Kármán equations**: `notes_archived/bl_solvers/Von Kármán Momentum Integral.md`
- **Coupled solver design**: `notes_archived/solver_implementation/06_coupled_solvers.md`
- **Theory doc (placeholder)**: `docs/theory/boundary_layers.md`
- **Target directory**: `src/solvers/boundary_layer/` (currently has README.md only)

## Von Kármán Momentum Integral Equations
The integral momentum equation for 2D steady incompressible flow:

$$\frac{d\theta}{ds} + \frac{\theta}{U_e}\frac{dU_e}{ds}(2 + H) = \frac{c_f}{2}$$

Where:
- θ(s) = momentum thickness
- H = δ*/θ = shape factor
- Ue(s) = edge velocity (from panel solver)
- cf/2 = wall shear stress coefficient

Closure requires a **velocity profile assumption** that provides relationships between
H, cf, and θ (or Reθ). This is where the different velocity profiles come in.

## Architecture

### Velocity Profile Interface
```python
@dataclass
class VelocityProfile(ABC):
    """Base class for BL velocity profile parameterizations."""
    
    @abstractmethod
    def shape_factor(self, *params) -> float:
        """H = δ*/θ as function of profile parameter."""
    
    @abstractmethod
    def skin_friction(self, Re_theta: float, *params) -> float:
        """cf/2 as function of Re_θ and profile parameter."""
    
    @abstractmethod
    def displacement_thickness_ratio(self, *params) -> float:
        """δ*/δ as function of profile parameter."""
```

### Profiles to Implement (experiment with all)
1. **Blasius** — flat-plate analytical: H = 2.59, cf/2 = 0.332/√Reθ
2. **Pohlhausen (4th-order polynomial)** — Λ parameter via pressure gradient
3. **Falkner-Skan** — wedge flow similarity: β parameter family
4. **Power-law** (1/n) — turbulent approximation: u/Ue = (y/δ)^(1/n)
5. **Thwaites' correlation** — one-parameter method: λ = θ²/ν · dUe/ds

### BL Solver Class
```python
@dataclass
class BoundaryLayerSolver:
    """Von Kármán integral BL solver with pluggable velocity profiles."""
    
    edge_velocity: NDArray     # Ue(s) from panel solver, shape (M,)
    arc_length: NDArray        # s coordinates, shape (M,)
    nu: float                  # kinematic viscosity
    profile: VelocityProfile   # pluggable velocity profile
    
    def solve(self) -> BoundaryLayerResult:
        """Integrate momentum equation along surface.
        
        Returns BoundaryLayerResult with:
        - theta(s): momentum thickness
        - delta_star(s): displacement thickness  
        - cf(s): skin friction coefficient
        - H(s): shape factor
        - transition_s: estimated transition location (if applicable)
        """
```

### Integration Approach
- March from stagnation point (s=0) outward along each surface
- For bluff bodies: stagnation point is where Ue ≈ 0 (front of body)
- Use RK4 or adaptive RK45 (scipy.integrate.solve_ivp) for the ODE
- Handle Ue=0 singularity at stagnation with Thwaites' starting procedure

## File Structure
```
src/solvers/boundary_layer/
├── __init__.py
├── base.py              # BoundaryLayerSolver, BoundaryLayerResult
├── profiles/
│   ├── __init__.py
│   ├── base.py          # VelocityProfile ABC
│   ├── blasius.py       # Blasius profile
│   ├── pohlhausen.py   # Pohlhausen 4th-order
│   ├── falkner_skan.py  # Falkner-Skan family
│   ├── power_law.py     # Power-law (turbulent)
│   └── thwaites.py      # Thwaites' correlation
└── transition.py        # Transition prediction (Michel, e^N)
```

## Validation Approach
1. **Flat plate (Blasius)**: Compare θ, δ*, cf against analytical Blasius solution
2. **Cylinder**: Compare against Thwaites' method analytical result
3. **OpenFOAM comparison**: Extract BL quantities from OF wallShearStress, compare cf(s)
4. **Profile comparison**: Run all 5 profiles on same case, plot δ*, θ, cf side-by-side,
   identify which matches OF best for bluff bodies

## Integration with Panel Method
The BL solver receives Ue(s) from `SurfaceDataExtractor.extract().Vt` — this is the
direct connection between the inviscid panel solution and viscous BL computation.

For viscous-inviscid coupling (later phase):
```python
# Iterate: panel → BL → modify BC → re-solve panel
for iteration in range(max_iter):
    solver.solve()
    Ue = extractor.extract().Vt
    bl_result = bl_solver.solve(Ue)
    transpiration_velocity = compute_transpiration(bl_result)
    solver.update_bc(transpiration_velocity)  # modify Neumann BC
    if converged(bl_result):
        break
```

## Success Criteria
- Flat plate cf within 1% of Blasius analytical
- BL thickness trends match OF qualitatively on cylinder and rounded shapes
- At least 3 velocity profiles implemented and compared
- Clean API that accepts any VelocityProfile subclass

## Post-Implementation
- Register in `SolverFactory` under category `"boundary_layer"`
- Update `.agent/modules/solver.md`
- Record which velocity profile works best in `.agent/decisions/`
- Update `docs/theory/boundary_layers.md` with implemented equations
