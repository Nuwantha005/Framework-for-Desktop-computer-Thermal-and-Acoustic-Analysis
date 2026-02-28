# Prompt: Validate Tangential Velocity (Vt) Against OpenFOAM

## Context
The tangential velocity profile along body surfaces is the primary accuracy metric for the
panel method. Currently using potential differentiation (central difference of φ over arc
length) which improved accuracy over direct influence summation, but still has discrepancies
against OpenFOAM — particularly near corners and stagnation points. This prompt guides the
iterative debugging/comparison workflow.

## Workflow

### 1. Run Panel Solver
```bash
# Pick a case (e.g., cylinder, rounded_square, two_rounded_rects)
cd /path/to/panel-method-solver
python demos/demo_surface_comparison.py cases/<case_name> validation_results/<case_name>/openfoam --mesh-level <N> --output validation_results/<case_name>/comparison --show
```
Or use Python MCP to run programmatically:
```python
from core.io import CaseLoader
from postprocessing.surface import SurfaceDataExtractor

case = CaseLoader.load_case("cases/<case_name>")
solver = case.create_solver()
solver.solve()
solver.validate()  # Check Vn RMS < 1e-10

extractor = SurfaceDataExtractor(solver)
surface_data = extractor.extract()
# surface_data.arc_length, surface_data.Vt, surface_data.Cp
```

### 2. Extract OpenFOAM Reference
```python
from validation.adapters.openfoam.surface_extractor import OpenFOAMSurfaceExtractor

of_extractor = OpenFOAMSurfaceExtractor("validation_results/<case_name>/openfoam")
of_surface = of_extractor.extract()
```

### 3. Generate Envelope Comparison Plot
```python
from visualization.surface_envelope import plot_surface_envelope
from visualization.comparison import ComparisonVisualizer

comp_viz = ComparisonVisualizer()
comp_viz.compare_surface_distributions(
    panel_surface=surface_data,
    reference_surface=of_surface,
    quantities=["Vt", "Cp"],
    output_path="validation_results/<case>/comparison/"
)
```

### 4. Compute Error Metrics
```python
from validation.convergence.metrics import ErrorMetrics

metrics = ErrorMetrics.compute(panel_vt, of_vt_interpolated)
print(f"L2: {metrics.L2:.6f}, L∞: {metrics.Linf:.6f}, RMS: {metrics.rms:.6f}")
```

### 5. Diagnose and Iterate
Check these common Vt accuracy issues:
- **Near corners**: Panel discretization at sharp corners — increase panel density locally
- **Stagnation points**: φ differentiation can be noisy near V=0 — check numerical scheme
- **Panel count**: Insufficient discretization — try 2x panels and check convergence
- **Higher-order panels**: If constant panels plateau in accuracy, implement linear-strength
  (see `.agent/prompts/implement-solver.md`)
- **Potential gradient method**: Currently using central differences for dφ/ds — verify
  periodicity handling at wrap-around point

## Key Files
| File | Purpose |
|------|---------|
| `src/solvers/panel2d/spm.py` | `_compute_surface_potential()`, `_compute_surface_velocity()` |
| `src/postprocessing/surface.py` | `SurfaceDataExtractor` — extracts Vt, Cp along arc length |
| `src/visualization/surface_envelope.py` | Envelope plots (Vt/Cp wrapped around geometry) |
| `src/visualization/comparison.py` | `ComparisonVisualizer.compare_surface_distributions()` |
| `validation/comparison/surface.py` | Surface comparison utilities with interpolation |
| `validation/adapters/openfoam/surface_extractor.py` | OF postProcess VTP reader |
| `demos/demo_surface_comparison.py` | End-to-end CLI for panel vs OF surface comparison |

## Validation Cases (in order of complexity)
1. **cylinder_flow** — analytical solution available (Vt = 2V∞sinθ), best for debugging
2. **single_square** — sharp corners, tests discretization limits
3. **rounded_square** — smooth corners, good for higher-order panel validation
4. **two_rounded_rects** — multi-body interaction, tests cross-influence accuracy

## Success Criteria
- **Cylinder**: Vt L2 error < 2% against analytical, < 5% against OF at N=128
- **Rounded shapes**: Vt RMS error < 10% against OF (improves with higher-order panels)
- **Convergence**: Error should decrease monotonically with mesh refinement

## After Each Iteration
Record in `.agent/decisions/`:
- What was tried (method, mesh level, panel order)
- Metrics achieved (L2, L∞, RMS)
- What worked / didn't work
- Next step decision
