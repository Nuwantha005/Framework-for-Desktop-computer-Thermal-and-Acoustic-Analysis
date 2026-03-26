# Validation Module State
**Last modified**: 2026-03-26

## Files
- `validation/__init__.py` — exports OpenFOAMCaseGenerator, Runner, etc.
- `validation/geometry_mapper.py` — `GeometryMapper`: projects query points onto reference polyline, computes arc length
- `validation/adapters/openfoam/case_generator.py` (1137 lines) — `MeshSettings`, `SolverType`, `OpenFOAMCaseGenerator`
- `validation/adapters/openfoam/foamlib_generator.py` — `FoamlibCaseGenerator` (preferred generator using foamlib)
- `validation/adapters/openfoam/runner.py` — `OpenFOAMRunner`
- `validation/adapters/openfoam/geometry_converter.py` — `GeometryConverter` (mesh → STL via trimesh)
- `validation/adapters/openfoam/surface_extractor.py` (518 lines) — `OpenFOAMSurfaceExtractor` (reads VTP from postProcessing)
- `validation/comparison/{grid,surface,probe}.py` — grid creation, surface velocity comparison, probe comparison
- `validation/convergence/{metrics,panel,openfoam}.py` — `ErrorMetrics`, `GCIResult`, panel/OpenFOAM convergence runners
- `validation/scripts/` — legacy OF pipeline + current BL Fluent plotting scripts
- `src/validation/adapters/fluent/` — Fluent BL adapter stack:
  - `ascii_reader.py` (load `filed_data` / `wall_data`)
  - `bl_extractor.py` (extract Ue, Cf, delta along BL paths)
  - `interpolator.py` (map Fluent velocity onto BL field grid)
  - `comparison.py` (`BLComparisonRunner`, `BLComparisonResult`, metrics)

## Public API
- `FoamlibCaseGenerator(case, output_dir).generate()` — creates complete OpenFOAM case
- `OpenFOAMRunner(case_dir).run()` — executes OpenFOAM solver
- `OpenFOAMSurfaceExtractor(case_dir).extract() -> SurfaceData`
- `compute_error_metrics(ref, test) -> ErrorMetrics` — L2, L∞, RMS, MAE
- `compute_gci(solutions, spacings) -> GCIResult` — Grid Convergence Index

## Pipeline Flow
### Legacy panel-vs-OpenFOAM flow
1. `generate_base_case.py` — Case YAML → OpenFOAM case (0/, constant/, system/, STL)
2. `run_of_convergence.py` — blockMesh → snappyHexMesh → potentialFoam at multiple refinement levels
3. `run_panel_convergence.py` — panel method at multiple mesh_levels from case.yaml
4. `visualize.py` — comparison plots
5. `compare_surface.py` — surface Cp/Vt comparison with error metrics

### Current BL-vs-Fluent flow
1. Run BL solve with reconstruction (`BoundaryLayerRunner.run(..., reconstruct=True)`)
2. Build Fluent comparison (`BLComparisonRunner(case, bl_result).run(profile_name=...)`)
3. Generate plots via scripts:
   - `plot_bl_fluent_difference.py` (difference-oriented)
   - `plot_bl_fluent_side_by_side.py` (absolute side-by-side)

## Dependencies
- Internal: `core.io.Case`, `core.geometry.Mesh`, `postprocessing.surface.SurfaceData`
- External: foamlib, trimesh, meshio, numpy, matplotlib

## What's Next
- Automated CI validation (run pipeline on push)
- Additional validation cases (multi-body, non-circular)
- Migrate remaining docs from OpenFOAM-first wording to Fluent-first BL validation wording

## Known Issues
- `foamlib` and `trimesh` not in requirements.txt
- Parallel case file resolution can fail (documented in notes)
