Final Implementation Plan
Based on the refactoring document and your answers, here's the complete plan. I've organized it into phases with dependencies noted.
Phase 1: Tabulated Data Infrastructure
Goal: Generate and store ODE solutions needed by profiles for stagnation patching and velocity reconstruction.
1. Create data/bl-solver-profiles/ directory
2. Blasius table generation script (scripts/generate_blasius_table.py)
   - Solve Blasius ODE f''' + ½f·f'' = 0 via scipy shooting method
   - Store f'(eta) on ~100-point eta grid (0 to 6.0), plus constants f''(0), I_1, I_2, eta_99
   - Output: data/bl-solver-profiles/blasius.json
3. Falkner-Skan table generation script (scripts/generate_falkner_skan_table.py)
   - Solve F-S ODE for ~25 beta values from -0.1988 to 2.0
   - Store f'(eta) profiles on common eta grid, plus f''(0)(beta), I_1(beta), I_2(beta), H(beta), S(beta), eta_99(beta)
   - Output: data/bl-solver-profiles/falkner_skan.json
4. Table loader utility (src/solvers/boundary_layer/profiles/tables.py)
   - Load JSON tables at module import (lazy singleton)
   - Provide interpolation functions: blasius_fprime(eta), falkner_skan_fprime(beta, eta), falkner_skan_constants(beta)
   - Fallback: if query is out of table range, solve ODE on-the-fly
Phase 2: Exact Stagnation Point & Velocity Gradient K
Goal: Replace threshold-based stagnation skipping with sign-change interpolation and linear regression for K.
5. Refactor BoundaryLayerRunner._find_stagnation_points() → add exact stagnation interpolation
   - After splitting into upper/lower paths, find where Ue changes sign (or minimum |Ue|) on each path
   - Interpolate between adjacent panels to find exact s_stag where Ue = 0
   - Recompute arc-length arrays with s = 0 at the interpolated stagnation point
6. Compute K per streamline via linear regression
   - Use panels near stagnation (|Ue| < some threshold of max) for the fit
   - K = Σ(Ue_i · s_i) / Σ(s_i²) (forced-through-origin regression)
   - Store K_upper, K_lower in BoundaryLayerPathResult
7. Refactor BoundaryLayerSolver._find_start_index() → use the first real panel (no more 10% threshold skip)
   - Integration now starts at s = s_first_panel with theta0 from stagnation patching
Phase 3: Stagnation Patching per Profile
Goal: Each profile computes analytically correct theta0 at stagnation using K.
8. Add stagnation_theta(nu: float, K: float) -> float to VelocityProfile ABC
   - Replaces the current initial_theta(nu, Ue0) which used ad-hoc estimates
   - Keep initial_theta as a fallback/deprecated alias
9. Implement stagnation patching in each profile:
   - BlasiusProfile.stagnation_theta: theta = sqrt(0.04803 * nu/K)
   - FalknerSkanProfile.stagnation_theta: theta = sqrt(0.08547 * nu/K) (Hiemenz exact)
   - PohlhausenProfile.stagnation_theta: theta = sqrt(0.0770 * nu/K) (Lambda_stag=7.052)
   - ThwaitesProfile.stagnation_theta: theta = sqrt(0.075 * nu/K)
   - PowerLawProfile.stagnation_theta: raise NotImplementedError (deferred — needs laminar start)
10. Update BoundaryLayerSolver.solve() to accept and use K parameter for stagnation_theta
Phase 4: BL Post-Processing / Velocity Reconstruction
Goal: Reconstruct the full velocity field u(s, y) and BL thickness delta(s) from integral results.
11. Add reconstruction methods to VelocityProfile ABC:
    - compute_delta(theta: float, H: float) -> float — returns delta or delta_99
    - reconstruct_velocity(y: NDArray, theta: float, H: float, Ue: float) -> NDArray — returns u(y)
12. Implement per-profile:
    - Blasius: L = theta/0.6641, delta_99 = 5.0 * L, u = Ue * f'(y/L) from table
    - Falkner-Skan: invert H → beta from table, L = theta/I_2(beta), delta_99 = eta_99(beta) * L, u = Ue * f'_beta(y/L) from table
    - Pohlhausen: solve quadratic H → Lambda, delta = theta/Phi(Lambda), u = Ue * g(y/delta; Lambda) (algebraic)
    - Power-Law: delta = ((n+1)(n+2)/n) * theta, u = Ue * (y/delta)^(1/n) (algebraic)
    - Thwaites + Falkner-Skan pairing: map lambda → beta via lambda = beta * I_2²(beta), then use F-S reconstruction
    - Thwaites + Pohlhausen pairing: map H → Lambda, then use Pohlhausen reconstruction
13. Add config option for Thwaites reconstruction pairing ("falkner_skan" or "pohlhausen")
14. Create BLFieldData dataclass to hold the reconstructed 2D field:
    - s: arc-length stations (M,)
    - y: wall-normal coordinates per station (M, Ny) or ragged
    - u: velocity values (M, Ny)
    - delta: BL thickness per station (M,)
    - Ue: edge velocity per station (M,)
15. Batch reconstruction function in a new file or in base.py:
    - Takes BoundaryLayerResult + VelocityProfile → BLFieldData
    - Evaluates at configurable y resolution and domain extent (e.g., 1.2δ)
Phase 5: Visualization
Goal: Add the new plot types from the refactoring doc.
16. s-y contour plot (viz #2): Arc-length x-axis, wall-normal y-axis, velocity contour. Upper bound at delta(s). Parameterized to show up to 1.2δ (viz #6).
17. Normalized y contour (viz #4): Same as #2 but with y/delta on y-axis → uniform rectangle.
18. Wrapped envelope with velocity color (viz #5): Extend existing plot_bl_envelope to accept a colormap based on the reconstructed velocity field inside the envelope.
19. Per-streamline handling (viz #7): All new plots accept a BoundaryLayerPathResult for one side, and a convenience wrapper generates both sides.
20. Placeholder for OpenFOAM comparison (viz #3): Create function signatures and layout for RMS difference plot, with of_field parameter defaulting to None. When None, skip the comparison subplot. No OF extraction implemented now.
Phase 6: Integration & Cleanup
21. Update BoundaryLayerRunner.run() to:
    - Use new stagnation interpolation and K computation
    - Pass K to stagnation_theta() instead of initial_theta()
    - Optionally run post-processing to generate BLFieldData
22. Update BoundaryLayerCaseResult to hold BLFieldData per profile per side
23. Update demo script (demos/demo_boundary_layer.py) with new features
24. Update config schema if needed (Thwaites pairing option, reconstruction resolution)
25. Update .agent/ docs (PROJECT_CONTEXT, modules/solver.md, decision record)
---
What Stays the Same
- VelocityProfile ABC structure (extended, not replaced)
- BoundaryLayerResult dataclass (fields added, none removed)
- BoundaryLayerRunner orchestration pattern
- BoundaryLayerSolver ODE integration (RK45 via solve_ivp)
- SurfaceDataExtractor → BL solver data flow
- All panel solvers, post-processing pipeline, existing visualization
- Factory patterns, case I/O, geometry module
Estimated Effort by Phase
Phase	Files Modified	Files Created	Complexity
1. Tables	0	4 (2 scripts, 2 JSON, 1 loader)	Medium — ODE solving
2. Stagnation	2 (runner.py, base.py)	0	Medium — interpolation
3. Patching	6 (ABC + 5 profiles)	0	Low — formulas from notes
4. Reconstruction	6 (ABC + 5 profiles + new dataclass)	1 (bl_field.py or similar)	High — per-profile logic
5. Visualization	1 (bl_plots.py)	0	Medium-High — 4 new plot types
6. Integration	3-4 (runner, demo, config, docs)	0	Low
Shall I proceed with implementation? If so, I'd start with Phase 1 (table generation) since Phases 3 and 4 depend on having those tables available.
