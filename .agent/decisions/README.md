# Architectural Decision Records

Record significant design decisions here as they are made during implementation.
Each decision gets its own section with date, context, and rationale.

## Template
```markdown
### YYYY-MM-DD — [Short Title]
**Context**: What prompted this decision?
**Decision**: What was decided?
**Alternatives considered**: What else was evaluated?
**Consequences**: What changes as a result?
**Validation**: How was the decision verified?
```

## Decisions

### 2026-02-28 — Linear vortex panels with zero-circulation closure for bluff bodies
**Context**: Need a panel method that directly outputs surface tangential velocity ($V_t$) for downstream boundary layer integration. Standard K&P vortex panel methods (Ch 11.4) use wake panels + Kutta condition, which is fundamentally incompatible with non-lifting closed bluff bodies (desktop PC components).
**Decision**: Implement linear-strength vortex panels (K&P §11.4.2) adapted for non-lifting bodies by replacing the Kutta condition with a Zero Net Circulation constraint ($\Gamma = \sum \frac{\gamma_j + \gamma_{j+1}}{2} S_j = 0$) per connected component. The overdetermined (N+C) × (N+1) system is solved via `np.linalg.lstsq`. Vortex influences are derived from linear source influences via the rotation identity $u_\text{vortex} = w_\text{source}$, $w_\text{vortex} = -u_\text{source}$.
**Alternatives considered**: (1) Constant-strength doublet panels with Kutta — rejected because wake panels break for closed non-lifting bodies. (2) Higher-order source panels with numerical Vt differentiation — rejected because $dφ/ds$ adds noise at corners. (3) Source+vortex combined — unnecessarily complex for bluff bodies.
**Consequences**: New solver `LinearVortexPanelSolver` registered as `("vortex", "linear", "flat")`. Vortex strength at nodes directly equals local tangential velocity, eliminating need for noisy potential differentiation. Comparison alias `"vortex"` added to `SolverComparisonRunner`.
**Validation**: Rounded square at 256 panels vs OpenFOAM: Vt relative RMS = 3.78%, comparable to linear source (3.71%). Both are ~13× better than constant source (50.7%).

<!-- Example:
### 2026-02-27 — Continuous vs discontinuous linear-strength formulation
**Context**: Implementing linear-strength source panels; two formulations exist.
**Decision**: Continuous formulation (N+1 node-based unknowns, shared at panel edges).
**Alternatives**: Discontinuous (2N unknowns, independent per panel) — more flexible but
larger system and jumps at panel edges.
**Consequences**: Need to restructure influence matrix to be (N, N+1) instead of (N, N);
surface potential is continuous by construction.
**Validation**: Cylinder Vt convergence rate should be O(h²) vs O(h) for constant panels.
-->
