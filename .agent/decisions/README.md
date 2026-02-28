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

(None yet — add entries as implementation progresses.)

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
