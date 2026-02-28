# Panel Methods Overview

## Potential Flow

The solver assumes incompressible, irrotational, inviscid flow. Under these assumptions, the velocity field can be expressed as the gradient of a scalar potential:

$$\mathbf{V} = \nabla\phi$$

where the velocity potential $\phi$ satisfies Laplace's equation everywhere outside the body:

$$\nabla^2 \phi = 0$$

## Panel Method Concept

Instead of solving Laplace's equation on a volume mesh, panel methods reformulate the problem as a boundary integral equation. The body surface is discretized into $N$ flat **panels**, and a singularity distribution (source, vortex, or doublet) is placed on each panel. The strengths of these singularities are determined by enforcing boundary conditions at **control points** (panel midpoints).

This reduces the problem from a 2D/3D field solve to a system of linear equations — a significant computational advantage.

## Current Limitations

- **Source, vortex, and doublet panels**: Source panels for non-lifting bodies; linear vortex panels with zero-circulation closure for direct $V_t$ extraction; Dirichlet doublet (Morino) for combined source+doublet formulation with both constant and linear variants. No lift generation.
- **Inviscid**: No boundary layer, no separation, no wake.
- **2D only**: 3D panel methods planned for future phases.

## Implemented Methods

- [Constant-Strength Source Panels](constant_source_panels.md) — Neumann BC, constant $\sigma$ per panel
- [Linear-Strength Source Panels](linear_source_panels.md) — Neumann BC, linear $\sigma$ at nodes
- [Linear-Strength Vortex Panels](linear_vortex_panels.md) — Neumann BC, linear $\gamma$ with zero-circulation closure
- [Dirichlet Doublet Panels (Morino)](dirichlet_doublet_panels.md) — Dirichlet BC, constant $\mu$ + $\sigma$
- [Linear Source/Doublet Panels](linear_source_doublet_panels.md) — Dirichlet BC, linear $\mu$ + $\sigma$ (K&P §11.5.1)

