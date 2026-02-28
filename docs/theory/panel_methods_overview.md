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

- **Source panels only**: No lift generation (no circulation). Suitable for non-lifting bodies.
- **Inviscid**: No boundary layer, no separation, no wake.
- **2D only**: 3D panel methods planned for future phases.


- [Constant-Strength Source Panels](constant_source_panels.md)
- [Linear-Strength Source Panels](linear_source_panels.md)

