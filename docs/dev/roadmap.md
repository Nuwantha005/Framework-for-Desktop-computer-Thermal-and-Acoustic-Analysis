# Roadmap

## Phase 1: Foundation (Complete)
- [x] Constant-strength source panel solver
- [x] Parametric geometry generation (circle, rectangle, rounded rectangle)
- [x] YAML case file I/O with Pydantic validation
- [x] Scene assembly (multi-component with transforms)
- [x] Visualization (contours, streamlines, Cp, surface envelopes)
- [x] Post-processing pipeline (pressure, velocity potential, stream function, vorticity)
- [x] OpenFOAM validation pipeline (legacy panel-method comparison)
- [x] Fluent-based BL comparison plotting pipeline
- [x] Grid Convergence Index (GCI) computation

## Phase 2: Extended Panel Methods
- [ ] Constant-strength vortex panels
- [ ] Kutta condition implementation for lifting bodies
- [ ] Linear-strength source panels
- [ ] Linear-strength vortex panels
- [ ] Combined source-vortex formulation
- [ ] Wake modeling for steady-state lift

## Phase 3: Boundary Layer Coupling
- [ ] Integral laminar boundary layer solver (Von Kármán momentum integral)
- [ ] Turbulent boundary layer solver (Head's method)
- [ ] Transition prediction
- [ ] Viscous-inviscid coupling iteration (displacement thickness feedback)
- [ ] Thermal boundary layer solver (BDIM)
- [ ] Surface heat transfer coefficient computation

## Phase 4: 3D Extension
- [ ] 3D panel method with quadrilateral panels
- [ ] 3D influence coefficient computation
- [ ] Actuator disk model for fan representation
- [ ] 3D mesh import (FreeCAD/SALOME/Netgen)

## Phase 5: Applications
- [ ] Desktop computer case thermal analysis
- [ ] Fan curve coupling with actuator disk model
- [ ] Compact thermal model for CPU/GPU die temperature
- [ ] Acoustic analysis (fan noise superposition)
- [ ] User-facing configuration interface
