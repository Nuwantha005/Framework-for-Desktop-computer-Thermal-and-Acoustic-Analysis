# Visualization Module

The visualization module (`visualization`) provides matplotlib-based plotting for panel method results, including contours, streamlines, pressure distributions, surface envelopes, and comparison plots.

## Core Classes

### Visualizer

The main facade for creating figures:

```python
from visualization import Visualizer

viz = Visualizer(output_dir="cases/cylinder_flow/out/", protect_overwrite=False)

# Create a multi-panel figure
viz.create_figure(subplots=(2, 2), figsize=(14, 12), title="Cylinder Flow")

# Plot into specific subplot positions
viz.plot_mesh(mesh, ax_index=0, show_normals=True)
viz.plot_contours(XX, YY, Vx, Vy, mesh, ax_index=1, levels=25)
viz.plot_streamlines(XX, YY, Vx, Vy, mesh, ax_index=2, density=1.2)
viz.plot_cp(mesh, solver.Cp, ax_index=3)

# Save and/or display
viz.finalize(save="combined.png", show=False, dpi=150)
```

### VelocityField2D

Computes velocity on a structured grid using the solver's `velocity_at()` method:

```python
from visualization.field2d import VelocityField2D

field = VelocityField2D(solver, mesh=case.mesh)
XX, YY, Vx, Vy = field.compute(
    x_range=(-2.0, 3.0),
    y_range=(-2.0, 2.0),
    resolution=(200, 160)
)
```

Features:

- **Body masking**: Automatically masks grid points inside solid bodies using per-component boundary detection
- **Caching**: Results are cached; subsequent calls with the same parameters return cached data
- **Multiprocessing**: Field computation can be parallelized across CPU cores

### ComparisonVisualizer

For comparing two solutions (e.g., panel method vs OpenFOAM):

```python
from visualization.comparison import ComparisonVisualizer, FieldSeries

comp = ComparisonVisualizer(output_dir="output/")
fig = comp.compare_contours(
    fields=[
        FieldSeries(name="Panel Method", XX=XX1, YY=YY1, values=speed1),
        FieldSeries(name="OpenFOAM", XX=XX2, YY=YY2, values=speed2),
    ],
    mesh=mesh,
    title="Velocity Comparison"
)

# Compute error metrics
metrics = comp.compute_metrics(field1=speed1, field2=speed2)
print(f"L2: {metrics.l2:.6f}, L∞: {metrics.linf:.6f}")
```

### Surface Envelope Plots

Visualize distributions wrapped directly around the body geometry:

```python
from visualization.surface_envelope import plot_surface_envelope

fig, ax = plot_surface_envelope(
    x=centers[:, 0], y=centers[:, 1],
    values=solver.Cp,
    scale=0.3,
    quantity_name="Cp",
    colormap='RdBu_r'
)
```

## Additional Plotters

### MeshPlotter

Quick geometry visualization:

```python
from visualization.mesh_plot import quick_plot_mesh, quick_plot_scene

quick_plot_mesh(mesh, show_normals=True, show_centers=True)
quick_plot_scene(scene, show_freestream=True)
```

### StreamlinePlotter / ContourPlotter

Lower-level plotters in `visualization.plotters`:

```python
from visualization.plotters.contours import ContourPlotter
from visualization.plotters.streamlines import StreamlinePlotter

ContourPlotter(mesh).plot_velocity_magnitude(XX, YY, Vx, Vy, levels=20)
StreamlinePlotter(mesh).plot(XX, YY, Vx, Vy, density=1.0, seed_style='left')
```

## File Layout

| File | Lines | Contents |
|------|-------|----------|
| `visualizer.py` | 654 | `OutputManager`, `Visualizer` facade |
| `field2d.py` | 204 | `VelocityField2D` — grid computation, body masking, caching |
| `comparison.py` | 978 | `ComparisonVisualizer`, `FieldSeries`, `ComparisonMetrics` |
| `panel2d.py` | 184 | `PanelVisualizer2D` (legacy) |
| `mesh_plot.py` | 364 | `MeshPlotter`, `quick_plot_*` helpers |
| `surface_envelope.py` | 509 | Surface envelope plotting functions |
| `streamlines.py` | 346 | `StreamlineVisualizer` (legacy, multiprocessing) |
| `plotters/contours.py` | 109 | `ContourPlotter` |
| `plotters/streamlines.py` | 133 | `StreamlinePlotter` |
