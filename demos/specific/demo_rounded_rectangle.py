"""
Demo: Rounded Rectangle - Standalone Visualization

Shows how to use Visualizer and PostProcessing without a case file:
- Single mesh visualization
- Scene with multiple components
- Post-processing pipeline for pressure, potential, etc.
- Exporting to case folder for later reuse
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from solvers.panel2d.spm import SourcePanelSolver
from core.io.geometry_io import generate_rounded_rectangle
from core.io.case_exporter import CaseExporter
from core.geometry import Component, Scene, Transform
from visualization import Visualizer
from visualization.field2d import VelocityField2D
from postprocessing import (
    FieldData, FluidState, 
    ProcessorPipeline, PressureProcessor, VelocityPotentialProcessor
)
from postprocessing.velocity_potential import VorticityProcessor


def one_rect():
    """Single rounded rectangle - standalone mesh usage."""
    print("=== Single Rounded Rectangle ===")
    
    # Create mesh directly (no case file needed)
    mesh = generate_rounded_rectangle(
        center=(0.0, 0.0),
        width=1.5,
        height=1.5,
        corner_radius=0.4,
        num_panels_per_side=5,
        num_panels_per_arc=4
    )
    print(f"Created mesh with {mesh.num_panels} panels")
    
    # Visualizer works directly with Mesh - no case file required!
    viz = Visualizer()  # No output_dir = won't save, just show
    viz.create_figure(figsize=(10, 8))
    viz.plot_mesh(mesh, show_normals=True, title="Single Rounded Rectangle")
    viz.finalize(show=True)
    
    # Solve
    v_inf = 10.0
    aoa = 0.0
    solver = SourcePanelSolver(mesh, v_inf=v_inf, aoa=aoa)
    solver.solve()
    print(f"Cp range: [{min(solver.Cp):.4f}, {max(solver.Cp):.4f}]")
    
    # Post-processing visualization
    field = VelocityField2D(mesh, v_inf, aoa, solver.sigma)
    XX, YY, Vx, Vy = field.compute((-2, 4), (-2, 2), (150, 100), num_cores=6)
    
    viz2 = Visualizer()
    viz2.create_figure(subplots=(1, 2), figsize=(14, 6))
    viz2.plot_contours(XX, YY, Vx, Vy, mesh, ax_index=0, title="Velocity Contours")
    viz2.plot_streamlines(XX, YY, Vx, Vy, mesh, ax_index=1, title="Streamlines")
    viz2.finalize(show=True)


def two_rects_scene():
    """Two rounded rectangles using Scene - demonstrates multi-body."""
    print("\n=== Two Rounded Rectangles (Scene) ===")
    
    # Create local mesh (will be reused for both components)
    local_mesh = generate_rounded_rectangle(
        center=(0.0, 0.0),  # Local origin
        width=2.0,
        height=1.0,
        corner_radius=0.2,
        num_panels_per_side=20,
        num_panels_per_arc=10
    )
    
    # Component 1: Front rectangle (no transform)
    comp1 = Component(
        name="rect_front",
        local_mesh=local_mesh,
        transform=Transform.from_2d(tx=0.0, ty=0.0, angle_deg=0.0),
        bc_type="wall"
    )
    
    # Component 2: Back rectangle (offset and slightly rotated)
    comp2 = Component(
        name="rect_back",
        local_mesh=local_mesh,
        transform=Transform.from_2d(tx=-3.0, ty=0.5, angle_deg=10.0),
        bc_type="wall"
    )
    
    # Create scene
    scene = Scene(
        name="Two Rounded Rectangles",
        components=[comp1, comp2],
        freestream=np.array([10.0, 0.0, 0.0])
    )
    print(f"Scene: {scene.num_components} components")
    
    # Visualize scene directly (no case file!)
    viz = Visualizer()
    viz.create_figure(figsize=(12, 8))
    viz.plot_scene(scene, show_normals=True, show_freestream=True)
    viz.finalize(show=True)
    
    # Assemble into single mesh for solver
    mesh = scene.assemble()
    print(f"Assembled mesh: {mesh.num_panels} panels")
    
    # Solve
    v_inf = 10.0
    aoa = 0.0
    solver = SourcePanelSolver(mesh, v_inf=v_inf, aoa=aoa)
    solver.solve()
    print(f"Cp range: [{min(solver.Cp):.4f}, {max(solver.Cp):.4f}]")
    
    # =========================================================================
    # Post-processing WITHOUT a case file - FluidState specified directly
    # =========================================================================
    print("\n--- Standalone Post-Processing ---")
    
    # Compute velocity field
    field = VelocityField2D(mesh, v_inf, aoa, solver.sigma)
    XX, YY, Vx, Vy = field.compute((-6, 4), (-3, 3), (200, 120), num_cores=6)
    
    # Create FieldData container
    fields = FieldData(XX, YY)
    fields.add_vector("velocity", Vx, Vy, units="m/s")
    fields.set_metadata("v_inf", v_inf)
    
    # Create FluidState directly (no case file needed!)
    fluid = FluidState.air_standard()  # Air at standard conditions
    print(f"Fluid: {fluid}")
    
    # Build and run post-processing pipeline
    pipeline = ProcessorPipeline()
    pipeline.add(PressureProcessor())
    pipeline.add(VelocityPotentialProcessor())
    pipeline.add(VorticityProcessor())
    
    pipeline.run(fields, fluid)
    print(f"Computed fields: {fields.available}")
    
    # Visualize using the new plot_field interface
    viz2 = Visualizer()
    viz2.create_figure(subplots=(2, 3), figsize=(18, 12), title="Two Rounded Rectangles - Full Analysis")
    
    viz2.plot_scene(scene, ax_index=0, show_normals=True, title="Geometry")
    viz2.plot_field("velocity", fields, mesh, ax_index=1, component="magnitude", title="|V|")
    viz2.plot_field("pressure_coefficient", fields, mesh, ax_index=2, 
                    title="Cp Field", cmap='RdBu_r', symmetric=True, show_iso=True)
    viz2.plot_streamlines(XX, YY, Vx, Vy, mesh, ax_index=3, title="Streamlines")
    viz2.plot_field("stream_function", fields, mesh, ax_index=4,
                    title="Stream Function ψ", cmap='coolwarm', show_iso=True)
    viz2.plot_field("vorticity", fields, mesh, ax_index=5,
                    title="Vorticity ω", cmap='RdBu_r', symmetric=True)
    
    viz2.finalize(show=True)


def export_to_case():
    """Export programmatic geometry to a case folder for later reuse."""
    print("\n=== Export to Case Folder ===")
    
    # Create geometry programmatically
    local_mesh = generate_rounded_rectangle(
        center=(0.0, 0.0),
        width=2.0,
        height=1.0,
        corner_radius=0.2,
        num_panels_per_side=50,
        num_panels_per_arc=40
    )
    
    # Build scene
    comp1 = Component(
        name="rect_front",
        local_mesh=local_mesh,
        transform=Transform.from_2d(tx=0.0, ty=0.0, angle_deg=0.0),
        bc_type="wall"
    )
    
    comp2 = Component(
        name="rect_back", 
        local_mesh=local_mesh,
        transform=Transform.from_2d(tx=-3.0, ty=0.5, angle_deg=10.0),
        bc_type="wall"
    )
    
    scene = Scene(
        name="Two Rounded Rectangles",
        components=[comp1, comp2],
        freestream=np.array([10.0, 0.0, 0.0]),
        description="Two rounded rectangles with offset - exported from code"
    )
    
    # Export to case folder with fluid properties
    exporter = CaseExporter.from_scene(scene)
    exporter.set_visualization_domain((-6, 5), (-4, 4), (200, 150))
    exporter.set_fluid(density=1.225, gravity=0.0, reference_pressure=101325.0)
    
    case_dir = Path(__file__).parent.parent.parent / "cases" / "two_rounded_rects_fine"
    exporter.export(case_dir, overwrite=True)
    
    print(f"\nNow you can run:")
    print(f"  python demos/demo_combined.py cases/two_rounded_rects_fine --show")

if __name__ == "__main__":
    #one_rect()
    #two_rects_scene()
    export_to_case()