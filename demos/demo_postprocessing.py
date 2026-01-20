#!/usr/bin/env python3
"""
Demo: Post-Processing Pipeline

Shows the new modular post-processing system:
- FieldData container for all computed fields
- FluidState for fluid properties
- ProcessorPipeline for computing derived quantities
- Visualization of multiple field types

Usage:
    python demo_postprocessing.py <case_dir> [--show] [--save]
    
Example:
    python demo_postprocessing.py ../cases/cylinder_flow --show
    python demo_postprocessing.py ../cases/two_rounded_rects --save
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.io import CaseLoader
from solvers.panel2d.spm import SourcePanelSolver
from visualization import Visualizer
from visualization.field2d import VelocityField2D
from postprocessing import (
    FieldData, FluidState, 
    ProcessorPipeline, PressureProcessor, VelocityPotentialProcessor
)
from postprocessing.velocity_potential import VorticityProcessor


def main():
    parser = argparse.ArgumentParser(description="Post-processing pipeline demo")
    parser.add_argument("case_dir", type=str, help="Path to case directory")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    parser.add_argument("--save", action="store_true", help="Save plots to case_dir/out/")
    parser.add_argument("--cores", type=int, default=6, help="Number of CPU cores")
    args = parser.parse_args()
    
    if not args.show and not args.save:
        args.save = True
    
    # =========================================================================
    # 1. Load case
    # =========================================================================
    case_dir = Path(args.case_dir).resolve()
    case = CaseLoader.load_case(case_dir)
    
    print(f"Loaded: {case.name}")
    print(f"  Panels: {case.num_panels}")
    print(f"  V_inf: {case.v_inf} m/s, AoA: {case.aoa}°")
    print(f"  Density: {case.density} kg/m³")
    print(f"  Reference pressure: {case.reference_pressure} Pa")
    
    # =========================================================================
    # 2. Solve panel method
    # =========================================================================
    print("\nSolving...")
    solver = SourcePanelSolver(case.mesh, v_inf=case.v_inf, aoa=case.aoa)
    solver.solve()
    print(f"  Cp range: [{min(solver.Cp):.4f}, {max(solver.Cp):.4f}]")
    
    # =========================================================================
    # 3. Compute velocity field
    # =========================================================================
    x_range = case.x_range
    y_range = case.y_range
    resolution = case.resolution
    
    print(f"\nComputing velocity field ({resolution[0]}×{resolution[1]})...")
    vfield = VelocityField2D(case.mesh, case.v_inf, case.aoa, solver.sigma)
    XX, YY, Vx, Vy = vfield.compute(x_range, y_range, resolution, num_cores=args.cores)
    
    # =========================================================================
    # 4. Create FieldData container
    # =========================================================================
    fields = FieldData(XX, YY)
    fields.add_vector("velocity", Vx, Vy, units="m/s")
    fields.set_metadata("v_inf", case.v_inf)
    fields.set_metadata("aoa", case.aoa)
    
    print(f"\nInitial fields: {fields.available}")
    
    # =========================================================================
    # 5. Setup post-processing pipeline
    # =========================================================================
    fluid = case.get_fluid_state()
    print(f"\nFluid: {fluid}")
    
    pipeline = ProcessorPipeline()
    pipeline.add(PressureProcessor(include_gravity=case.gravity != 0))
    pipeline.add(VelocityPotentialProcessor())
    pipeline.add(VorticityProcessor())
    
    print(f"Pipeline will produce: {pipeline.available_outputs()}")
    
    # =========================================================================
    # 6. Run pipeline
    # =========================================================================
    print("\nRunning post-processing pipeline...")
    pipeline.run(fields, fluid)
    
    print(f"\nAll available fields: {fields.available}")
    print(fields.summary())
    
    # =========================================================================
    # 7. Visualize results
    # =========================================================================
    output_dir = case.output_dir if args.save else None
    viz = Visualizer(output_dir=output_dir)
    
    # --- Figure 1: Velocity components ---
    viz.create_figure(subplots=(2, 2), figsize=(14, 12), title=f"{case.name} - Velocity")
    
    viz.plot_field("velocity", fields, case.mesh, ax_index=0, 
                   component="magnitude", title="Velocity Magnitude |V|")
    viz.plot_field("velocity", fields, case.mesh, ax_index=1,
                   component="x", title="Velocity X-Component (Vx)", cmap='RdBu_r', symmetric=True)
    viz.plot_field("velocity", fields, case.mesh, ax_index=2,
                   component="y", title="Velocity Y-Component (Vy)", cmap='RdBu_r', symmetric=True)
    viz.plot_streamlines(XX, YY, Vx, Vy, case.mesh, ax_index=3, title="Streamlines")
    
    viz.finalize(save="velocity_fields.png" if args.save else None, show=args.show)
    
    # --- Figure 2: Pressure ---
    viz2 = Visualizer(output_dir=output_dir)
    viz2.create_figure(subplots=(2, 2), figsize=(14, 12), title=f"{case.name} - Pressure")
    
    viz2.plot_field("pressure", fields, case.mesh, ax_index=0,
                    title="Absolute Pressure", cmap='jet')
    viz2.plot_field("pressure_gauge", fields, case.mesh, ax_index=1,
                    title="Gauge Pressure (P - P_ref)", cmap='RdBu_r', symmetric=True)
    viz2.plot_field("pressure_coefficient", fields, case.mesh, ax_index=2,
                    title="Pressure Coefficient Cp", cmap='RdBu_r', symmetric=True,
                    show_iso=True)
    viz2.plot_cp(case.mesh, solver.Cp, ax_index=3, title="Cp on Body Surface")
    
    viz2.finalize(save="pressure_fields.png" if args.save else None, show=args.show)
    
    # --- Figure 3: Derived quantities ---
    viz3 = Visualizer(output_dir=output_dir)
    viz3.create_figure(subplots=(2, 2), figsize=(14, 12), title=f"{case.name} - Derived Fields")
    
    viz3.plot_field("velocity_potential", fields, case.mesh, ax_index=0,
                    title="Velocity Potential φ", cmap='viridis', show_iso=True)
    viz3.plot_field("stream_function", fields, case.mesh, ax_index=1,
                    title="Stream Function ψ", cmap='coolwarm', show_iso=True)
    viz3.plot_field("vorticity", fields, case.mesh, ax_index=2,
                    title="Vorticity ω", cmap='RdBu_r', symmetric=True)
    viz3.plot_field("pressure_total", fields, case.mesh, ax_index=3,
                    title="Total Pressure P₀", cmap='jet')
    
    viz3.finalize(save="derived_fields.png" if args.save else None, show=args.show)
    
    print("\nDone.")


if __name__ == "__main__":
    main()
