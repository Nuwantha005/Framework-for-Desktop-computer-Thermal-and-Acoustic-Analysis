#!/usr/bin/env python3
"""
Thermal Boundary Layer Analysis Demo
=====================================

Run the complete thermal analysis pipeline:
    Panel Method -> Viscous BL -> Thermal BL

Demonstrates coupling the viscous boundary layer solver to the thermal solver.
Supports two thermal solvers:
- Reynolds Analogy (Chilton-Colburn): Surface-only, fast, approximate
- BDIM: Full domain integral method, accurate, requires field reconstruction

Usage::

    python demos/demo_thermal_bl.py cases/cylinder_flow
    python demos/demo_thermal_bl.py cases/cylinder_flow --profile thwaites
    python demos/demo_thermal_bl.py cases/cylinder_flow --q-wall 1000
    python demos/demo_thermal_bl.py cases/cylinder_flow --show-plots
    
    # Use BDIM solver (requires --reconstruct for field data)
    python demos/demo_thermal_bl.py cases/cylinder_flow --thermal-solver bdim --reconstruct
"""

import sys
import argparse
from pathlib import Path

import numpy as np

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.io.case_loader import CaseLoader
from solvers.boundary_layer.runner import BoundaryLayerRunner
from solvers.thermal.base import extract_thermal_input, ThermalSolverConfig
from solvers.thermal.factory import ThermalSolverFactory, create_thermal_solver
from visualization.thermal_plots import (
    ThermalCaseResult,
    plot_thermal_two_sides,
    plot_thermal_summary,
    plot_thermal_envelope_two_sides,
    plot_thermal_field_contour,
    plot_thermal_field_two_sides,
)


def _print_bl_summary(bl) -> None:
    """Print brief BL summary."""
    for side_name, path in bl.sides.items():
        n_panels = len(path.panel_indices)
        print(f"  {side_name.upper()}: {n_panels} panels, s_max = {path.s[-1]:.4f} m")
        for name, res in path.results.items():
            valid = ~np.isnan(res.theta)
            n_valid = int(valid.sum())
            print(f"    [{name}] {n_valid} valid stations")


def _print_thermal_summary(result, side: str) -> None:
    """Print thermal result summary."""
    print(f"  {side.upper()} ({result.num_stations} stations):")
    print(f"    T_w:  [{result.wall_temperature.min():.2f}, "
          f"{result.wall_temperature.max():.2f}] K")
    print(f"    h:    [{result.heat_transfer_coeff.min():.2f}, "
          f"{result.heat_transfer_coeff.max():.2f}] W/m²K")
    print(f"    Nu:   [{result.nusselt.min():.2f}, "
          f"{result.nusselt.max():.2f}]")
    print(f"    q_w:  [{result.wall_heat_flux.min():.2f}, "
          f"{result.wall_heat_flux.max():.2f}] W/m²")
    print(f"    Q:    {result.total_heat_rate:.4f} W/m (per span)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run thermal boundary layer analysis pipeline",
    )
    parser.add_argument("case_dir", type=Path,
                        help="Path to case directory (e.g. cases/cylinder_flow)")
    parser.add_argument("--solver-type", type=str, default=None,
                        help="Panel solver type (default: from case config)")
    parser.add_argument("--mesh-level", type=int, default=-1,
                        help="Mesh refinement level index (default: finest)")
    parser.add_argument("--profile", type=str, default="thwaites",
                        help="BL profile to use for thermal (default: thwaites)")
    parser.add_argument("--thermal-solver", type=str, default="reynolds_analogy",
                        choices=["reynolds_analogy", "bdim"],
                        help="Thermal solver type (default: reynolds_analogy)")
    parser.add_argument("--reconstruct", action="store_true",
                        help="Reconstruct BL field (required for BDIM solver)")
    parser.add_argument("--q-wall", type=float, default=None,
                        help="Wall heat flux BC [W/m²] (overrides case config)")
    parser.add_argument("--T-wall", type=float, default=None,
                        help="Wall temperature BC [K] (overrides case config)")
    parser.add_argument("--T-inf", type=float, default=300.0,
                        help="Freestream temperature [K] (default: 300)")
    parser.add_argument("--envelope-scale", type=float, default=0.15,
                        help="Envelope displacement scale factor")
    parser.add_argument("--show-plots", action="store_true",
                        help="Display plots interactively")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: <case>/out/thermal)")
    args = parser.parse_args()

    case_dir = args.case_dir.resolve()
    if not case_dir.exists():
        print(f"Error: Case directory not found: {case_dir}")
        return 1
    
    # Check that BDIM has reconstruct flag
    if args.thermal_solver == "bdim" and not args.reconstruct:
        print("Warning: BDIM solver requires --reconstruct flag. Enabling automatically.")
        args.reconstruct = True

    case_dir = args.case_dir.resolve()
    if not case_dir.exists():
        print(f"Error: Case directory not found: {case_dir}")
        return 1

    # ------------------------------------------------------------------
    # 1. Load case and solve panel method
    # ------------------------------------------------------------------
    print(f"Loading case: {case_dir}")
    case = CaseLoader.load_case(case_dir, mesh_level_index=args.mesh_level)
    print(f"  Case: {case.name}, {case.mesh.num_panels} panels")

    solver = case.create_solver(solver_type=args.solver_type)
    print(f"  Solver: {solver.__class__.__name__}")
    print("  Solving panel method ...")
    solver.solve()
    print("  Done.\n")

    # ------------------------------------------------------------------
    # 2. Run viscous boundary layer analysis
    # ------------------------------------------------------------------
    print("Running viscous boundary layer analysis ...")
    runner = BoundaryLayerRunner(case, solver)
    bl_cfg = case.config.boundary_layer
    
    # Ensure the requested profile is included
    profiles = list(bl_cfg.profiles)
    if args.profile not in profiles:
        profiles.append(args.profile)
    
    # Run with reconstruction if needed for BDIM
    reconstruct = args.reconstruct or args.thermal_solver == "bdim"
    
    bl = runner.run(
        profiles=profiles,
        nu=None,  # Use from fluid config
        transition_model=bl_cfg.transition_model,
        reconstruct=reconstruct,
    )
    
    print(f"  nu = {bl.nu:.3e} m²/s")
    if reconstruct:
        print(f"  Field reconstruction: enabled")
    _print_bl_summary(bl)
    print()

    # ------------------------------------------------------------------
    # 3. Extract thermal input and run thermal solver
    # ------------------------------------------------------------------
    print(f"Running thermal boundary layer analysis ...")
    print(f"  Profile: {args.profile}")
    print(f"  Solver: {args.thermal_solver}")
    
    # Determine boundary condition
    q_wall = args.q_wall
    T_wall = args.T_wall
    
    # Try to get from case config if not specified
    if q_wall is None and T_wall is None:
        # Check component BC
        comp = case.scene.components[0] if case.scene.components else None
        if comp is not None and hasattr(comp, 'bc_heat_flux') and comp.bc_heat_flux is not None:
            q_wall = comp.bc_heat_flux
            print(f"  Using heat flux from case config: {q_wall:.2f} W/m²")
        else:
            # Default for demo
            q_wall = 500.0
            print(f"  No BC specified, using default q_wall = {q_wall:.2f} W/m²")
    elif q_wall is not None:
        print(f"  Heat flux BC: q_wall = {q_wall:.2f} W/m²")
    else:
        print(f"  Temperature BC: T_wall = {T_wall:.2f} K")
    
    print(f"  T_inf = {args.T_inf:.2f} K")
    
    # Get fluid properties from case config
    fluid_cfg = case.config.fluid
    rho = fluid_cfg.density
    mu = fluid_cfg.viscosity if fluid_cfg.viscosity else 1.81e-5
    k = fluid_cfg.thermal_conductivity if fluid_cfg.thermal_conductivity else 0.026
    cp = fluid_cfg.specific_heat_cp if fluid_cfg.specific_heat_cp else 1005.0
    Pr = (cp * mu) / k
    
    print(f"  Pr = {Pr:.3f}, k = {k:.4f} W/mK")
    
    # Build config
    config = ThermalSolverConfig(
        T_inf=args.T_inf,
        Pr=Pr,
        k=k,
        rho=rho,
        cp=cp,
        q_wall=q_wall,
        T_wall=T_wall,
    )
    
    # Run thermal solver on both sides
    results = {}
    
    if args.thermal_solver == "bdim":
        # BDIM solver - requires field data
        from solvers.thermal.factory import create_bdim_solver
        
        for side_name, path_result in bl.sides.items():
            try:
                # Find the profile name (case-insensitive match)
                available_profiles = list(path_result.results.keys())
                profile_name = None
                for p in available_profiles:
                    if p.lower() == args.profile.lower():
                        profile_name = p
                        break
                if profile_name is None:
                    raise KeyError(
                        f"Profile '{args.profile}' not found. Available: {available_profiles}"
                    )
                
                # Check for field data
                if profile_name not in path_result.fields:
                    raise ValueError(
                        f"No field data for profile '{profile_name}'. "
                        f"Run with --reconstruct flag."
                    )
                
                bl_field = path_result.fields[profile_name]
                solver = create_bdim_solver(path_result, bl_field, config)
                results[side_name] = solver.solve()
                
            except (KeyError, ValueError, ImportError) as e:
                print(f"  Warning: Could not run BDIM on {side_name}: {e}")
                results[side_name] = None
    else:
        # Reynolds Analogy solver - surface data only
        for side_name, path_result in bl.sides.items():
            try:
                # Find the profile name (case-insensitive match)
                available_profiles = list(path_result.results.keys())
                profile_name = None
                for p in available_profiles:
                    if p.lower() == args.profile.lower():
                        profile_name = p
                        break
                if profile_name is None:
                    raise KeyError(
                        f"Profile '{args.profile}' not found. Available: {available_profiles}"
                    )
                
                thermal_input = extract_thermal_input(path_result, profile_name)
                thermal_solver = create_thermal_solver(
                    args.thermal_solver, thermal_input, config
                )
                results[side_name] = thermal_solver.solve()
            except (KeyError, ValueError) as e:
                print(f"  Warning: Could not run thermal on {side_name}: {e}")
                results[side_name] = None
    
    print()
    print("Thermal results:")
    for side_name, result in results.items():
        if result is not None:
            _print_thermal_summary(result, side_name)
    print()

    # ------------------------------------------------------------------
    # 4. Generate plots
    # ------------------------------------------------------------------
    import matplotlib
    if not args.show_plots:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = args.output_dir or (case.output_dir / "thermal")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to: {output_dir}")

    upper = results.get("upper")
    lower = results.get("lower")
    
    if upper is None or lower is None:
        print("  Warning: One or both sides missing, skipping some plots")
    
    # Get surface coordinates
    nodes = case.mesh.nodes[:, :2]  # (N, 2)
    surface_x = nodes[:, 0]
    surface_y = nodes[:, 1]
    
    # Get panel indices for mapping - only the VALID (non-separated) stations
    # The thermal result only contains valid stations, so we need to filter panel_indices
    def get_valid_panel_indices(bl_path, profile_name):
        """Get panel indices for valid (non-NaN) thermal stations."""
        if profile_name not in bl_path.results:
            return []
        bl_result = bl_path.results[profile_name]
        valid_mask = ~np.isnan(bl_result.theta)
        return list(np.array(bl_path.panel_indices)[valid_mask])
    
    # Find the actual profile name used (case-insensitive)
    profile_name_upper = None
    profile_name_lower = None
    for p in bl.upper.results.keys():
        if p.lower() == args.profile.lower():
            profile_name_upper = p
            break
    for p in bl.lower.results.keys():
        if p.lower() == args.profile.lower():
            profile_name_lower = p
            break
    
    upper_indices = get_valid_panel_indices(bl.upper, profile_name_upper) if profile_name_upper else []
    lower_indices = get_valid_panel_indices(bl.lower, profile_name_lower) if profile_name_lower else []
    
    print(f"  Valid panel indices: upper={len(upper_indices)}, lower={len(lower_indices)}")
    
    if upper is not None and lower is not None:
        # Two-sided line plots
        plot_thermal_two_sides(
            upper, lower,
            title=f"Thermal BL — {case.name}",
            output_path=output_dir / "thermal_lines.png",
        )
        plt.close("all")
        print("  + thermal_lines.png")
        
        # Envelope plot (wall temperature)
        plot_thermal_envelope_two_sides(
            upper, lower,
            surface_x, surface_y,
            upper_indices, lower_indices,
            quantity="wall_temperature",
            scale=args.envelope_scale,
            title=f"Wall Temperature — {case.name}",
            output_path=output_dir / "thermal_Tw_envelope.png",
        )
        plt.close("all")
        print("  + thermal_Tw_envelope.png")
        
        # Envelope plot (heat transfer coefficient)
        plot_thermal_envelope_two_sides(
            upper, lower,
            surface_x, surface_y,
            upper_indices, lower_indices,
            quantity="heat_transfer_coeff",
            scale=args.envelope_scale,
            title=f"Heat Transfer Coefficient — {case.name}",
            output_path=output_dir / "thermal_h_envelope.png",
        )
        plt.close("all")
        print("  + thermal_h_envelope.png")
        
        # Summary plot
        plot_thermal_summary(
            upper, lower,
            surface_x, surface_y,
            upper_indices, lower_indices,
            envelope_quantity="wall_temperature",
            envelope_scale=args.envelope_scale,
            title=f"Thermal Summary — {case.name}",
            output_path=output_dir / "thermal_summary.png",
        )
        plt.close("all")
        print("  + thermal_summary.png")
        
        # BDIM field contour plots (if available)
        if upper.has_field or lower.has_field:
            print("  Generating BDIM field plots...")
            
            # Temperature field contours for each side
            if upper.has_field:
                plot_thermal_field_contour(
                    upper,
                    quantity="T",
                    title=f"Temperature Field (Upper) — {case.name}",
                    output_path=output_dir / "thermal_field_upper.png",
                )
                plt.close("all")
                print("  + thermal_field_upper.png")
            
            if lower.has_field:
                plot_thermal_field_contour(
                    lower,
                    quantity="T",
                    title=f"Temperature Field (Lower) — {case.name}",
                    output_path=output_dir / "thermal_field_lower.png",
                )
                plt.close("all")
                print("  + thermal_field_lower.png")
            
            # Combined two-side field comparison
            if upper.has_field and lower.has_field:
                plot_thermal_field_two_sides(
                    upper, lower,
                    quantity="T",
                    title=f"Temperature Field — {case.name}",
                    output_path=output_dir / "thermal_field_comparison.png",
                )
                plt.close("all")
                print("  + thermal_field_comparison.png")

    if args.show_plots:
        plt.show()

    print("\nThermal boundary layer analysis complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
