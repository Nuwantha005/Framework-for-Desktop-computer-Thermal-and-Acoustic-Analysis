#!/usr/bin/env python3
"""Generate thermal wall envelope plots (solver vs Fluent)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from core.io.case_loader import CaseLoader
from solvers.boundary_layer.runner import BoundaryLayerRunner
from solvers.thermal.base import ThermalSolverConfig, extract_thermal_input
from solvers.thermal.factory import create_bdim_solver
from validation.adapters.fluent import ThermalComparisonRunner
from visualization.thermal_fluent_comparison_plots import (
    plot_thermal_wall_envelope_overlay,
    plot_thermal_wall_envelope_side_by_side,
)


def _resolve_profile(path_result, profile: str) -> str:
    for name in path_result.results.keys():
        if name.lower() == profile.lower():
            return name
    raise KeyError(f"Profile '{profile}' not found. Available: {list(path_result.results.keys())}")


def _build_thermal_config(case, T_inf: float, q_wall: float) -> ThermalSolverConfig:
    fluid = case.config.fluid
    rho = fluid.density
    mu = fluid.viscosity if fluid.viscosity else 1.81e-5
    k = fluid.thermal_conductivity if fluid.thermal_conductivity else 0.026
    cp = fluid.specific_heat_cp if fluid.specific_heat_cp else 1005.0
    Pr = (cp * mu) / k
    return ThermalSolverConfig(T_inf=T_inf, Pr=Pr, k=k, rho=rho, cp=cp, q_wall=q_wall)


def _run_thermal_bdim(case, bl_result, profile: str, cfg: ThermalSolverConfig):
    results = {}
    for side, path in bl_result.sides.items():
        pname = _resolve_profile(path, profile)
        if pname not in path.fields:
            raise ValueError(f"No BL field reconstruction for {side}/{pname}. Run with reconstruct=True.")
        solver = create_bdim_solver(path, path.fields[pname], cfg)
        results[side] = solver.solve()
    return results["upper"], results["lower"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate thermal wall envelope Fluent comparison plots")
    parser.add_argument("case_dir", type=Path, help="Path to case directory")
    parser.add_argument("--solver-type", type=str, default=None, help="Panel solver type")
    parser.add_argument("--mesh-level", type=int, default=-1, help="Mesh refinement level index")
    parser.add_argument("--profiles", nargs="+", default=None, help="BL profiles")
    parser.add_argument("--profile", type=str, default="thwaites", help="BL profile for thermal solve")
    parser.add_argument("--transition", type=str, default=None, choices=["michel", "en"], help="Transition model")
    parser.add_argument("--nu", type=float, default=None, help="Kinematic viscosity override [m^2/s]")
    parser.add_argument("--T-inf", type=float, default=300.0, help="Freestream temperature [K]")
    parser.add_argument("--q-wall", type=float, default=500.0, help="Wall heat flux [W/m^2]")
    parser.add_argument("--scale", type=float, default=0.15, help="Envelope displacement scale")
    parser.add_argument("--mode", type=str, default="both", choices=["side_by_side", "overlay", "both"], help="Plot mode")
    parser.add_argument("--quantities", nargs="+", default=["wall_temperature", "heat_transfer_coeff"], help="Wall quantities")
    parser.add_argument("--show-plots", action="store_true", help="Display plots interactively")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir (default: <case>/out/thermal/fluent_comparison/wall_envelopes)",
    )
    args = parser.parse_args()

    if not args.show_plots:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    case_dir = args.case_dir.resolve()
    if not case_dir.exists():
        print(f"Error: Case directory not found: {case_dir}")
        return 1

    case = CaseLoader.load_case(case_dir, mesh_level_index=args.mesh_level)
    solver = case.create_solver(solver_type=args.solver_type)
    solver.solve()

    runner = BoundaryLayerRunner(case, solver)
    bl_cfg = case.config.boundary_layer
    profiles = args.profiles or list(bl_cfg.profiles)
    transition = args.transition or bl_cfg.transition_model
    bl_result = runner.run(profiles=profiles, nu=args.nu, transition_model=transition, reconstruct=True)

    thermal_cfg = _build_thermal_config(case, args.T_inf, args.q_wall)
    upper_thermal, lower_thermal = _run_thermal_bdim(case, bl_result, args.profile, thermal_cfg)
    comp = ThermalComparisonRunner(case, bl_result, upper_thermal, lower_thermal).run()

    output_dir = args.output_dir or (case.output_dir / "thermal" / "fluent_comparison" / "wall_envelopes")
    output_dir.mkdir(parents=True, exist_ok=True)

    for qty in args.quantities:
        if args.mode in ("side_by_side", "both"):
            plot_thermal_wall_envelope_side_by_side(
                comp,
                quantity=qty,
                scale=args.scale,
                title=f"{qty} envelope - Thermal vs Fluent - {case.name}",
                output_path=output_dir / f"thermal_wall_{qty}_side_by_side.png",
            )
            plt.close("all")

        if args.mode in ("overlay", "both"):
            plot_thermal_wall_envelope_overlay(
                comp,
                quantity=qty,
                scale=args.scale,
                title=f"{qty} envelope - Thermal vs Fluent - {case.name}",
                output_path=output_dir / f"thermal_wall_{qty}_overlay.png",
                show_difference=True,
            )
            plt.close("all")

    if args.show_plots:
        plt.show()
    print(f"Saved thermal wall envelope plots to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
