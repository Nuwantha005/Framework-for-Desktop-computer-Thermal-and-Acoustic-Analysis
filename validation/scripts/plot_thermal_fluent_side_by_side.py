#!/usr/bin/env python3
"""Generate thermal Fluent side-by-side absolute plots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from core.io.case_loader import CaseLoader
from solvers.boundary_layer.runner import BoundaryLayerRunner
from solvers.thermal.base import ThermalSolverConfig
from solvers.thermal.factory import create_bdim_solver
from validation.adapters.fluent import ThermalComparisonRunner
from visualization.thermal_fluent_comparison_plots import (
    plot_thermal_fluent_contour_normalized_side_by_side,
    plot_thermal_fluent_contour_side_by_side,
    plot_thermal_fluent_envelope_side_by_side,
)


def _resolve_profile(path_result, profile: str) -> str:
    for name in path_result.results.keys():
        if name.lower() == profile.lower():
            return name
    raise KeyError(f"Profile '{profile}' not found. Available: {list(path_result.results.keys())}")


def _thermal_config(case, T_inf: float, q_wall: float) -> ThermalSolverConfig:
    fluid = case.config.fluid
    rho = fluid.density
    mu = fluid.viscosity if fluid.viscosity else 1.81e-5
    k = fluid.thermal_conductivity if fluid.thermal_conductivity else 0.026
    cp = fluid.specific_heat_cp if fluid.specific_heat_cp else 1005.0
    Pr = (cp * mu) / k
    return ThermalSolverConfig(T_inf=T_inf, Pr=Pr, k=k, rho=rho, cp=cp, q_wall=q_wall)


def _run_thermal(case, bl_result, profile: str, cfg: ThermalSolverConfig):
    out = {}
    for side, path in bl_result.sides.items():
        pname = _resolve_profile(path, profile)
        if pname not in path.fields:
            raise ValueError(f"No BL field reconstruction for {side}/{pname}. Run with reconstruct=True.")
        out[side] = create_bdim_solver(path, path.fields[pname], cfg).solve()
    return out["upper"], out["lower"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate thermal Fluent side-by-side plots")
    parser.add_argument("case_dir", type=Path, help="Path to case directory")
    parser.add_argument("--solver-type", type=str, default=None, help="Panel solver type")
    parser.add_argument("--mesh-level", type=int, default=-1, help="Mesh refinement level index")
    parser.add_argument("--profiles", nargs="+", default=None, help="BL profiles")
    parser.add_argument("--profile", type=str, default="thwaites", help="BL profile for thermal solve")
    parser.add_argument("--transition", type=str, default=None, choices=["michel", "en"], help="Transition model")
    parser.add_argument("--nu", type=float, default=None, help="Kinematic viscosity override [m^2/s]")
    parser.add_argument("--T-inf", type=float, default=300.0, help="Freestream temperature [K]")
    parser.add_argument("--q-wall", type=float, default=500.0, help="Wall heat flux [W/m^2]")
    parser.add_argument("--show-plots", action="store_true", help="Display plots interactively")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir (default: <case>/out/thermal/fluent_comparison/side_by_side)",
    )
    args = parser.parse_args()

    if not args.show_plots:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    case_dir = args.case_dir.resolve()
    case = CaseLoader.load_case(case_dir, mesh_level_index=args.mesh_level)
    solver = case.create_solver(solver_type=args.solver_type)
    solver.solve()

    bl_runner = BoundaryLayerRunner(case, solver)
    bl_cfg = case.config.boundary_layer
    profiles = args.profiles or list(bl_cfg.profiles)
    transition = args.transition or bl_cfg.transition_model
    bl_result = bl_runner.run(profiles=profiles, nu=args.nu, transition_model=transition, reconstruct=True)

    cfg = _thermal_config(case, args.T_inf, args.q_wall)
    upper_thermal, lower_thermal = _run_thermal(case, bl_result, args.profile, cfg)
    comp = ThermalComparisonRunner(case, bl_result, upper_thermal, lower_thermal).run()

    out_dir = args.output_dir or (case.output_dir / "thermal" / "fluent_comparison" / "side_by_side")
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_thermal_fluent_envelope_side_by_side(
        comp,
        title=f"Absolute temperature envelope - Thermal vs Fluent - {case.name}",
        output_path=out_dir / "thermal_fluent_envelope_abs.png",
    )
    plt.close("all")

    for side, thermal_res, fluent_field in [
        ("upper", upper_thermal, comp.upper_fluent_field),
        ("lower", lower_thermal, comp.lower_fluent_field),
    ]:
        plot_thermal_fluent_contour_side_by_side(
            thermal_res,
            fluent_field,
            title=f"{side.capitalize()} absolute temperature - Thermal vs Fluent - {case.name}",
            output_path=out_dir / f"thermal_fluent_contour_abs_{side}.png",
        )
        plt.close("all")

        plot_thermal_fluent_contour_normalized_side_by_side(
            thermal_res,
            fluent_field,
            title=f"{side.capitalize()} normalized temperature - Thermal vs Fluent - {case.name}",
            output_path=out_dir / f"thermal_fluent_contour_normalized_abs_{side}.png",
        )
        plt.close("all")

    if args.show_plots:
        plt.show()
    print(f"Saved thermal side-by-side plots to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
