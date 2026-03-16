"""Generate tabulated Falkner-Skan ODE solutions for boundary layer reconstruction.

Solves the Falkner-Skan equation: f''' + f f'' + beta (1 - f'^2) = 0
BCs: f(0) = 0, f'(0) = 0, f'(inf) = 1

For beta=0 this reduces to f''' + f f'' = 0 (NOT the Blasius ODE which has
the 1/2 factor). The beta=0 case uses the Topfer scaling trick since standard
shooting fails (solutions don't diverge for wrong f''(0)).

For each beta value, outputs:
  - f_prime(eta): velocity profile
  - fpp0: wall shear f''(0)
  - I_1, I_2: displacement and momentum integrals
  - H: shape factor
  - S: shear parameter f''(0) * I_2
  - eta_99: 99% thickness
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


def falkner_skan_ode(eta: float, y: np.ndarray, beta: float) -> np.ndarray:
    """Falkner-Skan ODE system: f''' + f f'' + beta(1-f'^2) = 0."""
    f, fp, fpp = y
    return np.array([fp, fpp, -f * fpp - beta * (1.0 - fp**2)])


def _solve_beta_zero(eta_max: float, n_points: int) -> dict:
    """Solve F-S at beta=0 (f'''+f*f''=0) via Topfer scaling.

    This is NOT the Blasius equation (which has 0.5*f*f''). The beta=0 F-S
    equation has the same scaling symmetry as Blasius, so Topfer works.
    """
    # Solve auxiliary with fpp0=1
    eta_aux = np.linspace(0.0, eta_max * 2, n_points * 4)
    sol_aux = solve_ivp(
        falkner_skan_ode,
        [0.0, eta_max * 2],
        [0.0, 0.0, 1.0],
        args=(0.0,),
        method="RK45",
        t_eval=eta_aux,
        rtol=1e-12,
        atol=1e-14,
    )
    A = sol_aux.y[1, -1]  # f'(inf) for auxiliary

    # Topfer: correct fpp0 = alpha / A^(3/2) for f'''+f*f''=0
    # (same scaling as Blasius: g(xi)=a*F(a*xi), g'(inf)=a^2*A=1, a=1/sqrt(A))
    fpp0 = 1.0 / A**1.5

    # Solve with correct fpp0
    eta_eval = np.linspace(0.0, eta_max, n_points)
    sol = solve_ivp(
        falkner_skan_ode,
        [0.0, eta_max],
        [0.0, 0.0, fpp0],
        args=(0.0,),
        method="RK45",
        t_eval=eta_eval,
        rtol=1e-12,
        atol=1e-14,
    )

    return _extract_results(sol, fpp0, beta=0.0)


def _solve_beta_nonzero(
    beta: float,
    eta_max: float,
    n_points: int,
    fpp0_lo: float,
    fpp0_hi: float,
) -> dict | None:
    """Solve F-S for beta != 0 via shooting method.

    For beta > 0: solutions with too-high fpp0 overshoot (f' > 1 and diverge).
    For beta < 0: solutions with too-low fpp0 undershoot (f' goes negative).
    """

    def shoot(fpp0: float) -> float:
        # Terminal events to catch divergence
        def overshoot(eta, y, beta):
            return y[1] - 2.0

        overshoot.terminal = True
        overshoot.direction = 1

        def undershoot(eta, y, beta):
            return y[1] + 1.0

        undershoot.terminal = True
        undershoot.direction = -1

        try:
            sol = solve_ivp(
                falkner_skan_ode,
                [0, eta_max],
                [0, 0, fpp0],
                args=(beta,),
                method="RK45",
                events=[overshoot, undershoot],
                rtol=1e-8,
                atol=1e-10,
                max_step=0.5,
            )
            if sol.t_events[0].size > 0:
                return 1.0  # fpp0 too high
            if sol.t_events[1].size > 0:
                return -2.0  # fpp0 too low
            if sol.success and np.isfinite(sol.y[1, -1]):
                return sol.y[1, -1] - 1.0
        except Exception:
            pass
        return 1e6

    # Verify bracket
    r_lo = shoot(fpp0_lo)
    r_hi = shoot(fpp0_hi)

    if r_lo * r_hi >= 0:
        # Try adaptive bracket search with fewer points
        test_vals = np.linspace(fpp0_lo, fpp0_hi, 20)
        found = False
        prev = r_lo
        for i, tv in enumerate(test_vals[1:], 1):
            cur = shoot(tv)
            if prev * cur < 0:
                fpp0_lo = test_vals[i - 1]
                fpp0_hi = tv
                found = True
                break
            prev = cur
        if not found:
            return None

    try:
        fpp0 = brentq(shoot, fpp0_lo, fpp0_hi, xtol=1e-12)
    except ValueError:
        return None

    # Full solution with correct fpp0
    eta_eval = np.linspace(0.0, eta_max, n_points)
    sol = solve_ivp(
        falkner_skan_ode,
        [0.0, eta_max],
        [0.0, 0.0, fpp0],
        args=(beta,),
        method="RK45",
        t_eval=eta_eval,
        rtol=1e-12,
        atol=1e-14,
    )

    if not sol.success:
        return None

    return _extract_results(sol, fpp0, beta)


def _extract_results(sol, fpp0: float, beta: float) -> dict:
    """Extract integral quantities and profile from ODE solution."""
    eta = sol.t
    fp = sol.y[1]

    I_1 = float(np.trapz(1.0 - fp, eta))
    I_2 = float(np.trapz(fp * (1.0 - fp), eta))
    H = I_1 / I_2 if I_2 > 1e-15 else float("inf")
    S = float(fpp0) * I_2

    # eta_99
    idx_99 = np.searchsorted(fp, 0.99)
    if 0 < idx_99 < len(eta):
        t = (0.99 - fp[idx_99 - 1]) / (fp[idx_99] - fp[idx_99 - 1])
        eta_99 = float(eta[idx_99 - 1] + t * (eta[idx_99] - eta[idx_99 - 1]))
    elif idx_99 == 0:
        eta_99 = float(eta[0])
    else:
        eta_99 = float(eta[-1])

    m = beta / (2.0 - beta) if abs(2.0 - beta) > 1e-10 else float("inf")

    return {
        "beta": float(beta),
        "m": float(m),
        "fpp0": float(fpp0),
        "I_1": float(I_1),
        "I_2": float(I_2),
        "H": float(H),
        "S": float(S),
        "eta_99": float(eta_99),
        "f_prime": fp.tolist(),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root / "data" / "bl-solver-profiles" / "falkner_skan.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Beta values: dense near separation and stagnation
    beta_values = [
        -0.19880,  # Near separation
        -0.18,
        -0.15,
        -0.12,
        -0.10,
        -0.05,
        0.0,  # Flat plate (F-S convention, NOT Blasius ODE)
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        1.00,  # Hiemenz (stagnation)
        1.10,
        1.20,
        1.40,
        1.60,
        1.80,
        2.00,
    ]

    eta_max = 12.0
    n_points = 200

    print(f"Solving Falkner-Skan ODE for {len(beta_values)} beta values...")
    print(f"  eta_max={eta_max}, n_points={n_points}")
    print()

    # Known approximate f''(0) values to guide bracketing
    # (from literature: Hartree 1937, White Table 4-2)
    known_fpp0 = {
        -0.1988: 0.005, -0.18: 0.13, -0.15: 0.22, -0.12: 0.28,
        -0.10: 0.32, -0.05: 0.40, 0.05: 0.53, 0.10: 0.59,
        0.15: 0.64, 0.20: 0.69, 0.25: 0.73, 0.30: 0.77,
        0.40: 0.85, 0.50: 0.93, 0.60: 1.00, 0.70: 1.07,
        0.80: 1.12, 0.90: 1.18, 1.00: 1.23, 1.10: 1.28,
        1.20: 1.34, 1.40: 1.43, 1.60: 1.52, 1.80: 1.60,
        2.00: 1.69,
    }

    profiles = []
    for beta in beta_values:
        if abs(beta) < 1e-10:
            # Beta=0: use Topfer scaling
            result = _solve_beta_zero(eta_max, n_points)
        else:
            # Use known value to set a tight bracket
            approx = known_fpp0.get(beta)
            if approx is not None:
                lo = max(approx * 0.5, 1e-4)
                hi = approx * 2.0
            elif beta < 0:
                lo, hi = 1e-4, 0.5
            elif beta < 1:
                lo, hi = 0.3, 1.5
            else:
                lo, hi = 0.5, 3.0

            result = _solve_beta_nonzero(beta, eta_max, n_points, lo, hi)

        if result is not None:
            profiles.append(result)
            print(
                f"  beta={beta:+.4f}  f''(0)={result['fpp0']:.6f}  "
                f"H={result['H']:.4f}  S={result['S']:.6f}  eta_99={result['eta_99']:.3f}"
            )
        else:
            print(f"  beta={beta:+.4f}  FAILED")

    # Common eta grid (all profiles solved on the same grid)
    eta_grid = np.linspace(0.0, eta_max, n_points).tolist()

    output = {
        "eta": eta_grid,
        "profiles": profiles,
        "metadata": {
            "n_profiles": len(profiles),
            "eta_max": eta_max,
            "n_points": n_points,
            "ode": "f''' + f f'' + beta(1-f'^2) = 0",
            "description": (
                "Falkner-Skan similarity solutions. Note: beta=0 here "
                "corresponds to f'''+f*f''=0 (F-S convention), not the "
                "Blasius ODE f'''+0.5*f*f''=0. The two differ by a "
                "sqrt(2) scaling of the similarity variable."
            ),
        },
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWritten {len(profiles)} profiles to {output_path}")

    # Validate
    print("\nValidation:")
    known = {
        0.0: {"fpp0": 0.4696, "H": 2.591, "name": "Flat plate (F-S)"},
        1.0: {"fpp0": 1.23259, "H": 2.216, "name": "Hiemenz"},
    }
    ok = True
    for beta_check, expected in known.items():
        for p in profiles:
            if abs(p["beta"] - beta_check) < 1e-6:
                err_fpp = abs(p["fpp0"] - expected["fpp0"])
                err_H = abs(p["H"] - expected["H"])
                # Relaxed tolerance for beta=0 (different ODE scaling)
                fpp_tol = 0.001 if beta_check != 0 else 0.01
                status = "OK" if err_fpp < fpp_tol and err_H < 0.02 else "FAIL"
                if status == "FAIL":
                    ok = False
                print(
                    f"  {expected['name']} (beta={beta_check}): "
                    f"f''(0)={p['fpp0']:.6f} (err={err_fpp:.2e}), "
                    f"H={p['H']:.4f} (err={err_H:.2e}) [{status}]"
                )
                break

    if ok:
        print("All validation checks passed.")
    else:
        print("Some checks failed — review output.")
        sys.exit(1)


if __name__ == "__main__":
    main()
