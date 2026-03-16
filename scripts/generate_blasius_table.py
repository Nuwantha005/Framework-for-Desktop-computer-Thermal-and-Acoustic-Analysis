"""Generate tabulated Blasius ODE solution for boundary layer reconstruction.

Solves the Blasius equation: f''' + (1/2) f f'' = 0
BCs: f(0) = 0, f'(0) = 0, f'(inf) = 1
Similarity variable: eta = y * sqrt(U_inf / (nu * x))

Uses the Topfer scaling trick: solve once with f''(0)=1, then rescale.

Outputs a JSON file with:
  - eta: similarity variable grid
  - f: stream function f(eta)
  - f_prime: velocity profile f'(eta) = u/U_e
  - f_double_prime: shear f''(eta)
  - constants: f''(0), I_1, I_2, eta_99, H
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


def blasius_ode(eta: float, y: np.ndarray) -> np.ndarray:
    """Blasius ODE system: f''' + 0.5*f*f'' = 0."""
    f, fp, fpp = y
    return np.array([fp, fpp, -0.5 * f * fpp])


def solve_blasius(eta_max: float = 10.0, n_points: int = 200) -> dict:
    """Solve the Blasius equation via Topfer scaling trick.

    Procedure:
      1. Solve with f''(0) = 1 (arbitrary) on a fine grid
      2. Get the asymptotic f'(inf) = L
      3. Correct solution: f_correct(eta) = L^(1/2) * F(L^(1/2) * eta)
         where F is the auxiliary solution, which gives f''_correct(0) = L^(3/2)
         Wait — we want f'_correct(inf)=1. The Topfer transform for
         f''' + 0.5*f*f'' = 0 is:
           g(xi) = F(xi/c)/c  with c = sqrt(L)
         so g'(inf) = F'(inf)/L = 1, g''(0) = F''(0)*c = 1/sqrt(L)
      Actually, the standard Topfer for f'''+af*f''=0:
         solve with fpp0=alpha, get fp_inf=A.
         correct fpp0 = alpha * A^(-3/2) (White 4.10, Schlichting 7.30)

    Args:
        eta_max: Upper bound of similarity variable domain.
        n_points: Number of output grid points.

    Returns:
        Dictionary with eta, f, f_prime, f_double_prime arrays and constants.
    """
    # Step 1: Solve auxiliary problem with f''(0) = 1
    alpha = 1.0
    eta_aux_max = eta_max * 2  # Solve on a larger domain for the auxiliary
    n_aux = n_points * 4

    sol_aux = solve_ivp(
        blasius_ode,
        [0.0, eta_aux_max],
        [0.0, 0.0, alpha],
        method="RK45",
        t_eval=np.linspace(0.0, eta_aux_max, n_aux),
        rtol=1e-12,
        atol=1e-14,
    )

    A = sol_aux.y[1, -1]  # f'(inf) for auxiliary solution

    # Step 2: Apply Topfer scaling
    # For f''' + 0.5*f*f'' = 0:
    # If F(eta) solves with F''(0)=alpha, F'(inf)=A, then
    # f(eta) = F(eta*c)/c with c = sqrt(A)
    # gives f''(0) = alpha*c = alpha*sqrt(A) and f'(inf) = A/A = 1
    # Wait, let me derive carefully:
    # f(eta) = F(c*eta)/c
    # f'(eta) = F'(c*eta)
    # f''(eta) = c * F''(c*eta)
    # f'''(eta) = c^2 * F'''(c*eta)
    # Substituting: c^2*F''' + 0.5*(F/c)*c*F'' = c^2*F''' + 0.5*F*F'' = 0 ✓
    # f'(inf) = F'(inf) = A  ... that's not 1!
    #
    # Alternative: f(eta) = c * F(eta/c)
    # f'(eta) = F'(eta/c)
    # f''(eta) = F''(eta/c)/c
    # f'''(eta) = F'''(eta/c)/c^2
    # Substituting: F'''/c^2 + 0.5*c*F*F''/c = F'''/(c^2) + 0.5*F*F'' ≠ 0
    # unless c=1. So that doesn't work either.
    #
    # Correct Topfer for f'''+0.5*f*f''=0:
    # g(xi) = b*F(a*xi), g'(xi) = ab*F'(a*xi), g''(xi)=a^2*b*F''(a*xi),
    # g'''(xi)=a^3*b*F'''(a*xi)
    # g'''+0.5*g*g'' = a^3*b*F''' + 0.5*b*F*a^2*b*F'' = a^2*b(a*F'''+0.5*b*F*F'')
    # For this to equal zero: a*F''' + 0.5*b*F*F'' = 0
    # We need b = a (comparing with F'''+0.5*F*F''=0)
    # So g(xi) = a*F(a*xi), g'(inf) = a^2*F'(inf) = a^2*A
    # We want g'(inf)=1: a = 1/sqrt(A)
    # g''(0) = a^3*F''(0) = alpha/A^(3/2)
    fpp0 = alpha / A ** 1.5
    c = 1.0 / np.sqrt(A)

    # Step 3: Recompute the solution with the correct f''(0)
    # (More accurate than rescaling the auxiliary solution)
    eta_eval = np.linspace(0.0, eta_max, n_points)

    sol = solve_ivp(
        blasius_ode,
        [0.0, eta_max],
        [0.0, 0.0, fpp0],
        method="RK45",
        t_eval=eta_eval,
        rtol=1e-12,
        atol=1e-14,
    )

    eta = sol.t
    f = sol.y[0]
    fp = sol.y[1]
    fpp = sol.y[2]

    # Integral quantities via trapezoidal rule
    I_1 = float(np.trapz(1.0 - fp, eta))
    I_2 = float(np.trapz(fp * (1.0 - fp), eta))
    H = I_1 / I_2

    # eta_99: where f'(eta) >= 0.99
    idx_99 = np.searchsorted(fp, 0.99)
    if 0 < idx_99 < len(eta):
        t = (0.99 - fp[idx_99 - 1]) / (fp[idx_99] - fp[idx_99 - 1])
        eta_99 = float(eta[idx_99 - 1] + t * (eta[idx_99] - eta[idx_99 - 1]))
    else:
        eta_99 = float(eta_max)

    return {
        "eta": eta.tolist(),
        "f": f.tolist(),
        "f_prime": fp.tolist(),
        "f_double_prime": fpp.tolist(),
        "constants": {
            "fpp0": float(fpp0),
            "I_1": I_1,
            "I_2": I_2,
            "H": H,
            "eta_99": eta_99,
        },
        "metadata": {
            "ode": "f''' + 0.5*f*f'' = 0",
            "similarity_variable": "eta = y * sqrt(U_inf / (nu * x))",
            "velocity_profile": "u/U_e = f'(eta)",
            "method": "Topfer scaling trick",
            "n_points": n_points,
            "eta_max": eta_max,
        },
    }


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root / "data" / "bl-solver-profiles" / "blasius.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Solving Blasius ODE via Topfer scaling trick...")
    result = solve_blasius(eta_max=10.0, n_points=200)

    c = result["constants"]
    print(f"  f''(0) = {c['fpp0']:.6f}  (expected: 0.332060)")
    print(f"  I_1    = {c['I_1']:.6f}  (expected: 1.720800)")
    print(f"  I_2    = {c['I_2']:.6f}  (expected: 0.664110)")
    print(f"  H      = {c['H']:.6f}  (expected: 2.591100)")
    print(f"  eta_99 = {c['eta_99']:.4f}")
    print(f"  Grid points: {len(result['eta'])}")

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Written to {output_path}")

    # Validate against known values
    tol = 1e-3
    checks = [
        ("f''(0)", c["fpp0"], 0.33206, tol),
        ("I_1", c["I_1"], 1.72080, tol),
        ("I_2", c["I_2"], 0.66411, tol),
        ("H", c["H"], 2.59110, 0.01),
    ]
    ok = True
    for name, got, expected, t in checks:
        if abs(got - expected) > t:
            print(f"  WARNING: {name} = {got:.6f}, expected {expected:.6f}")
            ok = False
    if ok:
        print("All validation checks passed.")
    else:
        print("Some checks failed — review output.")
        sys.exit(1)


if __name__ == "__main__":
    main()
