"""Tabulated ODE solutions for Blasius and Falkner-Skan velocity profiles.

Provides lazy-loaded, interpolation-ready access to pre-computed similarity
solutions stored as JSON in ``data/bl-solver-profiles/``.  When a query falls
outside the tabulated range, the ODE is solved on-the-fly as a fallback.

Public API
----------
- ``blasius_table()`` — returns the singleton ``BlasiusTable``.
- ``falkner_skan_table()`` — returns the singleton ``FalknerSkanTable``.

Each table object exposes typed interpolation methods; see class docstrings.

Notes
-----
The Blasius ODE (``f''' + 0.5 f f'' = 0``) and the Falkner-Skan ODE at
β = 0 (``f''' + f f'' = 0``) are *different* equations related by a √2
rescaling of the similarity variable.  They are stored in separate tables.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# Data directory (relative to project root, resolved at import time)
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).resolve().parents[4] / "data" / "bl-solver-profiles"


# ===================================================================== #
#                          Blasius table                                  #
# ===================================================================== #

@dataclass(frozen=True)
class BlasiusConstants:
    """Integral constants from the Blasius ODE solution.

    Attributes:
        fpp0: Wall shear f''(0) ≈ 0.33206.
        I_1: Displacement integral ∫(1 − f') dη.
        I_2: Momentum integral ∫ f'(1 − f') dη.
        H: Shape factor I_1 / I_2.
        eta_99: Similarity variable where f' = 0.99.
    """
    fpp0: float
    I_1: float
    I_2: float
    H: float
    eta_99: float


class BlasiusTable:
    """Lazy-loaded, interpolation-ready Blasius ODE solution.

    The table is loaded from ``data/bl-solver-profiles/blasius.json`` on first
    access.  All arrays are stored as contiguous float64 for fast interp.
    """

    def __init__(self) -> None:
        self._eta: NDArray[np.float64] | None = None
        self._f: NDArray[np.float64] | None = None
        self._f_prime: NDArray[np.float64] | None = None
        self._f_double_prime: NDArray[np.float64] | None = None
        self._constants: BlasiusConstants | None = None
        self._lock = threading.Lock()

    # -- lazy load --------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._eta is not None:
            return
        with self._lock:
            if self._eta is not None:  # double-check after acquiring lock
                return
            path = _DATA_DIR / "blasius.json"
            if not path.exists():
                raise FileNotFoundError(
                    f"Blasius table not found at {path}.  "
                    "Run scripts/generate_blasius_table.py first."
                )
            with open(path) as fh:
                data = json.load(fh)

            self._eta = np.ascontiguousarray(data["eta"], dtype=np.float64)
            self._f = np.ascontiguousarray(data["f"], dtype=np.float64)
            self._f_prime = np.ascontiguousarray(data["f_prime"], dtype=np.float64)
            self._f_double_prime = np.ascontiguousarray(
                data["f_double_prime"], dtype=np.float64,
            )
            c = data["constants"]
            self._constants = BlasiusConstants(
                fpp0=c["fpp0"], I_1=c["I_1"], I_2=c["I_2"],
                H=c["H"], eta_99=c["eta_99"],
            )

    # -- public accessors -------------------------------------------------

    @property
    def eta(self) -> NDArray[np.float64]:
        """Similarity-variable grid (N,)."""
        self._ensure_loaded()
        assert self._eta is not None
        return self._eta

    @property
    def constants(self) -> BlasiusConstants:
        """Integral constants (fpp0, I_1, I_2, H, eta_99)."""
        self._ensure_loaded()
        assert self._constants is not None
        return self._constants

    # -- interpolation ----------------------------------------------------

    def fprime(self, eta: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Interpolate the Blasius velocity profile f'(η) = u/U_e.

        Args:
            eta: Similarity variable value(s).  Scalar or array.

        Returns:
            f'(η) interpolated from the table.  Values beyond the table
            domain are clamped (0 at η < 0, 1 at η > η_max).
        """
        self._ensure_loaded()
        assert self._f_prime is not None
        eta_arr = np.asarray(eta, dtype=np.float64)
        return np.interp(eta_arr, self._eta, self._f_prime)  # type: ignore[arg-type]

    def f(self, eta: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Interpolate the stream-function f(η).

        Args:
            eta: Similarity variable value(s).

        Returns:
            f(η) interpolated from the table.
        """
        self._ensure_loaded()
        assert self._f is not None
        eta_arr = np.asarray(eta, dtype=np.float64)
        return np.interp(eta_arr, self._eta, self._f)  # type: ignore[arg-type]

    def fdoubleprime(self, eta: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Interpolate the wall-shear profile f''(η).

        Args:
            eta: Similarity variable value(s).

        Returns:
            f''(η) interpolated from the table.
        """
        self._ensure_loaded()
        assert self._f_double_prime is not None
        eta_arr = np.asarray(eta, dtype=np.float64)
        return np.interp(eta_arr, self._eta, self._f_double_prime)  # type: ignore[arg-type]


# ===================================================================== #
#                       Falkner-Skan table                               #
# ===================================================================== #

@dataclass(frozen=True)
class FalknerSkanConstants:
    """Per-β integral constants from the Falkner-Skan ODE.

    Attributes:
        beta: Pressure-gradient parameter.
        m: Wedge exponent Ue ∝ x^m; m = β / (2 − β).
        fpp0: Wall shear f''(0).
        I_1: Displacement integral.
        I_2: Momentum integral.
        H: Shape factor I_1 / I_2.
        S: Shear parameter f''(0) · I_2.
        eta_99: η where f' = 0.99.
    """
    beta: float
    m: float
    fpp0: float
    I_1: float
    I_2: float
    H: float
    S: float
    eta_99: float


class FalknerSkanTable:
    """Lazy-loaded Falkner-Skan ODE solutions for a range of β values.

    The table is loaded from ``data/bl-solver-profiles/falkner_skan.json``.
    Interpolation across β is linear; profile interpolation across η uses
    ``numpy.interp`` per β endpoint, then linearly blends.

    If a requested β lies outside the tabulated range, an on-the-fly ODE
    solve is attempted as a fallback.
    """

    def __init__(self) -> None:
        self._eta: NDArray[np.float64] | None = None
        self._betas: NDArray[np.float64] | None = None
        # (n_beta, n_eta) array of f'(eta) profiles
        self._fprime_grid: NDArray[np.float64] | None = None
        # Per-beta constants stored as parallel arrays for fast vectorised interp
        self._fpp0: NDArray[np.float64] | None = None
        self._I_1: NDArray[np.float64] | None = None
        self._I_2: NDArray[np.float64] | None = None
        self._H: NDArray[np.float64] | None = None
        self._S: NDArray[np.float64] | None = None
        self._eta_99: NDArray[np.float64] | None = None
        self._m: NDArray[np.float64] | None = None
        self._lock = threading.Lock()

    # -- lazy load --------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._eta is not None:
            return
        with self._lock:
            if self._eta is not None:
                return
            path = _DATA_DIR / "falkner_skan.json"
            if not path.exists():
                raise FileNotFoundError(
                    f"Falkner-Skan table not found at {path}.  "
                    "Run scripts/generate_falkner_skan_table.py first."
                )
            with open(path) as fh:
                data = json.load(fh)

            self._eta = np.ascontiguousarray(data["eta"], dtype=np.float64)

            profiles = data["profiles"]
            n_beta = len(profiles)
            n_eta = len(self._eta)

            self._betas = np.empty(n_beta, dtype=np.float64)
            self._fpp0 = np.empty(n_beta, dtype=np.float64)
            self._I_1 = np.empty(n_beta, dtype=np.float64)
            self._I_2 = np.empty(n_beta, dtype=np.float64)
            self._H = np.empty(n_beta, dtype=np.float64)
            self._S = np.empty(n_beta, dtype=np.float64)
            self._eta_99 = np.empty(n_beta, dtype=np.float64)
            self._m = np.empty(n_beta, dtype=np.float64)
            self._fprime_grid = np.empty((n_beta, n_eta), dtype=np.float64)

            for i, p in enumerate(profiles):
                self._betas[i] = p["beta"]
                self._fpp0[i] = p["fpp0"]
                self._I_1[i] = p["I_1"]
                self._I_2[i] = p["I_2"]
                self._H[i] = p["H"]
                self._S[i] = p["S"]
                self._eta_99[i] = p["eta_99"]
                self._m[i] = p["m"]
                self._fprime_grid[i] = np.asarray(p["f_prime"], dtype=np.float64)

    # -- public accessors -------------------------------------------------

    @property
    def eta(self) -> NDArray[np.float64]:
        """Common η grid shared by all profiles (N_eta,)."""
        self._ensure_loaded()
        assert self._eta is not None
        return self._eta

    @property
    def betas(self) -> NDArray[np.float64]:
        """Sorted array of tabulated β values (N_beta,)."""
        self._ensure_loaded()
        assert self._betas is not None
        return self._betas

    @property
    def beta_range(self) -> tuple[float, float]:
        """(β_min, β_max) covered by the table."""
        b = self.betas
        return float(b[0]), float(b[-1])

    # -- scalar constant interpolation ------------------------------------

    def constants(self, beta: float) -> FalknerSkanConstants:
        """Interpolate all integral constants at a given β.

        If *beta* is outside the tabulated range, falls back to on-the-fly
        ODE solution (slower but correct).

        Args:
            beta: Falkner-Skan pressure-gradient parameter.

        Returns:
            FalknerSkanConstants with all fields interpolated/computed.
        """
        self._ensure_loaded()
        assert self._betas is not None

        b_min, b_max = float(self._betas[0]), float(self._betas[-1])
        if beta < b_min or beta > b_max:
            return self._solve_ode_fallback(beta)

        return FalknerSkanConstants(
            beta=beta,
            m=float(np.interp(beta, self._betas, self._m)),  # type: ignore[arg-type]
            fpp0=float(np.interp(beta, self._betas, self._fpp0)),  # type: ignore[arg-type]
            I_1=float(np.interp(beta, self._betas, self._I_1)),  # type: ignore[arg-type]
            I_2=float(np.interp(beta, self._betas, self._I_2)),  # type: ignore[arg-type]
            H=float(np.interp(beta, self._betas, self._H)),  # type: ignore[arg-type]
            S=float(np.interp(beta, self._betas, self._S)),  # type: ignore[arg-type]
            eta_99=float(np.interp(beta, self._betas, self._eta_99)),  # type: ignore[arg-type]
        )

    def constants_from_H(self, H_target: float) -> FalknerSkanConstants:
        """Invert H → β and return all constants at that β.

        Useful for Thwaites-to-Falkner-Skan pairing where the integral
        solver provides H but reconstruction needs the F-S profile.

        Args:
            H_target: Shape factor to invert.

        Returns:
            FalknerSkanConstants at the β that gives H = H_target.

        Raises:
            ValueError: If H_target is outside the tabulated H range.
        """
        self._ensure_loaded()
        assert self._H is not None and self._betas is not None

        H_min, H_max = float(self._H.min()), float(self._H.max())
        if H_target < H_min or H_target > H_max:
            raise ValueError(
                f"H = {H_target:.4f} is outside the tabulated range "
                f"[{H_min:.4f}, {H_max:.4f}]."
            )
        # H is monotonically *decreasing* with increasing β, so we must
        # flip for np.interp which expects increasing x.
        beta = float(np.interp(
            H_target,
            self._H[::-1],   # ascending H (reversed)
            self._betas[::-1],  # corresponding β (reversed)
        ))
        return self.constants(beta)

    # -- profile interpolation --------------------------------------------

    def fprime(
        self,
        beta: float,
        eta: float | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Interpolate f'(η) at arbitrary β.

        Performs bilinear interpolation: linear in β between the two
        bracketing tabulated profiles, and ``numpy.interp`` in η within
        each profile.

        Args:
            beta: Pressure-gradient parameter.
            eta: Similarity variable value(s).  Scalar or array.

        Returns:
            Interpolated f'(η) values as a float64 array.
        """
        self._ensure_loaded()
        assert self._betas is not None and self._fprime_grid is not None

        eta_arr = np.asarray(eta, dtype=np.float64)

        b_min, b_max = float(self._betas[0]), float(self._betas[-1])
        if beta < b_min or beta > b_max:
            # Fallback: solve ODE and interpolate the result
            c = self._solve_ode_fallback(beta)
            # Re-solve to get the full profile (constants alone aren't enough)
            return self._solve_fprime_fallback(beta, eta_arr)

        # Find bracketing indices
        idx = np.searchsorted(self._betas, beta)
        if idx == 0:
            return np.interp(eta_arr, self._eta, self._fprime_grid[0])  # type: ignore[arg-type]
        if idx >= len(self._betas):
            return np.interp(eta_arr, self._eta, self._fprime_grid[-1])  # type: ignore[arg-type]

        # Linear blend in β
        b_lo, b_hi = self._betas[idx - 1], self._betas[idx]
        t = (beta - b_lo) / (b_hi - b_lo) if b_hi != b_lo else 0.0

        fp_lo = np.interp(eta_arr, self._eta, self._fprime_grid[idx - 1])  # type: ignore[arg-type]
        fp_hi = np.interp(eta_arr, self._eta, self._fprime_grid[idx])  # type: ignore[arg-type]

        return (1.0 - t) * fp_lo + t * fp_hi

    def fprime_from_H(
        self,
        H_target: float,
        eta: float | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Invert H → β, then interpolate f'(η).

        Convenience wrapper combining ``constants_from_H`` and ``fprime``.

        Args:
            H_target: Shape factor to invert.
            eta: Similarity variable value(s).

        Returns:
            Interpolated f'(η) at the β corresponding to H_target.
        """
        c = self.constants_from_H(H_target)
        return self.fprime(c.beta, eta)

    # -- ODE fallback -----------------------------------------------------

    @staticmethod
    def _falkner_skan_ode(
        eta: float, y: NDArray[np.float64], beta: float,
    ) -> NDArray[np.float64]:
        """f''' + f f'' + β(1 − f'²) = 0."""
        f, fp, fpp = y
        return np.array([fp, fpp, -f * fpp - beta * (1.0 - fp**2)])

    def _solve_ode_fallback(self, beta: float) -> FalknerSkanConstants:
        """Solve the F-S ODE on-the-fly for an out-of-table β.

        Uses the same shooting strategy as the generation script.
        """
        eta_max = 15.0
        n_points = 200

        if abs(beta) < 1e-10:
            result = self._solve_beta_zero(eta_max, n_points)
        else:
            result = self._solve_beta_nonzero(beta, eta_max, n_points)

        if result is None:
            raise RuntimeError(
                f"ODE fallback failed to converge for beta = {beta:.6f}."
            )
        return result

    def _solve_fprime_fallback(
        self, beta: float, eta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Solve the F-S ODE on-the-fly and return f'(η) at requested points."""
        eta_max = max(15.0, float(eta.max()) * 1.2) if eta.size > 0 else 15.0
        n_points = 400

        if abs(beta) < 1e-10:
            fpp0 = self._topfer_fpp0(eta_max, n_points)
        else:
            fpp0 = self._shoot_fpp0(beta, eta_max)

        sol = solve_ivp(
            self._falkner_skan_ode,
            [0.0, eta_max],
            [0.0, 0.0, fpp0],
            args=(beta,),
            method="RK45",
            t_eval=np.sort(np.unique(np.clip(eta, 0.0, eta_max))),
            rtol=1e-10,
            atol=1e-12,
        )
        # Interpolate back to the requested eta
        return np.interp(eta, sol.t, sol.y[1])

    # -- shooting helpers (mirror generation script logic) ----------------

    @classmethod
    def _topfer_fpp0(cls, eta_max: float, n_points: int) -> float:
        """Topfer trick for β = 0."""
        sol_aux = solve_ivp(
            cls._falkner_skan_ode,
            [0.0, eta_max * 2],
            [0.0, 0.0, 1.0],
            args=(0.0,),
            method="RK45",
            t_eval=np.linspace(0.0, eta_max * 2, n_points * 4),
            rtol=1e-12,
            atol=1e-14,
        )
        A = sol_aux.y[1, -1]
        return 1.0 / A**1.5

    @classmethod
    def _shoot_fpp0(cls, beta: float, eta_max: float) -> float:
        """Shooting to find f''(0) for β ≠ 0."""

        def shoot(fpp0: float) -> float:
            def overshoot(eta, y, beta):
                return y[1] - 2.0
            overshoot.terminal = True  # type: ignore[attr-defined]
            overshoot.direction = 1  # type: ignore[attr-defined]

            def undershoot(eta, y, beta):
                return y[1] + 1.0
            undershoot.terminal = True  # type: ignore[attr-defined]
            undershoot.direction = -1  # type: ignore[attr-defined]

            try:
                sol = solve_ivp(
                    cls._falkner_skan_ode,
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
                    return 1.0
                if sol.t_events[1].size > 0:
                    return -2.0
                if sol.success and np.isfinite(sol.y[1, -1]):
                    return sol.y[1, -1] - 1.0
            except Exception:
                pass
            return 1e6

        # Initial bracket based on beta sign
        if beta > 0:
            lo, hi = 0.1, max(3.0, beta * 2.0)
        else:
            lo, hi = 1e-4, 0.5

        # Verify bracket, scan if needed
        r_lo, r_hi = shoot(lo), shoot(hi)
        if r_lo * r_hi >= 0:
            test_vals = np.linspace(lo, hi, 30)
            prev = r_lo
            found = False
            for i, tv in enumerate(test_vals[1:], 1):
                cur = shoot(tv)
                if prev * cur < 0:
                    lo, hi = test_vals[i - 1], tv
                    found = True
                    break
                prev = cur
            if not found:
                raise RuntimeError(
                    f"ODE fallback: could not bracket f''(0) for beta={beta:.6f}"
                )

        return float(brentq(shoot, lo, hi, xtol=1e-12))

    def _solve_beta_zero(
        self, eta_max: float, n_points: int,
    ) -> FalknerSkanConstants:
        """Solve β = 0 via Topfer and extract constants."""
        fpp0 = self._topfer_fpp0(eta_max, n_points)
        eta_eval = np.linspace(0.0, eta_max, n_points)
        sol = solve_ivp(
            self._falkner_skan_ode,
            [0.0, eta_max],
            [0.0, 0.0, fpp0],
            args=(0.0,),
            method="RK45",
            t_eval=eta_eval,
            rtol=1e-12,
            atol=1e-14,
        )
        return self._extract_constants(sol, fpp0, 0.0)

    def _solve_beta_nonzero(
        self, beta: float, eta_max: float, n_points: int,
    ) -> FalknerSkanConstants | None:
        """Solve β ≠ 0 via shooting and extract constants."""
        try:
            fpp0 = self._shoot_fpp0(beta, eta_max)
        except RuntimeError:
            return None

        eta_eval = np.linspace(0.0, eta_max, n_points)
        sol = solve_ivp(
            self._falkner_skan_ode,
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
        return self._extract_constants(sol, fpp0, beta)

    @staticmethod
    def _extract_constants(sol, fpp0: float, beta: float) -> FalknerSkanConstants:
        """Compute integral constants from an ODE solution."""
        eta = sol.t
        fp = sol.y[1]

        I_1 = float(np.trapz(1.0 - fp, eta))
        I_2 = float(np.trapz(fp * (1.0 - fp), eta))
        H = I_1 / I_2 if I_2 > 1e-15 else float("inf")
        S = fpp0 * I_2

        idx_99 = np.searchsorted(fp, 0.99)
        if 0 < idx_99 < len(eta):
            t = (0.99 - fp[idx_99 - 1]) / (fp[idx_99] - fp[idx_99 - 1])
            eta_99 = float(eta[idx_99 - 1] + t * (eta[idx_99] - eta[idx_99 - 1]))
        elif idx_99 == 0:
            eta_99 = float(eta[0])
        else:
            eta_99 = float(eta[-1])

        m = beta / (2.0 - beta) if abs(2.0 - beta) > 1e-10 else float("inf")

        return FalknerSkanConstants(
            beta=beta, m=m, fpp0=fpp0,
            I_1=I_1, I_2=I_2, H=H, S=S, eta_99=eta_99,
        )


# ===================================================================== #
#                      Module-level singletons                           #
# ===================================================================== #

_blasius_instance: BlasiusTable | None = None
_falkner_skan_instance: FalknerSkanTable | None = None
_module_lock = threading.Lock()


def blasius_table() -> BlasiusTable:
    """Return the module-level Blasius table singleton (lazy-loaded)."""
    global _blasius_instance
    if _blasius_instance is None:
        with _module_lock:
            if _blasius_instance is None:
                _blasius_instance = BlasiusTable()
    return _blasius_instance


def falkner_skan_table() -> FalknerSkanTable:
    """Return the module-level Falkner-Skan table singleton (lazy-loaded)."""
    global _falkner_skan_instance
    if _falkner_skan_instance is None:
        with _module_lock:
            if _falkner_skan_instance is None:
                _falkner_skan_instance = FalknerSkanTable()
    return _falkner_skan_instance
