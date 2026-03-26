"""
Boundary layer solver using the Von Kármán momentum integral method.

Integrates the momentum integral ODE **forward** along a single surface
streamline using a pluggable velocity profile for closure.  The edge
velocity Ue(s) is supplied externally — typically from a panel method
surface solution, split into upper and lower paths by
:class:`BoundaryLayerRunner`.

Theory
------
The 2-D steady incompressible momentum integral equation is:

    dθ/ds + (θ / Ue) (dUe/ds) (2 + H) = cf / 2

where θ is the momentum thickness, H = δ*/θ the shape factor, and
cf/2 the wall-friction coefficient.  Closure comes from a
:class:`VelocityProfile` subclass that maps (Re_θ, λ) → (H, cf/2).

Integration starts just past the forward stagnation point (where
Ue ≈ 0) and marches forward along the streamline until reaching the
rear stagnation or encountering flow separation (H > H_sep).

References
----------
* Katz & Plotkin, *Low-Speed Aerodynamics* (2nd ed.), §14.
* Cebeci & Bradshaw, *Physical and Computational Aspects of Convective
  Heat Transfer*, Springer, 1984.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .profiles.base import VelocityProfile, ProfileClosureData


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BoundaryLayerResult:
    """
    Integral boundary-layer solution along a single surface streamline.

    All arrays share length *K* (number of arc-length stations on the path).
    Stations in the stagnation-skip region or past separation contain *NaN*.

    Attributes:
        s: Arc-length from forward stagnation [m], shape (K,).
        theta: Momentum thickness θ(s) [m], shape (K,).
        delta_star: Displacement thickness δ*(s) [m], shape (K,).
        cf: Freestream-normalized skin-friction coefficient
            C_f(s) = tau_w / (0.5 rho U_ref^2), shape (K,).
        H: Shape factor H(s) = δ*/θ, shape (K,).
        Re_theta: Momentum-thickness Reynolds number Re_θ(s), shape (K,).
        Ue: Edge velocity used for the computation [m/s], shape (K,).
        transition_s: Estimated arc-length location of laminar–turbulent
            transition, or ``None`` if not predicted.
        profile_name: Name of the velocity profile used.
        converged: Whether the integration completed without early failure.
    """
    s: NDArray[np.float64]
    theta: NDArray[np.float64]
    delta_star: NDArray[np.float64]
    cf: NDArray[np.float64]
    H: NDArray[np.float64]
    Re_theta: NDArray[np.float64]
    Ue: NDArray[np.float64]
    transition_s: Optional[float] = None
    profile_name: str = ""
    converged: bool = True


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

@dataclass
class BoundaryLayerSolver:
    """
    Von Kármán momentum integral BL solver — single forward march.

    Integrates the momentum integral ODE forward along a surface streamline
    supplied by the caller.  For a 2-D closed body the caller (typically
    :class:`BoundaryLayerRunner`) splits the surface into upper and lower
    paths and invokes this solver once per path.

    The starting station is automatically detected as the first station
    where ``|Ue| > 10 %`` of the peak edge velocity, skipping the
    stagnation region near *s = 0*.  Stations before this point receive
    *NaN*.  Integration terminates early if the shape factor *H* exceeds
    a separation threshold (*H > 4*).

    Args:
        edge_velocity: Ue(s) along the path [m/s], shape (K,).
        arc_length: Monotonically increasing arc-length [m], starting
            from 0 at the forward stagnation point, shape (K,).
        nu: Kinematic viscosity [m²/s].
        profile: Pluggable velocity profile providing closure relations.
        rtol: Relative tolerance for the ODE integrator.
        atol: Absolute tolerance for the ODE integrator.

    Example::

        from solvers.boundary_layer import BoundaryLayerSolver
        from solvers.boundary_layer.profiles import BlasiusProfile

        bl = BoundaryLayerSolver(
            edge_velocity=Ue_upper,
            arc_length=s_upper,
            nu=1.5e-5,
            profile=BlasiusProfile(),
        )
        result = bl.solve()
    """

    edge_velocity: NDArray[np.float64]
    arc_length: NDArray[np.float64]
    nu: float
    profile: VelocityProfile
    u_ref: float = 1.0
    rtol: float = 1e-6
    atol: float = 1e-9

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, K: float | None = None) -> BoundaryLayerResult:
        """
        Integrate the momentum-integral ODE forward along the path.

        Args:
            K: Velocity gradient dUe/ds at the stagnation point [1/s].
                When provided, enables analytical stagnation patching
                (``profile.stagnation_theta(nu, K)`` if available).
                Otherwise falls back to the legacy ``initial_theta``.

        Returns:
            :class:`BoundaryLayerResult` with θ, δ*, cf, H, Re_θ arrays.
            Stations in the stagnation-skip region and past separation
            contain *NaN*.
        """
        self._validate_inputs()

        s = self.arc_length
        Ue = self.edge_velocity
        nu = self.nu
        K_val = K  # store locally to avoid shadowing
        K = len(s)

        # Pre-compute dUe/ds via finite differences
        dUe_ds = np.gradient(Ue, s)

        # Find first downstream station with non-negligible Ue
        i0 = self._find_start_index(s, Ue)

        # Correct dUe/ds at the start station: np.gradient uses a central
        # difference which, when i0 > 0, averages across the stagnation
        # plateau (s < 0 panels).  Replace with a one-sided forward
        # difference so the BL sees the true velocity gradient at i0.
        if i0 > 0 and i0 + 1 < K:
            ds_fwd = s[i0 + 1] - s[i0]
            if ds_fwd > 1e-30:
                dUe_ds[i0] = (Ue[i0 + 1] - Ue[i0]) / ds_fwd

        # Compute initial theta: prefer stagnation patching (Phase 3)
        # if K and profile.stagnation_theta are available; else legacy.
        theta0 = self._compute_initial_theta(K_val, Ue[i0])

        # Initialise output — NaN everywhere, fill only where solved
        theta = np.full(K, np.nan)
        delta_star = np.full(K, np.nan)
        cf = np.full(K, np.nan)
        H_arr = np.full(K, np.nan)
        Re_theta_arr = np.full(K, np.nan)

        # Seed starting station
        cl0 = self._closure_at(theta0, Ue[i0], dUe_ds[i0])
        theta[i0] = theta0
        H_arr[i0] = cl0.H
        delta_star[i0] = cl0.H * theta0
        cf_local_0 = 2.0 * cl0.cf_2
        cf[i0] = cf_local_0 * (Ue[i0] / self.u_ref) ** 2
        Re_theta_arr[i0] = Ue[i0] * theta0 / nu

        # Forward march from i0 → K-1
        Ue_peak = float(np.max(np.abs(Ue)))
        converged = True
        if i0 < K - 1:
            theta_seg, ok = self._integrate_segment(
                s[i0:], Ue[i0:], dUe_ds[i0:], theta0, Ue_peak=Ue_peak,
            )
            if not ok:
                converged = False

            for j in range(len(theta_seg)):
                idx = i0 + j
                theta[idx] = theta_seg[j]
                if np.isnan(theta_seg[j]) or j == 0:
                    continue
                cl = self._closure_at(theta[idx], Ue[idx], dUe_ds[idx])
                H_arr[idx] = cl.H
                delta_star[idx] = cl.H * theta[idx]
                cf_local = 2.0 * cl.cf_2
                cf[idx] = cf_local * (Ue[idx] / self.u_ref) ** 2
                Re_theta_arr[idx] = Ue[idx] * theta[idx] / nu

        return BoundaryLayerResult(
            s=s.copy(),
            theta=theta,
            delta_star=delta_star,
            cf=cf,
            H=H_arr,
            Re_theta=Re_theta_arr,
            Ue=Ue.copy(),
            profile_name=self.profile.name,
            converged=converged,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_inputs(self) -> None:
        """Check array shapes and physical consistency."""
        if len(self.edge_velocity) != len(self.arc_length):
            raise ValueError(
                f"edge_velocity length ({len(self.edge_velocity)}) != "
                f"arc_length length ({len(self.arc_length)})"
            )
        if len(self.arc_length) < 3:
            raise ValueError("Need at least 3 arc-length stations")
        if self.nu <= 0:
            raise ValueError(f"Kinematic viscosity must be positive, got {self.nu}")
        if self.u_ref <= 0:
            raise ValueError(f"Reference velocity u_ref must be positive, got {self.u_ref}")
        if np.all(np.abs(self.edge_velocity) < 1e-14):
            raise ValueError("Edge velocity is zero everywhere — no BL to compute")

    @staticmethod
    def _find_start_index(
        s: NDArray[np.float64],
        Ue: NDArray[np.float64],
    ) -> int:
        """
        First downstream station suitable for starting the BL march.

        With exact stagnation re-zeroing (Phase 2), the arc-length array
        has s=0 at the interpolated stagnation point and the first panel
        beyond it has s > 0 with a small but finite Ue.  We start at the
        first such panel.

        Falls back to the legacy 10% threshold if s[0] < 0 (pre-Phase-2
        arc-length arrays that haven't been re-zeroed).

        Args:
            s: Arc-length array re-zeroed at stagnation (K,).
            Ue: Edge velocity array (K,).

        Returns:
            Index of the starting station.
        """
        # Phase 2 logic: start at first station with s > 0 and Ue > 0
        downstream = np.where((s > 1e-14) & (Ue > 1e-14))[0]
        if len(downstream) > 0:
            return int(downstream[0])

        # Legacy fallback: first station with |Ue| > 10% of peak
        Ue_abs = np.abs(Ue)
        Ue_max = Ue_abs.max()
        if Ue_max < 1e-14:
            raise ValueError("Peak |Ue| is essentially zero")
        threshold = 0.10 * Ue_max
        candidates = np.where(Ue_abs >= threshold)[0]
        if len(candidates) == 0:
            raise ValueError("No station with |Ue| >= 10 % of peak")
        return int(candidates[0])

    def _compute_initial_theta(
        self,
        K: float | None,
        Ue0: float,
    ) -> float:
        """Compute the initial momentum thickness at the starting station.

        Tries the analytical stagnation-patching method first (Phase 3):
        ``profile.stagnation_theta(nu, K)``.  If K is ``None`` or the
        profile does not implement ``stagnation_theta``, falls back to the
        legacy ``profile.initial_theta(nu, Ue0)``.

        Args:
            K: Velocity gradient at stagnation [1/s], or None.
            Ue0: Edge velocity at the starting station [m/s].

        Returns:
            Initial θ₀ [m].
        """
        if K is not None and hasattr(self.profile, "stagnation_theta"):
            try:
                return self.profile.stagnation_theta(self.nu, K)
            except NotImplementedError:
                pass  # Profile doesn't support stagnation patching (e.g. PowerLaw)
        return self.profile.initial_theta(self.nu, Ue0)

    def _closure_at(
        self,
        theta_val: float,
        Ue_val: float,
        dUe_ds_val: float,
    ) -> ProfileClosureData:
        """
        Evaluate profile closure at a single station.

        Computes Re_θ and the pressure-gradient parameter λ = θ²/ν · dUe/ds
        (Thwaites convention), then delegates to the profile.
        """
        Re_theta = abs(Ue_val) * max(theta_val, 1e-15) / self.nu
        lambda_param = max(theta_val, 1e-15) ** 2 / self.nu * dUe_ds_val
        return self.profile.compute_closure(Re_theta, lambda_param)

    def _momentum_ode(
        self,
        s_val: float,
        theta_vec: NDArray,
        Ue_interp,
        dUe_interp,
    ) -> NDArray:
        """
        RHS of the momentum-integral ODE: dθ/ds.

        dθ/ds = cf/2 − (θ/Ue)(dUe/ds)(2 + H)
        """
        theta_val = theta_vec[0]
        Ue_val = float(Ue_interp(s_val))
        dUe_val = float(dUe_interp(s_val))

        if abs(Ue_val) < 1e-14:
            return np.array([0.0])

        # Clamp θ to prevent runaway
        if theta_val <= 0.0:
            theta_val = 1e-15

        closure = self._closure_at(theta_val, Ue_val, dUe_val)
        dtheta_ds = closure.cf_2 - (theta_val / Ue_val) * dUe_val * (2.0 + closure.H)
        return np.array([dtheta_ds])

    def _integrate_segment(
        self,
        s_seg: NDArray[np.float64],
        Ue_seg: NDArray[np.float64],
        dUe_seg: NDArray[np.float64],
        theta0: float,
        Ue_peak: Optional[float] = None,
    ) -> tuple[NDArray[np.float64], bool]:
        """
        Integrate θ forward along a monotonic arc-length segment.

        Uses ``scipy.integrate.solve_ivp`` (RK45) with three terminal
        events that stop integration when:

        * **H > 3.5** — laminar separation (shape factor criterion).
        * **Ue < 5 % Ue_peak** — approaching rear stagnation singularity.
        * **θ > 5 % × path length** — runaway growth safety valve.

        Returns:
            (theta_array, success_flag).  Stations past termination are NaN.
        """
        from scipy.interpolate import interp1d

        s_abs = s_seg - s_seg[0]  # shift to start from 0
        Ue_func = interp1d(s_abs, Ue_seg, kind="linear", fill_value="extrapolate")
        dUe_func = interp1d(s_abs, dUe_seg, kind="linear", fill_value="extrapolate")

        def rhs(s, y):
            return self._momentum_ode(s, y, Ue_func, dUe_func)

        # --- Event 1: H-based separation ----------------------------------
        H_SEP = 3.5

        def _sep_H(s, y):
            """Returns negative when H > H_SEP (laminar separation)."""
            theta_val = max(y[0], 1e-15)
            Ue_val = float(Ue_func(s))
            dUe_val = float(dUe_func(s))
            if abs(Ue_val) < 1e-14:
                return -1.0
            Re_theta = abs(Ue_val) * theta_val / self.nu
            lam = theta_val ** 2 / self.nu * dUe_val
            cl = self.profile.compute_closure(Re_theta, lam)
            return H_SEP - cl.H

        _sep_H.terminal = True
        _sep_H.direction = -1

        events = [_sep_H]

        # --- Event 2: Ue floor (rear stagnation singularity guard) --------
        if Ue_peak is not None and Ue_peak > 1e-14:
            Ue_floor = 0.05 * Ue_peak

            def _ue_floor(s, y):
                """Returns negative when Ue drops below 5 % of peak."""
                return float(Ue_func(s)) - Ue_floor

            _ue_floor.terminal = True
            _ue_floor.direction = -1
            events.append(_ue_floor)

        # --- Event 3: θ ceiling (runaway growth safety valve) -------------
        path_length = max(s_abs[-1], 1e-10)
        theta_max = 0.05 * path_length

        def _theta_ceil(s, y):
            """Returns negative when θ exceeds 5 % of path length."""
            return theta_max - y[0]

        _theta_ceil.terminal = True
        _theta_ceil.direction = -1
        events.append(_theta_ceil)

        s_span = (s_abs[0], s_abs[-1])
        sol = solve_ivp(
            rhs,
            s_span,
            [theta0],
            method="RK45",
            t_eval=s_abs,
            rtol=self.rtol,
            atol=self.atol,
            max_step=np.diff(s_abs).max() if len(s_abs) > 1 else 1.0,
            events=events,
        )

        if sol.success and sol.status != 1:
            # Completed without event termination
            return sol.y[0], True
        else:
            # Separation detected or solver failed — fill remaining with NaN
            n_got = len(sol.t)
            theta_out = np.full(len(s_seg), np.nan)
            theta_out[:n_got] = sol.y[0]
            return theta_out, sol.status == 1  # status=1 = event-terminated
