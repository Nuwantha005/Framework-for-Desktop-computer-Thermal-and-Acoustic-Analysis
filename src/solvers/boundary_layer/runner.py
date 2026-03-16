"""
Boundary layer runner — connect panel method results to the BL solver.

Provides :class:`BoundaryLayerRunner` which orchestrates:

1. Identify forward and rear stagnation points via ``n · V∞``.
2. Split the closed body surface into upper and lower streamlines.
3. Extract ``Ue(s) = |Vt|`` for each path from the panel method solution.
4. Run :class:`BoundaryLayerSolver` on each path for each velocity profile.
5. Optionally apply transition prediction.
6. Return a :class:`BoundaryLayerCaseResult` with per-side, per-profile results.

Stagnation point identification
-------------------------------
The forward stagnation panel is the one whose outward normal is most
opposed to the freestream (``argmin(n · V̂∞)``).  The rear stagnation
is most aligned (``argmax(n · V̂∞)``).  These two points partition the
closed body into two surface streamlines — "upper" and "lower" — each
marched independently from forward to rear stagnation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from solvers.boundary_layer.base import BoundaryLayerResult, BoundaryLayerSolver
from solvers.boundary_layer.field import BLFieldData, reconstruct_bl_field
from solvers.boundary_layer.profiles.base import VelocityProfile
from solvers.boundary_layer.profiles.blasius import BlasiusProfile
from solvers.boundary_layer.profiles.pohlhausen import PohlhausenProfile
from solvers.boundary_layer.profiles.falkner_skan import FalknerSkanProfile
from solvers.boundary_layer.profiles.power_law import PowerLawProfile
from solvers.boundary_layer.profiles.thwaites import ThwaitesProfile
from solvers.boundary_layer.transition import (
    TransitionResult,
    michel_criterion,
    en_criterion,
)


# -------------------------------------------------------------------------
# Profile factory
# -------------------------------------------------------------------------

_PROFILE_MAP = {
    "blasius": lambda **kw: BlasiusProfile(),
    "pohlhausen": lambda **kw: PohlhausenProfile(),
    "falkner_skan": lambda **kw: FalknerSkanProfile(),
    "power_law": lambda **kw: PowerLawProfile(n=kw.get("power_law_n", 7)),
    "thwaites": lambda **kw: ThwaitesProfile(
        reconstruction=kw.get("reconstruction", "falkner_skan"),
    ),
}


def create_profile(name: str, **kwargs) -> VelocityProfile:
    """
    Instantiate a velocity profile by short name.

    Args:
        name: One of ``"blasius"``, ``"pohlhausen"``, ``"falkner_skan"``,
            ``"power_law"``, ``"thwaites"``.
        **kwargs: Forwarded to the profile constructor (e.g. ``power_law_n``).

    Returns:
        VelocityProfile instance.

    Raises:
        ValueError: Unknown profile name.
    """
    factory = _PROFILE_MAP.get(name)
    if factory is None:
        raise ValueError(
            f"Unknown BL profile '{name}'. "
            f"Available: {list(_PROFILE_MAP.keys())}"
        )
    return factory(**kwargs)


# -------------------------------------------------------------------------
# Result containers
# -------------------------------------------------------------------------

@dataclass
class BoundaryLayerPathResult:
    """
    BL results for a single surface streamline (fwd stag → rear stag).

    One instance per body side (upper / lower).

    Attributes:
        side: ``"upper"`` or ``"lower"``.
        panel_indices: Indices into the full surface panel array, shape (K,).
        s: Arc-length from forward stagnation [m], shape (K,).
        x: Surface x-coordinates for this path, shape (K,).
        y: Surface y-coordinates for this path, shape (K,).
        Ue: Edge velocity ``|Vt|`` along this path [m/s], shape (K,).
        K: Velocity gradient dUe/ds at the stagnation point [1/s],
            computed via forced-through-origin linear regression.
            ``None`` if not yet computed.
        results: Profile name → :class:`BoundaryLayerResult`.
        transitions: Profile name → :class:`TransitionResult`.
        fields: Profile name → :class:`BLFieldData` (populated when
            ``reconstruct=True`` is passed to :meth:`BoundaryLayerRunner.run`).
    """
    side: str
    panel_indices: List[int]
    s: NDArray[np.float64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    Ue: NDArray[np.float64]
    K: Optional[float] = None
    results: Dict[str, BoundaryLayerResult] = field(default_factory=dict)
    transitions: Dict[str, TransitionResult] = field(default_factory=dict)
    fields: Dict[str, BLFieldData] = field(default_factory=dict)

    @property
    def profile_names(self) -> List[str]:
        """Ordered list of profile names solved on this path."""
        return list(self.results.keys())


@dataclass
class BoundaryLayerCaseResult:
    """
    Two-sided BL results for a closed 2-D body.

    Holds upper and lower :class:`BoundaryLayerPathResult` instances plus
    full-body geometry needed for envelope plots.

    Attributes:
        case_name: Case identifier.
        surface_x: Full body x-coordinates (M,).
        surface_y: Full body y-coordinates (M,).
        i_fwd_stag: Forward stagnation panel index (min ``n · V̂∞``).
        i_rear_stag: Rear stagnation panel index (max ``n · V̂∞``).
        upper: Upper-surface BL results.
        lower: Lower-surface BL results.
        nu: Kinematic viscosity used [m²/s].
    """
    case_name: str
    surface_x: NDArray[np.float64]
    surface_y: NDArray[np.float64]
    i_fwd_stag: int
    i_rear_stag: int
    upper: BoundaryLayerPathResult
    lower: BoundaryLayerPathResult
    nu: float = 0.0

    @property
    def profile_names(self) -> List[str]:
        """Profile names (shared between upper and lower)."""
        return self.upper.profile_names

    @property
    def sides(self) -> Dict[str, BoundaryLayerPathResult]:
        """Convenience mapping ``{"upper": ..., "lower": ...}``."""
        return {"upper": self.upper, "lower": self.lower}

    def full_body_quantity(
        self,
        quantity: str,
        profile_name: str,
    ) -> NDArray[np.float64]:
        """
        Map a per-path BL quantity back to the full-body panel array.

        Returns an (M,) array with *NaN* for panels not covered by the
        BL computation (stagnation skip, separation, rear wake, etc.).
        """
        M = len(self.surface_x)
        full = np.full(M, np.nan)
        for path in (self.upper, self.lower):
            if profile_name not in path.results:
                continue
            res = path.results[profile_name]
            vals = getattr(res, quantity)
            for j, idx in enumerate(path.panel_indices):
                if j < len(vals):
                    full[idx] = vals[j]
        return full


# -------------------------------------------------------------------------
# Stagnation detection & velocity gradient
# -------------------------------------------------------------------------


def _interpolate_stagnation(
    s: NDArray[np.float64],
    Vt: NDArray[np.float64],
) -> float:
    """Find the exact arc-length of the stagnation point via interpolation.

    Looks for the location where the signed tangential velocity Vt changes
    sign.  If a sign change is found between adjacent panels, linearly
    interpolates to find the exact s where Vt = 0.  If no sign change
    exists (e.g. the stagnation point coincides with the first panel), falls
    back to the location of minimum |Vt|.

    Args:
        s: Raw arc-length array (K,), monotonically increasing.
        Vt: Signed tangential velocity along the path (K,).

    Returns:
        Arc-length s_stag at the interpolated stagnation point.
    """
    # Look for sign changes in Vt
    signs = np.sign(Vt)
    sign_changes = np.where(np.diff(signs) != 0)[0]

    if len(sign_changes) > 0:
        # Use the first sign change (closest to the forward stagnation)
        i = sign_changes[0]
        Vt_a, Vt_b = Vt[i], Vt[i + 1]
        dVt = Vt_b - Vt_a
        if abs(dVt) > 1e-30:
            t = -Vt_a / dVt  # fraction between i and i+1
            t = np.clip(t, 0.0, 1.0)
            return float(s[i] + t * (s[i + 1] - s[i]))
        else:
            return float(0.5 * (s[i] + s[i + 1]))
    else:
        # No sign change — stagnation is at minimum |Vt|
        i_min = int(np.argmin(np.abs(Vt)))
        return float(s[i_min])


def _compute_K(
    s: NDArray[np.float64],
    Ue: NDArray[np.float64],
    threshold_fraction: float = 0.10,
    min_points: int = 3,
) -> float:
    """Compute velocity gradient K = dUe/ds at the stagnation point.

    Uses forced-through-origin linear regression on near-stagnation panels:

        K = Σ(Ue_i · s_i) / Σ(s_i²)

    where the sum is taken over panels with |Ue| < threshold_fraction × max|Ue|.

    Args:
        s: Arc-length re-zeroed at stagnation (K,).  May contain negative
           values for panels before the interpolated stagnation point.
        Ue: Edge velocity |Vt| (K,), non-negative.
        threshold_fraction: Fraction of peak Ue below which panels are
            considered "near stagnation" for the regression.
        min_points: Minimum number of points required for the fit.

    Returns:
        Positive velocity gradient K [1/s].  If the fit fails (e.g. too few
        points), falls back to the simple forward-difference Ue[1]/s[1] at
        the first post-stagnation panel.
    """
    Ue_max = float(np.max(Ue))
    if Ue_max < 1e-14:
        return 1.0  # degenerate: no flow

    threshold = threshold_fraction * Ue_max

    # Select near-stagnation panels with positive s (downstream of stagnation)
    mask = (Ue < threshold) & (s > 1e-14)
    s_fit = s[mask]
    Ue_fit = Ue[mask]

    # If not enough points from the threshold, take the first `min_points`
    # downstream panels instead
    if len(s_fit) < min_points:
        downstream = np.where(s > 1e-14)[0]
        if len(downstream) >= min_points:
            sel = downstream[:min_points]
        elif len(downstream) > 0:
            sel = downstream
        else:
            # All panels are at or before stagnation — very unusual
            return float(Ue_max / (s[-1] - s[0])) if s[-1] > s[0] else 1.0
        s_fit = s[sel]
        Ue_fit = Ue[sel]

    # Forced-through-origin regression: K = Σ(Ue·s) / Σ(s²)
    denom = float(np.sum(s_fit**2))
    if denom < 1e-30:
        # Degenerate: all points at the same s
        return 1.0
    K = float(np.sum(Ue_fit * s_fit) / denom)
    return max(K, 1e-10)  # K must be positive


# -------------------------------------------------------------------------
# Runner
# -------------------------------------------------------------------------

class BoundaryLayerRunner:
    """
    Run boundary layer analysis on a solved panel-method case.

    For each velocity profile the runner:

    1. Identifies forward / rear stagnation via ``n · V∞``.
    2. Splits the surface into upper and lower streamlines.
    3. Marches the BL ODE along each streamline.

    Typical usage::

        case = CaseLoader.load_case("cases/rounded_square", mesh_level_index=3)
        solver = case.create_solver(solver_type="linear_source")
        solver.solve()

        runner = BoundaryLayerRunner(case, solver)
        bl = runner.run()

        bl.upper.results["Thwaites"].cf   # upper-side skin friction
        bl.lower.results["Thwaites"].cf   # lower-side skin friction
    """

    def __init__(self, case, solver):
        """
        Args:
            case: Loaded :class:`Case` instance (config, mesh, fluid).
            solver: Solved panel-method solver with ``.Vt`` and ``.Cp``.
        """
        self.case = case
        self.solver = solver

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        profiles: Optional[List[str]] = None,
        nu: Optional[float] = None,
        transition_model: Optional[str] = None,
        n_crit: float = 9.0,
        power_law_n: int = 7,
        reconstruct: bool = False,
        reconstruction_n_y: int = 80,
        thwaites_reconstruction: Optional[str] = None,
    ) -> BoundaryLayerCaseResult:
        """
        Execute the BL solver on both sides of the body.

        Parameters fall back to ``case.yaml → boundary_layer:`` section
        when not provided explicitly.

        Args:
            profiles: Profile short names (overrides case config if given).
            nu: Kinematic viscosity [m²/s] (overrides fluid config).
            transition_model: ``"michel"`` or ``"en"`` (overrides case config).
            n_crit: e^N critical factor (default 9.0).
            power_law_n: Power-law exponent (default 7).
            reconstruct: If *True*, run velocity-field reconstruction after
                the ODE solve and store :class:`BLFieldData` in each
                path's ``fields`` dict.
            reconstruction_n_y: Number of wall-normal grid points for
                velocity reconstruction (default 80).
            thwaites_reconstruction: Reconstruction pairing for the
                Thwaites profile: ``"falkner_skan"`` or ``"pohlhausen"``.
                Falls back to ``case.yaml`` then ``"falkner_skan"``.

        Returns:
            :class:`BoundaryLayerCaseResult` with upper/lower paths.
        """
        # --- Resolve parameters from case config --------------------------
        bl_cfg = self.case.config.boundary_layer

        if profiles is None:
            profiles = list(bl_cfg.profiles)
        if transition_model is None:
            transition_model = bl_cfg.transition_model
        if power_law_n == 7 and bl_cfg.power_law_n != 7:
            power_law_n = bl_cfg.power_law_n
        if n_crit == 9.0 and bl_cfg.n_crit != 9.0:
            n_crit = bl_cfg.n_crit

        # Resolve Thwaites reconstruction pairing
        if thwaites_reconstruction is None:
            thwaites_reconstruction = getattr(
                bl_cfg, "thwaites_reconstruction", "falkner_skan",
            )
        # Resolve reconstruction_n_y from config if available
        cfg_n_y = getattr(bl_cfg, "reconstruction_n_y", None)
        if reconstruction_n_y == 80 and cfg_n_y is not None:
            reconstruction_n_y = cfg_n_y

        nu = self._resolve_nu(nu)

        # --- Extract surface data -----------------------------------------
        from postprocessing.surface import SurfaceDataExtractor

        extractor = SurfaceDataExtractor(self.case.mesh, self.solver)
        surface = extractor.extract(arc_length=True)

        x = surface.x       # panel-centre x (M,)
        y = surface.y       # panel-centre y (M,)
        Vt = surface.Vt     # tangential velocity (M,)

        # --- Stagnation point detection via n · V∞ ------------------------
        i_fwd, i_rear = self._find_stagnation_points()
        print(
            f"  Stagnation: fwd panel {i_fwd} "
            f"({x[i_fwd]:.4f}, {y[i_fwd]:.4f}), "
            f"rear panel {i_rear} "
            f"({x[i_rear]:.4f}, {y[i_rear]:.4f})"
        )

        # --- Split body into upper / lower paths -------------------------
        upper_idx, lower_idx = self._split_paths(i_fwd, i_rear, x, y)

        upper_path = self._build_path("upper", upper_idx, x, y, Vt)
        lower_path = self._build_path("lower", lower_idx, x, y, Vt)

        print(
            f"  Upper path: {len(upper_idx)} panels, "
            f"s_max = {upper_path.s[-1]:.4f} m, "
            f"K = {upper_path.K:.2f} 1/s"
        )
        print(
            f"  Lower path: {len(lower_idx)} panels, "
            f"s_max = {lower_path.s[-1]:.4f} m, "
            f"K = {lower_path.K:.2f} 1/s"
        )

        # --- Run BL for each profile on each side -------------------------
        for pname in profiles:
            profile = create_profile(
                pname,
                power_law_n=power_law_n,
                reconstruction=thwaites_reconstruction,
            )
            for path in (upper_path, lower_path):
                bl_solver = BoundaryLayerSolver(
                    edge_velocity=path.Ue,
                    arc_length=path.s,
                    nu=nu,
                    profile=profile,
                )
                result = bl_solver.solve(K=path.K)
                path.results[result.profile_name] = result

                # Transition prediction on this path
                if transition_model is not None:
                    tr = self._predict_transition(
                        transition_model, path.s, path.Ue,
                        result.theta, nu, n_crit,
                    )
                    path.transitions[result.profile_name] = tr

                # Velocity-field reconstruction (optional)
                if reconstruct:
                    try:
                        fld = reconstruct_bl_field(
                            result, profile, n_y=reconstruction_n_y,
                        )
                        path.fields[result.profile_name] = fld
                    except NotImplementedError:
                        pass  # profile lacks reconstruction (e.g. future stubs)

        return BoundaryLayerCaseResult(
            case_name=self.case.name,
            surface_x=x,
            surface_y=y,
            i_fwd_stag=i_fwd,
            i_rear_stag=i_rear,
            upper=upper_path,
            lower=lower_path,
            nu=nu,
        )

    # ------------------------------------------------------------------
    # Stagnation & path splitting
    # ------------------------------------------------------------------

    def _find_stagnation_points(self) -> Tuple[int, int]:
        """
        Identify forward and rear stagnation panels via ``n · V̂∞``.

        Forward stagnation
            Most negative ``n · V̂∞`` — outward normal opposes the
            freestream (points into the wind).

        Rear stagnation
            Most positive ``n · V̂∞`` — outward normal is aligned with
            the freestream (points downwind).
        """
        normals_2d = self.case.mesh.normals[:, :2]         # (M, 2)
        v_inf_2d = self.solver.v_inf_vector[:2]             # (2,)
        v_inf_unit = v_inf_2d / np.linalg.norm(v_inf_2d)

        n_dot_v = normals_2d @ v_inf_unit                   # (M,)
        return int(np.argmin(n_dot_v)), int(np.argmax(n_dot_v))

    def _split_paths(
        self,
        i_fwd: int,
        i_rear: int,
        x: NDArray,
        y: NDArray,
    ) -> Tuple[List[int], List[int]]:
        """
        Split the closed panel ring into two paths fwd → rear.

        One path follows ascending (CCW) panel indices, the other
        follows descending (CW) indices.  The path whose panels have
        higher perpendicular-to-freestream coordinate is labelled
        **upper**; the other is **lower**.

        Returns:
            ``(upper_indices, lower_indices)``
        """
        M = len(x)

        # Path A: ascending indices (CCW) from fwd → rear
        path_a: List[int] = []
        i = i_fwd
        while True:
            path_a.append(i)
            if i == i_rear:
                break
            i = (i + 1) % M

        # Path B: descending indices (CW) from fwd → rear
        path_b: List[int] = []
        i = i_fwd
        while True:
            path_b.append(i)
            if i == i_rear:
                break
            i = (i - 1) % M

        # Label upper/lower by perpendicular-to-freestream height
        v_inf_2d = self.solver.v_inf_vector[:2]
        perp = np.array([-v_inf_2d[1], v_inf_2d[0]])   # 90° CCW = "up"
        perp = perp / np.linalg.norm(perp)
        centroid = np.array([x.mean(), y.mean()])

        def _mean_height(indices: List[int]) -> float:
            pts = np.column_stack([x[indices], y[indices]])
            return float(np.mean((pts - centroid) @ perp))

        if _mean_height(path_a) >= _mean_height(path_b):
            return path_a, path_b
        else:
            return path_b, path_a

    # ------------------------------------------------------------------
    # Path construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_path(
        side: str,
        indices: List[int],
        x: NDArray,
        y: NDArray,
        Vt: NDArray,
    ) -> BoundaryLayerPathResult:
        """
        Build a :class:`BoundaryLayerPathResult` for one surface streamline.

        Procedure:
        1. Compute raw arc-length from panel-centre distances.
        2. Detect the exact stagnation point via sign-change interpolation
           in the signed tangential velocity Vt.  Re-zero arc-length so
           that s = 0 at the interpolated stagnation location.
        3. Edge velocity Ue = |Vt|, always non-negative.
        4. Compute the stagnation velocity gradient K = dUe/ds|_{s=0}
           via forced-through-origin linear regression on near-stagnation
           panels.
        """
        xi = x[indices]
        yi = y[indices]
        ds = np.sqrt(np.diff(xi) ** 2 + np.diff(yi) ** 2)
        s_raw = np.concatenate([[0.0], np.cumsum(ds)])
        Vt_path = Vt[indices]

        # -- Exact stagnation point via sign-change interpolation ----------
        s_stag = _interpolate_stagnation(s_raw, Vt_path)
        s_path = s_raw - s_stag  # re-zero so s = 0 at stagnation
        Ue_path = np.abs(Vt_path)

        # -- Velocity gradient K via forced-through-origin regression ------
        K = _compute_K(s_path, Ue_path)

        return BoundaryLayerPathResult(
            side=side,
            panel_indices=list(indices),
            s=s_path,
            x=xi.copy(),
            y=yi.copy(),
            Ue=Ue_path,
            K=K,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _resolve_nu(self, nu_override: Optional[float]) -> float:
        """Derive kinematic viscosity from case fluid config or override."""
        if nu_override is not None:
            return nu_override
        fluid = self.case.config.fluid
        if fluid.viscosity is not None:
            return fluid.viscosity / fluid.density
        return 1.5e-5  # fallback: air at STP

    @staticmethod
    def _predict_transition(
        model: str,
        s: NDArray,
        Ue: NDArray,
        theta: NDArray,
        nu: float,
        n_crit: float,
    ) -> TransitionResult:
        """Dispatch to the requested transition criterion."""
        if model == "michel":
            return michel_criterion(s, Ue, theta, nu)
        elif model == "en":
            return en_criterion(s, Ue, theta, nu, n_crit=n_crit)
        else:
            raise ValueError(f"Unknown transition model '{model}'")
