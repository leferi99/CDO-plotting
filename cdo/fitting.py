r"""Fit the analytic avalanche distribution to a DREAM f(p, xi).

Two forms of eq. (2.17) of Embréus et al., J. Plasma Phys. 84 (2018)
905840506 (arXiv:1807.04460, eq. 2.12), both with the beam at xi = s
(``direction``, +1 by default as in :mod:`cdo.avalanche`):

.. math::

    f(p, \xi) = \frac{C}{2\pi\gamma_0 p^2} e^{-\gamma/\gamma_0}
        \frac{A\,e^{-A(1 - s\xi)}}{1 - e^{-2A}}

- ``"plain"``: :math:`A = \hat A\gamma`, as in the paper, where
  :math:`\hat A = (E/E_c + 1)/(Z_{\rm tot} + 1)`. Free: C, gamma0, A_hat.
- ``"extended"``: :math:`A = \gamma_{\rm ref}\hat A(\gamma/\gamma_{\rm ref})^\beta`.
  A_hat is A/gamma at gamma = gamma_ref (20 by default), beta = 1 is the
  plain form. Free: C, gamma0, A_hat, beta.

C is about the density the f integrates to, sum 2 pi p^2 f dxi dp, when the
grid covers the bulk of the beam, so about 1 for an f per electron. Both
forms are averaged over each pitch cell in closed form
(:func:`cdo.avalanche.ln_pitch_cell_mean`), since DREAM stores cell averages
and the beam can be narrower than a cell. The momentum dependence is taken
at the cell centres.

The fit is a least-squares fit of ln f over the fit cells (:func:`fit_cells`),
unweighted by default, from several starting points. :func:`fit_joint` fits
several distributions with a shared shape and one C each.
:func:`distribution_from_run` gives the radially volume-averaged f per
electron of a run at one time, the usual input.

A beam narrower than the pitch cells next to it fixes only a lower limit of
A. :class:`FitResult` therefore reports, per distribution, the number of
pitch rows with fit cells and the smallest A dxi of the fit cell nearest the
beam over the fitted momenta, and counts the beam as resolved when there are
at least two rows and that A dxi is at most :data:`MAX_BEAM_A_DXI`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import avalanche, derived

#: Default gamma at which A_hat of the extended form is A/gamma.
GAMMA_REF = 20.0
#: Default lower momentum limit of the fit cells, in m_e c.
P_MIN = 5.0
#: Default lower limit of f in the fit cells, relative to the maximum of f.
F_THRESH = 1e-6
#: Default bounds of the shape parameters. C is unbounded.
BOUNDS = {"gamma0": (1.0, 500.0), "A_hat": (0.01, 1000.0), "beta": (-2.0, 4.0)}
#: Shape parameters of each form, shared between the distributions of a joint fit.
FORMS = {"plain": ("gamma0", "A_hat"), "extended": ("gamma0", "A_hat", "beta")}

_DEFAULT = avalanche.AvalancheParams()
#: Starting points (gamma0, A_hat) of the plain form, around the defaults of
#: :class:`~cdo.avalanche.AvalancheParams`.
PLAIN_STARTS = tuple(
    (g0, ah)
    for g0 in (0.4 * _DEFAULT.gamma0, _DEFAULT.gamma0)
    for ah in (2.0, (_DEFAULT.E_over_Ec + 1.0) / (_DEFAULT.Z_tot + 1.0), 100.0)
)
#: Starting values of beta of the extended form, from the best plain fit.
BETA_STARTS = (1.0, 0.5, 1.5)
#: Distance in fit space (ln gamma0, ln A_hat, beta) below which a parameter
#: counts as at a bound.
AT_BOUND_TOL = 1e-3
#: Largest A dxi of the fit cell nearest the beam, at the best-resolved
#: fitted momentum, for which the beam counts as resolved. Above it the cell
#: mean hardly depends on A, and the fit gives only a lower limit of A_hat.
MAX_BEAM_A_DXI = 5.0


def _check_direction(direction):
    if direction not in (1, -1):
        raise ValueError(f"direction must be +1 or -1, got {direction!r}")


@dataclass
class Distribution:
    """f(p, xi) on one p/xi grid, shape ``(xi cells, p)``.

    ``p`` are the momentum cell centres in m_e c, ``xi_edges`` the pitch cell
    edges. ``p_edges`` gives the momentum widths, needed for the content
    weighting only. ``xi_pass`` is the default lower pitch limit of the fit
    cells. From :func:`distribution_from_run` it is the largest
    trapped-passing boundary over radius, and ``density`` (m^-3) is the
    volume-averaged density the f was divided by.
    """

    f: np.ndarray
    p: np.ndarray
    xi_edges: np.ndarray
    p_edges: np.ndarray | None = None
    xi_pass: float = 0.0
    time_index: int | None = None
    time_ms: float | None = None
    density: float | None = None
    grid: str | None = None

    def __post_init__(self):
        self.f = np.asarray(self.f, dtype=float)
        self.p = np.asarray(self.p, dtype=float)
        self.xi_edges = np.asarray(self.xi_edges, dtype=float)
        if self.p_edges is not None:
            self.p_edges = np.asarray(self.p_edges, dtype=float)
            if self.p_edges.shape != (self.p.size + 1,):
                raise ValueError(f"p_edges needs {self.p.size + 1} entries, "
                                 f"got {self.p_edges.shape}")
        shape = (self.xi_edges.size - 1, self.p.size)
        if self.f.shape != shape:
            raise ValueError(f"f must have shape (xi cells, p) = {shape}, got {self.f.shape}")

    @property
    def xi(self) -> np.ndarray:
        """Pitch cell centres."""
        return 0.5 * (self.xi_edges[:-1] + self.xi_edges[1:])

    @property
    def dxi(self) -> np.ndarray:
        """Pitch cell widths."""
        return np.abs(np.diff(self.xi_edges))

    @property
    def dp(self) -> np.ndarray:
        """Momentum cell widths, from ``p_edges``."""
        if self.p_edges is None:
            raise ValueError("p_edges is not set")
        return np.diff(self.p_edges)

    def content(self) -> np.ndarray:
        """Particle content of each cell, 2 pi p^2 f dxi dp, shape ``(xi, p)``."""
        return 2.0 * np.pi * self.p[None, :] ** 2 * self.f * self.dxi[:, None] * self.dp[None, :]


@dataclass
class FitResult:
    """Parameters and quality of a fit.

    ``C`` has one amplitude per distribution, one for :func:`fit_plain` and
    :func:`fit_extended`. ``beta`` is 1 for the plain form.

    ``sigma`` holds the 1 sigma uncertainties from the Jacobian,
    cov = (J^T J)^-1 SSR / (N - k), under the parameter names, ``"C"`` as a
    tuple. They are statistical only. A fixed beta has sigma 0, a parameter at
    a bound NaN. With ``weighted=True`` they come from the weighted residuals
    and measure the curvature of that cost, not a proper uncertainty.

    Per distribution: ``rms_log10`` is the RMS of log10(f / fit) over the fit
    cells and ``n_cells`` their number, ``n_pitch_rows`` the number of pitch
    cells holding fit cells, ``beam_A_dxi`` the smallest A dxi of the fit
    cell nearest the beam over the fitted momenta, and ``resolved`` whether
    the beam is resolved (see :data:`MAX_BEAM_A_DXI`). ``at_bound`` names the
    shape parameters at a bound. ``converged`` is the solver's success flag,
    ``success`` also needs every beam resolved and finite parameters.
    """

    form: str
    C: tuple
    gamma0: float
    A_hat: float
    beta: float
    sigma: dict
    rms_log10: tuple
    n_cells: tuple
    n_pitch_rows: tuple
    beam_A_dxi: tuple
    resolved: tuple
    converged: bool
    success: bool
    at_bound: tuple
    gamma_ref: float = GAMMA_REF
    direction: int = 1
    weighted: bool = False
    cost: float = math.nan
    masks: list = field(default_factory=list, repr=False)

    def A(self, p) -> np.ndarray:
        """Pitch parameter A at momenta ``p`` in m_e c."""
        return pitch_parameter(p, self.A_hat, self.beta, gamma_ref=self.gamma_ref)

    def A_over_gamma(self, gamma) -> np.ndarray:
        """A/gamma at Lorentz factors ``gamma``, A_hat at gamma = gamma_ref."""
        gamma = np.asarray(gamma, dtype=float)
        return self.A_hat * (gamma / self.gamma_ref) ** (self.beta - 1.0)

    def model(self, p, xi_edges, dataset: int = 0) -> np.ndarray:
        """The fitted f, averaged over each pitch cell, shape ``(xi cells, p)``."""
        return model(p, xi_edges, self.C[dataset], self.gamma0, self.A_hat, self.beta,
                     gamma_ref=self.gamma_ref, direction=self.direction)


# --- models -----------------------------------------------------------------


def pitch_parameter(p, A_hat, beta=1.0, *, gamma_ref=GAMMA_REF) -> np.ndarray:
    """A = gamma_ref A_hat (gamma/gamma_ref)^beta, which is A_hat gamma for beta = 1.

    ``A_hat``, ``beta`` and ``gamma_ref`` are scalars.
    """
    gamma = np.sqrt(1.0 + np.asarray(p, dtype=float) ** 2)
    if float(beta) == 1.0:
        return A_hat * gamma
    return gamma_ref * A_hat * (gamma / gamma_ref) ** beta


def ln_model(p, xi_edges, C, gamma0, A_hat, beta=1.0, *, gamma_ref=GAMMA_REF,
             direction: int = 1) -> np.ndarray:
    """ln f of the fitted forms, averaged over each pitch cell, ``(xi cells, p)``.

    ``beta = 1`` is the plain form. ``C`` must be positive.
    """
    _check_direction(direction)
    A = pitch_parameter(p, A_hat, beta, gamma_ref=gamma_ref)
    return avalanche.ln_distribution(p, xi_edges, C, gamma0, A, direction)


def model(p, xi_edges, C, gamma0, A_hat, beta=1.0, *, gamma_ref=GAMMA_REF,
          direction: int = 1) -> np.ndarray:
    """f of the fitted forms, the exponential of :func:`ln_model`."""
    return np.exp(ln_model(p, xi_edges, C, gamma0, A_hat, beta,
                           gamma_ref=gamma_ref, direction=direction))


# --- fit cells --------------------------------------------------------------


def fit_cells(f, p, xi_edges, *, xi_pass=0.0, p_min=P_MIN, f_thresh=F_THRESH,
              direction: int = 1) -> np.ndarray:
    """Cells used in a fit, a boolean mask of shape ``(xi cells, p)``.

    A cell is used when its momentum centre is above ``p_min``, the whole
    pitch cell lies at s xi >= ``xi_pass`` (s = ``direction``), and f is
    positive, finite and above ``f_thresh`` times the largest f.
    """
    _check_direction(direction)
    f = np.asarray(f, dtype=float)
    p = np.asarray(p, dtype=float)
    s_edges = direction * np.asarray(xi_edges, dtype=float)
    beam_side = np.minimum(s_edges[:-1], s_edges[1:]) >= xi_pass
    finite = np.isfinite(f)
    if not finite.any():
        return np.zeros(f.shape, dtype=bool)
    fmax = np.max(f[finite])
    with np.errstate(invalid="ignore"):
        good = finite & (f > 0) & (f > f_thresh * fmax)
    return good & (p[None, :] > p_min) & beam_side[:, None]


def beam_resolution(mask, p, xi_edges, A_hat, beta=1.0, *, gamma_ref=GAMMA_REF,
                    direction: int = 1) -> tuple:
    """Number of pitch rows with fit cells, and the smallest beam-cell A dxi.

    For each momentum with fit cells, the fit cell nearest the beam (largest
    s xi) gives A(p) dxi; the smallest over the momenta is returned. NaN when
    there are no fit cells.
    """
    _check_direction(direction)
    mask = np.asarray(mask, dtype=bool)
    edges = np.asarray(xi_edges, dtype=float)
    rows = int(np.count_nonzero(mask.any(axis=1)))
    columns = np.flatnonzero(mask.any(axis=0))
    if columns.size == 0:
        return rows, math.nan
    s_top = np.maximum(direction * edges[:-1], direction * edges[1:])
    dxi = np.abs(np.diff(edges))
    A = pitch_parameter(np.asarray(p, dtype=float)[columns], A_hat, beta, gamma_ref=gamma_ref)
    nearest = [np.flatnonzero(mask[:, j])[np.argmax(s_top[mask[:, j]])] for j in columns]
    return rows, float(np.min(A * dxi[nearest]))


# --- fitting ----------------------------------------------------------------


def fit_plain(data: Distribution, **kwargs) -> FitResult:
    """Fit the plain form (A = A_hat gamma) to one distribution.

    Keywords as for :func:`fit_joint`.
    """
    return fit_joint([data], form="plain", **kwargs)


def fit_extended(data: Distribution, **kwargs) -> FitResult:
    """Fit the extended form (A = gamma_ref A_hat (gamma/gamma_ref)^beta) to one distribution.

    Keywords as for :func:`fit_joint`.
    """
    return fit_joint([data], form="extended", **kwargs)


def fit_joint(datasets, *, form: str = "extended", xi_pass=None, p_min=P_MIN,
              f_thresh=F_THRESH, masks=None, weighted: bool = False,
              gamma_ref=GAMMA_REF, direction: int = 1, starts=None,
              bounds=None) -> FitResult:
    """Fit one form to several distributions with a shared shape.

    gamma0, A_hat and, for ``form="extended"``, beta are shared; each
    distribution has its own amplitude C. The residuals are ln f(model) -
    ln f(data) over the fit cells of every distribution.

    ``xi_pass``, ``p_min``, ``f_thresh`` and ``direction`` select the fit
    cells, see :func:`fit_cells`; ``xi_pass=None`` takes each distribution's
    own ``xi_pass``. ``masks`` gives the cells directly, one mask per
    distribution, where f must be finite and positive. ``weighted=True``
    multiplies each residual by the square root of the cell content
    2 pi p^2 f dxi dp, normalised to a mean of one within each distribution
    (this needs ``p_edges``); ``rms_log10`` is always unweighted. ``starts``
    lists starting shapes, (gamma0, A_hat) or (gamma0, A_hat, beta); by
    default the plain form starts from :data:`PLAIN_STARTS`, the extended form
    from the best plain fit with each beta of :data:`BETA_STARTS`. ``bounds``
    replaces entries of :data:`BOUNDS`.

    Each distribution needs more fit cells than a fit to it alone has
    parameters (C and the shape), so at least one degree of freedom of its
    own; ``ValueError`` otherwise.
    """
    if form not in FORMS:
        raise ValueError(f"form must be one of {list(FORMS)}, got {form!r}")
    _check_direction(direction)
    unknown = set(bounds or {}) - set(BOUNDS)
    if unknown:
        raise ValueError(f"unknown bounds {sorted(unknown)}, known: {list(BOUNDS)}")
    datasets = list(datasets)
    if not datasets:
        raise ValueError("no distributions to fit")
    if masks is None:
        masks = [fit_cells(d.f, d.p, d.xi_edges,
                           xi_pass=d.xi_pass if xi_pass is None else xi_pass,
                           p_min=p_min, f_thresh=f_thresh, direction=direction)
                 for d in datasets]
    else:
        masks = [np.asarray(m, dtype=bool) for m in masks]
        if len(masks) != len(datasets):
            raise ValueError(f"{len(masks)} masks for {len(datasets)} distributions")
        for k, (d, m) in enumerate(zip(datasets, masks)):
            if m.shape != d.f.shape:
                raise ValueError(f"mask {k} has shape {m.shape}, f has {d.f.shape}")
            values = d.f[m]
            if not np.all(np.isfinite(values) & (values > 0)):
                raise ValueError(f"mask {k} selects cells where f is not finite and positive")
    names = FORMS[form]
    for k, m in enumerate(masks):
        if m.sum() <= len(names) + 1:
            raise ValueError(f"distribution {k} has {int(m.sum())} fit cells, too few for "
                             f"the {form} form (at least {len(names) + 2})")
    limits = {**BOUNDS, **(bounds or {})}
    lo = [math.log(limits["gamma0"][0]), math.log(limits["A_hat"][0])]
    hi = [math.log(limits["gamma0"][1]), math.log(limits["A_hat"][1])]
    if form == "extended":
        lo.append(limits["beta"][0])
        hi.append(limits["beta"][1])
    problem = _Problem(datasets, masks, form, weighted, gamma_ref, direction,
                       np.array(lo), np.array(hi))

    if starts is None:
        if form == "plain":
            starts = PLAIN_STARTS
        else:
            plain = fit_joint(datasets, form="plain", masks=masks, weighted=weighted,
                              gamma_ref=gamma_ref, direction=direction, bounds=bounds)
            starts = [(plain.gamma0, plain.A_hat, b) for b in BETA_STARTS]
    best = None
    for start in starts:
        sol = problem.solve(start)
        if best is None or sol.cost < best.cost:
            best = sol
    return problem.result(best)


class _Problem:
    """Residuals and bookkeeping of one fit. Fit space: ln gamma0, ln A_hat,
    beta (extended form), then ln C per distribution."""

    def __init__(self, datasets, masks, form, weighted, gamma_ref, direction, lo, hi):
        self.datasets, self.masks, self.form = datasets, masks, form
        self.weighted, self.gamma_ref, self.direction = weighted, gamma_ref, direction
        self.n_shape = len(FORMS[form])
        self.lo, self.hi = lo, hi
        self.y = [np.log(d.f[m]) for d, m in zip(datasets, masks)]
        self.sw = [1.0] * len(datasets)
        if weighted:
            for k, (d, m) in enumerate(zip(datasets, masks)):
                if d.p_edges is None:
                    raise ValueError(f"distribution {k} has no p_edges, "
                                     "which the content weighting needs")
                w = d.content()[m]
                self.sw[k] = np.sqrt(w / w.sum() * w.size)

    def shape(self, x):
        beta = x[2] if self.form == "extended" else 1.0
        return math.exp(x[0]), math.exp(x[1]), float(beta)

    def raw(self, x, k):
        """Unweighted residuals ln model - ln f of distribution k."""
        d = self.datasets[k]
        gamma0, A_hat, beta = self.shape(x)
        ln_f = ln_model(d.p, d.xi_edges, 1.0, gamma0, A_hat, beta,
                        gamma_ref=self.gamma_ref, direction=self.direction)
        return ln_f[self.masks[k]] + x[self.n_shape + k] - self.y[k]

    def residuals(self, x):
        return np.concatenate([self.raw(x, k) * self.sw[k] for k in range(len(self.datasets))])

    def solve(self, start):
        from scipy.optimize import least_squares  # keeps `import cdo` light

        start = tuple(start)
        if len(start) not in (2, 3):
            raise ValueError(f"a start is (gamma0, A_hat) or (gamma0, A_hat, beta), got {start}")
        shape = [math.log(start[0]), math.log(start[1])]
        if self.form == "extended":
            shape.append(start[2] if len(start) == 3 else 1.0)
        shape = np.clip(shape, self.lo + 1e-6, self.hi - 1e-6)
        # Each ln C starts where the start shape matches the data in the median.
        flat = np.concatenate([shape, np.zeros(len(self.y))])
        amplitudes = [-float(np.median(self.raw(flat, k))) for k in range(len(self.y))]
        x0 = np.concatenate([shape, amplitudes])
        n_c = len(amplitudes)
        lo = np.concatenate([self.lo, np.full(n_c, -np.inf)])
        hi = np.concatenate([self.hi, np.full(n_c, np.inf)])
        return least_squares(self.residuals, x0, bounds=(lo, hi), method="trf",
                             x_scale="jac", max_nfev=4000)

    def result(self, sol) -> FitResult:
        x = sol.x
        n, k = sol.fun.size, x.size
        J = sol.jac
        with np.errstate(invalid="ignore", divide="ignore"):
            cov = np.linalg.pinv(J.T @ J) * float(sol.fun @ sol.fun) / max(n - k, 1)
        s = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        names = FORMS[self.form]
        bound = [min(x[i] - self.lo[i], self.hi[i] - x[i]) < AT_BOUND_TOL
                 for i in range(self.n_shape)]
        s[:self.n_shape][np.array(bound, dtype=bool)] = np.nan
        gamma0, A_hat, beta = self.shape(x)
        C = tuple(math.exp(float(v)) for v in x[self.n_shape:])
        sigma = {"gamma0": float(gamma0 * s[0]), "A_hat": float(A_hat * s[1]),
                 "beta": float(s[2]) if self.form == "extended" else 0.0,
                 "C": tuple(float(c * sk) for c, sk in zip(C, s[self.n_shape:]))}
        at_bound = tuple(name for name, b in zip(names, bound) if b)
        rms = tuple(float(np.sqrt(np.mean(self.raw(x, j) ** 2)) / math.log(10.0))
                    for j in range(len(self.datasets)))
        rows, beam = zip(*(beam_resolution(m, d.p, d.xi_edges, A_hat, beta,
                                           gamma_ref=self.gamma_ref, direction=self.direction)
                           for d, m in zip(self.datasets, self.masks)))
        resolved = tuple(r >= 2 and b <= MAX_BEAM_A_DXI for r, b in zip(rows, beam))
        converged = bool(sol.success)
        return FitResult(
            form=self.form, C=C, gamma0=gamma0, A_hat=A_hat, beta=beta, sigma=sigma,
            rms_log10=rms, n_cells=tuple(int(m.sum()) for m in self.masks),
            n_pitch_rows=tuple(rows), beam_A_dxi=tuple(beam), resolved=resolved,
            converged=converged,
            success=converged and all(resolved) and bool(np.all(np.isfinite(x))),
            at_bound=at_bound, gamma_ref=float(self.gamma_ref), direction=self.direction,
            weighted=self.weighted, cost=float(sol.cost), masks=self.masks,
        )


# --- data from a run --------------------------------------------------------


def distribution_from_run(run, *, time_ms=None, time_index=None,
                          grid: str = "runaway") -> Distribution:
    """The radially volume-averaged f(p, xi) of a run per electron, at one time.

    Give either ``time_ms``, a time in ms or ``"end"``, snapped to the nearest
    stored step, or ``time_index``, a step of ``run.timegrid``, negative from
    the end. ``grid`` is ``"runaway"`` or ``"hottail"``. Each radial cell
    counts with its volume, f_V = sum_r f(r) dV_r / V, and f_V is divided by
    its density sum 2 pi p^2 f_V dxi dp over the whole grid, so the result
    sums to one over the grid. On the runaway grid this is the f behind the
    ``favg_re_ravg`` figures: its pitch mean is their per-electron curve.
    ``xi_pass`` is the largest trapped-passing boundary over radius. Only the
    one time step of f is read.
    """
    from .selection import snap_times  # selection imports plotting, which imports avalanche

    if (time_ms is None) == (time_index is None):
        raise ValueError("give one of time_ms and time_index")
    n = run.timegrid_length
    if time_index is None:
        time_index = snap_times(run.timegrid_ms, [time_ms], lines=False)[0].index
    elif not -n <= int(time_index) < n:
        raise IndexError(f"time index {time_index} outside {-n}..{n - 1}")
    time_index = int(time_index) % n
    if grid not in ("runaway", "hottail"):
        raise ValueError(f"grid must be 'runaway' or 'hottail', got {grid!r}")
    mm = run.moments(grid)
    if mm is None:
        raise ValueError(f"run has no {grid} grid")
    f = run.field_at("f_re" if grid == "runaway" else "f_hot", time_index)
    if f is None:
        raise ValueError(f"run has no {grid} distribution at time step {time_index}")
    prefix = "re" if grid == "runaway" else "hot"
    p = np.asarray(run.reference.read(f"{prefix}_p"), dtype=float)
    p_edges = np.asarray(run.reference.read(f"{prefix}_p_edges"), dtype=float)
    volumes = np.asarray(run.cell_volumes, dtype=float)
    f_vol = derived.volume_average(np.moveaxis(f, 0, -1), volumes)  # (xi, p)
    weights = 2.0 * np.pi * p[None, :] ** 2 * mm.dxi[:, None] * np.diff(p_edges)[None, :]
    density = float(np.nansum(f_vol * weights))
    if not density > 0:
        raise ValueError(f"no electrons on the {grid} grid at time step {time_index}")
    return Distribution(
        f=f_vol / density, p=p, xi_edges=np.asarray(mm.xi_edges, dtype=float),
        p_edges=p_edges, xi_pass=float(np.max(mm.xi0_trapped)), time_index=time_index,
        time_ms=float(run.timegrid_ms[time_index]), density=density, grid=grid,
    )
