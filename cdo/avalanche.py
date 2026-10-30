r"""Analytic avalanche runaway distribution, for overlays on the DREAM data.

The distribution is eq. (2.17) of Embréus et al., J. Plasma Phys. 84 (2018)
905840506 (arXiv:1807.04460), after Fülöp et al., Phys. Plasmas 13 (2006):

.. math::

    f_{\rm RE}(p, \xi) = \frac{n_{\rm RE} A}{2\pi\gamma_0 p^2}
        \frac{\exp[-\gamma/\gamma_0 - A(1 - s\xi)]}{1 - e^{-2A}},\qquad
    A = \frac{E/E_c + 1}{Z_{\rm tot} + 1}\gamma,\qquad
    \gamma_0 = c_Z \ln\Lambda,

with p in m_e c, so f is in m^-3 (m_e c)^-3 like DREAM's ``f_re``. It holds
in a uniform, fully ionised plasma with constant E, for an avalanche-dominated
runaway population. The paper uses c_Z = sqrt(5 + Z_tot).

``s`` is the beam direction. The paper has electrons accelerated towards
xi = -1 (s = -1). DREAM with E > 0 puts the runaways at xi = +1, so the
default here is s = +1.

DREAM stores cell averages, and the analytic f varies over a pitch scale of
1/A, about 1e-3 at 20 MeV with the default parameters, narrower than the
pitch cells near the beam. :func:`distribution` therefore returns the mean of
f over each pitch cell, integrated in closed form, and the angle averages are
taken through the same :class:`~cdo.moments.MomentumMoments` code as the data.

E/E_c, Z_tot and lnΛ can also come from the run, at each time and radius:
:meth:`AvalancheParams.from_run` leaves them unset, and :meth:`~AvalancheParams.resolve`
fills them from ``|E_field| / Ectot``, ``Zeff`` and ``lnLambdaC``. Ectot is the
Connor-Hastie field with the total electron density, the E_c of the paper.
DREAM's Zeff equals the paper's Z_tot only in a fully ionised plasma. The
pitch-averaged f does not depend on E/E_c, so on the angle averages the run
values change the curve only through gamma0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np


#: Parameters that :meth:`AvalancheParams.from_run` takes from the run.
RUN_PARAMETERS = ("E_over_Ec", "Z_tot", "lnLambda")


@dataclass
class AvalancheParams:
    """Parameters of the analytic avalanche distribution.

    The defaults are the DEMO termination case (E/E_c = 50, Z_tot = 1.4,
    lnΛ = 14.3, mean energy about 18.5 MeV). ``E_over_Ec``, ``Z_tot`` or
    ``lnLambda`` set to ``None`` are taken from the run at each time and
    radius, from the fields named by ``Ec_field``, ``Z_field`` and
    ``lnLambda_field``; :meth:`from_run` sets all three so. ``c_Z`` defaults
    to sqrt(5 + Z_tot). ``n_re`` (m^-3) fixes the density; ``None`` takes the
    run's ``n_re``. ``direction`` is +1 for a beam at xi = +1 (DREAM with
    E > 0), -1 for xi = -1 (the paper).
    """

    E_over_Ec: float | None = 50.0
    Z_tot: float | None = 1.4
    lnLambda: float | None = 14.3
    c_Z: float | None = None
    n_re: float | None = None
    direction: int = 1
    Ec_field: str = "Ectot"
    Z_field: str = "Zeff"
    lnLambda_field: str = "lnLambdaC"

    def __post_init__(self):
        if self.direction not in (1, -1):
            raise ValueError(f"direction must be +1 or -1, got {self.direction!r}")

    @classmethod
    def from_run(cls, **kwargs) -> "AvalancheParams":
        """E/E_c, Z_tot and lnΛ from the run; other fields as given."""
        return cls(**{name: None for name in RUN_PARAMETERS}, **kwargs)

    @property
    def run_parameters(self) -> tuple:
        """Names of the parameters taken from the run."""
        return tuple(name for name in RUN_PARAMETERS if getattr(self, name) is None)

    @property
    def kind(self) -> str:
        """Legend text: which values the curve uses."""
        run = self.run_parameters
        if not run:
            return "set values"
        return "run values" if len(run) == len(RUN_PARAMETERS) else "run and set values"

    def _require(self):
        if self.run_parameters:
            raise ValueError(f"{', '.join(self.run_parameters)} come from the run; "
                             "call resolve(run, time_index, radial_cell) first")

    @property
    def cZ(self) -> float:
        self._require()
        return math.sqrt(5.0 + self.Z_tot) if self.c_Z is None else float(self.c_Z)

    @property
    def gamma0(self) -> float:
        """Mean Lorentz factor of the runaways, c_Z lnΛ."""
        return self.cZ * self.lnLambda

    def A(self, p) -> np.ndarray:
        """Pitch-width parameter A(p), the inverse width of the beam in xi."""
        self._require()
        gamma = np.sqrt(1.0 + np.asarray(p, dtype=float) ** 2)
        return (self.E_over_Ec + 1.0) / (self.Z_tot + 1.0) * gamma

    @property
    def values_text(self) -> str:
        """E/E_c, Z_tot, lnΛ and gamma0, "run" for those taken from the run."""
        def show(value, fmt):
            return "run" if value is None else format(value, fmt)
        text = (f"$E/E_c$ = {show(self.E_over_Ec, '.3g')}, "
                f"$Z$ = {show(self.Z_tot, '.3g')}, ln$\\Lambda$ = {show(self.lnLambda, '.3g')}")
        if not self.run_parameters:
            text += f", $\\gamma_0$ = {self.gamma0:.3g}"
        return text

    @property
    def label(self) -> str:
        return f"analytic avalanche ({self.kind}): {self.values_text}"

    def resolve(self, run, time_index: int, radial_cell: int) -> "Resolved":
        """These parameters with the run's values filled in at one time and radius."""
        values, missing = {}, []

        def read(name):
            data = run.field(name)
            if data is None:
                return None
            value = float(data[time_index, radial_cell])
            return value if math.isfinite(value) else None

        if self.E_over_Ec is None:
            E, Ec = read("E_field"), read(self.Ec_field)
            if E is None or Ec is None or not Ec > 0:
                missing.append(f"E_field/{self.Ec_field}")
            else:
                values["E_over_Ec"] = abs(E) / Ec
        if self.Z_tot is None:
            Z = read(self.Z_field)
            if Z is None:
                missing.append(self.Z_field)
            else:
                values["Z_tot"] = Z
        if self.lnLambda is None:
            lnL = read(self.lnLambda_field)
            if lnL is None:
                missing.append(self.lnLambda_field)
            else:
                values["lnLambda"] = lnL
        n_re = density_at(run, self, time_index, radial_cell)
        params = None if missing else replace(self, **values)
        return Resolved(params, n_re, values, missing)


@dataclass
class Resolved:
    """:class:`AvalancheParams` at one time and radius, with what came from the run.

    ``params`` is ``None`` when a run field was missing, listed in ``missing``.
    """

    params: AvalancheParams | None
    n_re: float
    from_run: dict = field(default_factory=dict)
    missing: list = field(default_factory=list)

    @property
    def drawable(self) -> bool:
        return self.params is not None and self.n_re > 0

    @property
    def text(self) -> str:
        """The values used, for the annotation box."""
        if self.missing:
            return "run has no " + ", ".join(self.missing)
        text = self.params.values_text
        return text + f", $n_{{\\rm RE}}$ = {self.n_re:.3g} m$^{{-3}}$"


def distribution(params: AvalancheParams, p, xi_edges, n_re: float) -> np.ndarray:
    """Mean of f over each pitch cell, shape ``(xi cells, p)``, in m^-3 (m_e c)^-3.

    ``p`` are momentum cell centres in m_e c, ``xi_edges`` the pitch cell
    edges. The momentum dependence is taken at the cell centres.
    """
    p = np.asarray(p, dtype=float)
    edges = np.asarray(xi_edges, dtype=float)
    gamma = np.sqrt(1.0 + p**2)
    A = params.A(p)[None, :]
    # u = 1 - s xi is the distance from the beam; per cell u_lo <= u <= u_lo + width.
    u = 1.0 - params.direction * edges
    u_lo = np.minimum(u[:-1], u[1:])[:, None]
    width = np.abs(np.diff(edges))[:, None]
    Aw = A * width
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        # Cell mean of exp(-A u): exp(-A u_lo) (1 - exp(-A w)) / (A w), tending to
        # exp(-A u_lo) for small A w.
        shape = np.where(Aw > 1e-12, -np.expm1(-Aw) / Aw, 1.0)
        pitch = np.exp(-A * u_lo) * shape
        norm = A / -np.expm1(-2.0 * A)
    radial = n_re * np.exp(-gamma / params.gamma0) / (2.0 * np.pi * params.gamma0 * p**2)
    return radial[None, :] * norm * pitch


def on_grid(run, grid: str, params: AvalancheParams, n_re: float) -> np.ndarray:
    """:func:`distribution` on a run's momentum grid, ``(xi cells, p)``."""
    prefix = "hot" if grid == "hottail" else "re"
    read = run.reference.read
    return distribution(params, read(f"{prefix}_p"), read(f"{prefix}_xi_edges"), n_re)


def angle_average(run, grid: str, radial_cell: int, params: AvalancheParams,
                  n_re: float, moment: str = "distribution") -> np.ndarray:
    """Angle average of the analytic f on a run's grid, as for the data.

    ``moment`` is ``"distribution"`` (mean over pitch) or ``"density"``
    (dn/dp, with the run's phase-space weights at ``radial_cell``).
    """
    mm = run.moments(grid)
    if mm is None:
        raise ValueError(f"run has no {grid} grid")
    f = on_grid(run, grid, params, n_re)
    if moment == "distribution":
        return mm.distribution(f)
    if moment == "density":
        return mm.density(f)[radial_cell]
    raise ValueError(f"moment must be 'distribution' or 'density', got {moment!r}")


def density_at(run, params: AvalancheParams, time_index: int, radial_cell: int) -> float:
    """n_RE for the analytic f: ``params.n_re`` if set, the run's ``n_re`` otherwise."""
    if params.n_re is not None:
        return float(params.n_re)
    n_re = run.field("n_re")
    return float(n_re[time_index, radial_cell]) if n_re is not None else float("nan")


def radial_average(run, params: AvalancheParams, time_index: int,
                   moment: str = "distribution") -> tuple:
    """Volume average over radius of the analytic angle average on the runaway grid.

    Each radial cell takes its own n_RE and, for parameters from the run, its
    own E/E_c, Z_tot and lnΛ, and is weighted by its volume, as the data is.
    ``moment`` is as in :func:`angle_average`. Returns the values over the
    runaway-grid momenta, ``None`` when no cell can be drawn, and the run
    fields missing at some radius.
    """
    if moment not in ("distribution", "density"):
        raise ValueError(f"moment must be 'distribution' or 'density', got {moment!r}")
    mm = run.moments("runaway")
    if mm is None:
        raise ValueError("run has no runaway grid")
    volumes = np.asarray(run.cell_volumes, dtype=float)
    total, missing, last, per_cell = None, [], None, None
    for cell, volume in enumerate(volumes):
        resolved = params.resolve(run, time_index, cell)
        missing += [name for name in resolved.missing if name not in missing]
        if not resolved.drawable:
            continue
        if resolved.params != last:
            f = on_grid(run, "runaway", resolved.params, 1.0)
            per_cell = mm.distribution(f) if moment == "distribution" else mm.density(f)
            last = resolved.params
        term = resolved.n_re * volume * (per_cell if moment == "distribution" else per_cell[cell])
        total = term if total is None else total + term
    if total is None:
        return None, missing
    return total / volumes.sum(), missing
