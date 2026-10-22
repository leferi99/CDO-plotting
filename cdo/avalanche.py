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
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class AvalancheParams:
    """Parameters of the analytic avalanche distribution.

    The defaults are the DEMO termination case (E/E_c = 50, Z_tot = 1.4,
    lnΛ = 14.3, mean energy about 18.5 MeV). ``c_Z`` defaults to
    sqrt(5 + Z_tot). ``n_re`` (m^-3) fixes the density; ``None`` takes the
    run's ``n_re`` at each time and radius. ``direction`` is +1 for a beam at
    xi = +1 (DREAM with E > 0), -1 for xi = -1 (the paper).
    """

    E_over_Ec: float = 50.0
    Z_tot: float = 1.4
    lnLambda: float = 14.3
    c_Z: float | None = None
    n_re: float | None = None
    direction: int = 1

    def __post_init__(self):
        if self.direction not in (1, -1):
            raise ValueError(f"direction must be +1 or -1, got {self.direction!r}")

    @property
    def cZ(self) -> float:
        return math.sqrt(5.0 + self.Z_tot) if self.c_Z is None else float(self.c_Z)

    @property
    def gamma0(self) -> float:
        """Mean Lorentz factor of the runaways, c_Z lnΛ."""
        return self.cZ * self.lnLambda

    def A(self, p) -> np.ndarray:
        """Pitch-width parameter A(p), the inverse width of the beam in xi."""
        gamma = np.sqrt(1.0 + np.asarray(p, dtype=float) ** 2)
        return (self.E_over_Ec + 1.0) / (self.Z_tot + 1.0) * gamma

    @property
    def label(self) -> str:
        return (f"analytic avalanche: $E/E_c$ = {self.E_over_Ec:g}, "
                f"$Z_{{\\rm tot}}$ = {self.Z_tot:g}, ln$\\Lambda$ = {self.lnLambda:g}, "
                f"$\\gamma_0$ = {self.gamma0:.3g}")


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
