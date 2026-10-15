"""Quantities derived from the raw datasets: volumes, integrals, ions.

Kept apart from concatenation so the formulas can be read and tested on their
own. Everything here takes plain arrays or a resolved run, never an open file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def cell_volumes(VpVol: np.ndarray, dr: np.ndarray, R0: np.ndarray) -> np.ndarray:
    r"""Physical volume of each radial cell in m^3, :math:`R_0\,V'_{\rm vol}\,\mathrm{d}r`.

    DREAM stores ``VpVol`` from a Jacobian normalised to :math:`R/R_0` rather
    than :math:`R`, so ``VpVol`` is the physical spatial Jacobian divided by the
    major radius. This is explicit in the kernel: ``JacobianAtTheta`` returns
    ``r * (R/R0) * normalizedJacobian``, with ``normalizedJacobian`` documented
    as normalised to ``r*R``. The physical volume element therefore carries one
    factor of ``R0`` back, giving ``R0 * VpVol * dr``.

    Verified on the analytic-geometry reference file: summed over the grid this
    gives 2361 m^3, matching the elongation-corrected plasma volume
    :math:`2\pi^2 R_0 a^2 \langle\kappa\rangle \approx 2357` m^3 for its
    :math:`\langle\kappa\rangle \approx 1.53`; on the ITER case it gives 790 m^3.
    DREAM's own ``Grid.integrate`` uses the bare ``VpVol * dr``, so it is a
    flux-label integral rather than a physical volume, smaller by ``R0``. Totals
    here are physical, so they equal ``R0`` times ``DREAMOutput``'s ``integral()``.
    """
    return VpVol * dr * np.asarray(R0).ravel()[0]


def radial_integral(density: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """Integrate a per-volume quantity over radius.

    ``density`` has radius as its last axis, any leading axes are kept, so a
    ``(t, r)`` density yields a ``(t,)`` total.
    """
    return np.tensordot(density, volumes, axes=([-1], [0]))


@dataclass
class IonSpecies:
    """One ion species and the charge-state rows it occupies in ``n_i``.

    DREAM stacks every charge state of every species along one axis of ``n_i``,
    ordered by species then by charge 0..Z. ``rows`` indexes that axis.
    """

    name: str
    Z: int
    rows: np.ndarray  # indices into the ion axis of n_i, length Z + 1

    def total(self, n_i: np.ndarray) -> np.ndarray:
        """Density summed over this species' charge states, ``(t, r)``."""
        return n_i[:, self.rows, :].sum(axis=1)

    def charged(self, n_i: np.ndarray) -> np.ndarray:
        """Density of the ionised states only, excluding the neutral."""
        return n_i[:, self.rows[1:], :].sum(axis=1)

    def mean_charge(self, n_i: np.ndarray) -> np.ndarray:
        r"""Mean charge :math:`\sum_j j\,n_j / \sum_j n_j` per time and radius.

        Zero where the species is absent, avoiding a divide-by-zero, rather than
        propagating the accumulator across time as the retired code did.
        """
        charges = np.arange(self.Z + 1)
        weighted = np.tensordot(charges, n_i[:, self.rows, :], axes=([0], [1]))
        total = self.total(n_i)
        return np.divide(
            weighted, total, out=np.zeros_like(weighted), where=total > 0
        )


def parse_ions(names: list[str], Z: np.ndarray) -> list[IonSpecies]:
    """Map species names and atomic numbers to their rows in ``n_i``.

    ``names`` comes from ``ionmeta/names`` with one entry per species, ``Z`` from
    ``ionmeta/Z`` likewise. Each species occupies ``Z + 1`` consecutive rows.
    Replaces the hardcoded index slices in the retired code, which assumed a
    fixed D, T, Ar layout and broke on any other.
    """
    species = []
    row = 0
    for name, z in zip(names, np.asarray(Z).ravel().tolist()):
        count = int(z) + 1
        species.append(
            IonSpecies(name=name, Z=int(z), rows=np.arange(row, row + count))
        )
        row += count
    return species


#: Prefactor of the Rosenbluth-Putvinski knock-on source, e^4 / ((4 pi eps0 m_e c)^2 c).
_E, _EPS0, _ME, _C = 1.602176634e-19, 8.8541878128e-12, 9.1093837015e-31, 299792458.0
_RP_PREFACTOR = _E**4 / ((4 * np.pi * _EPS0 * _ME * _C) ** 2 * _C)


def rp_avalanche_coefficient(p_cut: float) -> float:
    """Rosenbluth-Putvinski knock-on rate above ``p_cut`` per primary and target.

    ``n_re * n_tot * C(p_cut)`` is the rate of secondaries born above
    ``p_cut`` (in m_e c), in 1/(s m^3). DREAM uses this form for the n_re
    avalanche term of kinetic RP runs with both momentum grids, with ``p_cut``
    the hot-tail grid's p_max (AvalancheSourceRP.cpp, FLUID mode).
    """
    gamma = np.sqrt(1.0 + p_cut**2)
    return 2.0 * np.pi * _RP_PREFACTOR * (gamma + 1.0) / p_cut**2


@dataclass
class RateChannel:
    """One runaway rate channel over (time, radius), in 1/(s m^3).

    ``kind`` is ``"net"`` for the total dn_RE/dt, ``"term"`` for a term of the
    n_RE equation, ``"remainder"`` for the net rate minus the known terms, and
    ``"estimate"`` for a fluid-formula diagnostic that is not in the equations.
    """

    key: str
    label: str
    values: np.ndarray
    kind: str
    note: str = ""


def _setting(run, path, default):
    value = run.setting(path)
    return default if value is None else value


def rate_channels(run, floor: float = 1e-12) -> list[RateChannel]:
    """The runaway rate channels a run carries, labelled by what they are.

    Reads the run's settings to tell terms of the n_RE equation from fluid
    diagnostics that DREAM writes regardless of the model in use:

    - ``net``: ``runawayRate``, the total dn_RE/dt over the last internal step.
    - ``hot_flux``: ``gammaFhot``, the net flux out of the hot-tail grid
      across its p_max. In kinetic runs this is where Dreicer and hot-tail
      generation show up, together, plus any avalanche, Compton and tritium
      electrons born on the hot-tail grid. It can be negative.
    - ``avalanche``: ``GammaAva * n_re`` with fluid avalanche (modes 2, 3).
      With kinetic RP (mode 4) and both grids, ``n_re * n_tot * C(p_max,hot)``.
      Otherwise ``GammaAva * n_re`` is only an estimate.
    - ``dreicer``, ``compton``, ``tritium``, ``hottail``: terms when that
      model is on, estimates or left out otherwise.
    - ``remainder``: net minus the terms. With the Boltzmann operator (mode 5)
      and a runaway grid this is the re-re knock-on source, which DREAM does
      not write out, plus anything unaccounted for (transport, losses).

    Channels whose peak magnitude stays below ``floor`` times the strongest
    one are dropped, as are all-zero channels.
    """
    hot = bool(_setting(run, "hottailgrid/enabled", 0))
    re_grid = bool(_setting(run, "runawaygrid/enabled", 0))
    ava = int(_setting(run, "eqsys/n_re/avalanche", 1))
    dreicer = int(_setting(run, "eqsys/n_re/dreicer", 1))
    compton = int(_setting(run, "eqsys/n_re/compton/mode", 1))
    tritium = int(_setting(run, "eqsys/n_re/tritium", 1))
    n_re = run.field("n_re")

    net, terms, estimates = [], [], []

    rate = run.field("runawayRate")
    if rate is not None:
        net.append(RateChannel("net", r"net $dn_{\rm RE}/dt$", rate, "net",
                               "runawayRate, total change incl. losses"))

    flux = run.field("gammaFhot") if hot else None
    if flux is not None:
        extra = [name for name, on in (("avalanche", ava in (4, 5)),
                                       ("Compton", compton == 3),
                                       ("tritium", tritium == 3)) if on]
        note = "Dreicer + hot-tail flux across p_max,hot"
        if extra:
            note += ", incl. " + ", ".join(extra) + " electrons born on the hot grid"
        terms.append(RateChannel("hot_flux", r"$\gamma_{\rm hot+D}$: flux out of hot-tail grid",
                                 flux, "term", note))

    analytic = run.field("gammaHottail")
    if analytic is not None:
        terms.append(RateChannel("hottail", r"$\gamma_{\rm hot}$ (analytic hot-tail)",
                                 analytic, "term"))

    gamma_ava = run.field("GammaAva")
    if ava in (2, 3) and gamma_ava is not None and n_re is not None:
        terms.append(RateChannel("avalanche", r"$\Gamma_{\rm ava} n_{\rm RE}$ (fluid avalanche)",
                                 gamma_ava * n_re, "term"))
    elif ava == 4 and hot and re_grid and n_re is not None:
        n_tot = run.field("n_tot")
        p_max = float(_setting(run, "hottailgrid/pmax", np.nan))
        if n_tot is not None and np.isfinite(p_max):
            terms.append(RateChannel(
                "avalanche", r"avalanche above $p_{\rm max,hot}$ (RP)",
                n_re * n_tot * rp_avalanche_coefficient(p_max), "term",
                f"n_re n_tot C(p = {p_max:g}); secondaries below are in the hot-grid flux"))
    elif gamma_ava is not None and n_re is not None:
        estimates.append(RateChannel("avalanche_estimate",
                                     r"$\Gamma_{\rm ava} n_{\rm RE}$ fluid estimate",
                                     gamma_ava * n_re, "estimate", "not in the equations"))

    gamma_d = run.field("gammaDreicer")
    if gamma_d is not None:
        if dreicer in (2, 3, 4):
            terms.append(RateChannel("dreicer", r"$\gamma_{\rm D}$ (fluid Dreicer)", gamma_d, "term"))
        else:
            estimates.append(RateChannel("dreicer_estimate", r"$\gamma_{\rm D}$ fluid estimate",
                                         gamma_d, "estimate",
                                         "not in the equations" + ("; kinetic Dreicer is in the hot-grid flux" if hot else "")))

    for key, field, mode, symbol in (("compton", "gammaCompton", compton, "C"),
                                     ("tritium", "gammaTritium", tritium, "T")):
        value = run.field(field)
        if value is None or mode == 1:
            continue
        where = " born above $p_{\\rm max,hot}$" if (mode == 3 and hot) else ""
        terms.append(RateChannel(key, rf"$\gamma_{{\rm {symbol}}}${where}", value, "term"))

    lcfs = run.field("gammaLCFSLoss")
    if lcfs is not None:
        terms.append(RateChannel("lcfs_loss", "LCFS loss", lcfs, "term",
                                 "sign convention not verified"))

    remainder = []
    sources = [c for c in terms if c.key != "lcfs_loss"]
    if net and sources:
        rest = net[0].values - sum(c.values for c in sources)
        label = ("re-re knock-on + unaccounted (net - terms)" if ava == 5 and re_grid
                 else "unaccounted (net - terms)")
        remainder.append(RateChannel("remainder", label, rest, "remainder"))

    channels = net + terms + remainder + estimates
    peaks = {c.key: float(np.nanmax(np.abs(c.values))) if np.isfinite(c.values).any() else 0.0
             for c in channels}
    strongest = max(peaks.values(), default=0.0)
    keep = []
    for c in channels:
        if peaks[c.key] == 0.0:
            continue
        threshold = 1e-3 if c.kind == "remainder" else floor
        if peaks[c.key] > threshold * strongest:
            keep.append(c)
    return keep


def volume_average(field: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """Volume-weighted mean over radius, the last axis, ignoring NaN cells.

    NaN cells drop out of both the sum and the total volume, so a missing cell
    does not pull the mean towards zero. A time step with no finite cell is NaN.
    """
    finite = np.isfinite(field)
    weights = np.where(finite, volumes, 0.0)
    total = np.sum(weights, axis=-1)
    weighted = np.sum(np.where(finite, field, 0.0) * weights, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, weighted / np.where(total > 0, total, 1.0), np.nan)
