"""Momentum-grid moments computed directly from an HDF5 output file.

DREAM's own ``DREAMOutput`` provides ``j.current()`` and
``f.angleAveraged(moment=...)`` for these, and the retired ``CDOconcat`` opened
every file a second time through ``DREAMOutput`` to reach them. That dependency
is dropped here: the current ``DREAM`` package cannot even open the older output
files, raising in its settings parser, so relying on it for the files this
package exists to support is not viable.

The formulas reproduce the DREAM kernel and were checked against ``DREAMOutput``
on a p/xi output to floating-point round-off. See ``tests/test_moments.py``.
Only p/xi momentum grids are handled, which is what DREAM writes for the hottail
and runaway grids.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.constants

from .schema import Resolver

C = scipy.constants.c
E = scipy.constants.e


def current(
    j: np.ndarray,
    VpVol: np.ndarray,
    dr: np.ndarray,
    GR0: np.ndarray,
    Bmin: np.ndarray,
    FSA_R02OverR2: np.ndarray,
) -> np.ndarray:
    r"""Radially integrate a current density into a current.

    Reproduces ``DREAM.Output.CurrentDensity.current``: the flux-surface weight
    is :math:`GR0 / B_{\min} \cdot \langle R_0^2/R^2 \rangle`, and the integral
    is :math:`\sum_r j\, V'_{\rm vol}\, \mathrm{d}r\, w / (2\pi)`.

    ``j`` has its radial axis last, any leading axes (such as time) are kept.
    """
    weight = VpVol * dr * (GR0 / Bmin * FSA_R02OverR2)
    return np.tensordot(j, weight, axes=([-1], [0])) / (2 * np.pi)


@dataclass
class MomentumMoments:
    """Angle-averaging weights for one p/xi momentum grid.

    Built once from a file's grid data, then applied to any distribution on that
    grid. ``Vprime`` and the derived ``Vprime_VpVol`` have shape ``(r, xi, p)``;
    a distribution has shape ``(t, r, xi, p)`` and the average sums the xi axis.
    """

    p: np.ndarray  # (p,) normalised momentum, cell centres
    xi: np.ndarray  # (xi,) pitch cosine, cell centres
    xi_edges: np.ndarray  # (xi + 1,) pitch cosine, cell edges
    dxi: np.ndarray  # (xi,) pitch cell widths
    Vprime_VpVol: np.ndarray  # (r, xi, p)
    xi0_trapped: np.ndarray  # (r,) trapped-passing boundary per radius

    @classmethod
    def from_resolver(cls, resolver: Resolver, grid: str) -> "MomentumMoments":
        """Read the weights for ``grid`` ("hottail" or "runaway").

        Returns ``None`` when the grid was not enabled for the run.
        """
        if not resolver.grids.get(grid, False):
            return None

        Vprime = resolver.read(_grid_field(grid, "Vprime"))
        VpVol = resolver.read("VpVol")
        p = resolver.read(_grid_field(grid, "p"))
        xi = resolver.read(_grid_field(grid, "xi"))
        xi_edges = resolver.read(_grid_field(grid, "xi_edges"))
        dxi = resolver.read(_grid_field(grid, "dxi"))
        xi0 = resolver.read("xi0_trapped_boundary")

        if dxi is None:
            dxi = np.abs(np.diff(xi_edges))
        if xi0 is None:
            # Cylindrical or otherwise untrapped: nothing is trapped.
            xi0 = np.zeros(VpVol.shape)

        return cls(
            p=p,
            xi=xi,
            xi_edges=xi_edges,
            dxi=dxi,
            Vprime_VpVol=Vprime / VpVol[:, None, None],
            xi0_trapped=np.asarray(xi0).ravel(),
        )

    # --- moments -----------------------------------------------------------

    def distribution(self, f: np.ndarray) -> np.ndarray:
        r"""Angle average :math:`\langle f\rangle = \int f\,\mathrm{d}\xi_0`.

        DREAM divides by the pitch range of 2, giving the mean over pitch.
        """
        return np.sum(f / 2.0 * self.dxi[:, None], axis=-2)

    def density(self, f: np.ndarray) -> np.ndarray:
        r"""Density moment :math:`\langle V' f\rangle`, giving dn/dp per radius."""
        return np.sum(f * self.Vprime_VpVol * self.dxi[:, None], axis=-2)

    def current_density(self, f: np.ndarray) -> np.ndarray:
        r"""Parallel-current moment :math:`\langle v\xi_0 V' f\rangle e`.

        Uses the bounce-averaged parallel velocity, so trapped pitch cells,
        those straddling the trapped-passing boundary, contribute nothing. This
        matches ``f.angleAveraged(moment="current")`` rather than the simpler
        ``currentDensity`` that ignores trapping.
        """
        weighted = f * self._bounce_averaged_vpar() * self.Vprime_VpVol * E
        return np.sum(weighted * self.dxi[:, None], axis=-2)

    def _bounce_averaged_vpar(self) -> np.ndarray:
        """The ``(r, xi, p)`` weight from DREAM's ``getBounceAveragedVpar``.

        A pitch cell counts only if it lies wholly on one side of the
        trapped-passing boundary, or wholly spans it. Its contribution then uses
        the cell-centre pitch. The Vprime_VpVol division here is undone by the
        multiplication in the moment, kept so the expression tracks DREAM.
        """
        nr = self.xi0_trapped.shape[0]
        nxi = self.xi.shape[0]
        np_ = self.p.shape[0]
        integrand = np.zeros((nr, nxi, np_))

        v = C * self.p / np.sqrt(1.0 + self.p**2)  # (p,)

        for ir in range(nr):
            xi0t = self.xi0_trapped[ir]
            for j in range(nxi):
                lo, hi = sorted((self.xi_edges[j], self.xi_edges[j + 1]))
                passing = (
                    hi <= -xi0t or lo >= xi0t or (lo <= -xi0t and hi >= xi0t)
                )
                if not passing or self.xi[j] == 0:
                    continue
                integrand[ir, j, :] = (
                    2 * np.pi * self.p**2 * v * self.xi[j] / self.Vprime_VpVol[ir, j, :]
                )
        return integrand


_GRID_PREFIX = {"hottail": "hot", "runaway": "re"}


def _grid_field(grid: str, suffix: str) -> str:
    """Canonical field name for a per-grid quantity, e.g. ("runaway", "p") -> "re_p"."""
    return f"{_GRID_PREFIX[grid]}_{suffix}"
