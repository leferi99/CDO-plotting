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


def flux_to_re(
    runawayRate: np.ndarray,
    n_re: np.ndarray,
    GammaAva: np.ndarray,
    gammaTritium: np.ndarray,
    gammaCompton: np.ndarray,
) -> np.ndarray:
    """Net non-avalanche runaway source.

    The total rate less the avalanche, tritium and Compton contributions, as in
    the retired code. Leaves the Dreicer plus hottail flux.
    """
    return runawayRate - n_re * GammaAva - gammaTritium - gammaCompton
