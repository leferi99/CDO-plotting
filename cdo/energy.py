"""Energy-space view of the runaway distribution.

The runaway momentum grid is uniform in normalised momentum, not in energy, so
turning an angle-averaged distribution into a spectrum in energy needs the
change of variable ``dn/dE = dn/dp * dp/dE``. This gathers the energy grid and
that transform, carried over from the retired ``distribution.py``.

Total energy here includes the rest mass, so the grid starts at
:math:`m_e c^2`. Kinetic energy is total minus rest mass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.constants

M_E = scipy.constants.m_e
C = scipy.constants.c
MEC2_J = M_E * C**2  # rest energy in joules
MEC2_EV = 510998.95  # rest energy in electronvolts


@dataclass
class EnergyGrid:
    """Runaway energy grid and the momentum-to-energy spectrum transform."""

    p: np.ndarray  # (p,) normalised momentum, cell centres
    p_edges: np.ndarray  # (p + 1,) cell edges
    total_energy_eV: np.ndarray  # (p,) total energy including rest mass
    total_energy_J: np.ndarray  # (p,)
    kinetic_energy_edges_J: np.ndarray  # (p + 1,) kinetic energy at cell edges
    velocity: np.ndarray  # (p,) particle speed in m/s

    @classmethod
    def from_run(cls, run) -> "EnergyGrid | None":
        """Build from a :class:`cdo.concat.Run`, or ``None`` without a runaway grid."""
        if not run.reference.grids.get("runaway", False):
            return None
        p = run.reference.read("re_p")
        p_edges = run.reference.read("re_p_edges")
        return cls.from_momentum(p, p_edges)

    @classmethod
    def from_momentum(cls, p: np.ndarray, p_edges: np.ndarray) -> "EnergyGrid":
        gamma = np.sqrt(p**2 + 1)
        return cls(
            p=p,
            p_edges=p_edges,
            total_energy_eV=(gamma - 1) * MEC2_EV + MEC2_EV,
            total_energy_J=(gamma - 1) * MEC2_J + MEC2_J,
            kinetic_energy_edges_J=(np.sqrt(p_edges**2 + 1) - 1) * MEC2_J,
            velocity=p * C / gamma,
        )

    @property
    def dp_dE(self) -> np.ndarray:
        r"""The Jacobian :math:`\mathrm{d}p/\mathrm{d}E` at each grid point.

        With momentum normalised to :math:`m_e c` and :math:`E` the total energy,
        :math:`\mathrm{d}E/\mathrm{d}p = m_e c^2\, p/\gamma`, so
        :math:`\mathrm{d}p/\mathrm{d}E = \gamma/(m_e c^2 p)`. Written in energy
        variables this is :math:`E/(m_e c^2\sqrt{E^2-(m_e c^2)^2})`, since
        :math:`\sqrt{E^2-(m_e c^2)^2}=m_e c^2 p`. The two forms are identical and
        the Jacobian does not depend on which moment is being transformed.
        """
        return self.total_energy_J / (
            MEC2_J * np.sqrt(self.total_energy_J**2 - MEC2_J**2)
        )

    def to_energy(self, moment_per_dp: np.ndarray) -> np.ndarray:
        r"""Re-express a per-momentum spectrum as a per-energy spectrum.

        Multiplies by :math:`\mathrm{d}p/\mathrm{d}E`, turning a quantity whose
        integral over :math:`\mathrm{d}p` is meaningful into one whose integral
        over :math:`\mathrm{d}E` is the same quantity. Applies to any moment on
        the runaway grid: the density moment ``Run.angle_average("runaway",
        "density")`` becomes :math:`\mathrm{d}n/\mathrm{d}E`, and the current
        moment ``Run.angle_average("runaway", "current")`` becomes
        :math:`\mathrm{d}j/\mathrm{d}E`.

        The pitch weighting stays inside the moment. Building a current by
        multiplying the density spectrum by speed, as the retired
        ``distribution.py`` did in ``alternate_current``, drops that weighting
        and overcounts by a large factor; use the current moment instead.
        """
        return moment_per_dp * self.dp_dE

    def integrate(self, moment_per_dp: np.ndarray) -> np.ndarray:
        r"""Integrate a per-momentum moment over the energy grid, ``(t, r)``.

        Equivalent to integrating over ``dp`` up to the midpoint-rule error of
        evaluating the spectrum at cell centres against energy widths taken at
        cell edges, which falls with grid resolution. On the density moment this
        returns the runaway density; on the current moment, the current density.
        """
        dE = np.diff(self.kinetic_energy_edges_J)
        return np.tensordot(self.to_energy(moment_per_dp), dE, axes=([-1], [0]))
