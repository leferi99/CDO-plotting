"""Checks for the runaway energy-space transform.

These pin the formulas carried over from the retired ``distribution.py``. Note
that the integrated energy-space density is not expected to equal ``eqsys/n_re``;
that cross-check was flagged as uncertain in the original and is not a
correctness criterion here.
"""

import numpy as np
import pytest
import scipy.constants

from cdo.energy import MEC2_EV, MEC2_J, EnergyGrid


@pytest.fixture
def grid():
    p_edges = np.linspace(3.0, 60.0, 41)
    p = 0.5 * (p_edges[:-1] + p_edges[1:])  # cell centres
    return EnergyGrid.from_momentum(p, p_edges)


def test_total_energy_starts_above_rest_mass(grid):
    # The runaway grid begins well above thermal, so p is order unity or more.
    assert np.all(grid.total_energy_eV > MEC2_EV)
    assert grid.total_energy_J[0] == pytest.approx(
        (np.sqrt(3.0**2 + 1) - 1) * MEC2_J + MEC2_J
    )


def test_ev_and_joule_grids_agree_through_the_elementary_charge(grid):
    assert grid.total_energy_eV == pytest.approx(
        grid.total_energy_J / scipy.constants.e
    )


def test_velocity_is_relativistic_and_below_c(grid):
    gamma = np.sqrt(grid.p**2 + 1)
    assert grid.velocity == pytest.approx(grid.p * scipy.constants.c / gamma)
    assert np.all(grid.velocity < scipy.constants.c)


def test_dp_dE_jacobian_matches_the_momentum_form(grid):
    # Energy form E/(mc^2 sqrt(E^2-(mc^2)^2)) equals momentum form gamma/(mc^2 p).
    gamma = np.sqrt(grid.p**2 + 1)
    assert grid.dp_dE == pytest.approx(gamma / (MEC2_J * grid.p))


def test_to_energy_applies_the_jacobian_to_any_moment(grid):
    f = np.ones((2, 3, grid.p.size))
    assert grid.to_energy(f) == pytest.approx(f * grid.dp_dE)


def test_integrate_over_energy_equals_integrate_over_momentum(grid):
    """A per-dp moment integrated over dE recovers its dp integral, up to
    the midpoint discretisation between cell centres and edges."""
    rng = np.random.default_rng(0)
    moment = rng.random((2, 3, grid.p.size))
    dp = np.diff(grid.p_edges)
    over_p = np.tensordot(moment, dp, axes=([-1], [0]))
    over_E = grid.integrate(moment)
    assert over_E == pytest.approx(over_p, rel=0.02)
    assert over_E.shape == (2, 3)


def test_from_run_without_runaway_grid_returns_none():
    class FakeRef:
        grids = {"runaway": False}

    class FakeRun:
        reference = FakeRef()

    assert EnergyGrid.from_run(FakeRun()) is None
