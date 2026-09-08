"""Checks for momentum-grid moments.

The formulas are validated for real against ``DREAMOutput`` in
``test_dream_crosscheck.py`` when DREAM is importable. Here they are pinned by
their defining properties, which hold without a DREAM install.
"""

import numpy as np
import pytest

from cdo.moments import MomentumMoments, current


def make_moments(nr=2, nxi=6, np_=5, trapped=0.0):
    xi_edges = np.linspace(-1, 1, nxi + 1)
    xi = 0.5 * (xi_edges[:-1] + xi_edges[1:])
    dxi = np.diff(xi_edges)
    p = np.linspace(0.1, 2.0, np_)
    Vprime = np.ones((nr, nxi, np_))
    return MomentumMoments(
        p=p,
        xi=xi,
        xi_edges=xi_edges,
        dxi=dxi,
        Vprime_VpVol=Vprime,  # VpVol = 1
        xi0_trapped=np.full(nr, trapped),
    )


def test_distribution_of_constant_is_the_mean_over_pitch():
    mm = make_moments()
    f = np.ones((3, 2, 6, 5))  # (t, r, xi, p)
    # Integral of f/2 over xi from -1 to 1 with f=1 is 1.
    assert mm.distribution(f) == pytest.approx(np.ones((3, 2, 5)))


def test_density_uses_vprime_weight():
    mm = make_moments()
    mm.Vprime_VpVol = np.full_like(mm.Vprime_VpVol, 2.0)
    f = np.ones((1, 2, 6, 5))
    # sum over xi of f * 2 * dxi = 2 * (sum dxi) = 2 * 2 = 4
    assert mm.density(f) == pytest.approx(np.full((1, 2, 5), 4.0))


def test_single_pitch_cell_grid_is_handled():
    """output_7_.h5 has f_hot with one pitch cell."""
    mm = make_moments(nxi=1)
    f = np.ones((1, 2, 1, 5))
    assert mm.distribution(f).shape == (1, 2, 5)
    assert mm.density(f).shape == (1, 2, 5)
    assert mm.current_density(f).shape == (1, 2, 5)


def test_current_moment_excludes_trapped_cells():
    """With a wide trapped region the central pitch cells contribute nothing."""
    free = make_moments(nxi=6, trapped=0.0)
    trapped = make_moments(nxi=6, trapped=0.9)
    f = np.ones((1, 2, 6, 5))
    cur_free = free.current_density(f)
    cur_trapped = trapped.current_density(f)
    # Excluding cells can only reduce the magnitude of the summed current.
    assert np.all(np.abs(cur_trapped) <= np.abs(cur_free) + 1e-30)
    assert np.any(np.abs(cur_trapped) < np.abs(cur_free))


def test_current_moment_is_antisymmetric_in_pitch():
    """Reversing every pitch reverses the parallel current."""
    mm = make_moments()
    f = np.random.default_rng(0).random((1, 2, 6, 5))
    forward = mm.current_density(f)
    reversed_pitch = mm.current_density(f[:, :, ::-1, :])
    assert forward == pytest.approx(-reversed_pitch)


def test_current_integrates_density_with_flux_weight():
    j = np.ones((4, 3))  # (t, r)
    VpVol = np.array([2.0, 3.0, 4.0])
    dr = np.full(3, 0.5)
    GR0 = np.full(3, 2.0)
    Bmin = np.full(3, 1.0)
    FSA = np.full(3, 1.0)
    # weight = VpVol*dr*GR0/Bmin*FSA = [2,3,4]*0.5*2 = [2,3,4]; sum=9; /2pi
    expected = 9.0 / (2 * np.pi)
    assert current(j, VpVol, dr, GR0, Bmin, FSA) == pytest.approx(
        np.full(4, expected)
    )
