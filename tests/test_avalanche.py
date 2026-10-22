"""Analytic avalanche distribution: conventions, normalisation, overlays."""

import numpy as np
import pytest
from scipy.integrate import quad

from cdo import avalanche
from cdo.avalanche import AvalancheParams
from cdo.moments import MomentumMoments


def _shape(params, p, xi):
    """The paper's f at a point, with the beam at xi = s, n_RE = 1."""
    gamma = np.sqrt(1 + p**2)
    A = params.A(p)
    s = params.direction
    return (A / (2 * np.pi * params.gamma0 * p**2)
            * np.exp(-gamma / params.gamma0 - A * (1 - s * xi)) / (1 - np.exp(-2 * A)))


def test_defaults_match_the_slide_mean_energy():
    params = AvalancheParams()
    assert params.cZ == pytest.approx(np.sqrt(6.4))
    assert params.gamma0 * 0.51099895 == pytest.approx(18.5, abs=0.1)
    assert AvalancheParams(c_Z=2.0).gamma0 == pytest.approx(28.6)
    with pytest.raises(ValueError):
        AvalancheParams(direction=0)


@pytest.mark.parametrize("direction", [1, -1])
def test_cell_mean_matches_the_integral_over_the_cell(direction):
    params = AvalancheParams(direction=direction)
    p = np.array([2.0, 20.0, 80.0])
    edges = np.array([-1.0, -0.999, -0.99, -0.5, 0.5, 0.99, 0.999, 1.0])
    f = avalanche.distribution(params, p, edges, n_re=1.0)
    for k, pk in enumerate(p):
        for j in range(len(edges) - 1):
            lo, hi = edges[j], edges[j + 1]
            exact = quad(lambda x: _shape(params, pk, x), lo, hi, epsabs=0, epsrel=1e-10,
                         points=[-1.0, 1.0])[0] / (hi - lo)
            assert f[j, k] == pytest.approx(exact, rel=1e-8, abs=1e-300)


def test_direction_mirrors_the_pitch_axis():
    p = np.array([5.0, 40.0])
    edges = np.linspace(-1, 1, 11)
    forward = avalanche.distribution(AvalancheParams(direction=1), p, edges, 1.0)
    backward = avalanche.distribution(AvalancheParams(direction=-1), p, edges, 1.0)
    np.testing.assert_allclose(forward, backward[::-1], rtol=1e-12)
    assert forward[-1, 0] > forward[0, 0]


def test_angle_average_is_the_pitch_integrated_formula():
    params = AvalancheParams()
    p = np.linspace(1.0, 200.0, 50)
    edges = np.concatenate([np.linspace(-1, 0.99, 20), 1 - np.logspace(-2, -6, 12), [1.0]])
    f = avalanche.distribution(params, p, edges, n_re=3e16)
    mean = np.sum(f / 2.0 * np.diff(edges)[:, None], axis=0)
    gamma = np.sqrt(1 + p**2)
    expected = 3e16 * np.exp(-gamma / params.gamma0) / (4 * np.pi * params.gamma0 * p**2)
    np.testing.assert_allclose(mean, expected, rtol=1e-10)


def test_momentum_integral_gives_n_re():
    params = AvalancheParams()
    edges = np.linspace(0, 1500, 30001)
    p = 0.5 * (edges[1:] + edges[:-1])
    f = avalanche.distribution(params, p, np.array([-1.0, 1.0]), n_re=1.0)
    n = np.sum(4 * np.pi * p**2 * f[0] * np.diff(edges))  # f[0] is the pitch mean
    # 4 pi p^2 <f> integrates to exp(-1/gamma0) ~ 0.97, since p ~ gamma only at large p.
    assert n == pytest.approx(1.0, rel=0.05)
