"""Fits of the analytic avalanche distribution, on synthetic data."""

import os

import numpy as np
import pytest

from cdo import avalanche, fitting
from cdo.avalanche import AvalancheParams

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "output_7_.h5")

P_EDGES = np.linspace(2.5, 252.5, 101)
P = 0.5 * (P_EDGES[:-1] + P_EDGES[1:])
# Pitch edges dense towards xi = 1, down to 1 - xi = 1e-5, so the beam is
# resolved at every momentum, and coarse elsewhere.
XI_EDGES = np.sort(1.0 - np.concatenate([[0.0], np.geomspace(1e-5, 0.3, 40),
                                         np.linspace(0.4, 2.0, 9)]))
# Not the defaults, so no default start is the answer.
OTHER = AvalancheParams(E_over_Ec=12.0, Z_tot=3.0, lnLambda=17.0)


def a_hat(params):
    return (params.E_over_Ec + 1.0) / (params.Z_tot + 1.0)


def synthetic(params, n_re=1.0, xi_edges=XI_EDGES):
    f = avalanche.distribution(params, P, xi_edges, n_re)
    return fitting.Distribution(f, P, xi_edges, P_EDGES)


@pytest.mark.parametrize("params", [AvalancheParams(), OTHER], ids=["default", "other"])
@pytest.mark.parametrize("fit", [fitting.fit_plain, fitting.fit_extended], ids=["plain", "extended"])
def test_fit_returns_the_parameters_of_the_analytic_distribution(params, fit):
    result = fit(synthetic(params, n_re=3e16))
    assert result.success
    assert result.gamma0 == pytest.approx(params.gamma0, rel=1e-6)
    assert result.A_hat == pytest.approx(a_hat(params), rel=1e-6)
    assert result.beta == pytest.approx(1.0, abs=1e-6)
    assert result.C[0] == pytest.approx(3e16, rel=1e-6)
    assert result.rms_log10[0] < 1e-8
    assert result.at_bound == ()
    assert result.n_cells[0] > 100


def test_extended_fit_from_a_poor_start():
    result = fitting.fit_extended(synthetic(OTHER), starts=[(15.0, 1.0, 0.3)])
    assert result.gamma0 == pytest.approx(OTHER.gamma0, rel=1e-6)
    assert result.A_hat == pytest.approx(a_hat(OTHER), rel=1e-6)
    assert result.beta == pytest.approx(1.0, abs=1e-6)


def test_content_weighting_returns_the_same_parameters():
    result = fitting.fit_extended(synthetic(OTHER), weighted=True)
    assert result.weighted
    assert result.gamma0 == pytest.approx(OTHER.gamma0, rel=1e-6)
    assert result.A_hat == pytest.approx(a_hat(OTHER), rel=1e-6)


def test_weighting_needs_the_momentum_edges():
    data = synthetic(OTHER)
    data.p_edges = None
    with pytest.raises(ValueError, match="p_edges"):
        fitting.fit_plain(data, weighted=True)


def test_uncertainties_follow_from_the_jacobian():
    rng = np.random.default_rng(1)
    data = synthetic(OTHER)
    data.f = data.f * np.exp(0.05 * rng.standard_normal(data.f.shape))
    result = fitting.fit_extended(data)
    for name in ("gamma0", "A_hat", "beta"):
        assert 0 < result.sigma[name] < 0.05 * max(abs(getattr(result, name)), 1.0)
    assert result.gamma0 == pytest.approx(OTHER.gamma0, abs=5 * result.sigma["gamma0"])
    assert result.A_hat == pytest.approx(a_hat(OTHER), abs=5 * result.sigma["A_hat"])
    assert result.beta == pytest.approx(1.0, abs=5 * result.sigma["beta"])
    assert result.rms_log10[0] == pytest.approx(0.05 / np.log(10), rel=0.1)
    plain = fitting.fit_plain(data)
    assert plain.sigma["beta"] == 0.0


def test_extended_fit_recovers_beta():
    f = fitting.model(P, XI_EDGES, 1.0, 30.0, 5.0, 0.6)
    result = fitting.fit_extended(fitting.Distribution(f, P, XI_EDGES, P_EDGES))
    assert (result.gamma0, result.A_hat, result.beta) == pytest.approx((30.0, 5.0, 0.6), rel=1e-6)


def test_gamma_ref_changes_A_hat_but_not_the_fitted_A():
    beta = 0.6
    f = fitting.model(P, XI_EDGES, 1.0, 30.0, 5.0, beta, gamma_ref=20.0)
    data = fitting.Distribution(f, P, XI_EDGES, P_EDGES)
    at_20 = fitting.fit_extended(data, gamma_ref=20.0)
    at_50 = fitting.fit_extended(data, gamma_ref=50.0)
    np.testing.assert_allclose(at_50.A(P), at_20.A(P), rtol=1e-6)
    assert at_50.A_hat == pytest.approx(at_20.A_hat * (20.0 / 50.0) ** (1.0 - beta), rel=1e-6)
    assert at_50.A_over_gamma(50.0) == pytest.approx(at_50.A_hat)
    assert at_20.A_over_gamma(50.0) == pytest.approx(at_50.A_hat, rel=1e-6)
    assert at_50.gamma0 == pytest.approx(at_20.gamma0, rel=1e-6)


def test_plain_form_does_not_depend_on_gamma_ref():
    a = fitting.ln_model(P, XI_EDGES, 1.0, 30.0, 5.0, 1.0, gamma_ref=20.0)
    b = fitting.ln_model(P, XI_EDGES, 1.0, 30.0, 5.0, 1.0, gamma_ref=7.0)
    np.testing.assert_array_equal(a, b)


def test_model_matches_the_analytic_distribution():
    params = AvalancheParams(direction=-1)
    expected = avalanche.distribution(params, P, XI_EDGES[::-1] * -1, 2.0)
    got = fitting.model(P, XI_EDGES[::-1] * -1, 2.0, params.gamma0, a_hat(params), direction=-1)
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_joint_fit_shares_the_shape():
    a = synthetic(OTHER, n_re=1.0)
    b = synthetic(OTHER, n_re=0.02)
    result = fitting.fit_joint([a, b])
    assert result.C == pytest.approx((1.0, 0.02), rel=1e-6)
    assert result.gamma0 == pytest.approx(OTHER.gamma0, rel=1e-6)
    assert result.A_hat == pytest.approx(a_hat(OTHER), rel=1e-6)
    assert result.beta == pytest.approx(1.0, abs=1e-6)
    assert len(result.rms_log10) == len(result.n_cells) == len(result.sigma["C"]) == 2
    np.testing.assert_allclose(result.model(P, XI_EDGES, 1), b.f, rtol=1e-6)


def test_joint_fit_of_different_grids():
    coarse = np.sort(1.0 - np.concatenate([[0.0], np.geomspace(1e-4, 0.3, 12), [1.0, 2.0]]))
    result = fitting.fit_joint([synthetic(OTHER, 1.0), synthetic(OTHER, 0.5, xi_edges=coarse)],
                               form="plain")
    assert result.form == "plain"
    assert result.C == pytest.approx((1.0, 0.5), rel=1e-6)
    assert result.gamma0 == pytest.approx(OTHER.gamma0, rel=1e-6)


def test_parameter_at_a_bound_is_reported():
    result = fitting.fit_plain(synthetic(OTHER), bounds={"gamma0": (1.0, 40.0)})
    assert result.at_bound == ("gamma0",)
    assert result.gamma0 == pytest.approx(40.0)


# --- fit cells ----------------------------------------------------------------


def test_fit_cells_selection():
    p = np.array([4.0, 5.0, 6.0, 7.0])
    edges = np.array([-1.0, 0.0, 0.6, 0.7, 0.8, 1.0])
    f = np.ones((5, 4))
    mask = fitting.fit_cells(f, p, edges, xi_pass=0.7, p_min=5.0)
    # p must exceed p_min; pitch cells wholly at xi >= xi_pass, an edge on xi_pass counts.
    expected = np.zeros((5, 4), dtype=bool)
    expected[3:, 2:] = True
    np.testing.assert_array_equal(mask, expected)
    # A cell straddling xi_pass is out.
    assert not fitting.fit_cells(f, p, edges, xi_pass=0.65, p_min=0.0)[2].any()


def test_fit_cells_drop_small_negative_and_missing_values():
    p = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    edges = np.array([0.9, 1.0])
    f = np.array([[1.0, 2e-6, 5e-7, -1.0, np.nan]])
    mask = fitting.fit_cells(f, p, edges, xi_pass=0.0, p_min=0.0)
    np.testing.assert_array_equal(mask, [[True, True, False, False, False]])
    assert not fitting.fit_cells(np.full((1, 5), np.nan), p, edges).any()
    assert not fitting.fit_cells(np.zeros((1, 5)), p, edges).any()


def test_fit_cells_follow_the_direction():
    p = np.array([10.0])
    edges = np.array([-1.0, -0.8, 0.0, 0.8, 1.0])
    f = np.ones((4, 1))
    forward = fitting.fit_cells(f, p, edges, xi_pass=0.7, p_min=0.0)
    backward = fitting.fit_cells(f, p, edges, xi_pass=0.7, p_min=0.0, direction=-1)
    np.testing.assert_array_equal(forward[:, 0], [False, False, False, True])
    np.testing.assert_array_equal(backward[:, 0], [True, False, False, False])


def test_fit_uses_the_distribution_xi_pass():
    data = synthetic(OTHER)
    data.xi_pass = 0.99
    assert fitting.fit_plain(data).n_cells[0] < fitting.fit_plain(data, xi_pass=0.5).n_cells[0]
    assert np.all(data.xi_edges[:-1][fitting.fit_plain(data).masks[0].any(axis=1)] >= 0.99)


def test_too_few_fit_cells_raise():
    data = synthetic(OTHER)
    with pytest.raises(ValueError, match="fit cells"):
        fitting.fit_extended(data, p_min=1e4)
    with pytest.raises(ValueError, match="form"):
        fitting.fit_joint([data], form="beam")
    with pytest.raises(ValueError, match="shape"):
        fitting.Distribution(data.f[:, :-1], P, XI_EDGES)


# --- data from a run ----------------------------------------------------------


@pytest.mark.skipif(not os.path.exists(SAMPLE), reason="output_7_.h5 not present")
def test_distribution_from_run_is_per_electron():
    import cdo
    from cdo.plotting import _momentum_weights, _volume_mean

    with cdo.Run([SAMPLE]) as run:
        data = fitting.distribution_from_run(run, time_ms="end")
        assert data.time_index == run.timegrid_length - 1
        assert data.content().sum() == pytest.approx(1.0, rel=1e-12)
        assert 0.0 <= data.xi_pass < 1.0
        # Its pitch mean is the per-electron curve of the favg_re_ravg figures.
        favg = _volume_mean(run.angle_average("runaway", "distribution"), run.cell_volumes)
        favg = favg[data.time_index]
        favg /= np.nansum(favg * _momentum_weights(run, "re", "distribution"))
        pitch_mean = np.sum(data.f * data.dxi[:, None] / 2.0, axis=0)
        np.testing.assert_allclose(pitch_mean, favg, rtol=1e-10, atol=1e-14 * favg.max())
        same = fitting.distribution_from_run(run, time_index=-1)
        np.testing.assert_array_equal(same.f, data.f)
        with pytest.raises(ValueError, match="one of"):
            fitting.distribution_from_run(run)
