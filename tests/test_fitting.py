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


# Starting shapes away from every truth used here, so the solver has to move.
POOR_STARTS = {"plain": [(15.0, 1.0)], "extended": [(15.0, 1.0, 0.5)]}
FITS = {"plain": fitting.fit_plain, "extended": fitting.fit_extended}


@pytest.mark.parametrize("params", [AvalancheParams(), OTHER], ids=["default", "other"])
@pytest.mark.parametrize("form", ["plain", "extended"])
def test_fit_returns_the_parameters_of_the_analytic_distribution(params, form):
    result = FITS[form](synthetic(params, n_re=3e16), starts=POOR_STARTS[form])
    assert result.success and result.converged and all(result.resolved)
    assert result.gamma0 == pytest.approx(params.gamma0, rel=1e-6)
    assert result.A_hat == pytest.approx(a_hat(params), rel=1e-6)
    assert result.beta == pytest.approx(1.0, abs=1e-6)
    assert result.C[0] == pytest.approx(3e16, rel=1e-6)
    assert result.rms_log10[0] < 1e-8
    assert result.at_bound == ()
    assert result.n_cells[0] > 100


def test_extended_fit_from_the_default_starts():
    result = fitting.fit_extended(synthetic(OTHER))
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
    with pytest.raises(ValueError, match="distribution 0 has no p_edges"):
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
    assert all(type(v) is float for v in (*result.C, *result.sigma["C"], result.sigma["gamma0"],
                                          result.sigma["A_hat"], result.sigma["beta"]))


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
    plain = fitting.ln_model(P, XI_EDGES, 1.0, 30.0, 5.0, 1.0)
    # Through the general branch, beta just off 1, at two gamma_ref.
    for gamma_ref in (7.0, 20.0, 300.0):
        general = fitting.ln_model(P, XI_EDGES, 1.0, 30.0, 5.0, 1.0 + 1e-12, gamma_ref=gamma_ref)
        np.testing.assert_allclose(general, plain, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("form", ["plain", "extended"])
def test_fit_of_a_beam_at_xi_minus_one(form):
    params = AvalancheParams(E_over_Ec=12.0, Z_tot=3.0, lnLambda=17.0, direction=-1)
    edges = np.sort(-XI_EDGES)
    data = fitting.Distribution(avalanche.distribution(params, P, edges, 1.0), P, edges, P_EDGES)
    # The default direction finds no cells on the side of xi = +1 above f_thresh.
    with pytest.raises(ValueError, match="fit cells"):
        FITS[form](data, xi_pass=0.5)
    result = FITS[form](data, direction=-1, xi_pass=0.5, starts=POOR_STARTS[form])
    assert result.success and result.direction == -1
    assert np.all(edges[1:][result.masks[0].any(axis=1)] <= -0.5)
    assert result.gamma0 == pytest.approx(params.gamma0, rel=1e-6)
    assert result.A_hat == pytest.approx(a_hat(params), rel=1e-6)
    assert result.beta == pytest.approx(1.0, abs=1e-6)


def test_unresolved_beam_is_flagged():
    # A_hat = 500 on 60 uniform pitch cells: A dxi is far above one at every p.
    edges = np.linspace(-1.0, 1.0, 61)
    f = fitting.model(P, edges, 1.0, 30.0, 500.0)
    result = fitting.fit_plain(fitting.Distribution(f, P, edges, P_EDGES))
    assert result.converged
    assert not result.success and result.resolved == (False,)
    assert result.beam_A_dxi[0] > fitting.MAX_BEAM_A_DXI
    resolved = fitting.fit_plain(synthetic(OTHER))
    assert resolved.beam_A_dxi[0] < 1.0 and resolved.n_pitch_rows[0] > 2


def test_one_pitch_row_is_not_resolved():
    data = synthetic(OTHER)
    mask = np.zeros(data.f.shape, dtype=bool)
    mask[-1, P > 5] = True
    result = fitting.fit_plain(data, masks=[mask])
    assert result.n_pitch_rows == (1,) and result.resolved == (False,) and not result.success


def test_beam_resolution_takes_the_cell_nearest_the_beam():
    edges = np.array([0.0, 0.5, 0.9, 1.0])
    mask = np.array([[True, True], [True, False], [False, True]])
    rows, beam = fitting.beam_resolution(mask, np.array([1.0, 3.0]), edges, 2.0)
    gamma = np.sqrt(1 + np.array([1.0, 3.0]) ** 2)
    assert rows == 3
    assert beam == pytest.approx(min(2.0 * gamma[0] * 0.4, 2.0 * gamma[1] * 0.1))


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
    assert np.isnan(result.sigma["gamma0"]) and np.isfinite(result.sigma["A_hat"])
    with pytest.raises(ValueError, match="unknown bounds"):
        fitting.fit_plain(synthetic(OTHER), bounds={"gamma_0": (1.0, 40.0)})


def test_masks_must_select_positive_finite_values():
    data = synthetic(OTHER)
    data.f[-1, -1] = 0.0
    other = synthetic(OTHER)
    good = fitting.fit_cells(other.f, P, XI_EDGES)
    with pytest.raises(ValueError, match="mask 1"):
        fitting.fit_joint([other, data], masks=[good, np.ones(data.f.shape, dtype=bool)])
    with pytest.raises(ValueError, match="mask 0 has shape"):
        fitting.fit_plain(data, masks=[good[:, :-1]])


def test_weighted_joint_fit_names_the_distribution_without_p_edges():
    a, b = synthetic(OTHER), synthetic(OTHER)
    b.p_edges = None
    with pytest.raises(ValueError, match="distribution 1 has no p_edges"):
        fitting.fit_joint([a, b], weighted=True)


def test_direction_is_checked():
    for call in (lambda: fitting.fit_cells(np.ones((1, 1)), [10.0], [0.0, 1.0], direction=0),
                 lambda: fitting.ln_model(P, XI_EDGES, 1.0, 30.0, 5.0, direction=2),
                 lambda: fitting.model(P, XI_EDGES, 1.0, 30.0, 5.0, direction=0),
                 lambda: fitting.fit_plain(synthetic(OTHER), direction=0)):
        with pytest.raises(ValueError, match="direction"):
            call()


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


def _synthetic_run(tmp_path):
    """Two files, 3 radial cells, a runaway grid, f_re different at every step."""
    from test_concat import NR, write_file

    from cdo.concat import Run

    xi_edges = np.array([-1.0, -0.5, 0.0, 0.5, 0.8, 0.95, 1.0])
    p_edges = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    rng = np.random.default_rng(3)
    files = []
    for k, nt in enumerate((3, 4)):
        f_re = rng.random((nt, NR, xi_edges.size - 1, p_edges.size - 1)) * 1e15
        path = tmp_path / f"output_{k}.h5"
        write_file(path, nt=nt, t0_index=10 * k, seed=k, runaway=(f_re, xi_edges, p_edges),
                   xi0=[0.1, 0.4, 0.3])
        files.append(path)
    return Run(files), xi_edges, p_edges


def test_distribution_from_run_on_a_synthetic_run(tmp_path):
    run, xi_edges, p_edges = _synthetic_run(tmp_path)
    p = 0.5 * (p_edges[:-1] + p_edges[1:])
    volumes = run.cell_volumes
    for i in range(run.timegrid_length):
        data = fitting.distribution_from_run(run, time_index=i)
        f = run.field("f_re")[i]  # (r, xi, p)
        f_vol = np.einsum("r,rjp->jp", volumes, f) / volumes.sum()
        density = np.sum(2 * np.pi * p**2 * f_vol * np.diff(xi_edges)[:, None] * np.diff(p_edges))
        assert data.density == pytest.approx(density, rel=1e-12)
        np.testing.assert_allclose(data.f, f_vol / density, rtol=1e-12)
        assert data.content().sum() == pytest.approx(1.0, rel=1e-12)
        assert data.time_index == i and data.time_ms == run.timegrid_ms[i]
    assert data.xi_pass == pytest.approx(0.4)
    np.testing.assert_array_equal(data.xi_edges, xi_edges)
    np.testing.assert_array_equal(data.p_edges, p_edges)
    # Times in ms snap to the nearest step, "end" is the last.
    t = run.timegrid_ms
    assert fitting.distribution_from_run(run, time_ms=t[2] + 1e-6).time_index == 2
    assert fitting.distribution_from_run(run, time_ms="end").time_index == run.timegrid_length - 1
    assert fitting.distribution_from_run(run, time_index=-1).time_index == run.timegrid_length - 1
    for bad in (run.timegrid_length, -run.timegrid_length - 1):
        with pytest.raises(IndexError):
            fitting.distribution_from_run(run, time_index=bad)
    with pytest.raises(ValueError, match="one of"):
        fitting.distribution_from_run(run, time_ms=1.0, time_index=1)
    with pytest.raises(ValueError, match="hottail"):
        fitting.distribution_from_run(run, time_index=0, grid="hottail")
    run.close()


@pytest.mark.skipif(not os.path.exists(SAMPLE), reason="output_7_.h5 not present")
def test_distribution_from_run_is_per_electron():
    import cdo
    from cdo.derived import volume_average

    with cdo.Run([SAMPLE]) as run:
        data = fitting.distribution_from_run(run, time_ms="end")
        assert data.time_index == run.timegrid_length - 1
        assert data.content().sum() == pytest.approx(1.0, rel=1e-12)
        assert 0.0 <= data.xi_pass < 1.0
        # Its pitch mean is the per-electron curve of the favg_re_ravg figures.
        favg = run.angle_average("runaway", "distribution")[data.time_index]  # (r, p)
        favg = volume_average(favg.T, run.cell_volumes)
        favg /= np.nansum(favg * 4.0 * np.pi * data.p**2 * data.dp)
        pitch_mean = np.sum(data.f * data.dxi[:, None] / 2.0, axis=0)
        np.testing.assert_allclose(pitch_mean, favg, rtol=1e-10, atol=1e-14 * favg.max())
        same = fitting.distribution_from_run(run, time_index=-1)
        np.testing.assert_array_equal(same.f, data.f)
        with pytest.raises(ValueError, match="one of"):
            fitting.distribution_from_run(run)
