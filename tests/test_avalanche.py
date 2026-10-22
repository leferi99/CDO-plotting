"""Analytic avalanche distribution: conventions, normalisation, overlays."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy.integrate import quad  # noqa: E402

from cdo import avalanche, plotting  # noqa: E402
from cdo.avalanche import AvalancheParams  # noqa: E402
from cdo.moments import MomentumMoments  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


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


# --- overlays on a stub run --------------------------------------------------------


class _StubReference:
    def __init__(self, arrays):
        self.arrays = arrays

    def read(self, name):
        return self.arrays.get(name)


class _StubRun:
    """Two times, one radius, an isotropic hot-tail grid and a 4-cell runaway grid."""

    def __init__(self, n_re=(0.0, 1e16)):
        self.timegrid_ms = np.array([0.0, 1.0])
        self.radialgrid = np.array([0.1])
        self.minor_radius = 0.4
        self.derived_cache = {}
        xi_edges = np.array([-1.0, 0.0, 0.9, 0.99, 1.0])
        xi = 0.5 * (xi_edges[1:] + xi_edges[:-1])
        self.p = np.array([5.0, 20.0, 60.0])
        self.reference = _StubReference({
            "hot_p": np.array([0.5, 1.5]), "hot_p_edges": np.array([0.0, 1.0, 2.0]),
            "hot_xi": np.array([0.0]),
            "re_p": self.p, "re_xi": xi, "re_xi_edges": xi_edges,
        })
        self._mm = MomentumMoments(
            p=self.p, xi=xi, xi_edges=xi_edges, dxi=np.diff(xi_edges),
            Vprime_VpVol=np.broadcast_to(2 * np.pi * self.p**2, (1, 4, 3)).copy(),
            xi0_trapped=np.zeros(1))
        self._fields = {
            "n_re": np.array(n_re)[:, None],
            "f_re": np.ones((2, 1, 4, 3)),
            "E_field": np.array([[0.5], [-3.0]]),
            "Ectot": np.array([[0.25], [0.25]]),
            "Zeff": np.array([[2.0], [3.0]]),
            "lnLambdaC": np.array([[15.0], [16.0]]),
        }

    def field(self, name):
        return self._fields.get(name)

    def moments(self, grid):
        return self._mm if grid == "runaway" else None

    def angle_average(self, grid, moment="distribution"):
        n = 2 if grid == "hottail" else 3
        return np.ones((2, 1, n))


def test_density_from_run_or_override():
    run = _StubRun()
    assert avalanche.density_at(run, AvalancheParams(), 1, 0) == 1e16
    assert avalanche.density_at(run, AvalancheParams(n_re=5.0), 1, 0) == 5.0


@pytest.mark.parametrize("moment", ["distribution", "density"])
def test_combined_plot_adds_analytic_on_runaway_momenta(moment):
    run = _StubRun()
    fig = plotting.angle_averaged_combined_1D(run, moment, times_ms=[0, 1],
                                              analytic=AvalancheParams())
    ax = fig.axes[0]
    dashed = [line for line in ax.lines
              if line.get_linestyle() == "--" and len(set(line.get_xdata())) > 1]  # not the boundary
    # n_re = 0 at t = 0, so only the second time gets an analytic curve.
    assert len(dashed) == 1
    np.testing.assert_array_equal(dashed[0].get_xdata(), run.p)
    params = AvalancheParams()
    gamma = np.sqrt(1 + run.p**2)
    if moment == "density":
        expected = 1e16 * np.exp(-gamma / params.gamma0) / params.gamma0
    else:
        expected = 1e16 * np.exp(-gamma / params.gamma0) / (4 * np.pi * params.gamma0 * run.p**2)
    np.testing.assert_allclose(dashed[0].get_ydata(), expected, rtol=1e-10)
    assert "set values: $E/E_c$ = 50" in _box_text(ax)


def test_pitch_lines_one_per_cell_with_dashed_analytic():
    run = _StubRun()
    fig = plotting.pitch_resolved_1D(run, times_ms=[0, 1], pitch_cells=[-1, -2],
                                     analytic=AvalancheParams())
    first, second = fig.axes[:2]
    assert [line.get_linestyle() for line in first.lines] == ["-", "-"]
    assert [line.get_linestyle() for line in second.lines] == ["-", "-", "--", "--"]
    assert "0.995" in fig.legends[0].get_texts()[1].get_text()


def test_pitch_lines_need_pitch_resolution():
    with pytest.raises(ValueError):
        plotting.pitch_resolved_1D(_StubRun(), "hottail")


def test_hottail_pitch_lines_get_no_analytic_curve():
    run = _StubRun()
    run.reference.arrays.update({"hot_xi": np.array([-0.5, 0.5])})
    run._fields["f_hot"] = np.ones((2, 1, 2, 2))
    fig = plotting.pitch_resolved_1D(run, "hottail", times_ms=["end"],
                                     analytic=AvalancheParams())
    assert all(line.get_linestyle() == "-" for line in fig.axes[0].lines)


def _box_text(ax):
    from matplotlib.offsetbox import AnchoredText
    return "\n".join(a.txt._text.get_text() for a in ax.artists if isinstance(a, AnchoredText))


def test_resolve_takes_run_values_with_the_field_magnitude():
    run = _StubRun()
    resolved = AvalancheParams.from_run().resolve(run, 1, 0)
    assert resolved.from_run == {"E_over_Ec": 12.0, "Z_tot": 3.0, "lnLambda": 16.0}
    assert resolved.params.gamma0 == pytest.approx(np.sqrt(8.0) * 16.0)
    assert resolved.n_re == 1e16 and resolved.drawable
    assert "12" in resolved.text and "1e+16" in resolved.text
    mixed = AvalancheParams(Z_tot=None).resolve(run, 0, 0)
    assert mixed.from_run == {"Z_tot": 2.0} and mixed.params.E_over_Ec == 50.0
    assert AvalancheParams(Z_tot=None).kind == "run and set values"
    with pytest.raises(ValueError):
        AvalancheParams.from_run().gamma0


def test_missing_run_field_gives_no_curve_and_says_so():
    run = _StubRun()
    del run._fields["Zeff"]
    resolved = AvalancheParams.from_run().resolve(run, 1, 0)
    assert not resolved.drawable and "Zeff" in resolved.text
    fig = plotting.angle_averaged_combined_1D(
        run, times_ms=["end"], analytic=[AvalancheParams(), AvalancheParams.from_run()])
    ax = fig.axes[0]
    styles = [line.get_linestyle() for line in ax.lines if len(set(line.get_xdata())) > 1]
    assert styles == ["-", "--"]
    assert "run has no Zeff" in _box_text(ax)


def test_single_time_combined_plot_uses_dark_data_and_two_analytic_styles():
    run = _StubRun()
    fig = plotting.angle_averaged_combined_1D(
        run, times_ms=["end"], analytic=[AvalancheParams(), AvalancheParams.from_run()])
    ax = fig.axes[0]
    curves = [line for line in ax.lines if len(set(line.get_xdata())) > 1]
    assert [line.get_linestyle() for line in curves] == ["-", "--", ":"]
    data_colour = matplotlib.colors.to_rgba(curves[0].get_color())
    assert data_colour == plt.cm.viridis(plotting.SINGLE_TIME_SHADE)
    assert all(matplotlib.colors.to_rgba(line.get_color()) == data_colour for line in curves)
    text = _box_text(ax)
    assert "set values" in text and "run values, t = 1 ms" in text and "12" in text


def test_pick_pitch_cells_spreads_from_the_beam():
    xi = np.cos(np.radians([60, 40, 21.5, 15.8, 10.1, 7.3, 4.5, 2.0]))
    assert list(plotting.pick_pitch_cells(xi, 3)) == [2, 4, 7]
    assert list(plotting.pick_pitch_cells(-xi, 3, direction=-1)) == [2, 4, 7]
    assert list(plotting.pick_pitch_cells(xi, 1)) == [7]
    assert list(plotting.pick_pitch_cells(xi, [-1, 0])) == [0, 7]
    assert len(plotting.pick_pitch_cells(xi, "all")) == 8
    assert list(plotting.pick_pitch_cells(xi, 30, max_angle_deg=3)) == [7]  # deduplicated


def test_pitch_panels_get_run_values_and_one_style_per_analytic_entry():
    run = _StubRun()
    fig = plotting.pitch_resolved_1D(run, times_ms=["end"], pitch_cells=2,
                                     analytic=[AvalancheParams(), AvalancheParams.from_run()])
    ax = fig.axes[0]
    assert [line.get_linestyle() for line in ax.lines] == ["-", "-", "--", "--", ":", ":"]
    colours = [matplotlib.colors.to_rgba(line.get_color()) for line in ax.lines]
    assert colours[0:2] == colours[2:4] == colours[4:6]  # analytic in the data colour per cell
    assert colours[0] != colours[1]
    assert "run values" in _box_text(ax)
    assert "set values" in fig._suptitle.get_text()
