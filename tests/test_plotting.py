"""Checks for the plotting helpers.

The figure builders are exercised on a non-interactive backend to confirm they
run, save, and leave no figures open, without asserting on pixels. The
time-index helpers are pure logic and are checked in full.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cdo import plotting


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_basic_1D_returns_a_figure_without_saving():
    fig = plotting.basic_1D([np.arange(5)], np.arange(5))
    assert fig is not None
    assert plt.fignum_exists(fig.number)


def test_basic_1D_saves_and_closes(tmp_path):
    fig = plotting.basic_1D(
        [np.arange(5), np.arange(5) ** 2],
        np.arange(5),
        labels=["a", "b"],
        legendloc="upper left",
        folder=str(tmp_path),
        savename="lines",
    )
    assert (tmp_path / "lines.png").exists()
    assert not plt.fignum_exists(fig.number)


def test_basic_1D_suffix_appends_to_filename(tmp_path):
    plotting.basic_1D(
        [np.arange(3)], np.arange(3),
        folder=str(tmp_path), savename="fig", suffix="_zoom",
    )
    assert (tmp_path / "fig_zoom.png").exists()


@pytest.mark.parametrize("normalization", ["lin", "log", "symlog"])
def test_basic_2D_runs_for_each_normalization(normalization):
    x = np.linspace(0, 1, 6)
    y = np.linspace(0, 2, 5)
    data = np.abs(np.outer(y, x)) + 1.0
    if normalization == "symlog":
        data = np.outer(y - 1, x - 0.5)
    fig = plotting.basic_2D(data, x, y, normalization=normalization)
    assert fig is not None


@pytest.mark.parametrize("normalization", ["lin", "log", "symlog"])
def test_basic_2D_mesh_runs_for_each_normalization(normalization):
    x = np.linspace(0, 1, 6)
    y = np.geomspace(1e-3, 2, 5)
    data = np.abs(np.outer(y, x)) + 1.0
    if normalization == "symlog":
        data = np.outer(y - 1, x - 0.5)
    fig = plotting.basic_2D(data, x, y, normalization=normalization, mesh=True,
                            linthresh=1e-3, yscale="log")
    assert fig is not None


def test_symlog_ticks_are_symmetric_and_capped():
    ticks = plotting.symlog_ticks(1e3, 1e20, per_sign=4)
    assert 0.0 in ticks
    assert np.allclose(ticks, -ticks[::-1])
    assert len(ticks) <= 2 * 4 + 1


def test_basic_2D_does_not_mutate_input():
    x = np.linspace(0, 1, 4)
    y = np.linspace(0, 1, 3)
    data = np.ones((3, 4))
    data[0, 0] = np.inf
    before = data.copy()
    plotting.basic_2D(data, x, y)
    assert np.array_equal(data, before, equal_nan=True)


def test_basic_2D_rejects_unknown_normalization():
    with pytest.raises(ValueError, match="normalization"):
        plotting.basic_2D(np.ones((3, 3)), np.arange(3), np.arange(3),
                          normalization="bogus")


def test_critical_field_ratio_2D_runs_and_saves(tmp_path):
    r = np.linspace(0, 1, 6)
    t = np.linspace(0, 2, 5)
    E = np.outer(np.linspace(0.1, 0.3, 5), np.ones(6))
    Eceff = np.full((5, 6), 0.2)  # ratio crosses one across the grid
    fig = plotting.critical_field_ratio_2D(
        E, Eceff, r, t, folder=str(tmp_path), savename="ratio")
    assert (tmp_path / "ratio.png").exists()
    assert not plt.fignum_exists(fig.number)


def test_critical_field_ratio_2D_tolerates_nonfinite_ratio():
    r = np.linspace(0, 1, 4)
    t = np.linspace(0, 1, 3)
    E = np.ones((3, 4))
    Eceff = np.ones((3, 4))
    Eceff[0, 0] = 0.0  # division by zero -> masked, not a crash
    fig = plotting.critical_field_ratio_2D(E, Eceff, r, t)
    assert fig is not None


def test_index_array_returns_exact_index_for_on_grid_times():
    timegrid = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert list(plotting.index_array(timegrid, [1.0, 3.0, 4.0])) == [1, 3, 4]


def test_index_array_rounds_up_for_off_grid_times():
    timegrid = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert list(plotting.index_array(timegrid, [2.5])) == [3]


def test_indices_every_spaces_frames_and_stays_in_bounds():
    timegrid = np.linspace(0.0, 10.0, 101)  # 0.1 spacing
    frames = plotting.indices_every(timegrid, step=2.0)
    assert frames[0] == 0
    assert frames[-1] <= len(timegrid) - 1
    # About every 2.0 in time, so grid steps of ~20.
    picked = timegrid[frames]
    gaps = np.diff(picked)
    assert np.allclose(gaps, 2.0, atol=0.1)


def test_indices_every_is_strictly_increasing():
    timegrid = np.linspace(0.0, 1.0, 11)
    frames = plotting.indices_every(timegrid, step=0.25)
    assert all(b > a for a, b in zip(frames, frames[1:]))


def test_indices_every_terminates_on_fine_step():
    timegrid = np.linspace(0.0, 1.0, 5)
    frames = plotting.indices_every(timegrid, step=1e-6)
    assert frames[-1] == len(timegrid) - 1


def test_time_marks_medium_run_uses_both_ladders_and_closes_at_the_end():
    timegrid = np.arange(0.2, 70.01, 0.2)
    frames = plotting.time_marks(timegrid)
    assert [round(timegrid[i], 1) for i in frames] == [0.2, 1.0, 10.0, 25.0, 50.0, 70.0]
    lines = plotting.time_marks(timegrid, lines=True)
    assert [round(timegrid[i], 1) for i in lines] == [
        0.2, 1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 70.0]


def test_time_marks_long_run_line_ladder():
    timegrid = np.linspace(0.1, 300.0, 3000)
    lines = plotting.time_marks(timegrid, lines=True)
    assert [round(timegrid[i]) for i in lines] == [
        0, 1, 10, 20, 30, 50, 100, 200, 300]


def test_time_marks_snap_to_the_nearest_step():
    timegrid = np.array([0.001, 0.12, 0.45, 0.7, 1.0])
    frames = plotting.time_marks(timegrid, lines=True)
    # Each mark takes the nearest step: 0.1 the 0.12 one, and 0.5 the earlier
    # 0.45 one, not the next step after it.
    assert [timegrid[i] for i in frames] == [0.001, 0.12, 0.45, 1.0]


def test_time_label_shows_the_first_step_as_zero():
    timegrid = np.array([1e-6, 0.5, 10.0])
    assert plotting.time_label(timegrid, 0) == "0 ms"
    assert plotting.time_label(timegrid, 2) == "10 ms"


def test_time_marks_short_run_resolves_sub_millisecond_steps():
    timegrid = np.linspace(0.001, 4.0, 400)
    frames = plotting.time_marks(timegrid)
    assert frames[0] == 0
    assert frames[-1] == len(timegrid) - 1
    assert len(frames) == len(set(frames)) == 7


class _StubReference:
    def __init__(self, arrays):
        self.arrays = arrays

    def read(self, name):
        return self.arrays[name]


class _StubRun:
    """Two time steps, two radii, a hot-tail grid up to p = 2 and a runaway grid above."""

    def __init__(self):
        self.timegrid_ms = np.array([0.0, 1.0])
        self.radialgrid = np.array([0.1, 0.3])
        self.minor_radius = 0.4
        self.derived_cache = {}
        self.reference = _StubReference({
            "hot_p": np.array([0.5, 1.5]),
            "hot_p_edges": np.array([0.0, 1.0, 2.0]),
            "re_p": np.array([3.0, 5.0, 7.0]),
        })

    def angle_average(self, grid, moment="distribution"):
        n = 2 if grid == "hottail" else 3
        values = np.ones((2, 2, n))
        values[..., 0] = 0.0  # a non-positive cell the log axis must tolerate
        return values


@pytest.mark.parametrize("moment", ["distribution", "density"])
def test_angle_averaged_combined_1D_joins_grids_and_marks_boundary(moment):
    fig = plotting.angle_averaged_combined_1D(_StubRun(), moment, times_ms=[0, 0.9, 1])
    ax = fig.axes[0]
    curves = [line for line in ax.lines if line.get_linestyle() == "-"]
    assert len(curves) == 2  # 0.9 and 1 ms snap to the same step
    np.testing.assert_array_equal(curves[0].get_xdata(), [0.5, 1.5, 3.0, 5.0, 7.0])
    boundary = [line for line in ax.lines if line.get_linestyle() == "--"]
    assert len(boundary) == 1 and boundary[0].get_xdata()[0] == 2.0


def test_angle_averaged_combined_1D_snaps_radius_and_trims_momentum():
    fig = plotting.angle_averaged_combined_1D(_StubRun(), times_ms=["end"], r_over_a=0.7,
                                              p_range=(2.5, None))
    ax = fig.axes[0]
    assert "r/a = 0.750" in ax.get_title()
    curves = [line for line in ax.lines if line.get_linestyle() == "-"]
    np.testing.assert_array_equal(curves[0].get_xdata(), [3.0, 5.0, 7.0])
    assert not [line for line in ax.lines if line.get_linestyle() == "--"]


def test_angle_averaged_combined_1D_rejects_unknown_moment():
    with pytest.raises(ValueError):
        plotting.angle_averaged_combined_1D(_StubRun(), "current")
