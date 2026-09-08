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
