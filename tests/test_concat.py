"""Checks for multi-file concatenation."""

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from cdo.concat import InconsistentRunWarning, Run

NR = 3


def _string_list(*items):
    text = "".join(f"{item};" for item in items)
    return np.array([c.encode() for c in text], dtype="S1")


def write_file(path, nt, t0_index, *, i_wall=True, nr=NR, seed=0, runaway=None, xi0=None):
    """One output file with ``nt`` time steps starting the local grid at zero.

    ``t0_index`` seeds the data so concatenated series can be checked for order.
    ``runaway`` = (f_re, xi_edges, p_edges) adds a runaway grid, f_re of shape
    (nt, nr, xi, p), with V'/VpVol = 2 pi p^2. ``xi0`` sets the trapped-passing
    boundary per radius.
    """
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        f["grid/t"] = np.linspace(0, (nt - 1) * 1e-4, nt)
        f["grid/r"] = np.linspace(0.1, 1.5, nr)
        f["grid/r_f"] = np.linspace(0.0, 1.6, nr + 1)
        f["grid/dr"] = np.full(nr, 0.5)
        f["grid/R0"] = np.array([6.0])
        f["grid/a"] = np.array([1.6])
        f["grid/VpVol"] = np.array([2.0, 3.0, 4.0])[:nr]
        f["grid/geometry/GR0"] = np.full(nr, 2.0)
        f["grid/geometry/Bmin"] = np.full(nr, 1.0)
        f["grid/geometry/FSA_R02OverR2"] = np.full(nr, 1.0)
        f["settings/hottailgrid/enabled"] = np.array([0])
        f["settings/runawaygrid/enabled"] = np.array([0 if runaway is None else 1])
        if xi0 is not None:
            f["grid/geometry/xi0TrappedBoundary"] = np.asarray(xi0, dtype=float)
        if runaway is not None:
            f_re, xi_edges, p_edges = (np.asarray(a, dtype=float) for a in runaway)
            p = 0.5 * (p_edges[:-1] + p_edges[1:])
            xi = 0.5 * (xi_edges[:-1] + xi_edges[1:])
            f["grid/runaway/p1"] = p
            f["grid/runaway/p1_f"] = p_edges
            f["grid/runaway/p2"] = xi
            f["grid/runaway/p2_f"] = xi_edges
            f["grid/runaway/dp1"] = np.diff(p_edges)
            f["grid/runaway/dp2"] = np.diff(xi_edges)
            vpvol = np.array([2.0, 3.0, 4.0])[:nr]
            f["grid/runaway/Vprime"] = (2 * np.pi * p**2)[None, None, :] * vpvol[:, None, None] \
                * np.ones((nr, xi.size, p.size))
            f["eqsys/f_re"] = f_re
        f["settings/other/include"] = _string_list("fluid")
        f["ionmeta/Z"] = np.array([1, 18])
        f["ionmeta/names"] = _string_list("D", "Ar")

        # A per-time marker: value at step k is t0_index + k.
        marker = (t0_index + np.arange(nt)).astype(float)
        f["eqsys/T_cold"] = np.repeat(marker[:, None], nr, axis=1)
        f["eqsys/I_p"] = marker[:, None]
        for name in ("E_field", "W_cold", "j_ohm", "j_re", "j_tot",
                     "n_cold", "n_re", "n_tot"):
            f[f"eqsys/{name}"] = rng.random((nt, nr))
        f["eqsys/n_i"] = rng.random((nt, 21, nr))  # D:2 + Ar:19 rows
        if i_wall:
            f["eqsys/I_wall"] = marker[:, None]


def test_two_files_concatenate_dropping_the_shared_initial_step(tmp_path):
    write_file(tmp_path / "output_0.h5", nt=3, t0_index=0)
    write_file(tmp_path / "output_1.h5", nt=3, t0_index=10)
    run = Run([tmp_path / "output_0.h5", tmp_path / "output_1.h5"])

    # Each file drops its first step: 2 + 2 = 4.
    assert run.timegrid_length == 4
    # T_cold markers: file0 steps 1,2 then file1 steps 1,2 -> [1, 2, 11, 12].
    assert list(run.field("T_cold")[:, 0]) == [1.0, 2.0, 11.0, 12.0]
    assert np.all(np.diff(run.timegrid) > 0)
    run.close()


def test_plasma_current_is_flattened(tmp_path):
    write_file(tmp_path / "output_0.h5", nt=2, t0_index=0)
    run = Run([tmp_path / "output_0.h5"])
    assert run.plasma_current.shape == (1,)
    run.close()


def test_field_absent_from_one_file_is_nan_filled_and_warns(tmp_path):
    write_file(tmp_path / "output_0.h5", nt=2, t0_index=0, i_wall=True)
    write_file(tmp_path / "output_1.h5", nt=2, t0_index=10, i_wall=False)
    run = Run([tmp_path / "output_0.h5", tmp_path / "output_1.h5"])
    with pytest.warns(InconsistentRunWarning, match="I_wall"):
        i_wall = run.field("I_wall")
    assert not np.isnan(i_wall[0]).any()
    assert np.isnan(i_wall[1]).any()
    run.close()


def test_differing_radial_grid_is_reported(tmp_path):
    write_file(tmp_path / "output_0.h5", nt=2, t0_index=0, nr=3)
    write_file(tmp_path / "output_1.h5", nt=2, t0_index=10, nr=4)
    with pytest.warns(InconsistentRunWarning, match="radial grid"):
        run = Run([tmp_path / "output_0.h5", tmp_path / "output_1.h5"])
    assert run.problems
    run.close()


def test_ions_addressed_by_name(tmp_path):
    write_file(tmp_path / "output_0.h5", nt=2, t0_index=0)
    run = Run([tmp_path / "output_0.h5"])
    assert [s.name for s in run.ions] == ["D", "Ar"]
    assert run.ion("Ar").Z == 18
    assert run.ion("Ar").rows.size == 19
    with pytest.raises(KeyError, match="no ion species"):
        run.ion("Ne")
    run.close()


def test_current_matches_direct_formula(tmp_path):
    write_file(tmp_path / "output_0.h5", nt=3, t0_index=0)
    run = Run([tmp_path / "output_0.h5"])
    j = run.field("j_re")
    # The current weight is VpVol*dr times the flux-surface factor GR0/Bmin*FSA,
    # with no R0. This is DREAM's convention and differs from cell_volumes.
    weight = np.array([2.0, 3.0, 4.0]) * 0.5 * (2.0 / 1.0 * 1.0)
    expected = (j * weight).sum(axis=-1) / (2 * np.pi)
    assert run.current("j_re") == pytest.approx(expected)
    run.close()


def test_empty_file_list_is_rejected():
    with pytest.raises(ValueError):
        Run([])


def _two_legs(tmp_path, *, ectot_in_second=True):
    """Two files of 3 and 4 steps, with Ectot (an other/ field, one step fewer)."""
    files = []
    for k, nt in enumerate((3, 4)):
        path = tmp_path / f"output_{k}.h5"
        write_file(path, nt=nt, t0_index=10 * k, seed=k)
        if k == 0 or ectot_in_second:
            with h5py.File(path, "a") as f:
                f["other/fluid/Ectot"] = 100.0 * k + np.arange(nt - 1)[:, None] * np.ones(NR)
        files.append(path)
    return files


@pytest.mark.parametrize("name", ["T_cold", "n_re", "Ectot"])
def test_field_at_equals_every_step_of_the_series(tmp_path, name):
    files = _two_legs(tmp_path)
    fresh = Run(files)  # never reads the whole series
    whole = Run(files)
    series = whole.field(name)
    n = fresh.timegrid_length
    assert n == 2 + 3
    for i in range(-n, n):
        np.testing.assert_array_equal(fresh.field_at(name, i), series[i])
        np.testing.assert_array_equal(whole.field_at(name, i), series[i])
    if name == "T_cold":
        # Markers: file 0 steps 1, 2, then file 1 steps 1, 2, 3.
        assert fresh.field_at(name, 0)[0] == 1.0
        assert fresh.field_at(name, 1)[0] == 2.0  # last step of file 0
        assert fresh.field_at(name, 2)[0] == 11.0  # first step of file 1
        assert fresh.field_at(name, -1)[0] == 13.0
    if name == "Ectot":
        assert fresh.field_at(name, 1)[0] == 1.0
        assert fresh.field_at(name, 2)[0] == 100.0
    for i in (n, -n - 1):
        with pytest.raises(IndexError):
            fresh.field_at(name, i)
    fresh.close()
    whole.close()


def test_field_at_of_absent_and_partial_fields(tmp_path):
    files = _two_legs(tmp_path, ectot_in_second=False)
    fresh = Run(files)
    assert fresh.field_at("f_re", 0) is None
    with pytest.warns(InconsistentRunWarning, match="Ectot"):
        missing = fresh.field_at("Ectot", -1)
    assert missing.shape == (NR,) and np.isnan(missing).all()
    np.testing.assert_array_equal(fresh.field_at("radius", 0), fresh.radialgrid)
    with pytest.raises(KeyError, match="not a known field"):
        fresh.field_at("no_such_field", 0)
    # The same answers once the whole series is cached.
    with pytest.warns(InconsistentRunWarning):
        fresh.field("Ectot")
    assert fresh.field("f_re") is None
    assert fresh.field_at("f_re", 0) is None
    assert np.isnan(fresh.field_at("Ectot", -1)).all()
    fresh.close()

