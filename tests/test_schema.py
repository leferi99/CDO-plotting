"""Checks for the field resolver against synthetic files in both DREAM layouts.

Real outputs from the older DREAM versions are not available in this
environment, so the old layout is reproduced here from the paths the retired
``CDOconcat`` module read. The point of these checks is that one canonical name
reaches the right dataset in either layout, and that an absent dataset is
classified by the reason it is absent.
"""

import warnings

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from cdo.schema import (
    DuplicatePathWarning,
    MissingFieldError,
    Presence,
    Resolver,
    decode_string_list,
    survey,
)

NT, NR, NION = 5, 4, 3


def _string_list(*items):
    """Encode as DREAM does: one dataset of single characters, ';' separated."""
    text = "".join(f"{item};" for item in items)
    return np.array([c.encode() for c in text], dtype="S1")


def _common(f, hottail=True, runaway=True, include=("fluid", "scalar")):
    f["grid/t"] = np.linspace(0, 1e-3, NT)
    f["grid/r"] = np.linspace(0.1, 1.6, NR)
    f["grid/r_f"] = np.linspace(0.0, 1.8, NR + 1)
    f["grid/dr"] = np.full(NR, 0.45)
    f["grid/VpVol"] = np.linspace(5.0, 50.0, NR)
    f["settings/hottailgrid/enabled"] = np.array([int(hottail)])
    f["settings/runawaygrid/enabled"] = np.array([int(runaway)])
    f["settings/other/include"] = _string_list(*include)
    f["ionmeta/Z"] = np.array([1, 18])
    f["ionmeta/names"] = _string_list("D", "Ar")

    for name in ("E_field", "T_cold", "W_cold", "j_ohm", "j_re", "j_tot",
                 "n_cold", "n_re", "n_tot"):
        f[f"eqsys/{name}"] = np.ones((NT, NR))
    f["eqsys/I_p"] = np.ones((NT, 1))
    f["eqsys/n_i"] = np.ones((NT, NION, NR))

    if hottail:
        f["grid/hottail/p1"] = np.linspace(0, 1, 8)
        f["grid/hottail/p2"] = np.linspace(-1, 1, 6)
        f["eqsys/f_hot"] = np.ones((NT, NR, 6, 8))
        f["eqsys/j_hot"] = np.ones((NT, NR))
        f["eqsys/n_hot"] = np.ones((NT, NR))
    if runaway:
        f["grid/runaway/p1"] = np.linspace(0, 10, 9)
        f["grid/runaway/p2"] = np.linspace(-1, 1, 6)
        f["eqsys/f_re"] = np.ones((NT, NR, 6, 9))


def old_layout(path, **kwargs):
    """A file as the DREAM versions the notebooks were written against wrote it."""
    with h5py.File(path, "w") as f:
        _common(f, **kwargs)
        f["settings/radialgrid/R0"] = np.array([6.2])
        f["settings/radialgrid/a"] = np.array([1.8])
        f["eqsys/I_wall"] = np.ones((NT, 1))
        f["eqsys/V_loop_w"] = np.ones((NT, 1))
        f["other/fluid/gammaHottail"] = np.ones((NT - 1, NR))
        f["other/fluid/Tcold_ohmic"] = np.ones((NT - 1, NR))
    return path


def new_layout(path, **kwargs):
    """A file as the checkouts under DREAM-runs write it."""
    with h5py.File(path, "w") as f:
        _common(f, **kwargs)
        f["grid/R0"] = np.array([6.2])
        f["grid/a"] = np.array([1.8])
        f["other/fluid/gammaFhot"] = np.ones((NT - 1, NR))
    return path


@pytest.fixture
def old(tmp_path):
    return old_layout(tmp_path / "old.h5")


@pytest.fixture
def new(tmp_path):
    return new_layout(tmp_path / "new.h5")


def open_resolver(path):
    return Resolver(h5py.File(path, "r"), name=path.name)


def test_major_radius_reachable_in_both_layouts(old, new):
    for path, expected_source in (
        (old, "settings/radialgrid/R0"),
        (new, "grid/R0"),
    ):
        r = open_resolver(path)
        assert r.read("major_radius")[0] == pytest.approx(6.2)
        assert r.resolved_path["major_radius"] == expected_source


def test_rename_is_followed(old, new):
    for path, expected_source in (
        (old, "other/fluid/gammaHottail"),
        (new, "other/fluid/gammaFhot"),
    ):
        r = open_resolver(path)
        assert r.read("gammaHottail").shape == (NT - 1, NR)
        assert r.resolved_path["gammaHottail"] == expected_source


def test_relocated_reports_non_primary_paths(old):
    r = open_resolver(old)
    r.read("major_radius")
    r.read("gammaHottail")
    r.read("minor_radius")
    assert r.relocated() == {
        "major_radius": "settings/radialgrid/R0",
        "minor_radius": "settings/radialgrid/a",
        "gammaHottail": "other/fluid/gammaHottail",
    }


def test_disabled_grid_is_gated_not_missing(tmp_path):
    r = open_resolver(new_layout(tmp_path / "nohot.h5", hottail=False))
    assert r.read("f_hot") is None
    assert r.presence["f_hot"] is Presence.GATED
    assert not r.expected("f_hot")
    assert "hottail grid disabled" in r.report()


def test_excluded_output_group_is_gated(tmp_path):
    r = open_resolver(new_layout(tmp_path / "nofluid.h5", include=("scalar",)))
    assert r.read("Eceff") is None
    assert r.presence["Eceff"] is Presence.GATED


def test_included_group_but_absent_quantity_is_not_produced(new):
    r = open_resolver(new)
    assert r.read("Tcold_ohmic") is None
    assert r.presence["Tcold_ohmic"] is Presence.NOT_PRODUCED
    assert not r.expected("Tcold_ohmic")


def test_absent_optional_eqsys_field_is_missing(new):
    r = open_resolver(new)
    assert r.read("I_wall") is None
    assert r.presence["I_wall"] is Presence.MISSING


def test_absent_required_field_raises(tmp_path):
    path = tmp_path / "broken.h5"
    new_layout(path)
    with h5py.File(path, "a") as f:
        del f["eqsys/n_re"]
    with pytest.raises(MissingFieldError, match="n_re"):
        open_resolver(path).read("n_re")


def test_grid_flags_and_groups_are_read(old):
    r = open_resolver(old)
    assert r.grids == {"hottail": True, "runaway": True}
    assert r.groups == {"fluid", "scalar"}


def test_string_lists_are_decoded(new):
    r = open_resolver(new)
    assert r.read_string_list("ion_names") == ["D", "Ar"]
    assert r.read_string_list("other_include") == ["fluid", "scalar"]


def test_decode_string_list_drops_trailing_separator():
    assert decode_string_list(_string_list("D", "T")) == ["D", "T"]


def test_group_paths_do_not_count_as_a_match(tmp_path):
    path = tmp_path / "group.h5"
    new_layout(path)
    with h5py.File(path, "a") as f:
        f.create_group("grid/geometry")  # a group, as in real outputs
    r = open_resolver(path)
    assert r.locate("B_min") is None
    assert r.read("B_min") is None


def test_duplicated_paths_that_agree_are_silent(tmp_path):
    """Old files carry both grid/R0 and settings/radialgrid/R0 with equal values."""
    path = tmp_path / "dup_ok.h5"
    new_layout(path)
    with h5py.File(path, "a") as f:
        f["settings/radialgrid/R0"] = np.array([6.2])
    r = open_resolver(path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert r.read("major_radius")[0] == pytest.approx(6.2)
    assert r.conflicts == {}


def test_duplicated_paths_that_disagree_warn(tmp_path):
    path = tmp_path / "dup_bad.h5"
    new_layout(path)
    with h5py.File(path, "a") as f:
        f["settings/radialgrid/R0"] = np.array([9.07])
    r = open_resolver(path)
    with pytest.warns(DuplicatePathWarning, match="major_radius"):
        value = r.read("major_radius")
    assert value[0] == pytest.approx(6.2)  # the preferred path wins
    assert r.conflicts["major_radius"] == ["grid/R0", "settings/radialgrid/R0"]
    assert "duplicated paths that disagree" in r.report()


def test_single_path_present_never_warns(new):
    r = open_resolver(new)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r.read("major_radius")
        r.read("gammaHottail")
    assert r.conflicts == {}


def test_unknown_field_names_are_rejected(new):
    with pytest.raises(KeyError, match="not a known field"):
        open_resolver(new).read("no_such_quantity")


def test_survey_classifies_every_field(new):
    r = survey(h5py.File(new, "r"), name="new")
    from cdo.schema import FIELDS

    assert set(r.presence) == set(FIELDS)
    assert "f_re" in r.presence and r.presence["f_re"] is Presence.PRESENT
