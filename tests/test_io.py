"""Checks for output file discovery."""

import pytest

from cdo.io import find_outputs, natural_key, newest_output

NAMES = [
    "output_0000.h5",
    "output_init_0.h5",
    "output_init_1.h5",
    "settings_0000.h5",
    "ITER_DThmode24_00400_LUKE.h5",
    "manifest.json",
]


@pytest.fixture
def run_folder(tmp_path):
    for name in NAMES:
        (tmp_path / name).touch()
    return tmp_path


def test_initialisation_files_are_excluded_by_default(run_folder):
    assert [p.name for p in find_outputs(run_folder)] == ["output_0000.h5"]


def test_initialisation_files_can_be_asked_for(run_folder):
    assert [p.name for p in find_outputs(run_folder, init=True)] == [
        "output_0000.h5",
        "output_init_0.h5",
        "output_init_1.h5",
    ]


def test_settings_and_unrelated_files_are_never_returned(run_folder):
    names = [p.name for p in find_outputs(run_folder, init=True)]
    assert not any(n.startswith("settings") for n in names)
    assert "ITER_DThmode24_00400_LUKE.h5" not in names
    assert "manifest.json" not in names


def test_numeric_ordering_beats_lexicographic(tmp_path):
    for i in (1, 2, 7, 10, 11):
        (tmp_path / f"output_{i}_.h5").touch()
    assert [p.name for p in find_outputs(tmp_path)] == [
        "output_1_.h5",
        "output_2_.h5",
        "output_7_.h5",
        "output_10_.h5",
        "output_11_.h5",
    ]


def test_newest_output_is_the_last_in_time_order(tmp_path):
    for i in (1, 2, 10):
        (tmp_path / f"output_{i}_.h5").touch()
    assert newest_output(tmp_path).name == "output_10_.h5"


def test_empty_folder_yields_no_files_and_no_newest(tmp_path):
    assert find_outputs(tmp_path) == []
    assert newest_output(tmp_path) is None


def test_missing_folder_raises(tmp_path):
    with pytest.raises(NotADirectoryError):
        find_outputs(tmp_path / "absent")


def test_prefix_selects_one_run_within_a_folder(tmp_path):
    (tmp_path / "output_a_1_.h5").touch()
    (tmp_path / "output_b_1_.h5").touch()
    assert [p.name for p in find_outputs(tmp_path, prefix="output_a")] == [
        "output_a_1_.h5"
    ]


def test_natural_key_orders_numbers_numerically():
    assert natural_key("output_2_.h5") < natural_key("output_10_.h5")
