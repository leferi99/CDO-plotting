"""Checks for the language label sets."""

import pytest

from cdo.labels import EN, HU, LANGUAGES, active


def test_axis_labels_differ_by_language():
    assert EN.time == "Time [ms]"
    assert HU.time == "Idő [ms]"
    assert EN.radius != HU.radius


def test_quantity_lookup_falls_back_to_the_key():
    assert EN.quantity("currents") == "Currents [MA]"
    assert EN.quantity("not_registered") == "not_registered"


def test_every_language_registers_the_same_quantity_keys():
    assert set(EN.quantities) == set(HU.quantities)


def test_active_selects_by_substring():
    assert [ls.code for ls in active("EN HU")] == ["EN", "HU"]
    assert [ls.code for ls in active("HU")] == ["HU"]
    assert [ls.code for ls in active("EN")] == ["EN"]


def test_active_none_returns_all():
    assert active(None) == list(LANGUAGES.values())


def test_label_sets_are_immutable():
    with pytest.raises(Exception):
        EN.time = "changed"
