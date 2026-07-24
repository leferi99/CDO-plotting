"""Checks for the label constants."""

from cdo import labels


def test_axis_labels_are_english():
    assert labels.TIME == "Time [ms]"
    assert labels.RADIUS == "Minor radius [m]"


def test_quantity_lookup_returns_registered_label():
    assert labels.quantity("currents") == "Currents [MA]"


def test_quantity_lookup_falls_back_to_the_key():
    assert labels.quantity("not_registered") == "not_registered"
