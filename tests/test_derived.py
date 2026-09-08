"""Checks for volumes, radial integrals and ion bookkeeping."""

import numpy as np
import pytest

from cdo.derived import (
    IonSpecies,
    cell_volumes,
    flux_to_re,
    parse_ions,
    radial_integral,
)


def test_cell_volume_is_vpvol_dr_r0():
    # Physical volume: VpVol is the Jacobian normalised to R/R0, so a factor of
    # R0 comes back. Verified against the shaped-geometry plasma volume.
    VpVol = np.array([4.0, 12.0, 20.0])
    dr = np.full(3, 0.5)
    R0 = np.array([6.0])
    assert cell_volumes(VpVol, dr, R0) == pytest.approx([12.0, 36.0, 60.0])


def test_radial_integral_keeps_leading_axes():
    density = np.ones((5, 3))  # (t, r)
    volumes = np.array([2.0, 3.0, 4.0])
    assert radial_integral(density, volumes) == pytest.approx(np.full(5, 9.0))


def test_parse_ions_assigns_consecutive_charge_state_rows():
    # D, T each have Z=1 (2 rows), Ar has Z=18 (19 rows): 2 + 2 + 19 = 23.
    species = parse_ions(["D", "T", "Ar"], np.array([1, 1, 18]))
    assert [s.name for s in species] == ["D", "T", "Ar"]
    assert list(species[0].rows) == [0, 1]
    assert list(species[1].rows) == [2, 3]
    assert species[2].rows[0] == 4 and species[2].rows[-1] == 22
    assert species[2].rows.size == 19


def test_ion_totals_and_charged_states():
    species = IonSpecies("D", Z=1, rows=np.array([0, 1]))
    n_i = np.zeros((2, 4, 3))
    n_i[:, 0, :] = 1.0  # neutral
    n_i[:, 1, :] = 2.0  # singly ionised
    assert species.total(n_i) == pytest.approx(np.full((2, 3), 3.0))
    assert species.charged(n_i) == pytest.approx(np.full((2, 3), 2.0))


def test_mean_charge_is_not_cumulative_across_time():
    """The retired code let the accumulator run across time; each step is on its own."""
    ar = IonSpecies("Ar", Z=2, rows=np.array([0, 1, 2]))
    n_i = np.zeros((3, 3, 1))
    # Fully stripped to charge 2 at every time step: mean charge is 2 each time.
    n_i[:, 2, :] = 5.0
    mean = ar.mean_charge(n_i)
    assert mean == pytest.approx(np.full((3, 1), 2.0))


def test_mean_charge_zero_where_species_absent():
    ar = IonSpecies("Ar", Z=2, rows=np.array([0, 1, 2]))
    n_i = np.zeros((1, 3, 2))
    n_i[0, 1, 0] = 4.0  # present only in the first radial cell
    mean = ar.mean_charge(n_i)
    assert mean[0, 0] == pytest.approx(1.0)
    assert mean[0, 1] == 0.0  # no divide-by-zero, no NaN


def test_flux_to_re_subtracts_known_sources():
    shape = (2, 3)
    rate = np.full(shape, 10.0)
    n_re = np.full(shape, 1.0)
    ava = np.full(shape, 2.0)
    trit = np.full(shape, 1.0)
    compton = np.full(shape, 1.0)
    # 10 - 1*2 - 1 - 1 = 6
    assert flux_to_re(rate, n_re, ava, trit, compton) == pytest.approx(
        np.full(shape, 6.0)
    )
