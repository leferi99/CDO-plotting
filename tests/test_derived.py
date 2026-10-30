"""Checks for volumes, radial integrals and ion bookkeeping."""

import numpy as np
import pytest

from cdo.derived import (
    IonSpecies,
    cell_volumes,
    parse_ions,
    radial_integral,
    rate_channels,
    rp_avalanche_coefficient,
    volume_average,
)


class FakeRun:
    """Serves named (time, radius) fields and settings, None for the rest."""

    def __init__(self, settings=None, **fields):
        self.fields = fields
        self.settings = settings or {}

    def field(self, name):
        return self.fields.get(name)

    def setting(self, path):
        return self.settings.get(path)


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


KINETIC_RP = {"hottailgrid/enabled": 1, "runawaygrid/enabled": 1,
              "eqsys/n_re/avalanche": 4, "eqsys/n_re/dreicer": 1,
              "hottailgrid/pmax": 2.5}


def test_rate_channels_fluid_avalanche_is_a_term():
    shape = (2, 3)
    run = FakeRun(
        settings={"eqsys/n_re/avalanche": 3, "eqsys/n_re/dreicer": 1},
        runawayRate=np.full(shape, 1e17),
        GammaAva=np.full(shape, 100.0),
        n_re=np.full(shape, 1e15),
        gammaDreicer=np.full(shape, 1e-29),
    )
    channels = {c.key: c for c in rate_channels(run)}
    assert channels["avalanche"].kind == "term"
    assert channels["avalanche"].values == pytest.approx(np.full(shape, 1e17))
    # The Dreicer estimate is negligible and dropped, the remainder is zero.
    assert set(channels) == {"net", "avalanche"}


def test_rate_channels_kinetic_rp_reconstructs_avalanche_and_flags_estimates():
    shape = (2, 3)
    n_re, n_tot = np.full(shape, 1e15), np.full(shape, 1e20)
    ava = n_re * n_tot * rp_avalanche_coefficient(2.5)
    flux = np.full(shape, 1e18)
    run = FakeRun(
        settings=KINETIC_RP,
        runawayRate=flux + ava,
        gammaFhot=flux,
        GammaAva=np.full(shape, 100.0),
        gammaDreicer=np.full(shape, 1e16),
        n_re=n_re, n_tot=n_tot,
    )
    channels = {c.key: c for c in rate_channels(run)}
    assert channels["hot_flux"].kind == "term"
    assert "avalanche electrons born on the hot grid" in channels["hot_flux"].note
    assert channels["avalanche"].values == pytest.approx(ava)
    assert channels["dreicer_estimate"].kind == "estimate"
    # Net equals flux plus avalanche, so no remainder is shown.
    assert "remainder" not in channels


def test_rate_channels_boltzmann_remainder_is_the_knock_on_source():
    shape = (2, 3)
    run = FakeRun(
        settings={**KINETIC_RP, "eqsys/n_re/avalanche": 5},
        runawayRate=np.full(shape, 1.1e18),
        gammaFhot=np.full(shape, 1e18),
        n_re=np.full(shape, 1e15),
    )
    channels = {c.key: c for c in rate_channels(run)}
    assert channels["remainder"].values == pytest.approx(np.full(shape, 1e17))
    assert "knock-on" in channels["remainder"].label


def test_rate_channels_skips_disabled_compton():
    shape = (2, 3)
    run = FakeRun(settings={"eqsys/n_re/compton/mode": 1},
                  runawayRate=np.ones(shape), gammaCompton=np.ones(shape))
    assert [c.key for c in rate_channels(run)] == ["net"]


def test_volume_average_weights_by_cell_volume():
    field = np.array([[1.0, 3.0], [2.0, np.nan]])
    volumes = np.array([1.0, 3.0])
    # The NaN cell drops out of the total volume, so the second step is 2.0.
    assert volume_average(field, volumes) == pytest.approx([2.5, 2.0])
