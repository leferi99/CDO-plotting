import numpy as np
import pytest

from cdo import selection as sel


T_MS = np.array([0.001, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0])
R = np.array([0.1, 0.3, 0.5, 0.7, 0.9])  # cell centres, a = 1.0


def test_snap_times_nearest_and_end():
    times = sel.snap_times(T_MS, [0.9, 4, "end"])
    assert [t.index for t in times] == [2, 4, 6]
    assert [t.ms for t in times] == [1.0, 5.0, 30.0]
    assert times[0].requested_ms == 0.9 and times[2].requested_ms is None


def test_snap_times_drops_repeated_steps():
    times = sel.snap_times(T_MS, [9, 10, 11])
    assert [t.index for t in times] == [5]


def test_snap_times_auto_uses_ladder():
    times = sel.snap_times(T_MS, "auto", lines=True)
    assert times[0].index == 0
    assert times[-1].index == len(T_MS) - 1


def test_snap_times_rejects_unknown_text():
    with pytest.raises(ValueError):
        sel.snap_times(T_MS, ["middle"])


def test_snap_radii_reports_actual_cell():
    radii = sel.snap_radii(R, 1.0, [0.0, 0.48, 1.0])
    assert [x.cell for x in radii] == [0, 2, 4]
    assert radii[1].r_over_a == pytest.approx(0.5)
    assert radii[1].r_m == pytest.approx(0.5)
    assert "r/a = 0.500" in radii[1].label and "r = 0.500 m" in radii[1].label


def test_snap_radii_scales_by_minor_radius():
    radii = sel.snap_radii(R * 2.0, 2.0, [0.5])
    assert radii[0].cell == 2 and radii[0].r_m == pytest.approx(1.0)


def test_time_window_inclusive_bounds():
    window = sel.time_window(T_MS, (1.0, 10.0))
    assert T_MS[window].tolist() == [1.0, 2.0, 5.0, 10.0]
    assert sel.time_window(T_MS, (None, None)) == slice(0, len(T_MS))


def test_energy_momentum_round_trip():
    E = np.array([0.0, 0.1, 1.0, 10.0])
    assert sel.momentum_to_energy(sel.energy_to_momentum(E)) == pytest.approx(E)
    # 1 m_e c of momentum: E = (sqrt(2) - 1) m_e c^2
    assert sel.momentum_to_energy(1.0) == pytest.approx((np.sqrt(2) - 1) * sel.MEC2_MEV)


def test_range_in_mev_masks_momentum_cells():
    rng = sel.Range(None, 1.0, "MeV")
    p = sel.energy_to_momentum([0.5, 1.0, 2.0])
    assert rng.mask(p).tolist() == [True, True, False]
    assert rng.tag == "Emin-1MeV"


def test_range_full_has_empty_tag():
    assert sel.Range(None, None).tag == ""
    assert sel.Range(None, None).is_full


def test_range_rejects_unknown_unit():
    with pytest.raises(ValueError):
        sel.Range(0, 1, "keV")


def test_selection_skip_and_only():
    s = sel.Selection(skip=["rate_map"])
    assert not s.wants("rate_map_net") and s.wants("temperature_map")
    s = sel.Selection(only=["currents", "f_re"])
    assert s.wants("f_re_ra0.00") and not s.wants("temperature_map")


def test_selection_wildcards_match_the_whole_name():
    s = sel.Selection(skip=["spectrum_*_Emin-10MeV"])
    assert not s.wants("spectrum_hot_ra0.05_Emin-10MeV")
    assert s.wants("spectrum_hot_ra0.05") and s.wants("spectrum_re_ra0.50_E1-5MeV")
    assert s.wants("f_hot_ra0.05_Emin-10MeV")
    s = sel.Selection(only=["*_ra0.05*", "currents"], skip=["f_*"])
    assert s.wants("spectrum_re_ra0.05_Emin-10MeV") and s.wants("currents")
    assert s.wants("electric_field_ra0.05")
    assert not s.wants("f_re_ra0.05") and not s.wants("spectrum_re_ra0.50")
    assert not sel.Selection(skip=["*_map"]).wants("temperature_map")
    assert sel.Selection(skip=["*_map"]).wants("temperature_map_extra")


def test_default_skip_keeps_profiles_maps_and_drops_field_traces():
    s = sel.Selection()
    for name in ("effective_charge_profiles", "electron_density_profiles",
                 "ohmic_current_density_profiles", "runaway_current_density_map",
                 "electric_field_profiles", "critical_field_ratio_map",
                 "temperature_profiles", "temperature_map", "runaway_density_map",
                 "hot_density_map", "radiated_power_map", "f_combined_ra0.05",
                 "dnde_combined_ra0.05", "pitch_re_ra0.05"):
        assert s.wants(name), name
    for name in ("effective_charge_traces", "effective_charge_map", "electron_density_map",
                 "electron_density_traces", "total_current_density_traces",
                 "hot_current_density_traces", "electric_field_ra0.05",
                 "electric_field_average", "electric_field_map", "temperature_traces",
                 "runaway_density_traces", "runaway_density_profiles", "hot_density_traces",
                 "hot_density_profiles", "radiated_power_traces", "radiated_power_profiles",
                 "dndp_combined_ra0.05"):
        assert not s.wants(name), name
    assert sel.Selection(skip=[]).wants("electric_field_average")
    assert not sel.Selection(skip=[*sel.DEFAULT_SKIP, "currents"]).wants("currents")


def test_field_profiles_leave_out_the_critical_field_by_default():
    assert sel.FluidSelection().critical_field_profiles is False


def test_pitch_plots_are_on_by_default():
    assert sel.KineticSelection().pitch_cells == 4
