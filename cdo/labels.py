"""Axis and quantity labels for the figures.

Figures are produced in English only. Earlier revisions carried a parallel
Hungarian set and looped over both languages; that has been dropped, so these are
plain module constants and a small quantity lookup.
"""

from __future__ import annotations

RADIUS = "Minor radius [m]"
TIME = "Time [ms]"
MOMENTUM = r"Momentum normalized to m$_e$c [-]"
ENERGY = "Energy [MeV]"
KINETIC_ENERGY = "Kinetic energy [MeV]"

#: Labels for named quantities, for reuse across figures.
QUANTITIES: dict[str, str] = {
    "currents": "Currents [MA]",
    "electron_temperature": "Electron temperature",
    "ohmic_current_density": "Ohmic current density",
    "re_generation_rates": "RE generation rates",
    "runaway_density": r"Runaway density [1/m$^3$]",
    "electric_field": "Electric field [V/m]",
    "electron_distribution": r"Electron distribution [1/m$^3$]",
    "pitch_angle": "Pitch angle [degrees]",
    "radiated_power": r"Radiated power density [W/m$^3$]",
    "critical_field_ratio": "Field over effective critical field",
}


def quantity(key: str) -> str:
    """Label for a named quantity, or the key itself if none is registered."""
    return QUANTITIES.get(key, key)
