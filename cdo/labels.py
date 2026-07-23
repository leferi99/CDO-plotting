"""Axis and quantity labels in the languages the figures are produced in.

The retired notebooks repeated every label as an ``if "EN" in language:`` block
followed by a matching ``if "HU" in language:`` block, so each plot carried two
near-identical copies. Here the labels live in one :class:`LabelSet` per
language and a plotting script loops over the active sets, rendering each figure
once per language into its own output folder.

Legend entries that are pure mathematics, such as ``$I_{\\rm p}$``, are the same
in every language and are not stored here; only the words that change are.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LabelSet:
    """Labels for one language, keyed by a language-neutral name."""

    code: str
    radius: str
    time: str
    momentum: str
    energy: str
    quantities: dict[str, str] = field(default_factory=dict)

    def quantity(self, key: str) -> str:
        """Label for a named quantity, or the key itself if none is registered."""
        return self.quantities.get(key, key)


EN = LabelSet(
    code="EN",
    radius="Minor radius [m]",
    time="Time [ms]",
    momentum=r"Momentum normalized to m$_e$c [-]",
    energy="Energy [MeV]",
    quantities={
        "currents": "Currents [MA]",
        "electron_temperature": "Electron temperature",
        "ohmic_current_density": "Ohmic current density",
        "re_generation_rates": "RE generation rates",
        "runaway_density": "Runaway density [1/m$^3$]",
        "electric_field": "Electric field [V/m]",
        "electron_distribution": r"Electron distribution [1/m$^3$]",
        "pitch_angle": "Pitch angle [degrees]",
    },
)

HU = LabelSet(
    code="HU",
    radius="Kissugár [m]",
    time="Idő [ms]",
    momentum=r"Lendület m$_e$c egységben [-]",
    energy="Energia [MeV]",
    quantities={
        "currents": "Áramok [MA]",
        "electron_temperature": "Elektronhőmérséklet",
        "ohmic_current_density": "Ohmikus áramsűrűség",
        "re_generation_rates": "Keletkezési ráták",
        "runaway_density": "Elfutó elektron sűrűség [1/m$^3$]",
        "electric_field": "Elektromos tér [V/m]",
        "electron_distribution": r"Elektroneloszlás [1/m$^3$]",
        "pitch_angle": "Menetemelkedési szög [fok]",
    },
)

#: Every language keyed by its code, for lookup and iteration.
LANGUAGES: dict[str, LabelSet] = {ls.code: ls for ls in (EN, HU)}


def active(selection: str | None = None) -> list[LabelSet]:
    """Label sets named in ``selection``, in registration order.

    ``selection`` is a string that may list several codes, matching the old
    ``language = "EN HU"`` convention. ``None`` returns every language.
    """
    if selection is None:
        return list(LANGUAGES.values())
    return [ls for code, ls in LANGUAGES.items() if code in selection]
