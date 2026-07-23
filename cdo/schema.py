"""Dataset name resolution across DREAM output eras.

DREAM has moved datasets between HDF5 paths and renamed them over time, and any
single output file contains only the datasets the run actually produced. Both
effects are handled here so the rest of the package can ask for a canonical name
and get either an array or ``None``.

Three things determine whether a dataset is present:

* Which DREAM version wrote the file. Handled by listing every path a quantity
  has lived at in :data:`FIELDS`, tried in order.
* Whether the hottail and runaway momentum grids were enabled. Handled by the
  ``grid`` gate on a field.
* Whether the quantity was requested through ``settings/other/include`` and
  whether the physics module producing it was active. Handled by the ``group``
  gate on a field.

A field that is absent while its gates are open is reported as unexpected. A
field that is absent because a gate is closed is not reported at all, since that
is a normal consequence of how the run was set up.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum

import h5py
import numpy as np


class DuplicatePathWarning(UserWarning):
    """Two paths for the same quantity exist in one file and disagree."""


class Presence(Enum):
    """Outcome of resolving one field against one file."""

    PRESENT = "present"
    GATED = "gated"  # absent because a momentum grid or output group was off
    NOT_PRODUCED = "not_produced"  # in an included other/ group, but not written
    MISSING = "missing"  # absent from eqsys or grid with every gate open


class MissingFieldError(KeyError):
    """A field marked required is absent from the file."""


class TimeAxis(Enum):
    """How a dataset's leading axis relates to ``grid/t``.

    ``grid/t`` has ``nt`` entries, the first of which is the initial state.
    Datasets under ``eqsys`` carry all ``nt`` entries. Datasets under ``other``
    are evaluated between time steps and carry ``nt - 1``. Concatenating runs
    means dropping the initial entry from the ``eqsys`` datasets so both line up,
    which is why the alignment is recorded per field rather than applied by hand
    at every read.
    """

    NONE = "none"  # no time axis
    FULL = "full"  # nt entries, drop the first when concatenating
    STEPS = "steps"  # nt - 1 entries, already aligned


@dataclass(frozen=True)
class Field:
    """One canonical quantity and every HDF5 path it has been written to."""

    paths: tuple[str, ...]
    time: TimeAxis = TimeAxis.NONE
    required: bool = False
    grid: str | None = None  # "hottail" or "runaway"
    group: str | None = None  # entry of settings/other/include
    unit: str = ""
    note: str = ""


def _f(*paths: str, **kwargs) -> Field:
    return Field(paths=paths, **kwargs)


# Fields under other/ are never required: DREAM writes them only when the
# quantity was included in the output and the module producing it was active.
def _fluid(*paths: str, **kwargs) -> Field:
    kwargs.setdefault("time", TimeAxis.STEPS)
    kwargs.setdefault("group", "fluid")
    return Field(paths=paths, **kwargs)


FIELDS: dict[str, Field] = {
    # --- grid and geometry -------------------------------------------------
    "time": _f("grid/t", required=True, unit="s"),
    "radius": _f("grid/r", required=True, unit="m"),
    "radius_edges": _f("grid/r_f", required=True, unit="m"),
    "radial_step": _f("grid/dr", required=True, unit="m"),
    "major_radius": _f(
        "grid/R0",
        "settings/radialgrid/R0",
        required=True,
        unit="m",
        note="moved out of settings/radialgrid; absent in cylindrical geometry",
    ),
    "minor_radius": _f("grid/a", "settings/radialgrid/a", required=True, unit="m"),
    "VpVol": _f(
        "grid/VpVol",
        required=True,
        note="cell volume is VpVol * dr * R0",
    ),
    "VpVol_edges": _f("grid/VpVol_f"),
    # --- flux surface averages, under the grid/geometry group --------------
    "B_min": _f("grid/geometry/Bmin", unit="T"),
    "B_max": _f("grid/geometry/Bmax", unit="T"),
    "toroidal_flux": _f("grid/geometry/toroidalFlux", unit="Vs"),
    # GR0 and FSA_R02OverR2 weight the radial integral that turns a current
    # density into a current, matching DREAM's CurrentDensity.current().
    "GR0": _f("grid/geometry/GR0", unit="Tm"),
    "FSA_R02OverR2": _f("grid/geometry/FSA_R02OverR2"),
    "passing_fraction": _f("grid/geometry/effectivePassingFraction"),
    "xi0_trapped_boundary": _f("grid/geometry/xi0TrappedBoundary"),
    # --- grid enable flags -------------------------------------------------
    "hottail_enabled": _f("settings/hottailgrid/enabled", required=True),
    "runaway_enabled": _f("settings/runawaygrid/enabled", required=True),
    "other_include": _f("settings/other/include", note="string list"),
    # --- hottail momentum grid --------------------------------------------
    "hot_p": _f("grid/hottail/p1", grid="hottail"),
    "hot_p_edges": _f("grid/hottail/p1_f", grid="hottail"),
    "hot_xi": _f("grid/hottail/p2", grid="hottail"),
    "hot_xi_edges": _f("grid/hottail/p2_f", grid="hottail"),
    "hot_dp": _f("grid/hottail/dp1", grid="hottail"),
    "hot_dxi": _f("grid/hottail/dp2", grid="hottail"),
    "hot_Vprime": _f("grid/hottail/Vprime", grid="hottail"),
    # --- runaway momentum grid --------------------------------------------
    "re_p": _f("grid/runaway/p1", grid="runaway"),
    "re_p_edges": _f("grid/runaway/p1_f", grid="runaway"),
    "re_xi": _f("grid/runaway/p2", grid="runaway"),
    "re_xi_edges": _f("grid/runaway/p2_f", grid="runaway"),
    "re_dp": _f("grid/runaway/dp1", grid="runaway"),
    "re_dxi": _f("grid/runaway/dp2", grid="runaway"),
    "re_Vprime": _f("grid/runaway/Vprime", grid="runaway"),
    # --- ion metadata ------------------------------------------------------
    "ion_Z": _f("ionmeta/Z", required=True),
    "ion_names": _f("ionmeta/names", required=True, note="string list"),
    # --- eqsys, always written --------------------------------------------
    "E_field": _f("eqsys/E_field", time=TimeAxis.FULL, required=True, unit="V/m"),
    "T_cold": _f("eqsys/T_cold", time=TimeAxis.FULL, required=True, unit="eV"),
    "W_cold": _f("eqsys/W_cold", time=TimeAxis.FULL, required=True, unit="J/m$^3$"),
    "I_p": _f("eqsys/I_p", time=TimeAxis.FULL, required=True, unit="A"),
    "j_ohm": _f("eqsys/j_ohm", time=TimeAxis.FULL, required=True, unit="A/m$^2$"),
    "j_re": _f("eqsys/j_re", time=TimeAxis.FULL, required=True, unit="A/m$^2$"),
    "j_tot": _f("eqsys/j_tot", time=TimeAxis.FULL, required=True, unit="A/m$^2$"),
    "n_cold": _f("eqsys/n_cold", time=TimeAxis.FULL, required=True, unit="1/m$^3$"),
    "n_re": _f("eqsys/n_re", time=TimeAxis.FULL, required=True, unit="1/m$^3$"),
    "n_tot": _f("eqsys/n_tot", time=TimeAxis.FULL, required=True, unit="1/m$^3$"),
    "n_i": _f("eqsys/n_i", time=TimeAxis.FULL, required=True, unit="1/m$^3$"),
    # --- eqsys, depends on the wall and transport model used ---------------
    "I_wall": _f("eqsys/I_wall", time=TimeAxis.FULL, unit="A"),
    "V_loop_wall": _f("eqsys/V_loop_w", time=TimeAxis.FULL, unit="V"),
    "psi_p": _f("eqsys/psi_p", time=TimeAxis.FULL, unit="Vs"),
    "psi_edge": _f("eqsys/psi_edge", time=TimeAxis.FULL, unit="Vs"),
    "psi_wall": _f("eqsys/psi_wall", time=TimeAxis.FULL, unit="Vs"),
    "S_particle": _f("eqsys/S_particle", time=TimeAxis.FULL),
    # --- eqsys, tied to a momentum grid ------------------------------------
    "f_hot": _f("eqsys/f_hot", time=TimeAxis.FULL, grid="hottail", unit="1/m$^3$"),
    "j_hot": _f("eqsys/j_hot", time=TimeAxis.FULL, grid="hottail", unit="A/m$^2$"),
    "n_hot": _f("eqsys/n_hot", time=TimeAxis.FULL, grid="hottail", unit="1/m$^3$"),
    "f_re": _f("eqsys/f_re", time=TimeAxis.FULL, grid="runaway", unit="1/m$^3$"),
    # --- other/fluid, electric fields --------------------------------------
    "Ectot": _fluid("other/fluid/Ectot", unit="V/m"),
    "Ecfree": _fluid("other/fluid/Ecfree", unit="V/m"),
    "Eceff": _fluid("other/fluid/Eceff", unit="V/m"),
    "EDreic": _fluid("other/fluid/EDreic", unit="V/m"),
    # --- other/fluid, plasma parameters ------------------------------------
    "Zeff": _fluid("other/fluid/Zeff"),
    "conductivity": _fluid("other/fluid/conductivity", unit="S/m"),
    "lnLambdaC": _fluid("other/fluid/lnLambdaC"),
    "lnLambdaT": _fluid("other/fluid/lnLambdaT"),
    "qR0": _fluid("other/fluid/qR0", unit="m"),
    "tauEERel": _fluid("other/fluid/tauEERel", unit="s"),
    "tauEETh": _fluid("other/fluid/tauEETh", unit="s"),
    "pCrit": _fluid("other/fluid/pCrit"),
    "pStar": _fluid("other/fluid/pStar"),
    # --- other/fluid, runaway generation rates -----------------------------
    "GammaAva": _fluid("other/fluid/GammaAva", "other/fluid/gammaAva", unit="1/s"),
    "gammaCompton": _fluid("other/fluid/gammaCompton", unit="1/(s m$^3$)"),
    "gammaDreicer": _fluid("other/fluid/gammaDreicer", unit="1/(s m$^3$)"),
    "gammaTritium": _fluid("other/fluid/gammaTritium", unit="1/(s m$^3$)"),
    "gammaHottail": _fluid(
        "other/fluid/gammaFhot",
        "other/fluid/gammaHottail",
        unit="1/(s m$^3$)",
        note=(
            "gammaHottail is an unconfirmed alias, kept as a fallback. No "
            "available file carries it, so the rename is not evidenced."
        ),
    ),
    "runawayRate": _fluid("other/fluid/runawayRate", unit="1/(s m$^3$)"),
    # --- other/fluid, energy balance ---------------------------------------
    "Tcold_ohmic": _fluid("other/fluid/Tcold_ohmic", unit="J/(s m$^3$)"),
    "Tcold_radiation": _fluid("other/fluid/Tcold_radiation", unit="J/(s m$^3$)"),
    "Tcold_transport": _fluid("other/fluid/Tcold_transport", unit="J/(s m$^3$)"),
    "Tcold_nre_coll": _fluid("other/fluid/Tcold_nre_coll", unit="J/(s m$^3$)"),
    "Tcold_ion_coll": _fluid("other/fluid/Tcold_ion_coll", unit="J/(s m$^3$)"),
    "Tcold_binding_energy": _fluid(
        "other/fluid/Tcold_binding_energy", unit="J/(s m$^3$)"
    ),
    "Wcold_Tcold_Drr": _fluid("other/fluid/Wcold_Tcold_Drr"),
    "W_hot": _fluid("other/fluid/W_hot", grid="hottail", unit="J/m$^3$"),
    "W_re": _fluid("other/fluid/W_re", grid="runaway", unit="J/m$^3$"),
    "Tcold_fhot_coll": _fluid(
        "other/fluid/Tcold_fhot_coll", grid="hottail", unit="J/(s m$^3$)"
    ),
    "Tcold_fre_coll": _fluid(
        "other/fluid/Tcold_fre_coll", grid="runaway", unit="J/(s m$^3$)"
    ),
    # --- other/fluid, ionization -------------------------------------------
    "ni_posIonization": _fluid("other/fluid/ni_posIonization"),
    "ni_negIonization": _fluid("other/fluid/ni_negIonization"),
    "ni_posRecombination": _fluid("other/fluid/ni_posRecombination"),
    "ni_negRecombination": _fluid("other/fluid/ni_negRecombination"),
    "tIoniz": _fluid("other/fluid/tIoniz", unit="s"),
    # --- other/scalar ------------------------------------------------------
    "E_mag": _fluid("other/scalar/E_mag", group="scalar", unit="J"),
    "l_i": _fluid("other/scalar/l_i", group="scalar"),
    "L_i": _fluid("other/scalar/L_i", group="scalar", unit="H"),
    "energyloss_T_cold": _fluid("other/scalar/energyloss_T_cold", group="scalar"),
    "radialloss_n_re": _fluid("other/scalar/radialloss_n_re", group="scalar"),
    # --- provenance --------------------------------------------------------
    "commit": _f("code/commit", note="string"),
    "code_datetime": _f("code/datetime_simulation", note="string"),
}


def decode_string(dataset) -> str:
    """Decode a DREAM string, stored as an array of one-byte characters."""
    value = dataset[()] if hasattr(dataset, "__getitem__") else dataset
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    return b"".join(np.asarray(value).ravel().tolist()).decode("utf-8")


def decode_string_list(dataset) -> list[str]:
    """Decode a semicolon-separated DREAM string list.

    ``ionmeta/names`` and ``settings/other/include`` both use this encoding, for
    example ``[b'D', b';', b'T', b';']`` for the ion names ``D`` and ``T``. A
    trailing separator is normal and produces no empty entry.
    """
    text = decode_string(dataset)
    return [part for part in text.split(";") if part]


class Resolver:
    """Reads canonical field names from one open DREAM output file.

    The gates are read once on construction, so every later lookup knows whether
    an absent dataset is explained by the run setup.
    """

    def __init__(self, handle, name: str = ""):
        self.handle = handle
        self.name = name or getattr(handle, "filename", "")
        self.presence: dict[str, Presence] = {}
        self.resolved_path: dict[str, str] = {}
        self.conflicts: dict[str, list[str]] = {}

        self.grids = {
            "hottail": self._flag("settings/hottailgrid/enabled"),
            "runaway": self._flag("settings/runawaygrid/enabled"),
        }

        if "settings/other/include" in handle:
            self.groups = set(decode_string_list(handle["settings/other/include"]))
        else:
            # Written by every version that has other/ at all. Treat an absent
            # include list as "whatever is on disk is what was asked for".
            self.groups = {"fluid", "scalar"}

    def _flag(self, path: str) -> bool:
        if path not in self.handle:
            return False
        return bool(np.asarray(self.handle[path]).ravel()[0])

    def _classify_absent(self, spec: Field) -> tuple[Presence, str]:
        """Explain why an absent field is absent.

        Quantities under ``other/`` are written only when the module computing
        them was active, so their absence from an included group is a normal
        consequence of the run setup and never an error. That also means a
        rename cannot be told apart from an inactive module by looking at the
        file, which is why renames belong in the alias table.
        """
        if spec.grid is not None and not self.grids.get(spec.grid, False):
            return Presence.GATED, f"{spec.grid} grid disabled"
        if spec.group is not None:
            if spec.group not in self.groups:
                return (
                    Presence.GATED,
                    f"other/{spec.group} not included in the output",
                )
            return (
                Presence.NOT_PRODUCED,
                "not written, the module computing it was inactive",
            )
        return Presence.MISSING, "not written by this run"

    def _gate_closed(self, spec: Field) -> str | None:
        """Return the reason a field is not expected, or ``None`` if expected."""
        presence, reason = self._classify_absent(spec)
        return None if presence is Presence.MISSING else reason

    def spec(self, name: str) -> Field:
        try:
            return FIELDS[name]
        except KeyError:
            raise KeyError(
                f"'{name}' is not a known field. Add it to cdo.schema.FIELDS."
            ) from None

    def locate_all(self, name: str) -> list[str]:
        """Every listed path for this field that exists as a dataset here.

        Groups do not count as a match. Several DREAM paths, ``grid/geometry``
        among them, are groups rather than datasets, and a field pointing at one
        is a mistake in :data:`FIELDS` that should surface as an absent field
        rather than as a type error at read time.
        """
        return [
            path
            for path in self.spec(name).paths
            if isinstance(self.handle.get(path), h5py.Dataset)
        ]

    def locate(self, name: str) -> str | None:
        """Return the path this field lives at in this file, if any."""
        found = self.locate_all(name)
        return found[0] if found else None

    def _check_duplicates(self, name: str, found: list[str]) -> None:
        """Warn when several paths hold the same quantity and disagree.

        A quantity written to more than one path is normal: older files carry
        both ``grid/R0`` and ``settings/radialgrid/R0``. That is only worth
        reporting when the copies differ, since then the alias order silently
        decides which value is used. One path present and the others absent
        needs no warning.
        """
        if len(found) < 2 or name in self.conflicts:
            return

        primary = np.asarray(self.handle[found[0]])
        disagreeing = []
        for path in found[1:]:
            other = np.asarray(self.handle[path])
            if primary.shape != other.shape or not np.allclose(
                primary, other, rtol=1e-10, atol=0.0, equal_nan=True
            ):
                disagreeing.append(path)

        if not disagreeing:
            return

        self.conflicts[name] = [found[0]] + disagreeing
        warnings.warn(
            f"{self.name}: '{name}' differs between {found[0]} and "
            f"{', '.join(disagreeing)}. Using {found[0]}.",
            DuplicatePathWarning,
            stacklevel=3,
        )

    def has(self, name: str) -> bool:
        return self.locate(name) is not None

    def expected(self, name: str) -> bool:
        """Whether the run setup implies this field should be present."""
        return self._gate_closed(self.spec(name)) is None

    def read(self, name: str, default=None):
        """Read a field by canonical name.

        Returns ``default`` when the field is absent. Raises
        :class:`MissingFieldError` when a required field is absent.
        """
        spec = self.spec(name)
        found = self.locate_all(name)

        if found:
            self._check_duplicates(name, found)
            path = found[0]
            self.presence[name] = Presence.PRESENT
            self.resolved_path[name] = path
            return np.asarray(self.handle[path])

        presence, _ = self._classify_absent(spec)
        self.presence[name] = presence

        if presence is Presence.MISSING and spec.required:
            tried = ", ".join(spec.paths)
            raise MissingFieldError(
                f"required field '{name}' not found in {self.name}. Tried: {tried}"
            )
        return default

    def read_string(self, name: str, default=None):
        path = self.locate(name)
        if path is None:
            return self.read(name, default)
        self.presence[name] = Presence.PRESENT
        self.resolved_path[name] = path
        return decode_string(self.handle[path])

    def read_string_list(self, name: str, default=None) -> list[str] | None:
        path = self.locate(name)
        if path is None:
            return self.read(name, default)
        self.presence[name] = Presence.PRESENT
        self.resolved_path[name] = path
        return decode_string_list(self.handle[path])

    def read_all(self, names) -> dict:
        return {name: self.read(name) for name in names}

    # --- reporting ---------------------------------------------------------

    def _names(self, presence: Presence) -> list[str]:
        return sorted(n for n, p in self.presence.items() if p is presence)

    def missing(self) -> list[str]:
        """Absent ``eqsys`` or ``grid`` fields, the only concerning category."""
        return self._names(Presence.MISSING)

    def gated(self) -> list[str]:
        """Absent because a momentum grid or an output group was switched off."""
        return self._names(Presence.GATED)

    def not_produced(self) -> list[str]:
        """Absent ``other/`` fields whose group was included in the output."""
        return self._names(Presence.NOT_PRODUCED)

    def relocated(self) -> dict[str, str]:
        """Fields found somewhere other than their first listed path."""
        return {
            name: path
            for name, path in self.resolved_path.items()
            if path != self.spec(name).paths[0]
        }

    def report(self) -> str:
        lines = [f"{self.name}"]
        grids = ", ".join(k for k, v in self.grids.items() if v) or "none"
        lines.append(f"  momentum grids enabled: {grids}")
        lines.append(f"  other groups included: {', '.join(sorted(self.groups))}")

        if self.conflicts:
            lines.append("  duplicated paths that disagree:")
            for name, paths in sorted(self.conflicts.items()):
                lines.append(f"    {name}: {' vs '.join(paths)}, using {paths[0]}")

        relocated = self.relocated()
        if relocated:
            lines.append("  read from an alternative path:")
            for name, path in sorted(relocated.items()):
                lines.append(f"    {name} <- {path}")

        gated = self.gated()
        if gated:
            lines.append("  switched off for this run:")
            for name in gated:
                lines.append(f"    {name}: {self._gate_closed(self.spec(name))}")

        not_produced = self.not_produced()
        if not_produced:
            lines.append("  not computed by this run:")
            lines.append("    " + ", ".join(not_produced))

        missing = self.missing()
        if missing:
            lines.append("  absent, check whether the run needed them:")
            for name in missing:
                tried = ", ".join(self.spec(name).paths)
                lines.append(f"    {name} (tried {tried})")

        return "\n".join(lines)


def survey(handle, name: str = "") -> Resolver:
    """Resolve every known field against a file, for inspection.

    Reads only the paths, not the data, so it is cheap on large outputs.
    """
    resolver = Resolver(handle, name=name)
    for field_name in FIELDS:
        found = resolver.locate_all(field_name)
        if found:
            resolver._check_duplicates(field_name, found)
            resolver.presence[field_name] = Presence.PRESENT
            resolver.resolved_path[field_name] = found[0]
        else:
            resolver.presence[field_name] = resolver._classify_absent(
                FIELDS[field_name]
            )[0]
    return resolver
