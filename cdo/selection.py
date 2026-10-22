"""Which times, radii and momentum ranges the figures show.

A :class:`Selection` holds two independent choices, one for fluid quantities
(temperature, densities, current densities, fields, rates) and one for kinetic
quantities (the hot-tail and runaway distributions). Edit it in a notebook and
pass it to the functions in :mod:`cdo.figures`.

Requested values are snapped to the saved data: times to the nearest stored
step, radii to the nearest radial cell. Figures print the snapped values, and
:func:`record` returns both the requested and the snapped values for saving
next to the figures.

Radii are given as r/a, so runs with different minor radii compare directly.
Momentum ranges are ``(min, max)`` tuples in MeV of kinetic energy or in m_e c,
set by ``range_unit``. ``None`` means the edge of the grid. Every range gives
its own figure file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from fnmatch import fnmatchcase
from typing import Sequence

import numpy as np

from . import plotting
from .avalanche import AvalancheParams

#: Electron rest energy in MeV.
MEC2_MEV = 0.51099895


@dataclass
class FluidSelection:
    """Times and radii for fluid quantities.

    ``times_ms`` picks the radial-profile snapshots: a list of times in ms,
    ``"end"`` for the last step, or ``"auto"`` for the standard ladder chosen
    by run length. ``radii`` (r/a) picks the time traces. ``time_range_ms``
    crops the time axis of every fluid figure, ``None`` meaning the run edge.
    ``critical_field_profiles`` adds E_c,eff, dashed, to the electric-field
    profiles.
    """

    times_ms: str | Sequence = "auto"
    radii: Sequence[float] = (0.0, 0.5, 0.9)
    time_range_ms: tuple = (None, None)
    critical_field_profiles: bool = False


@dataclass
class KineticSelection:
    """Times, radii and momentum ranges for the distribution functions.

    Every (radius, range) pair gives one figure, with one panel or curve per
    time. ``ranges`` is a list of ``(min, max)`` tuples in ``range_unit``,
    ``"MeV"`` (kinetic energy) or ``"mec"`` (momentum over m_e c). ``decades``
    sets the depth of the logarithmic colour scale.

    ``pitch_cells`` turns on the ``pitch_<grid>_*`` line plots and picks
    their pitch cells: a number N for N cells spread from the beam out to
    ``pitch_angle_max_deg``, ``"all"``, or indices into the xi cells (negative
    from the end). ``None`` leaves the plots out.

    ``analytic`` is an :class:`~cdo.avalanche.AvalancheParams`, or a list of
    them such as ``[AvalancheParams(), AvalancheParams.from_run()]``, adding
    the analytic avalanche distribution to the combined angle averages and the
    pitch line plots; ``None`` leaves it out. ``combined_per_time`` splits the
    combined figures into one per time, named ``..._t<ms>ms``.
    """

    times_ms: str | Sequence = "auto"
    radii: Sequence[float] = (0.0, 0.5)
    ranges: Sequence[tuple] = ((None, None),)
    range_unit: str = "MeV"
    decades: float = 12.0
    pitch_cells: int | str | Sequence[int] | None = None
    pitch_angle_max_deg: float = 20.0
    analytic: AvalancheParams | Sequence[AvalancheParams] | None = None
    combined_per_time: bool = False


#: Figures a :class:`Selection` leaves out unless ``skip`` is given: Z_eff and
#: n_cold only as radial profiles, current densities as profiles and maps, and
#: the electric field only as radial profiles, with the E/E_c,eff map. ``skip=[]`` makes every figure;
#: ``skip=[*DEFAULT_SKIP, ...]`` adds to the list.
DEFAULT_SKIP = (
    "effective_charge_traces",
    "effective_charge_map",
    "electron_density_traces",
    "electron_density_map",
    "*_current_density_traces",
    "electric_field_ra*",
    "electric_field_average",
    "electric_field_map",
)


@dataclass
class Selection:
    """What the figures show.

    ``skip`` lists figures to leave out, :data:`DEFAULT_SKIP` unless given.
    ``only``, when not empty, keeps just
    the figures that match one of its entries. A plain entry matches every name
    starting with it, so ``"rate_map"`` covers ``rate_map_net`` and the rest.
    An entry with ``*``, ``?`` or ``[...]`` is a wildcard pattern matched
    against the whole name: ``"spectrum_*_Emin-10MeV"`` covers the spectra of
    that range at every radius, and ``"*_ra0.05*"`` every figure at r/a 0.05.
    """

    fluid: FluidSelection = field(default_factory=FluidSelection)
    kinetic: KineticSelection = field(default_factory=KineticSelection)
    skip: Sequence[str] = DEFAULT_SKIP
    only: Sequence[str] = ()

    def wants(self, name: str) -> bool:
        if self.only and not any(_matches(name, p) for p in self.only):
            return False
        return not any(_matches(name, p) for p in self.skip)


def _matches(name: str, entry: str) -> bool:
    """A ``skip``/``only`` entry: a wildcard pattern over the whole name, or a prefix."""
    if any(c in entry for c in "*?["):
        return fnmatchcase(name, entry)
    return name.startswith(entry)


# --- snapping ---------------------------------------------------------------


@dataclass(frozen=True)
class Time:
    """A requested time snapped to a stored step."""

    index: int
    requested_ms: float | None
    ms: float

    @property
    def label(self) -> str:
        return f"t = {self.ms:.4g} ms"

    @property
    def tag(self) -> str:
        return f"t{self.ms:g}ms"


@dataclass(frozen=True)
class Radius:
    """A requested r/a snapped to a radial cell centre."""

    cell: int
    requested: float
    r_over_a: float
    r_m: float

    @property
    def label(self) -> str:
        return f"r/a = {self.r_over_a:.3f} (r = {self.r_m:.3f} m)"

    @property
    def tag(self) -> str:
        return f"ra{self.r_over_a:.2f}"


def snap_times(timegrid_ms, times="auto", *, lines=True) -> list[Time]:
    """Snap requested times (ms) to the nearest stored steps.

    ``times`` is a list of numbers and ``"end"`` entries, or ``"auto"`` for
    the standard ladder of :func:`plotting.time_marks`. Repeated steps are
    dropped, so the result can be shorter than the request.
    """
    t = np.asarray(timegrid_ms)
    if isinstance(times, str):
        if times == "auto":
            return [Time(int(i), None, float(t[i]))
                    for i in plotting.time_marks(t, lines=lines)]
        if times == "end":
            times = ["end"]
        else:
            raise ValueError(f"unknown time selection {times!r}")
    out, seen = [], set()
    for request in times:
        if isinstance(request, str):
            if request != "end":
                raise ValueError(f"unknown time {request!r}")
            i, requested = len(t) - 1, None
        else:
            i = int(plotting.nearest_indices(t, [float(request)])[0])
            requested = float(request)
        if i not in seen:
            seen.add(i)
            out.append(Time(i, requested, float(t[i])))
    return out


def snap_radii(radialgrid, minor_radius, radii) -> list[Radius]:
    """Snap requested r/a values to the nearest radial cell centres."""
    r = np.asarray(radialgrid)
    out, seen = [], set()
    for requested in radii:
        cell = int(np.argmin(np.abs(r / minor_radius - requested)))
        if cell not in seen:
            seen.add(cell)
            out.append(Radius(cell, float(requested), float(r[cell] / minor_radius),
                              float(r[cell])))
    return out


def time_window(timegrid_ms, time_range_ms) -> slice:
    """Index slice of the steps inside ``time_range_ms``, ``None`` for an open end."""
    t = np.asarray(timegrid_ms)
    lo, hi = time_range_ms
    start = 0 if lo is None else int(np.searchsorted(t, lo, side="left"))
    stop = len(t) if hi is None else int(np.searchsorted(t, hi, side="right"))
    return slice(start, stop)


# --- momentum / energy ranges ------------------------------------------------


def energy_to_momentum(E_MeV):
    """Momentum over m_e c of an electron with kinetic energy ``E_MeV``."""
    gamma = 1.0 + np.asarray(E_MeV, dtype=float) / MEC2_MEV
    return np.sqrt(gamma**2 - 1.0)


def momentum_to_energy(p):
    """Kinetic energy in MeV of an electron with momentum ``p`` over m_e c."""
    return (np.sqrt(1.0 + np.asarray(p, dtype=float) ** 2) - 1.0) * MEC2_MEV


@dataclass(frozen=True)
class Range:
    """One ``(min, max)`` momentum range, in MeV or m_e c."""

    lo: float | None
    hi: float | None
    unit: str = "MeV"

    def __post_init__(self):
        if self.unit not in ("MeV", "mec"):
            raise ValueError(f"range unit must be 'MeV' or 'mec', not {self.unit!r}")

    def momentum(self) -> tuple:
        """Bounds as momentum over m_e c, ``None`` for an open end."""
        conv = energy_to_momentum if self.unit == "MeV" else float
        return tuple(None if b is None else float(conv(b)) for b in (self.lo, self.hi))

    def energy(self) -> tuple:
        """Bounds as kinetic energy in MeV, ``None`` for an open end."""
        conv = float if self.unit == "MeV" else momentum_to_energy
        return tuple(None if b is None else float(conv(b)) for b in (self.lo, self.hi))

    @property
    def is_full(self) -> bool:
        return self.lo is None and self.hi is None

    @property
    def tag(self) -> str:
        """File-name part, empty for the full grid."""
        if self.is_full:
            return ""
        prefix = "E" if self.unit == "MeV" else "p"
        suffix = "MeV" if self.unit == "MeV" else "mec"
        fmt = lambda b, edge: edge if b is None else f"{b:g}"
        return f"{prefix}{fmt(self.lo, 'min')}-{fmt(self.hi, 'max')}{suffix}"

    @property
    def label(self) -> str:
        """Title text, in the chosen unit first and the other in brackets."""
        if self.is_full:
            return "full grid"

        def span(symbol, lo, hi, unit):
            if lo is None:
                return f"{symbol} ≤ {hi:.3g} {unit}"
            if hi is None:
                return f"{symbol} ≥ {lo:.3g} {unit}"
            return f"{symbol} = {lo:.3g}-{hi:.3g} {unit}"

        energy = span("E", *self.energy(), "MeV")
        momentum = span("p", *self.momentum(), "m$_e$c")
        first, second = (energy, momentum) if self.unit == "MeV" else (momentum, energy)
        return f"{first} ({second})"

    def mask(self, p) -> np.ndarray:
        """Boolean mask of the momentum cells inside the range."""
        p = np.asarray(p)
        lo, hi = self.momentum()
        keep = np.ones(p.shape, dtype=bool)
        if lo is not None:
            keep &= p >= lo
        if hi is not None:
            keep &= p <= hi
        return keep


def ranges(selection: KineticSelection) -> list[Range]:
    return [Range(lo, hi, selection.range_unit) for lo, hi in selection.ranges]


# --- record ------------------------------------------------------------------


def record(run, selection: Selection) -> dict:
    """The requested selection and what it snapped to on ``run``, as plain data."""
    t, r, a = run.timegrid_ms, run.radialgrid, run.minor_radius
    fluid, kinetic = selection.fluid, selection.kinetic
    return {
        "requested": asdict(selection),
        "fluid": {
            "times_ms": [x.ms for x in snap_times(t, fluid.times_ms, lines=True)],
            "radii": [asdict(x) for x in snap_radii(r, a, fluid.radii)],
        },
        "kinetic": {
            "times_ms": [x.ms for x in snap_times(t, kinetic.times_ms, lines=False)],
            "radii": [asdict(x) for x in snap_radii(r, a, kinetic.radii)],
            "ranges": [{"tag": x.tag or "full", "p_mec": x.momentum(), "E_MeV": x.energy()}
                       for x in ranges(kinetic)],
        },
        "run": {"files": run.files, "t_end_ms": float(t[-1]), "minor_radius_m": a},
    }
