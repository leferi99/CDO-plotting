"""The standard figures of a DREAM run, driven by a :class:`~cdo.selection.Selection`.

Each figure has a name. :func:`plan` lists the figures a run can make with a
given selection, :func:`build` draws one by name, and :func:`make_figures`
draws all of them, saving PNGs (and optionally PDFs) or showing them inline.
A figure the run cannot supply is skipped with a one-line reason.

Fluid figures (selection ``fluid``):

- ``currents``: I_p, I_ohm, I_hot, I_RE and I_wall, each on its own.
- ``<quantity>_traces``: the quantity against time at the chosen radii, with
  its volume average in black.
- ``<quantity>_profiles``: the quantity against radius at the chosen times.
- ``<quantity>_map``: the quantity over time and radius.
- ``electric_field_<r/a>`` and ``electric_field_average``: E, the effective
  and Connor-Hastie critical fields, and E/Eceff on the right axis.
- ``electric_field_profiles`` (|E|, with E_c,eff dashed when
  ``fluid.critical_field_profiles`` is set), ``electric_field_map`` (signed E, symlog) and
  ``critical_field_ratio_map`` (|E|/E_c,eff in discrete bands, readable by
  eye, with the ratio-one contour in black).

:data:`cdo.selection.DEFAULT_SKIP` leaves some of these out unless a
selection's ``skip`` is set.
- ``generation_rates`` and ``rate_map_<channel>``: the channels found by
  :func:`cdo.derived.rate_channels`, labelled as equation terms or estimates.

Kinetic figures (selection ``kinetic``), per grid, radius and range:

- ``f_<grid>_<r/a>[_<range>]``: momentum-pitch maps at the chosen times, or
  f(p) curves on an isotropic grid.
- ``spectrum_<grid>_<r/a>[_<range>]``: density per unit kinetic energy.
- ``favg_re_<r/a>[_<range>]``: angle-averaged f of the runaway grid against
  |p|, for ``ranges`` and ``runaway_zoom_ranges`` (0-40 MeV by default).
- ``pitch_<grid>_<r/a>[_<range>]``: f against momentum, one line per pitch
  cell, one panel per time, on grids with more than one pitch cell. Made
  only when ``kinetic.pitch_cells`` is set: a number of cells spread from the
  beam, ``"all"``, or a list of cells. ``pitch_re`` also for
  ``runaway_zoom_ranges``.

With ``kinetic.analytic`` set, the combined and pitch figures also show the
analytic avalanche distribution of :mod:`cdo.avalanche`: one set of curves
per entry, dashed, dotted, dash-dotted in list order, with the values used
in a box or the title.

Combined figures, when the run has both grids, per radius and range:

- ``f_combined_<r/a>[_<range>]``, ``dndp_combined_<r/a>[_<range>]`` and
  ``dnde_combined_<r/a>[_<range>]``: the angle-averaged f and dn/dp against
  total momentum |p|, and dn/dE against kinetic energy, of the hot-tail and
  runaway grids joined on one axis, with the hot-runaway boundary dashed in
  grey, for ``combined_ranges`` (0-10 MeV by default). With
  ``kinetic.combined_per_time`` one figure per time, ``..._t<ms>ms``.

Colour maps show missing data in white, and on log scales also zero and
negative values. Titles carry the snapped r/a, r in metres and times.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, patheffects

from . import derived, plotting
from .energy import MEC2_EV, EnergyGrid
from .selection import (
    Radius,
    Selection,
    Time,
    ranges,
    record,
    snap_radii,
    snap_times,
    time_window,
)

TIME_LABEL = "Time [ms]"
RADIUS_LABEL = "Minor radius r [m]"
MOMENTUM_LABEL = plotting.MOMENTUM_LABEL
ENERGY_LABEL = plotting.ENERGY_LABEL
GRID_NAMES = {"hottail": "hot-tail", "runaway": "runaway"}
FIELD_OF_GRID = {"hottail": "f_hot", "runaway": "f_re"}
PREFIX_OF_GRID = {"hottail": "hot", "runaway": "re"}


class Skip(Exception):
    """Raised by a figure builder when the run cannot supply the figure."""


# --- fluid quantities ----------------------------------------------------------


@dataclass(frozen=True)
class Quantity:
    """A fluid (time, radius) field and how to draw it.

    ``line_scale`` applies to traces and profiles, ``map_scale`` to the colour
    map. ``decades`` sets the depth of log scales. With ``abs_map`` the map
    shows the magnitude, for fields whose sign is not the point.
    """

    key: str
    field: str
    label: str
    unit: str
    line_scale: str = "log"
    map_scale: str = "log"
    decades: float = 5
    abs_map: bool = False


QUANTITIES = (
    Quantity("temperature", "T_cold", r"$T_{\rm cold}$", "eV", decades=5),
    Quantity("electron_density", "n_cold", r"$n_{\rm cold}$", r"m$^{-3}$", decades=3),
    Quantity("runaway_density", "n_re", r"$n_{\rm RE}$", r"m$^{-3}$", decades=10),
    Quantity("hot_density", "n_hot", r"$n_{\rm hot}$", r"m$^{-3}$", decades=8),
    Quantity("ohmic_current_density", "j_ohm", r"$j_{\rm ohm}$", r"A m$^{-2}$",
             line_scale="linear", decades=4, abs_map=True),
    Quantity("runaway_current_density", "j_re", r"$j_{\rm RE}$", r"A m$^{-2}$",
             line_scale="linear", decades=6, abs_map=True),
    Quantity("hot_current_density", "j_hot", r"$j_{\rm hot}$", r"A m$^{-2}$",
             line_scale="linear", decades=6, abs_map=True),
    Quantity("total_current_density", "j_tot", r"$j_{\rm tot}$", r"A m$^{-2}$",
             line_scale="linear", decades=4, abs_map=True),
    Quantity("radiated_power", "Tcold_radiation", r"$P_{\rm rad}$", r"W m$^{-3}$",
             decades=6, abs_map=True),
    Quantity("effective_charge", "Zeff", r"$Z_{\rm eff}$", "", line_scale="linear",
             map_scale="lin"),
)


# --- shared state --------------------------------------------------------------


class Context:
    """A run plus the selection, snapped once and shared by every builder."""

    def __init__(self, run, selection: Selection):
        self.run = run
        self.sel = selection
        self.t_ms = np.asarray(run.timegrid_ms)
        self.r = np.asarray(run.radialgrid)
        self.a = run.minor_radius
        self.volumes = run.cell_volumes
        self.window = time_window(self.t_ms, selection.fluid.time_range_ms)
        self.fluid_times = snap_times(self.t_ms, selection.fluid.times_ms, lines=True)
        self.fluid_radii = snap_radii(self.r, self.a, selection.fluid.radii)
        self.kinetic_times = snap_times(self.t_ms, selection.kinetic.times_ms, lines=False)
        self.kinetic_radii = snap_radii(self.r, self.a, selection.kinetic.radii)
        self.ranges = ranges(selection.kinetic)
        self.combined_ranges = ranges(selection.kinetic, "combined_ranges")
        self.zoom_ranges = ranges(selection.kinetic, "runaway_zoom_ranges")

    def field(self, name: str) -> np.ndarray:
        value = self.run.field(name)
        if value is None:
            raise Skip(f"run has no {name}")
        return value

    def grid_on(self, grid: str) -> bool:
        """Whether the run has this grid and its distribution, without loading it."""
        return bool(self.run.reference.grids.get(grid)) and \
            f"eqsys/{FIELD_OF_GRID[grid]}" in self.run.reference.handle

    @property
    def derived_cache(self) -> dict:
        """Per-run cache of derived arrays, shared by every Context of the run."""
        return self.run.derived_cache


def _time_limits(ctx: Context) -> tuple:
    """Time-axis limits: from 0 ms, or from the start of a cropped window."""
    t = ctx.t_ms[ctx.window]
    lower = 0.0 if ctx.sel.fluid.time_range_ms[0] is None else float(t[0])
    return lower, float(t[-1])


def _time_colours(n: int):
    """Early times light, late times dark, so the sequence reads left to right."""
    return plt.cm.viridis(np.linspace(0.9, 0.0, max(n, 1)))


def _legend_outside(ax, **kwargs):
    ax.legend(fontsize="small", loc="upper left", bbox_to_anchor=(1.01, 1.0), **kwargs)


def _set_scale(ax, scale: str, axis: str = "y"):
    getattr(ax, f"set_{axis}scale")("linear" if scale == "lin" else scale)


def _line_values(values: np.ndarray, scale: str) -> np.ndarray:
    """Values a log axis can draw: zero and negative entries become gaps."""
    if scale != "log":
        return values
    values = np.array(values, dtype=float)
    values[~(values > 0)] = np.nan
    return values


# --- fluid builders --------------------------------------------------------------


def _currents(ctx: Context):
    run, w = ctx.run, ctx.window
    t = ctx.t_ms[w]
    curves = [(r"$I_{\rm p}$ plasma current", run.plasma_current, dict(color="k", lw=2.5))]
    for label, value, style in (
        (r"$I_{\rm ohm}$ ohmic", run.ohmic_current, dict(color="tab:blue", ls="--")),
        (r"$I_{\rm hot}$ hot-tail grid, $p > p_{\rm thr}$", run.hot_current,
         dict(color="tab:orange")),
        (r"$I_{\rm RE}$ runaway electrons", run.runaway_current, dict(color="tab:red")),
    ):
        if value is not None:
            curves.append((label, value, style))
    i_wall = run.field("I_wall")
    if i_wall is not None:
        curves.append((r"$I_{\rm wall}$", i_wall.reshape(-1), dict(color="tab:gray", ls=":")))
    with plt.rc_context(plotting.LINE_RC):
        fig, ax = plt.subplots(figsize=(9, 5))
        for label, value, style in curves:
            ax.plot(t, np.asarray(value)[w] / 1e6, label=label, **style)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel(TIME_LABEL)
        ax.set_xlim(*_time_limits(ctx))
        ax.set_ylabel("Current [MA]")
        ax.set_title("Currents")
        _legend_outside(ax)
    return fig


def _traces(ctx: Context, q: Quantity):
    values = ctx.field(q.field)
    w = ctx.window
    t = ctx.t_ms[w]
    colours = plt.cm.plasma(np.linspace(0.0, 0.85, max(len(ctx.fluid_radii), 1)))
    with plt.rc_context(plotting.LINE_RC):
        fig, ax = plt.subplots(figsize=(9, 5))
        for c, rad in zip(colours, ctx.fluid_radii):
            ax.plot(t, _line_values(values[w, rad.cell], q.line_scale), color=c,
                    label=rad.label)
        average = derived.volume_average(values, ctx.volumes)[w]
        ax.plot(t, _line_values(average, q.line_scale), color="k", lw=2.5,
                label="volume average")
        _set_scale(ax, q.line_scale)
        ax.set_xlabel(TIME_LABEL)
        ax.set_xlim(*_time_limits(ctx))
        ax.set_ylabel(f"{q.label} [{q.unit}]" if q.unit else q.label)
        ax.set_title(q.label)
        _legend_outside(ax)
    return fig


def _profiles(ctx: Context, q: Quantity):
    values = ctx.field(q.field)
    with plt.rc_context(plotting.LINE_RC):
        fig, ax = plt.subplots(figsize=(9, 5))
        for c, tm in zip(_time_colours(len(ctx.fluid_times)), ctx.fluid_times):
            ax.plot(ctx.r, _line_values(values[tm.index], q.line_scale), color=c,
                    marker=".", label=tm.label)
        _set_scale(ax, q.line_scale)
        ax.set_xlabel(RADIUS_LABEL)
        ax.set_ylabel(f"{q.label} [{q.unit}]" if q.unit else q.label)
        ax.set_title(f"{q.label}, radial profiles (minor radius a = {ctx.a:.3f} m)")
        _legend_outside(ax)
    return fig


def _map(ctx: Context, q: Quantity):
    values = ctx.field(q.field)
    w = ctx.window
    data = np.abs(values[w]) if q.abs_map else values[w]
    label = (rf"$|${q.label}$|$" if q.abs_map else q.label)
    if q.map_scale == "log":
        finite = data[np.isfinite(data) & (data > 0)]
        if finite.size == 0:
            raise Skip(f"{q.field} has no positive values for a log scale")
    with plt.rc_context(plotting.MESH_RC):
        fig = plotting.basic_2D(
            data, ctx.r, ctx.t_ms[w], mesh=True, normalization=q.map_scale,
            logdiff=q.decades, xlabel=RADIUS_LABEL, ylabel=TIME_LABEL,
            ylim=_time_limits(ctx), title=label,
            cbarlabel=(f"{label} [{q.unit}]" if q.unit else label),
        )
    return fig


def _electric_field(ctx: Context, radius: Radius | None):
    E = ctx.field("E_field")
    eceff = ctx.field("Eceff")
    ectot = ctx.run.field("Ectot")
    w = ctx.window
    t = ctx.t_ms[w]
    if radius is None:
        pick = lambda v: derived.volume_average(v, ctx.volumes)[w]
        where = "volume average"
    else:
        pick = lambda v: v[w, radius.cell]
        where = radius.label
    with plt.rc_context(plotting.LINE_RC):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(t, _line_values(np.abs(pick(E)), "log"), color="k", lw=2,
                label=r"$|E_\parallel|$")
        ax.plot(t, _line_values(pick(eceff), "log"), color="tab:blue", ls="--",
                label=r"$E_{\rm c,eff}$ effective critical")
        if ectot is not None:
            ax.plot(t, _line_values(pick(ectot), "log"), color="tab:green", ls="-.",
                    label=r"$E_{\rm c,tot}$ Connor-Hastie, all electrons")
        ax.set_yscale("log")
        ax.set_xlabel(TIME_LABEL)
        ax.set_xlim(*_time_limits(ctx))
        ax.set_ylabel("Electric field [V/m]")
        ratio_ax = ax.twinx()
        ratio_ax.grid(False)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.abs(pick(E)) / pick(eceff)
        ratio_ax.plot(t, _line_values(ratio, "log"), color="tab:red", lw=1.2,
                      label=r"$|E_\parallel| / E_{\rm c,eff}$")
        ratio_ax.axhline(1.0, color="tab:red", lw=0.8, ls=":")
        ratio_ax.set_yscale("log")
        ratio_ax.set_ylabel(r"$|E_\parallel| / E_{\rm c,eff}$", color="tab:red")
        ratio_ax.tick_params(axis="y", colors="tab:red")
        ax.set_title(f"Electric field, {where}")
        lines = ax.get_lines() + ratio_ax.get_lines()[:1]
        ax.legend(lines, [l.get_label() for l in lines], fontsize="small",
                  loc="upper left", bbox_to_anchor=(1.12, 1.0))
    return fig


def _electric_field_profiles(ctx: Context):
    E = ctx.field("E_field")
    critical = ctx.sel.fluid.critical_field_profiles
    eceff = ctx.field("Eceff") if critical else None
    with plt.rc_context(plotting.LINE_RC):
        fig, ax = plt.subplots(figsize=(9, 5))
        for c, tm in zip(_time_colours(len(ctx.fluid_times)), ctx.fluid_times):
            ax.plot(ctx.r, _line_values(np.abs(E[tm.index]), "log"), color=c, marker=".",
                    label=tm.label)
            if critical:
                ax.plot(ctx.r, _line_values(eceff[tm.index], "log"), color=c, ls="--")
        ax.set_yscale("log")
        ax.set_xlabel(RADIUS_LABEL)
        ax.set_ylabel("Electric field [V/m]")
        ax.set_title(r"$|E_\parallel|$ (solid) and $E_{\rm c,eff}$ (dashed)" if critical
                     else r"$|E_\parallel|$, radial profiles")
        _legend_outside(ax)
    return fig


def _critical_field_ratio_map(ctx: Context):
    E = ctx.field("E_field")
    eceff = ctx.field("Eceff")
    w = ctx.window
    with plt.rc_context(plotting.MESH_RC):
        return plotting.critical_field_ratio_bands_2D(
            E[w], eceff[w], ctx.r, ctx.t_ms[w], xlabel=RADIUS_LABEL, ylabel=TIME_LABEL,
            ylim=_time_limits(ctx),
            title=r"$|E_\parallel| / E_{\rm c,eff}$",
        )


#: Half-width of the linear band of the electric field map, V/m.
E_MAP_LINTHRESH = 1e-2


def _electric_field_map(ctx: Context):
    E = ctx.field("E_field")
    w = ctx.window
    with plt.rc_context(plotting.MESH_RC):
        return plotting.basic_2D(
            E[w], ctx.r, ctx.t_ms[w], mesh=True, normalization="symlog",
            linthresh=E_MAP_LINTHRESH, xlabel=RADIUS_LABEL, ylabel=TIME_LABEL,
            ylim=_time_limits(ctx), title=r"$E_\parallel$",
            cbarlabel=r"$E_\parallel$ [V/m], symlog, linear within $\pm$"
                      f"{E_MAP_LINTHRESH:g}",
        )


RATE_STYLES = {
    "net": dict(color="k", lw=2.5),
    "term": dict(lw=1.5),
    "remainder": dict(lw=1.5, ls=":"),
    "estimate": dict(lw=1.0, ls="--", alpha=0.7),
}


def _rate_channels(ctx: Context):
    cache = ctx.derived_cache
    if "rate_channels" not in cache:
        cache["rate_channels"] = derived.rate_channels(ctx.run)
    return cache["rate_channels"]


def _generation_rates(ctx: Context):
    channels = _rate_channels(ctx)
    if not channels:
        raise Skip("no runaway rate outputs")
    w = ctx.window
    t = ctx.t_ms[w]
    curves = [(c, derived.volume_average(c.values, ctx.volumes)[w]) for c in channels]
    peak = max(float(np.nanmax(np.abs(v))) for _, v in curves)
    linthresh = peak * 1e-8
    with plt.rc_context(plotting.LINE_RC):
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for c, v in curves:
            kind = {"term": "", "remainder": " [remainder]", "estimate": " [estimate, not in eqs]",
                    "net": ""}[c.kind]
            ax.plot(t, v, label=c.label + kind, **RATE_STYLES[c.kind])
        ax.set_yscale("symlog", linthresh=linthresh)
        ax.set_yticks(plotting.symlog_ticks(linthresh, peak))
        ax.set_xlabel(TIME_LABEL)
        ax.set_xlim(*_time_limits(ctx))
        ax.set_ylabel(r"Volume-averaged rate [m$^{-3}$ s$^{-1}$]")
        ax.set_title("Runaway rates: net change and its terms")
        _legend_outside(ax)
        notes = [f"{c.key}: {c.note}" for c in channels if c.note]
        if notes:
            fig.text(0.01, -0.02, "\n".join(notes), fontsize="x-small", va="top")
    return fig


def _rate_map(ctx: Context, key: str):
    channel = next((c for c in _rate_channels(ctx) if c.key == key), None)
    if channel is None:
        raise Skip(f"no rate channel {key}")
    w = ctx.window
    rate = channel.values[w]
    peak = float(np.nanmax(np.abs(rate)))
    signed = np.nanmin(rate) < -1e-3 * peak
    with plt.rc_context(plotting.MESH_RC):
        return plotting.basic_2D(
            rate, ctx.r, ctx.t_ms[w], mesh=True,
            normalization="symlog" if signed else "log",
            logdiff=8, linthresh=peak * 1e-6,
            datamin=-peak if signed else None, datamax=peak if signed else None,
            xlabel=RADIUS_LABEL, ylabel=TIME_LABEL, ylim=_time_limits(ctx),
            title=channel.label, cbarlabel=r"Rate [m$^{-3}$ s$^{-1}$]",
        )


# --- kinetic builders ------------------------------------------------------------


def _grid_axes(ctx: Context, grid: str):
    prefix = PREFIX_OF_GRID[grid]
    read = ctx.run.reference.read
    p, p_edges = read(f"{prefix}_p"), read(f"{prefix}_p_edges")
    xi_edges = read(f"{prefix}_xi_edges")
    return p, p_edges, np.degrees(np.arccos(np.clip(xi_edges, -1, 1)))


def _range_columns(p, rng):
    keep = np.flatnonzero(rng.mask(p))
    if keep.size == 0:
        raise Skip(f"no momentum cells in {rng.label}")
    return slice(keep[0], keep[-1] + 1)


#: Trapped-passing boundary lines on momentum-pitch maps: white dashes with a
#: thin black outline, visible on both ends of the colour map.
TRAPPED_LINE = dict(color="white", ls="--", lw=1.0,
                    path_effects=[patheffects.withStroke(linewidth=2.2, foreground="black")])


def _trapped_band_deg(ctx: Context, radius: Radius) -> tuple:
    """Pitch angles (deg) bounding the trapped region at ``radius``, or ``()``.

    DREAM computes the boundary per radius, xi0_T = sqrt(1 - Bmin/Bmax), and
    stores it in grid/geometry/xi0TrappedBoundary. Particles with
    |xi0| < xi0_T are trapped, so the band is arccos(xi0_T) to
    180 deg - arccos(xi0_T). Zero means no trapping, as in a cylinder.
    """
    xi0 = ctx.run.reference.read("xi0_trapped_boundary")
    if xi0 is None:
        return ()
    xi0_r = float(np.asarray(xi0).ravel()[radius.cell])
    if not xi0_r > 0:
        return ()
    theta = math.degrees(math.acos(xi0_r))
    return (theta, 180.0 - theta)


def _kinetic_title(grid: str, radius: Radius, rng) -> str:
    return f"{GRID_NAMES[grid]} grid, {radius.label}, {rng.label}"


def _distribution(ctx: Context, grid: str, radius: Radius, rng):
    f = ctx.field(FIELD_OF_GRID[grid])
    p, p_edges, pitch_edges = _grid_axes(ctx, grid)
    cols = _range_columns(p, rng)
    frames = [(tm, f[tm.index, radius.cell, :, cols]) for tm in ctx.kinetic_times]
    if f.shape[2] == 1:
        return _isotropic_distribution(ctx, grid, radius, rng, p[cols], frames)
    positive = [d[np.isfinite(d) & (d > 0)] for _, d in frames]
    vmax = max((float(x.max()) for x in positive if x.size), default=0.0)
    if not vmax > 0:
        raise Skip("distribution has no positive values at the chosen times")
    norm = colors.LogNorm(vmin=vmax * 10.0 ** -ctx.sel.kinetic.decades, vmax=vmax)
    n = len(frames)
    ncols = min(n, 4)
    nrows = math.ceil(n / ncols)
    edges = p_edges[cols.start:cols.stop + 1]
    with plt.rc_context(plotting.MESH_RC):
        fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, squeeze=False,
                                 figsize=(3.6 * ncols + 1.4, 3.0 * nrows + 0.8))
        band = _trapped_band_deg(ctx, radius)
        for ax, (tm, data) in zip(axes.flat, frames):
            mesh = ax.pcolormesh(edges, pitch_edges, plotting.mask_for_log(data),
                                 norm=norm, cmap=plotting.masked_cmap())
            for angle in band:
                ax.axhline(angle, **TRAPPED_LINE)
            ax.set_title(tm.label, fontsize="small")
        for ax in axes.flat[n:]:
            ax.set_visible(False)
        axes.flat[0].set_xlim(left=rng.momentum()[0] or 0.0)
        for ax in axes[-1]:
            ax.set_xlabel(MOMENTUM_LABEL, fontsize="small")
        for ax in axes[:, 0]:
            ax.set_ylabel("Pitch angle [deg]", fontsize="small")
        cbar = fig.colorbar(mesh, ax=axes, shrink=0.9, extend="min")
        cbar.set_label(r"$f$ [m$^{-3}$ $(m_e c)^{-3}$]")
        title = _kinetic_title(grid, radius, rng)
        if band:
            title += (f"\ntrapped for {band[0]:.1f}°-{band[1]:.1f}° (dashed), "
                      f"$\\xi_{{0,T}}$ = {math.cos(math.radians(band[0])):.3f}")
        fig.suptitle(title, fontsize="medium")
    return fig


def _combined(ctx: Context, moment: str, radius: Radius, rng, time: Time | None = None):
    kin = ctx.sel.kinetic
    title = f"Hot-tail and runaway grids, {radius.label}, {rng.label}"
    if time is not None:
        title += f", {time.label}"
    return plotting.angle_averaged_combined_1D(
        ctx.run, moment, times_ms=kin.times_ms if time is None else [time.ms],
        r_over_a=radius.requested, p_range=rng.momentum(), analytic=kin.analytic,
        title=title)


def _runaway_average(ctx: Context, grid: str, radius: Radius, rng):
    kin = ctx.sel.kinetic
    return plotting.angle_averaged_combined_1D(
        ctx.run, "distribution", times_ms=kin.times_ms, r_over_a=radius.requested,
        p_range=rng.momentum(), analytic=kin.analytic, grids=("runaway",),
        title=f"Angle-averaged f, {_kinetic_title(grid, radius, rng)}")


def _pitch_lines(ctx: Context, grid: str, radius: Radius, rng):
    kin = ctx.sel.kinetic
    return plotting.pitch_resolved_1D(
        ctx.run, grid, times_ms=kin.times_ms, r_over_a=radius.requested,
        p_range=rng.momentum(), pitch_cells=kin.pitch_cells,
        max_angle_deg=kin.pitch_angle_max_deg, analytic=kin.analytic,
        decades=kin.decades, title=_kinetic_title(grid, radius, rng))


def _pitch_resolved(ctx: Context, grid: str) -> bool:
    if ctx.sel.kinetic.pitch_cells is None:
        return False
    xi = ctx.run.reference.read(f"{PREFIX_OF_GRID[grid]}_xi")
    return xi is not None and len(xi) > 1


def _isotropic_distribution(ctx, grid, radius, rng, p, frames):
    with plt.rc_context(plotting.LINE_RC):
        fig, ax = plt.subplots(figsize=(9, 5))
        for c, (tm, data) in zip(_time_colours(len(frames)), frames):
            ax.plot(p, _line_values(data[0], "log"), color=c, label=tm.label)
        ax.set_yscale("log")
        ax.set_xlim(left=rng.momentum()[0] or 0.0)
        ax.set_xlabel(MOMENTUM_LABEL)
        ax.set_ylabel(r"$f$ [m$^{-3}$ $(m_e c)^{-3}$]")
        ax.set_title(_kinetic_title(grid, radius, rng) + " (isotropic grid)", fontsize="medium")
        _legend_outside(ax)
    return fig


def _density_moment(ctx: Context, grid: str):
    """dn/dp over (t, r, p), cached per run and grid."""
    cache = ctx.derived_cache
    key = f"dndp_{grid}"
    if key not in cache:
        value = ctx.run.angle_average(grid, "density")
        if value is None:
            raise Skip(f"no {GRID_NAMES[grid]} grid")
        cache[key] = value
    return cache[key]


def _spectrum(ctx: Context, grid: str, radius: Radius, rng):
    dndp = _density_moment(ctx, grid)
    _, p_edges, _ = _grid_axes(ctx, grid)
    p = ctx.run.reference.read(f"{PREFIX_OF_GRID[grid]}_p")
    energy = EnergyGrid.from_momentum(p, p_edges)
    dnde = energy.to_energy(dndp, per="MeV")
    kinetic_MeV = (energy.total_energy_eV - MEC2_EV) / 1e6
    cols = _range_columns(p, rng)
    with plt.rc_context(plotting.LINE_RC):
        fig, ax = plt.subplots(figsize=(9, 5))
        for c, tm in zip(_time_colours(len(ctx.kinetic_times)), ctx.kinetic_times):
            ax.plot(kinetic_MeV[cols], _line_values(dnde[tm.index, radius.cell, cols], "log"),
                    color=c, label=tm.label)
        ax.set_yscale("log")
        ax.set_xlim(left=rng.energy()[0] or 0.0)
        ax.set_xlabel(ENERGY_LABEL)
        ax.set_ylabel(r"$dn/dE$ [m$^{-3}$ MeV$^{-1}$]")
        ax.set_title("Energy spectrum, " + _kinetic_title(grid, radius, rng), fontsize="medium")
        _legend_outside(ax)
    return fig


# --- planning and output ------------------------------------------------------------


def plan(run, selection: Selection | None = None) -> list[tuple[str, Callable]]:
    """The figures ``run`` can be asked for, as (name, builder) pairs, in order.

    Names follow the module docstring. Figures the selection excludes are left
    out. A builder may still skip when the data is missing.
    """
    ctx = Context(run, selection or Selection())
    items: list[tuple[str, Callable]] = [("currents", lambda: _currents(ctx))]
    for q in QUANTITIES:
        if run.field(q.field) is None:
            continue
        items += [
            (f"{q.key}_traces", lambda q=q: _traces(ctx, q)),
            (f"{q.key}_profiles", lambda q=q: _profiles(ctx, q)),
            (f"{q.key}_map", lambda q=q: _map(ctx, q)),
        ]
    for rad in ctx.fluid_radii:
        items.append((f"electric_field_{rad.tag}", lambda rad=rad: _electric_field(ctx, rad)))
    items += [
        ("electric_field_average", lambda: _electric_field(ctx, None)),
        ("electric_field_profiles", lambda: _electric_field_profiles(ctx)),
        ("electric_field_map", lambda: _electric_field_map(ctx)),
        ("critical_field_ratio_map", lambda: _critical_field_ratio_map(ctx)),
        ("generation_rates", lambda: _generation_rates(ctx)),
    ]
    for channel in _rate_channels(ctx):
        if channel.kind != "estimate":
            items.append((f"rate_map_{channel.key}",
                          lambda key=channel.key: _rate_map(ctx, key)))
    for grid in ("hottail", "runaway"):
        if not ctx.grid_on(grid):
            continue
        short = PREFIX_OF_GRID[grid]
        zooms = [z for z in ctx.zoom_ranges if z not in ctx.ranges] \
            if grid == "runaway" else []
        for rad in ctx.kinetic_radii:
            for rng in [*ctx.ranges, *zooms]:
                suffix = f"_{rad.tag}" + (f"_{rng.tag}" if rng.tag else "")
                builders = [] if rng in zooms else [("f", _distribution), ("spectrum", _spectrum)]
                if grid == "runaway":
                    builders.append(("favg", _runaway_average))
                if _pitch_resolved(ctx, grid):
                    builders.append(("pitch", _pitch_lines))
                for kind, builder in builders:
                    items.append((f"{kind}_{short}{suffix}",
                                  lambda b=builder, g=grid, r=rad, x=rng: b(ctx, g, r, x)))
    if ctx.grid_on("hottail") and ctx.grid_on("runaway"):
        for rad in ctx.kinetic_radii:
            for rng in ctx.combined_ranges:
                suffix = f"_{rad.tag}" + (f"_{rng.tag}" if rng.tag else "")
                for kind, moment in (("f", "distribution"), ("dndp", "density"),
                                     ("dnde", "energy")):
                    if not ctx.sel.kinetic.combined_per_time:
                        items.append((f"{kind}_combined{suffix}",
                                      lambda m=moment, r=rad, x=rng: _combined(ctx, m, r, x)))
                        continue
                    for tm in ctx.kinetic_times:
                        items.append((f"{kind}_combined{suffix}_{tm.tag}",
                                      lambda m=moment, r=rad, x=rng, t=tm: _combined(ctx, m, r, x, t)))
    sel = ctx.sel
    return [(name, fn) for name, fn in items if sel.wants(name)]


def build(run, name: str, selection: Selection | None = None):
    """Draw one figure by name and return it, or raise :class:`Skip`."""
    for item, fn in plan(run, selection):
        if item == name:
            return fn()
    raise KeyError(f"no figure {name!r}; see figures.names(run)")


def names(run, selection: Selection | None = None) -> list[str]:
    """Names of the figures :func:`make_figures` would try for ``run``."""
    return [name for name, _ in plan(run, selection)]


def make_figures(run, selection: Selection | None = None, out=None, *, pdf: bool = False,
                 dpi: int = 150, quiet: bool = False) -> dict:
    """Draw every planned figure of ``run``.

    With ``out`` each figure is saved as ``out/<name>.png``, plus
    ``out/pdf/<name>.pdf`` with ``pdf``, and ``out/plot_settings.json`` records
    the requested and snapped selection. Without ``out`` the figures are shown
    inline. Returns ``{"made": [...], "skipped": {name: reason}}``.
    """
    selection = selection or Selection()
    made, skipped = [], {}
    out = Path(out) if out is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        if pdf:
            (out / "pdf").mkdir(exist_ok=True)
    for name, fn in plan(run, selection):
        try:
            fig = fn()
        except Skip as reason:
            skipped[name] = str(reason)
        except Exception as exc:  # a figure must not stop the rest
            skipped[name] = f"error: {type(exc).__name__}: {exc}"
            plt.close("all")
        else:
            made.append(name)
            if out is None:
                plt.show()
            else:
                fig.savefig(out / f"{name}.png", dpi=dpi, bbox_inches="tight")
                if pdf:
                    fig.savefig(out / "pdf" / f"{name}.pdf", bbox_inches="tight")
            plt.close(fig)
        if name in skipped and not quiet:
            print(f"    [skip] {name}: {skipped[name]}")
    if out is not None:
        info = record(run, selection)
        info.update(figures=made, skipped=skipped)
        (out / "plot_settings.json").write_text(json.dumps(info, indent=1, default=str))
    return {"made": made, "skipped": skipped}


# --- cross-run comparison -------------------------------------------------------------


def compare(runs: dict, out=None, *, time_shift_ms: dict | None = None,
            references: dict | None = None, styles: dict | None = None,
            time_range_ms: tuple = (None, None), suffix: str = "",
            pdf: bool = False, dpi: int = 150) -> list[str]:
    """Overlay scalar time traces of several runs.

    ``time_range_ms`` limits the (shifted) time axis and fits the y range to
    what is inside it, for a zoomed copy. ``suffix`` is appended to every
    figure name, for example ``"_zoom"``.

    ``runs`` maps legend labels to :class:`cdo.Run`. ``time_shift_ms`` maps
    labels to an offset added to their time axis. ``references`` maps figure
    names to extra curves or points, ``[(label, t_ms, values_MA, style), ...]``.
    ``styles`` maps run labels to matplotlib line keyword dicts.
    Figures: ``compare_plasma_current``, ``compare_runaway_current``,
    ``compare_hot_current``, ``compare_temperature_average``.
    """
    time_shift_ms = time_shift_ms or {}
    references = references or {}
    styles = styles or {}
    default = plt.cm.turbo(np.linspace(0.05, 0.95, max(len(runs), 1)))
    looks = {label: styles.get(label, dict(color=c)) for label, c in zip(runs, default)}

    def series(run, what):
        if what == "temperature":
            T = run.field("T_cold")
            return None if T is None else derived.volume_average(T, run.cell_volumes)
        value = {"plasma": run.plasma_current, "runaway": run.runaway_current,
                 "hot": run.hot_current}[what]
        return None if value is None else np.asarray(value) / 1e6

    figures = (
        ("compare_plasma_current", "plasma", r"$I_{\rm p}$ [MA]", "linear"),
        ("compare_runaway_current", "runaway", r"$I_{\rm RE}$ runaway electrons [MA]", "linear"),
        ("compare_hot_current", "hot", r"$I_{\rm hot}$ hot-tail grid [MA]", "linear"),
        ("compare_temperature_average", "temperature", r"volume-averaged $T_{\rm cold}$ [eV]",
         "log"),
    )
    made = []
    out = Path(out) if out is not None else None
    t_lo, t_hi = time_range_ms

    def inside(t):
        keep = np.ones(t.shape, dtype=bool)
        if t_lo is not None:
            keep &= t >= t_lo
        if t_hi is not None:
            keep &= t <= t_hi
        return keep

    for name, what, ylabel, scale in figures:
        name = name + suffix
        with plt.rc_context(plotting.LINE_RC):
            fig, ax = plt.subplots(figsize=(11, 5))
            visible = []
            for label, run in runs.items():
                y = series(run, what)
                if y is None:
                    continue
                t = np.asarray(run.timegrid_ms) + time_shift_ms.get(label, 0.0)
                y = _line_values(y, scale)
                ax.plot(t, y, label=label, **looks[label])
                visible.append(y[inside(t)])
            for ref_label, t_ref, y_ref, style in references.get(name.removesuffix(suffix), ()):
                ax.plot(t_ref, y_ref, label=ref_label, **style)
            if not visible:
                plt.close(fig)
                continue
            _set_scale(ax, scale)
            if t_lo is not None or t_hi is not None:
                ax.set_xlim(t_lo, t_hi)
                values = np.concatenate(visible)
                values = values[np.isfinite(values)]
                if values.size:
                    if scale == "log":
                        ax.set_ylim(values.min() / 1.3, values.max() * 1.3)
                    else:
                        pad = 0.05 * (values.max() - values.min()) or 0.05
                        ax.set_ylim(values.min() - pad, values.max() + pad)
            if not time_shift_ms and t_lo is None:
                ax.set_xlim(left=0.0)
            ax.set_xlabel(TIME_LABEL + (" (shifted)" if time_shift_ms else ""))
            ax.set_ylabel(ylabel)
            _legend_outside(ax)
        made.append(name)
        if out is None:
            plt.show()
        else:
            out.mkdir(parents=True, exist_ok=True)
            fig.savefig(out / f"{name}.png", dpi=dpi, bbox_inches="tight")
            if pdf:
                (out / "pdf").mkdir(exist_ok=True)
                fig.savefig(out / "pdf" / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
    return made
