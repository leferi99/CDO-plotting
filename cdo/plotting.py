"""Figure building blocks lifted out of the plotting notebooks.

`basic_1D` and `basic_2D` are the general line and colour-mesh plotters the
notebooks called for almost every figure. The rcParams that were pasted into a
notebook cell are named style presets here, and the per-timestep momentum and
pitch distribution plots are carried over from the retired `CDO.py`, the one
part of that module worth keeping.

Saving is opt-in: pass a `savename` and a `folder` to write a PNG and close the
figure, or leave them out to return the figure for display or further editing.
"""

from __future__ import annotations

import math
import os
from bisect import bisect_left

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, ticker

# Figure sizes reused across the notebooks.
FS_1D_LINE = (6, 5)
FS_1D_DISTRIBUTION = (7, 5)
FS_2D_MESH = (6, 5)

#: Base rcParams: inward minor ticks, thin frameless legends, constrained layout.
BASE_RC = {
    "xtick.direction": "in",
    "xtick.labelsize": "small",
    "xtick.major.size": 5,
    "xtick.major.width": 0.7,
    "xtick.minor.size": 2.5,
    "xtick.minor.width": 0.5,
    "xtick.minor.visible": True,
    "xtick.top": False,
    "ytick.direction": "in",
    "ytick.labelsize": "small",
    "ytick.major.size": 5,
    "ytick.major.width": 0.7,
    "ytick.minor.size": 2.5,
    "ytick.minor.width": 0.5,
    "ytick.minor.visible": True,
    "ytick.right": True,
    "legend.frameon": False,
    "font.size": 20,
    "figure.dpi": 150,
    "lines.linewidth": 1.5,
    "figure.constrained_layout.use": True,
    "image.cmap": "inferno",
}

#: Line plots: a faint dashed grid behind the data.
LINE_RC = {
    "axes.grid": True,
    "grid.color": "black",
    "grid.linewidth": 0.3,
    "grid.linestyle": (0, (10, 10)),
}

#: Colour-mesh plots: no grid, ticks pointing out into black margins.
MESH_RC = {
    "axes.grid": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.labelcolor": "k",
    "ytick.labelcolor": "k",
}


def use_base_style(latex: bool = False, sansserif: bool = True, fontsize: int = 20):
    """Apply the base rcParams globally, with optional LaTeX text rendering.

    Without LaTeX the mathtext renderer is used, which needs no system TeX and
    covers the labels here. With LaTeX and a sans-serif face, the cm-bright
    package is loaded so maths matches the surrounding text.
    """
    plt.rcParams.update(BASE_RC)
    plt.rcParams.update({"font.size": fontsize})
    if latex:
        preamble = r"\usepackage{amsmath}\usepackage{amssymb}"
        plt.rcParams.update({"text.usetex": True})
        if sansserif:
            plt.rcParams.update({
                "font.family": "sans-serif",
                "text.latex.preamble": preamble + r"\usepackage{cmbright}",
                "mathtext.fontset": "stixsans",
            })
        else:
            plt.rcParams.update({
                "font.family": "serif",
                "font.serif": "STIXGeneral",
                "text.latex.preamble": preamble,
                "mathtext.fontset": "stix",
            })
    else:
        plt.rcParams.update({"text.usetex": False, "font.family": "sans-serif"})


def _finish(fig, folder, savename, dpi=150):
    """Save and close the figure, or leave it open for display."""
    if savename:
        os.makedirs(folder, exist_ok=True)
        fig.savefig(os.path.join(folder, savename + ".png"), dpi=dpi)
        plt.close(fig)
    return fig


def masked_cmap(cmap=None, under=True):
    """A colormap that shows missing data in white.

    Masked and non-finite cells take the "bad" colour, white. With ``under``,
    values below the colour floor take the darkest colour of the map, so a
    small value still looks different from a missing one. Pair it with
    ``extend="min"`` on the colour bar.
    """
    if cmap is None:
        cmap = plt.colormaps["inferno"]
    elif isinstance(cmap, str):
        cmap = plt.colormaps[cmap]
    extremes = {"bad": "white"}
    if under:
        extremes["under"] = cmap(0.0)
    return cmap.with_extremes(**extremes)


def mask_for_log(data):
    """Mask the cells a log colour scale cannot show: missing, zero and negative."""
    data = np.asarray(data, dtype=float)
    return np.ma.masked_where(~np.isfinite(data) | (data <= 0), data)


def basic_1D(
    ydata,
    xdata,
    *,
    figsize=FS_1D_LINE,
    labels=None,
    colorlist=None,
    linestyles=None,
    xlabel=None,
    ylabel=None,
    title=None,
    xscale="linear",
    yscale="linear",
    ylinthresh=1.0,
    xlim=(None, None),
    ylim=(None, None),
    yticks=None,
    legendloc=None,
    legendfontsize="small",
    folder=None,
    savename=None,
    suffix=None,
    dpi=150,
):
    """Plot one or more curves sharing an x-axis.

    ``ydata`` is a sequence of arrays. Per-curve ``labels``, ``colors`` and
    ``linestyles`` are applied by position when given. ``yscale`` may be
    ``symlog``, in which case ``ylinthresh`` sets the linear band around zero
    and :func:`symlog_ticks` gives legible ``yticks`` for a wide range.
    """
    fig = plt.figure(figsize=figsize)
    for i, data in enumerate(ydata):
        kwargs = {}
        if colorlist is not None:
            kwargs["color"] = colorlist[i]
        if labels is not None:
            kwargs["label"] = labels[i]
        if linestyles is not None:
            kwargs["ls"] = linestyles[i]
        plt.plot(xdata, data, **kwargs)

    axes = plt.gca()
    if yscale == "symlog":
        axes.set_yscale("symlog", linthresh=ylinthresh)
    else:
        axes.set_yscale(yscale)
    axes.set_xscale(xscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if yticks is not None:
        axes.set_yticks(yticks)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    if title is not None:
        axes.set_title(title)
    if legendloc is not None:
        axes.legend(loc=legendloc, fontsize=legendfontsize)

    name = (savename + suffix) if (savename and suffix) else savename
    return _finish(fig, folder, name, dpi)


def basic_2D(
    data,
    xgrid,
    ygrid,
    *,
    figsize=FS_2D_MESH,
    normalization="lin",
    datamin=None,
    datamax=None,
    logdiff=None,
    levels=None,
    linthresh=1e-2,
    cmap=None,
    xlabel=None,
    ylabel=None,
    title=None,
    cbarlabel=None,
    yscale="linear",
    ylim=(None, None),
    mesh=False,
    folder=None,
    savename=None,
    dpi=150,
):
    """Filled-contour plot of a ``(y, x)`` field, usually time against radius.

    ``normalization`` is ``lin``, ``log`` or ``symlog``. For ``log`` the colour
    range spans ``logdiff`` decades below the maximum. For ``symlog`` the range
    is symmetric about zero with a linear band of half-width ``linthresh``. The
    input is not modified.

    ``mesh`` draws a ``pcolormesh`` with nearest shading instead of contours,
    so each cell keeps its own value on a non-uniform grid. ``levels`` is then
    ignored.

    Missing (non-finite) cells are white. On a log scale zero and negative
    cells are white too, and positive values below the floor take the darkest
    colour, marked by the arrow at the bottom of the colour bar.
    """
    data = np.array(data, dtype=float)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError("no finite values to plot")

    if finite.max() < 0 and datamax is None:
        normalization = "symlog"
    if datamax is None:
        datamax = finite.max()
    if datamin is None:
        datamin = finite.min()
    if normalization == "lin" and datamax <= datamin:
        # A constant field still needs a colour range to draw.
        pad = abs(datamax) * 0.01 or 1.0
        datamin, datamax = datamin - pad, datamax + pad
    if levels is None:
        levels = 11
    cmap = masked_cmap(cmap)
    missing_masked = np.ma.masked_invalid(data)

    fig = plt.figure(figsize=figsize)
    axes = plt.subplot(1, 1, 1)

    if normalization == "log":
        if logdiff is None:
            logdiff = max(math.log10(datamax) - 1, 10)
        logmax = math.ceil(math.log10(datamax))
        logmin = logmax - logdiff
        norm = colors.LogNorm(vmin=10**logmin, vmax=datamax)
        if mesh:
            plot = axes.pcolormesh(xgrid, ygrid, mask_for_log(data),
                                   norm=norm, cmap=cmap, shading="nearest")
        else:
            contour_levels = np.logspace(logmin, logmax, levels)
            plot = axes.contourf(xgrid, ygrid, mask_for_log(data), levels=contour_levels,
                                 norm=norm, cmap=cmap, extend="min")
        decade_ticks = np.power(10.0, np.arange(math.ceil(logmin), logmax + 1))
        cbar = fig.colorbar(plot, ax=axes, ticks=decade_ticks, extend="min")
        cbar.formatter = ticker.LogFormatterSciNotation(base=10)
        cbar.update_ticks()
    elif normalization == "lin":
        if mesh:
            plot = axes.pcolormesh(xgrid, ygrid, missing_masked, cmap=cmap, vmin=datamin,
                                   vmax=datamax, shading="nearest")
        else:
            contour_levels = np.linspace(datamin, datamax, levels)
            plot = axes.contourf(xgrid, ygrid, missing_masked, levels=contour_levels,
                                 cmap=cmap, vmin=datamin, vmax=datamax)
        cbar = fig.colorbar(plot, ax=axes)
    elif normalization == "symlog":
        # Signed data: negative values carry meaning, only missing cells are white.
        if mesh:
            bound = max(abs(datamax), abs(datamin))
            norm = colors.SymLogNorm(linthresh=linthresh, vmin=-bound, vmax=bound)
            plot = axes.pcolormesh(xgrid, ygrid, missing_masked, norm=norm,
                                   cmap=masked_cmap("RdBu_r", under=False),
                                   shading="nearest")
            cbar = fig.colorbar(plot, ax=axes, ticks=symlog_ticks(linthresh, bound))
        else:
            norm = colors.SymLogNorm(linthresh=linthresh)
            plot = axes.contourf(xgrid, ygrid, missing_masked, levels=levels, norm=norm,
                                 cmap=masked_cmap("RdBu", under=False),
                                 vmin=-datamax, vmax=datamax)
            cbar = fig.colorbar(plot, ax=axes)
    else:
        raise ValueError(f"unknown normalization '{normalization}'")

    cbar.set_label(cbarlabel)
    axes.set_ylim(ylim)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.set_yscale(yscale)

    return _finish(fig, folder, savename, dpi)


def symlog_ticks(linthresh, bound, per_sign=4):
    """Decade ticks for a symmetric log axis, at most ``per_sign`` per sign.

    Spaces the decades between ``linthresh`` and ``bound`` evenly, so a range of
    many decades keeps legible labels, and adds zero.
    """
    lo = math.ceil(math.log10(linthresh))
    hi = math.floor(math.log10(bound))
    step = max(1, math.ceil((hi - lo + 1) / per_sign))
    decades = np.power(10.0, np.arange(hi, lo - 1, -step))
    return np.concatenate([-decades, [0.0], decades[::-1]])


def critical_field_ratio_2D(
    E,
    Eceff,
    xgrid,
    ygrid,
    *,
    figsize=FS_2D_MESH,
    levels=(-3, -2, -1, 0, 1),
    cmap="RdBu_r",
    xlabel=None,
    ylabel=None,
    title=None,
    cbarlabel=r"log$_{10}(E/E_{\rm c,eff})$",
    yscale="linear",
    ylim=(None, None),
    folder=None,
    savename=None,
    dpi=150,
):
    """Where the field exceeds the effective critical field, over time and radius.

    Plots ``log10(|E| / Eceff)`` as a few-level diverging contour. The ratio-one
    boundary sits at zero: cells above it, drawn warm, are where runaway
    generation can occur, and the ``E = Eceff`` contour is overdrawn in black.
    ``levels`` are the log10 boundaries; the default spans three decades below the
    threshold to one above. ``E`` and ``Eceff`` are ``(time, radius)`` and only
    their ratio matters, so a shared sign convention on ``E`` is not needed.
    """
    E = np.asarray(E, dtype=float)
    Eceff = np.asarray(Eceff, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.abs(E) / Eceff
    logr = np.log10(ratio)
    logr[~np.isfinite(logr)] = np.nan

    levels = list(levels)
    cmap = plt.colormaps[cmap] if isinstance(cmap, str) else cmap
    norm = colors.BoundaryNorm(levels, ncolors=cmap.N, clip=False)

    fig = plt.figure(figsize=figsize)
    axes = plt.subplot(1, 1, 1)
    plot = axes.contourf(xgrid, ygrid, logr, levels=levels, cmap=cmap,
                         norm=norm, extend="both")
    cbar = fig.colorbar(plot, ax=axes, ticks=levels)
    if 0.0 in levels:
        # The E = Eceff threshold: everything warm of this line can run away.
        axes.contour(xgrid, ygrid, logr, levels=[0.0], colors="k", linewidths=2)

    cbar.set_label(cbarlabel)
    axes.set_ylim(ylim)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.set_yscale(yscale)

    return _finish(fig, folder, savename, dpi)


# --- distribution plots, carried over from CDO.py ------------------------


def _pitch_degrees_edges(run, grid):
    """Pitch-angle cell edges in degrees for a momentum grid."""
    xi_edges = run.reference.read(f"{'hot' if grid == 'hottail' else 're'}_xi_edges")
    return np.degrees(np.arccos(xi_edges))


def distribution_momentum_2D(
    run,
    grid="runaway",
    *,
    timestep=0,
    radial_cell=0,
    figsize=(8, 5),
    momentum_range=(0, None),
    ylim=(None, None),
    xlabel=None,
    ylabel=None,
    title=None,
    cbarlabel=r"Electron distribution [1/m$^3$]",
    folder=None,
    savename=None,
    dpi=150,
):
    """Colour-mesh of the electron distribution over momentum and pitch angle.

    Shows one radial cell at one time step of a hottail or runaway grid, with
    momentum normalised to ``m_e c`` on the x-axis and pitch angle in degrees on
    the y-axis. The colour scale is logarithmic with a floor of one, matching the
    original notebook plots. ``momentum_range`` selects a slice of momentum cells.
    """
    f = run.field("f_hot" if grid == "hottail" else "f_re")
    lo, hi = momentum_range
    data = f[timestep, radial_cell, :, lo:hi]

    p_edges = run.reference.read(f"{'hot' if grid == 'hottail' else 're'}_p_edges")
    xi_deg_edges = _pitch_degrees_edges(run, grid)
    vmin = max(np.nanmin(data), 1.0)
    edges = p_edges[lo:None if hi is None else hi + 1]

    fig = plt.figure(figsize=figsize)
    axes = plt.subplot(1, 1, 1)
    mesh = axes.pcolormesh(edges, xi_deg_edges, mask_for_log(data), cmap=masked_cmap(),
                           norm=colors.LogNorm(vmin=vmin, vmax=np.nanmax(data)))
    axes.set_ylim(ylim)
    cbar = fig.colorbar(mesh, ax=axes, extend="min")
    cbar.set_label(cbarlabel)
    axes.set_xlabel(xlabel or r"Momentum normalized to $m_e c$")
    axes.set_ylabel(ylabel or "Pitch angle [degrees]")
    axes.set_title(title)

    return _finish(fig, folder, savename, dpi)


def distribution_momentum_panels(
    run,
    grid="runaway",
    *,
    timesteps,
    radial_cell=0,
    decades=12,
    ncols=None,
    panel_size=(4.2, 3.2),
    cbarlabel=r"Electron distribution [1/m$^3$]",
    suptitle=None,
    folder=None,
    savename=None,
    dpi=150,
):
    """Momentum-pitch colour-meshes of one radial cell at several time steps.

    One panel per entry of ``timesteps``, laid out in a grid with shared axes
    and one shared logarithmic colour scale, so the panels read as a time
    sequence. The scale tops out at the largest value over the chosen steps and
    spans ``decades`` below it. Positive values below that floor take the
    darkest colour. Zero, negative and missing cells are white.
    """
    prefix = "hot" if grid == "hottail" else "re"
    f = run.field("f_hot" if grid == "hottail" else "f_re")
    p_edges = run.reference.read(f"{prefix}_p_edges")
    xi_deg_edges = _pitch_degrees_edges(run, grid)
    t_ms = run.timegrid_ms

    frames = [f[i, radial_cell] for i in timesteps]
    vmax = max(float(np.nanmax(d)) for d in frames)
    if not vmax > 0:
        return None
    norm = colors.LogNorm(vmin=vmax * 10.0**-decades, vmax=vmax)

    n = len(frames)
    ncols = ncols or min(n, 4)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, sharex=True, sharey=True, squeeze=False,
        figsize=(panel_size[0] * ncols + 1.2, panel_size[1] * nrows + 0.6),
    )
    for ax, i, data in zip(axes.flat, timesteps, frames):
        mesh = ax.pcolormesh(p_edges, xi_deg_edges, mask_for_log(data),
                             norm=norm, cmap=masked_cmap())
        ax.set_title(f"t = {time_label(t_ms, i)}", fontsize="small")
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(r"$p$ [$m_e c$]", fontsize="small")
    for ax in axes[:, 0]:
        ax.set_ylabel("Pitch angle [deg]", fontsize="small")
    cbar = fig.colorbar(mesh, ax=axes, shrink=0.9, extend="min")
    cbar.set_label(cbarlabel)
    if suptitle is None:
        suptitle = f"{grid} grid, r = {run.radialgrid[radial_cell]:.2f} m"
    fig.suptitle(suptitle)

    return _finish(fig, folder, savename, dpi)


def angle_averaged_momentum_1D(
    run,
    grid="runaway",
    *,
    timestep=0,
    radial_cell=0,
    figsize=FS_1D_DISTRIBUTION,
    momentum_range=(0, None),
    xlabel=None,
    ylabel=None,
    title=None,
    folder=None,
    savename=None,
    dpi=150,
):
    """Angle-averaged distribution against momentum at one time and radius.

    Uses the density-normalised angle average, plotted on a log y-axis.
    """
    avg = run.angle_average(grid, "density")
    lo, hi = momentum_range
    data = avg[timestep, radial_cell, lo:hi]
    p = run.reference.read(f"{'hot' if grid == 'hottail' else 're'}_p")[lo:hi]

    fig = plt.figure(figsize=figsize)
    axes = plt.subplot(1, 1, 1)
    axes.scatter(p, data)
    axes.set_yscale("log")
    axes.grid(True, which="both", linestyle="--")
    axes.set_xlabel(xlabel or r"Momentum normalized to $m_e c$")
    axes.set_ylabel(ylabel or r"Angle-averaged distribution [1/m$^3$]")
    axes.set_title(title)

    return _finish(fig, folder, savename, dpi)


# --- time-index helpers, from the notebook -------------------------------


def index_array(timegrid, times):
    """Grid indices for a list of requested times.

    Each entry is the first index at or after the requested time. For times that
    land on the grid, as the evenly spaced millisecond marks in the notebooks do,
    this is the exact cell.
    """
    return np.array([bisect_left(timegrid, t) for t in times], dtype=int)


def indices_every(timegrid, step, start=None):
    """Frame indices spaced by ``step`` in time-grid units, snapped to the grid.

    Walks from ``start`` (the grid's first time if ``None``) in increments of
    ``step``, taking the nearest cell at each stop. Suited to picking evenly
    spaced time slices for an animation or a set of profiles.
    """
    if start is None:
        start = timegrid[0]
    n = len(timegrid)

    def nearest(target, lo):
        i = bisect_left(timegrid, target, lo=lo)
        if i >= n:
            return n - 1
        if i and timegrid[i] - target > target - timegrid[i - 1]:
            return i - 1
        return i

    frames = [nearest(start, 0)]
    target = timegrid[frames[0]] + step
    while frames[-1] < n - 1:
        i = nearest(target, frames[-1])
        if i <= frames[-1]:
            i = frames[-1] + 1
        if i >= n:
            break
        frames.append(i)
        target = timegrid[i] + step
    return frames


#: Snapshot times in ms for colour-mesh panels, by run length: runs of at least
#: ``SHORT_RUN_MS``, and shorter runs, which resolve the thermal quench.
MESH_MARKS_LONG = (0, 1, 10, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500)
MESH_MARKS_SHORT = (0, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10, 15)
#: Snapshot times in ms for line plots, fewer so the curves stay apart: runs
#: past ``MEDIUM_RUN_MS``, runs of ``SHORT_RUN_MS`` to ``MEDIUM_RUN_MS``, where
#: the 10-50 ms window holds the current quench and the avalanche, and shorter
#: runs.
LINE_MARKS_LONG = (0, 1, 10, 20, 30, 50, 100, 200, 300, 400, 500)
LINE_MARKS_MEDIUM = (0, 1, 5, 10, 20, 30, 40, 50)
LINE_MARKS_SHORT = (0, 0.1, 0.5, 1, 2, 4)
SHORT_RUN_MS = 20.0
MEDIUM_RUN_MS = 100.0


def nearest_indices(timegrid, times):
    """Index of the grid cell nearest to each requested time."""
    t = np.asarray(timegrid)
    i = np.clip(np.searchsorted(t, times), 1, len(t) - 1)
    left_closer = np.asarray(times) - t[i - 1] < t[i] - np.asarray(times)
    return np.where(left_closer, i - 1, i).astype(int)


def time_marks(timegrid_ms, lines=False):
    """Time-step indices of the standard snapshot times within one run.

    Picks the mark set by run length, from the ``MESH_MARKS_*`` sets, or the
    ``LINE_MARKS_*`` sets with ``lines``. Keeps the marks inside the run, closes
    with the final step when it lies clearly past the last mark, and snaps each
    mark to the nearest saved step. The 0 ms mark resolves to the first saved
    step, since the initial state is not part of a run.
    """
    t = np.asarray(timegrid_ms)
    end = t[-1]
    if lines:
        marks = (LINE_MARKS_SHORT if end < SHORT_RUN_MS
                 else LINE_MARKS_MEDIUM if end <= MEDIUM_RUN_MS
                 else LINE_MARKS_LONG)
    else:
        marks = MESH_MARKS_SHORT if end < SHORT_RUN_MS else MESH_MARKS_LONG
    marks = [m for m in marks if m <= end * 1.001]
    if end > marks[-1] * 1.05:
        marks.append(end)
    steps = nearest_indices(t, marks)
    return list(dict.fromkeys(int(i) for i in steps))


def time_label(timegrid_ms, step):
    """Legend or title text for a time step, with the first step shown as 0 ms."""
    return "0 ms" if step == 0 else f"{timegrid_ms[step]:.3g} ms"
