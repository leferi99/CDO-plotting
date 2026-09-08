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
    ``symlog``, in which case ``ylinthresh`` sets the linear band around zero.
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
    folder=None,
    savename=None,
    dpi=150,
):
    """Filled-contour plot of a ``(y, x)`` field, usually time against radius.

    ``normalization`` is ``lin``, ``log`` or ``symlog``. For ``log`` the colour
    range spans ``logdiff`` decades below the maximum. For ``symlog`` the range
    is symmetric about zero with a linear band of half-width ``linthresh``. Data
    is not modified in place: non-finite entries are zeroed on a copy.
    """
    data = np.array(data, dtype=float)
    data[~np.isfinite(data)] = 0.0

    if data.max() < 0 and datamax is None:
        normalization = "symlog"
    if datamax is None:
        datamax = data.max()
    if datamin is None:
        datamin = data.min()
    if levels is None:
        levels = 11
    if cmap is None:
        cmap = plt.colormaps["inferno"]

    fig = plt.figure(figsize=figsize)
    axes = plt.subplot(1, 1, 1)

    if normalization == "log":
        if logdiff is None:
            logdiff = max(math.log10(datamax) - 1, 10)
        logmax = math.ceil(math.log10(datamax))
        logmin = logmax - logdiff
        contour_levels = np.logspace(logmin, logmax, levels)
        norm = colors.LogNorm(vmin=10**logmin, vmax=datamax)
        plot = axes.contourf(xgrid, ygrid, data, levels=contour_levels, norm=norm, cmap=cmap)
        decade_ticks = np.power(10.0, np.arange(math.ceil(logmin), logmax + 1))
        cbar = fig.colorbar(plot, ax=axes, ticks=decade_ticks)
        cbar.formatter = ticker.LogFormatterExponent(base=10)
        cbar.update_ticks()
    elif normalization == "lin":
        contour_levels = np.linspace(datamin, datamax, levels)
        plot = axes.contourf(xgrid, ygrid, data, levels=contour_levels, cmap=cmap,
                             vmin=datamin, vmax=datamax)
        cbar = fig.colorbar(plot, ax=axes)
    elif normalization == "symlog":
        norm = colors.SymLogNorm(linthresh=linthresh)
        plot = axes.contourf(xgrid, ygrid, data, levels=levels, norm=norm,
                             cmap=plt.colormaps["RdBu"], vmin=-datamax, vmax=datamax)
        cbar = fig.colorbar(plot, ax=axes)
    else:
        raise ValueError(f"unknown normalization '{normalization}'")

    axes.set_facecolor("black")
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
    momentum_range=(0, -1),
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
    vmin = max(data.min(), 1.0)

    fig = plt.figure(figsize=figsize)
    axes = plt.subplot(1, 1, 1)
    mesh = axes.pcolormesh(p_edges[lo:hi], xi_deg_edges, data,
                           norm=colors.LogNorm(vmin=vmin, vmax=data.max()))
    axes.set_facecolor("black")
    axes.set_ylim(ylim)
    cbar = fig.colorbar(mesh, ax=axes)
    cbar.set_label(cbarlabel)
    axes.set_xlabel(xlabel or r"Momentum normalized to $m_e c$")
    axes.set_ylabel(ylabel or "Pitch angle [degrees]")
    axes.set_title(title)

    return _finish(fig, folder, savename, dpi)


def angle_averaged_momentum_1D(
    run,
    grid="runaway",
    *,
    timestep=0,
    radial_cell=0,
    figsize=FS_1D_DISTRIBUTION,
    momentum_range=(0, -1),
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
