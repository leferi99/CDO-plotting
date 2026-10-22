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


#: Band edges of :func:`critical_field_ratio_bands_2D`, linear values of E/E_c.
RATIO_BANDS = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0)


def _ratio_band_colours(levels, cmap):
    """One colour per band: blues from the bottom up to 1, reds above, from ``cmap``."""
    n_below = sum(1 for lo in levels[:-1] if lo < 1.0)
    n_above = len(levels) - 1 - n_below
    below = cmap(np.linspace(0.05, 0.42, n_below)) if n_below else np.empty((0, 4))
    above = cmap(np.linspace(0.58, 0.95, n_above)) if n_above else np.empty((0, 4))
    return np.vstack([below, above])


def critical_field_ratio_bands_2D(
    E,
    Ec,
    xgrid,
    ygrid,
    *,
    levels=RATIO_BANDS,
    figsize=FS_2D_MESH,
    cmap="RdBu_r",
    xlabel=None,
    ylabel=None,
    title=None,
    cbarlabel=r"$|E_\parallel| / E_{\rm c,eff}$",
    contour_label=r"$|E_\parallel| = E_{\rm c,eff}$",
    ylim=(None, None),
    folder=None,
    savename=None,
    dpi=150,
):
    """``|E| / Ec`` over time and radius in discrete colour bands, for reading by eye.

    Each cell is drawn (pcolormesh, nearest shading) in the colour of the band
    its value falls in. ``levels`` are the band edges in linear values, finer
    near 1 and coarser above; bands below 1 are blue, above 1 red, and values
    past the last edge take the darkest red, marked by the arrow on the colour
    bar. The bar gives every band the same height and labels the edges. The
    ``|E| = Ec`` contour is drawn in black, labelled ``contour_label`` in a
    legend above the axes. Missing cells are white. ``E`` and
    ``Ec`` are ``(time, radius)``.
    """
    E = np.asarray(E, dtype=float)
    Ec = np.asarray(Ec, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.abs(E) / Ec
    ratio = np.ma.masked_invalid(ratio)
    levels = list(levels)
    base = plt.colormaps[cmap] if isinstance(cmap, str) else cmap
    band_colours = _ratio_band_colours(levels, base)
    listed = colors.ListedColormap(band_colours).with_extremes(
        over=base(1.0), under=base(0.0), bad="white")
    norm = colors.BoundaryNorm(levels, ncolors=listed.N, clip=False)

    fig = plt.figure(figsize=figsize)
    axes = plt.subplot(1, 1, 1)
    mesh = axes.pcolormesh(xgrid, ygrid, ratio, cmap=listed, norm=norm, shading="nearest")
    if np.nanmin(ratio.filled(np.nan)) < 1.0 < np.nanmax(ratio.filled(np.nan)):
        axes.contour(xgrid, ygrid, ratio.filled(np.nan), levels=[1.0], colors="k",
                     linewidths=1.5)
        axes.legend([matplotlib.lines.Line2D([], [], color="k", linewidth=1.5)],
                    [contour_label], fontsize="small", loc="lower right",
                    bbox_to_anchor=(1.0, 1.0), borderaxespad=0.2)
    cbar = fig.colorbar(mesh, ax=axes, ticks=levels, spacing="uniform", extend="max")
    cbar.ax.set_yticklabels([f"{v:g}" for v in levels])
    cbar.set_label(cbarlabel)
    axes.set_ylim(ylim)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title)

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
    axes.set_xlabel(xlabel or MOMENTUM_LABEL)
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
        ax.set_xlabel(MOMENTUM_LABEL, fontsize="small")
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
    axes.set_xlabel(xlabel or MOMENTUM_LABEL)
    axes.set_ylabel(ylabel or r"Angle-averaged distribution [1/m$^3$]")
    axes.set_title(title)

    return _finish(fig, folder, savename, dpi)


#: Momentum axis label: DREAM's p/xi grids store the total momentum |p|.
MOMENTUM_LABEL = r"Total momentum $|p|$ [$m_e c$]"
#: Energy axis label.
ENERGY_LABEL = "Kinetic energy [MeV]"
#: Electron rest energy in MeV.
_MEC2_MEV = 0.51099895

#: y-axis labels of :func:`angle_averaged_combined_1D`, by moment.
_COMBINED_YLABELS = {
    "distribution": r"$\langle f\rangle$ [m$^{-3}$ $(m_e c)^{-3}$]",
    "density": r"$dn/dp$ [m$^{-3}$ $(m_e c)^{-1}$]",
    "energy": r"$dn/dE$ [m$^{-3}$ MeV$^{-1}$]",
}


def _kinetic_MeV(p):
    return (np.sqrt(1.0 + np.asarray(p, dtype=float) ** 2) - 1.0) * _MEC2_MEV


def _per_MeV(run, prefix):
    """Factor turning dn/dp on a grid into dn/dE per MeV, ``(p,)``."""
    from .energy import J_PER_ENERGY_UNIT, EnergyGrid

    grid = EnergyGrid.from_momentum(run.reference.read(f"{prefix}_p"),
                                    run.reference.read(f"{prefix}_p_edges"))
    return grid.dp_dE * J_PER_ENERGY_UNIT["MeV"]


def _start_at(axes, lower, axis="x"):
    """Start an axis at ``lower``, 0 for an open range, keeping the upper limit."""
    lower = 0.0 if lower is None else lower
    if axis == "x":
        axes.set_xlim(left=lower)
    else:
        axes.set_ylim(bottom=lower)

#: Line styles of the analytic curves, by position in the ``analytic`` list.
ANALYTIC_STYLES = ("--", ":", "-.")
#: Data colour of a single-time figure, from viridis.
SINGLE_TIME_SHADE = 0.3
#: Pitch cells drawn by default in :func:`pitch_resolved_1D`.
DEFAULT_PITCH_CELLS = 4
#: Largest angle from the beam, in degrees, of an automatic pitch-cell pick.
DEFAULT_PITCH_ANGLE_MAX_DEG = 20.0
#: Run-value lines in an annotation box before the rest is cut.
_BOX_LINES = 8


def _analytic_list(analytic) -> list:
    """``None``, one :class:`~cdo.avalanche.AvalancheParams` or several, as a list."""
    if analytic is None:
        return []
    if isinstance(analytic, (list, tuple)):
        return list(analytic)
    return [analytic]


def _annotate(axes, lines, loc="lower left"):
    """A small text box inside the axes, empty for decreasing f(p) at lower left."""
    if not lines:
        return
    from matplotlib.offsetbox import AnchoredText

    box = AnchoredText("\n".join(lines), loc=loc, prop=dict(fontsize="x-small"),
                       frameon=True, borderpad=0.4)
    box.patch.set(facecolor="white", alpha=0.85, edgecolor="0.6")
    axes.add_artist(box)


def _cached_angle_average(run, grid, moment):
    """``run.angle_average`` kept in the run's derived cache, shared with cdo.figures."""
    cache = getattr(run, "derived_cache", None)
    if cache is None:
        return run.angle_average(grid, moment)
    key = f"{'dndp' if moment == 'density' else 'favg'}_{grid}"
    if key not in cache:
        value = run.angle_average(grid, moment)
        if value is None:
            return None
        cache[key] = value
    return cache[key]


def angle_averaged_combined_1D(
    run,
    moment="distribution",
    *,
    times_ms="auto",
    r_over_a=0.0,
    p_range=(None, None),
    figsize=(9, 5),
    ylim=(None, None),
    analytic=None,
    grids=("hottail", "runaway"),
    xlabel=None,
    ylabel=None,
    title=None,
    folder=None,
    savename=None,
    dpi=150,
):
    """Hot-tail and runaway angle averages joined on one momentum axis.

    ``grids`` picks the grids, in order of momentum: both by default, or
    ``("runaway",)`` for the runaway grid alone, without the boundary line.

    ``moment`` is ``"distribution"`` for the pitch-averaged f, ``"density"``
    for dn/dp, both against total momentum |p|, or ``"energy"`` for dn/dE per
    MeV against kinetic energy. ``times_ms`` is a list of times in ms and ``"end"`` entries, or
    ``"auto"`` for the standard ladder of the kinetic figures. ``r_over_a`` is
    the radius as r/a. Both are snapped to the stored data, as a
    :class:`~cdo.selection.Selection` is. ``p_range`` keeps the momentum cells
    inside ``(min, max)`` in m_e c, ``None`` meaning the grid edge.

    Each time gives one curve, the hot-tail grid followed by the runaway grid
    in the same colour, early times light and late times dark. A single time
    is drawn in a darker blue. A grey dashed vertical line marks the
    hot-runaway boundary, the upper momentum edge of the hot-tail grid, when it
    lies inside ``p_range``. Non-positive values are left out of the log
    y-axis.

    ``analytic`` is one :class:`~cdo.avalanche.AvalancheParams` or a list of
    them, for example ``[AvalancheParams(), AvalancheParams.from_run()]``. Each
    adds the analytic avalanche distribution over the runaway-grid momenta,
    averaged the same way as the data, with the line style of its position in
    the list (``ANALYTIC_STYLES``), in the colour of the data of each time. A box at lower left lists the values used, per
    time for values taken from the run. Times with no runaways get no curve.
    """
    from .selection import snap_radii, snap_times  # selection imports this module

    if moment not in _COMBINED_YLABELS:
        raise ValueError(f"moment must be one of {list(_COMBINED_YLABELS)}, got {moment!r}")
    energy = moment == "energy"
    basis = "density" if energy else moment
    grids = tuple(grids)
    prefixes = {"hottail": "hot", "runaway": "re"}
    parts = []
    for grid in grids:
        values = _cached_angle_average(run, grid, basis)
        if values is None:
            raise ValueError(f"the {grid} grid must be enabled")
        if energy:
            values = values * _per_MeV(run, prefixes[grid])
        parts.append(values)
    data = np.concatenate(parts, axis=-1)
    p = np.concatenate([run.reference.read(f"{prefixes[g]}_p") for g in grids])
    joined = len(grids) > 1
    p_boundary = run.reference.read(f"{prefixes[grids[0]]}_p_edges")[-1] if joined else None
    x = _kinetic_MeV(p) if energy else p
    if joined:
        x_boundary = float(_kinetic_MeV(p_boundary)) if energy else p_boundary
    n_re = parts[-1].shape[-1] if grids[-1] == "runaway" else 0
    lo, hi = p_range
    keep = np.ones(p.shape, dtype=bool)
    if lo is not None:
        keep &= p >= lo
    if hi is not None:
        keep &= p <= hi
    keep_re = keep[len(p) - n_re:]
    times = snap_times(run.timegrid_ms, times_ms, lines=False)
    radius = snap_radii(run.radialgrid, run.minor_radius, [r_over_a])[0]
    models = _analytic_list(analytic) if n_re else []
    single = len(times) == 1

    box = []
    with plt.rc_context(LINE_RC):
        fig, axes = plt.subplots(figsize=figsize)
        if single:
            palette = [plt.cm.viridis(SINGLE_TIME_SHADE)]
        else:
            palette = plt.cm.viridis(np.linspace(0.9, 0.0, max(len(times), 1)))
        for c, tm in zip(palette, times):
            values = data[tm.index, radius.cell]
            values = np.where(values > 0, values, np.nan)
            axes.plot(x[keep], values[keep], color=c, label=tm.label)
            for k, params in enumerate(models):
                resolved = params.resolve(run, tm.index, radius.cell)
                if params.run_parameters:
                    box.append(f"{params.kind}, {tm.label}: {resolved.text}")
                if resolved.drawable:
                    _plot_analytic_average(axes, run, resolved, moment, radius.cell, keep_re,
                                           color=c,
                                           linestyle=ANALYTIC_STYLES[k % len(ANALYTIC_STYLES)])
        for k, params in enumerate(models):
            colour = palette[0] if single else "grey"
            axes.plot([], [], color=colour, linestyle=ANALYTIC_STYLES[k % len(ANALYTIC_STYLES)],
                      label=f"analytic, {params.kind}")
            if not params.run_parameters:
                box.insert(0, f"set values: {params.values_text}")
        if joined and (lo is None or p_boundary >= lo) and (hi is None or p_boundary <= hi):
            axes.axvline(x_boundary, color="grey", linestyle="--", linewidth=1.0)
        axes.set_yscale("log")
        axes.set_ylim(ylim)
        _start_at(axes, None if lo is None else (float(_kinetic_MeV(lo)) if energy else lo))
        axes.set_xlabel(xlabel or (ENERGY_LABEL if energy else MOMENTUM_LABEL))
        axes.set_ylabel(ylabel or _COMBINED_YLABELS[moment])
        axes.set_title(title or radius.label, fontsize="medium")
        axes.legend(fontsize="small", loc="upper left", bbox_to_anchor=(1.01, 1.0))
        _annotate(axes, _cap(box))

    return _finish(fig, folder, savename, dpi)


def _cap(lines, limit=_BOX_LINES):
    """At most ``limit`` lines, the rest replaced by a count."""
    if len(lines) <= limit:
        return lines
    return lines[:limit] + [f"... {len(lines) - limit} more"]


def _plot_analytic_average(axes, run, resolved, moment, radial_cell, keep, **style):
    """Analytic angle average on the runaway-grid momenta inside ``keep``."""
    from . import avalanche

    energy = moment == "energy"
    values = avalanche.angle_average(run, "runaway", radial_cell, resolved.params,
                                     resolved.n_re, "density" if energy else moment)
    p = run.reference.read("re_p")
    if energy:
        values = values * _per_MeV(run, "re")
    values = np.where(values > 0, values, np.nan)
    axes.plot((_kinetic_MeV(p) if energy else p)[keep], values[keep], **style)


def pick_pitch_cells(xi, cells=DEFAULT_PITCH_CELLS, *, direction=1,
                     max_angle_deg=DEFAULT_PITCH_ANGLE_MAX_DEG) -> np.ndarray:
    """Indices of the pitch cells to draw, sorted.

    ``cells`` is ``"all"``, a list of indices into ``xi`` (negative from the
    end), or a number N: N cells spread evenly in angle from the cell nearest
    the beam (xi = ``direction``) out to ``max_angle_deg`` from it, each
    target snapped to the nearest cell. Targets landing on one cell give one.
    """
    xi = np.asarray(xi, dtype=float)
    if cells is None or (isinstance(cells, str) and cells == "all"):
        return np.arange(len(xi))
    if isinstance(cells, str):
        raise ValueError(f"pitch cells must be 'all', a number or a list, got {cells!r}")
    if np.ndim(cells) == 0:
        n = int(cells)
        if n < 1:
            raise ValueError(f"need at least one pitch cell, got {cells!r}")
        angle = np.degrees(np.arccos(np.clip(direction * xi, -1, 1)))
        start = float(angle.min())
        targets = np.linspace(start, max(max_angle_deg, start), n) if n > 1 else [start]
        return np.unique([int(np.argmin(np.abs(angle - t))) for t in targets])
    return np.unique(np.arange(len(xi))[list(cells)])


def pitch_resolved_1D(
    run,
    grid="runaway",
    *,
    times_ms="auto",
    r_over_a=0.0,
    p_range=(None, None),
    pitch_cells=DEFAULT_PITCH_CELLS,
    max_angle_deg=DEFAULT_PITCH_ANGLE_MAX_DEG,
    analytic=None,
    decades=12,
    cmap="viridis",
    ncols=None,
    panel_size=(4.6, 3.6),
    title=None,
    folder=None,
    savename=None,
    dpi=150,
):
    """f against momentum, one line per pitch cell, at one radius and several times.

    One panel per time. ``pitch_cells`` picks the cells, see
    :func:`pick_pitch_cells`: by default 4 spread from the beam out to
    ``max_angle_deg``, or ``"all"``, or a list of indices (negative from the
    end, so ``[-1, -2, -3]`` are the three cells nearest xi = +1).
    ``run.reference.read("re_xi")`` lists the cell centres. Data lines are
    coloured by cell from ``cmap``.

    ``analytic`` is one :class:`~cdo.avalanche.AvalancheParams` or a list.
    Each adds the analytic f of every drawn cell, averaged over the cell, in
    the colour of that cell's data and the line style of its
    position in the list (``ANALYTIC_STYLES``), on the runaway grid only. Set
    values go in the title, run values in a box in each panel. The y-axis
    spans ``decades`` below the largest drawn data value. ``times_ms``,
    ``r_over_a`` and ``p_range`` (m_e c) are as in
    :func:`angle_averaged_combined_1D`.
    """
    from . import avalanche
    from .selection import snap_radii, snap_times  # selection imports this module

    prefix = "hot" if grid == "hottail" else "re"
    models = _analytic_list(analytic)
    if grid == "hottail":
        models = []  # the avalanche formula describes the runaways only
    read = run.reference.read
    xi = read(f"{prefix}_xi")
    if xi is None or len(xi) < 2:
        raise ValueError(f"the {grid} grid has no pitch resolution")
    f = run.field(f"f_{prefix}")
    p = read(f"{prefix}_p")
    direction = models[0].direction if models else 1
    cells = pick_pitch_cells(xi, pitch_cells, direction=direction, max_angle_deg=max_angle_deg)
    lo, hi = p_range
    keep = np.ones(p.shape, dtype=bool)
    if lo is not None:
        keep &= p >= lo
    if hi is not None:
        keep &= p <= hi
    times = snap_times(run.timegrid_ms, times_ms, lines=False)
    radius = snap_radii(run.radialgrid, run.minor_radius, [r_over_a])[0]

    shades = np.linspace(0.0, 0.9, len(cells)) if len(cells) > 1 else np.array([0.3])
    data_colours = plt.get_cmap(cmap)(shades)

    n = len(times)
    ncols = ncols or min(n, 3)
    nrows = math.ceil(n / ncols)
    vmax = 0.0
    with plt.rc_context(LINE_RC):
        fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, squeeze=False,
                                 layout="constrained",
                                 figsize=(panel_size[0] * ncols + 2.4, panel_size[1] * nrows + 0.8))
        for ax, tm in zip(axes.flat, times):
            data = f[tm.index, radius.cell][:, keep]
            for c, j in zip(data_colours, cells):
                values = np.where(data[j] > 0, data[j], np.nan)
                if np.any(np.isfinite(values)):
                    vmax = max(vmax, float(np.nanmax(values)))
                ax.plot(p[keep], values, color=c,
                        label=f"$\\xi$ = {xi[j]:.4g} ({np.degrees(np.arccos(np.clip(xi[j], -1, 1))):.3g}°)")
            box = []
            for k, params in enumerate(models):
                resolved = params.resolve(run, tm.index, radius.cell)
                if params.run_parameters:
                    box.append(f"{params.kind}: {resolved.text}")
                if not resolved.drawable:
                    continue
                model = avalanche.on_grid(run, grid, resolved.params, resolved.n_re)[:, keep]
                for c, j in zip(data_colours, cells):
                    ax.plot(p[keep], np.where(model[j] > 0, model[j], np.nan), color=c,
                            linestyle=ANALYTIC_STYLES[k % len(ANALYTIC_STYLES)])
            _annotate(ax, box)
            ax.set_yscale("log")
            ax.set_title(tm.label, fontsize="small")
        for ax in axes.flat[n:]:
            ax.set_visible(False)
        if vmax > 0:
            axes.flat[0].set_ylim(vmax * 10.0**-decades, vmax * 3)
        _start_at(axes.flat[0], lo)
        for ax in axes[-1]:
            ax.set_xlabel(MOMENTUM_LABEL, fontsize="small")
        for ax in axes[:, 0]:
            ax.set_ylabel(r"$f$ [m$^{-3}$ $(m_e c)^{-3}$]", fontsize="small")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        for k, params in enumerate(models):
            handles.append(matplotlib.lines.Line2D(
                [], [], color="grey", linestyle=ANALYTIC_STYLES[k % len(ANALYTIC_STYLES)]))
            labels.append(f"analytic, {params.kind}")
        fig.legend(handles, labels, fontsize="x-small", loc="outside right upper")
        heading = title or f"{'hot-tail' if grid == 'hottail' else 'runaway'} grid, {radius.label}"
        for params in models:
            if not params.run_parameters:
                heading += f"\nset values: {params.values_text}"
        fig.suptitle(heading, fontsize="medium")

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
