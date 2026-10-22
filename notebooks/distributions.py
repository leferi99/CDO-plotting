# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Distribution functions of one run
#
# The kinetic figures only: momentum-pitch maps of f, f per pitch cell and
# energy spectra per grid, and the angle-averaged f, dn/dp and dn/dE of both grids on
# one momentum axis, optionally with the analytic avalanche distribution.
# Each (radius, range) pair is one figure, with one panel or curve per time.

# %%
import os
import sys
from pathlib import Path

REPO = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "cdo" / "__init__.py").exists())
sys.path.insert(0, str(REPO))

import matplotlib.pyplot as plt

import cdo
from cdo import figures, plotting
from cdo.selection import KineticSelection, Selection

cdo.plotting.use_base_style(fontsize=11)

# %%
picker = cdo.pick_runs(multiple=False)

# %% editable=true deletable=true slideshow={"slide_type": ""}
DEFAULT_RUN = os.environ.get(
    "CDO_RUN", cdo.paths.run_folder("dream_runs", "iter-nrt/iter_dthmode24_rp_boltzmann-eqterm"))
RUN = picker.selected[0] if picker.selected else Path(DEFAULT_RUN)
if not picker.selected:
    print("No run selected in the picker, using DEFAULT_RUN. Select one and re-run this cell.")
run = cdo.Run.from_folder(RUN)
print(run.report())

# %% [markdown]
# ## Kinetic selection
# Only the `kinetic` part of the `Selection` matters here. Every value is
# snapped to the stored data, and titles and figure names show the snapped
# values. `run_overview.py` describes the fluid part.
#
# | Field | Values | Default |
# |---|---|---|
# | `times_ms` | list of ms and `"end"`, or `"auto"` for a standard set by run length | `"auto"` |
# | `radii` | r/a values | `(0.0, 0.5)` |
# | `ranges` | list of `(min, max)`, one figure each, `None` = grid edge | `[(None, None)]` |
# | `range_unit` | `"MeV"` (kinetic energy) or `"mec"` (momentum over m_e c) | `"MeV"` |
# | `decades` | depth of the log colour scale of the f maps and pitch plots | `12` |
# | `pitch_cells` | turns on the `pitch_` plots: a number N of cells spread from the beam, `"all"`, or indices into the xi cells, negative from the end | `None` (no pitch plots) |
# | `pitch_angle_max_deg` | largest angle from the beam when `pitch_cells` is a number | `20` |
# | `analytic` | `AvalancheParams`, or a list of them, for the analytic curves; `None` for none | `None` |
# | `combined_per_time` | one combined figure per time, named `..._t<ms>ms` | `False` |
# | `combined_ranges` | ranges of the combined figures, for the hot-runaway boundary | `[(None, 10)]` |
# | `runaway_zoom_ranges` | extra ranges of `favg_re` and `pitch_re` | `[(None, 40)]` |
#
# `only` and `skip` take figure-name prefixes, or wildcard patterns with `*`,
# `?` or `[...]` matched against the whole name. The kinetic names are
# `f_hot_`, `f_re_` (momentum-pitch maps), `spectrum_hot_`, `spectrum_re_`
# (dn/dE), `favg_re_` (angle-averaged f of the runaway grid against |p|),
# `pitch_hot_`, `pitch_re_` (f against |p|, one line per pitch cell, on grids
# with more than one pitch cell, when `pitch_cells` is set), each followed by
# `<r/a>[_<range>]`; `favg_re_` and `pitch_re_` also in the
# `runaway_zoom_ranges`. The combined figures `f_combined_`, `dndp_combined_`
# (angle-averaged f and dn/dp of both grids against total momentum |p|) and
# `dnde_combined_` (dn/dE against kinetic energy), with the hot-runaway
# boundary dashed grey, come only in the `combined_ranges`. `dndp_combined_` is in `DEFAULT_SKIP`, so it needs
# `skip=[]`. `only=["f_combined_", "dnde_combined_"]`
# keeps just the combined plots; `only=["f_", "spectrum_"]` keeps every map,
# spectrum and combined f. `skip=["spectrum_*_Emin-10MeV"]` drops the spectra
# of that range at every radius, `only=["*_combined_*_Emin-10MeV"]` keeps the
# combined plots of that range only.
#
# The `<r/a>` tag is the snapped radius and depends on the run's radial grid:
# on an 11-cell grid r/a = 0 becomes `ra0.05` and 1.0 becomes `ra0.95`. Range
# tags read `Emin-10MeV`, `E1-5MeV` or `pmin-10mec`; the full grid has no
# tag. Copy names from the printed `figures.names` list.

# %% [markdown]
# ## Analytic avalanche distribution
# Eq. (2.17) of Embréus et al., J. Plasma Phys. 84 (2018) 905840506
# (arXiv:1807.04460): an avalanche-dominated runaway population in a uniform
# plasma, f ∝ exp[-γ/γ0 - A(1 - ξ)] with A = (E/E_c + 1) γ / (Z_tot + 1)
# and γ0 = c_Z lnΛ. The defaults are the DEMO termination case,
# γ0 m_e c² ≈ 18.5 MeV.
#
# | Field | Meaning | Default |
# |---|---|---|
# | `E_over_Ec` | E/E_c | `50` |
# | `Z_tot` | total charge | `1.4` |
# | `lnLambda` | Coulomb logarithm | `14.3` |
# | `c_Z` | `None` for sqrt(5 + Z_tot) | `None` |
# | `n_re` | density in m^-3, `None` for the run's n_re at each time and radius | `None` |
# | `direction` | `+1` beam at ξ = +1 (DREAM with E > 0), `-1` at ξ = -1 (the paper) | `+1` |
#
# A parameter set to `None` comes from the run at each time and radius:
# E/E_c = |E_field| / Ectot, Z = Zeff, lnΛ = lnLambdaC. `AvalancheParams.from_run()`
# takes all three from the run. `ANALYTIC` is a list: each entry draws its own
# curves in the colour of the data, the first dashed, the second dotted, and a
# box on the figure lists the values used. `ANALYTIC = [SET_VALUES, RUN_VALUES]`
# adds the curves with the run's values. The pitch-averaged f does not depend on E/E_c, so on the
# combined plots the run values move the curve only through γ0; on the pitch
# plots E/E_c sets the width in pitch.
#
# The analytic f is averaged over each pitch cell, since its width in ξ
# (about 1/A) is below the cell size near the beam, and the combined plots
# average it the same way as the data. Set `ANALYTIC = None` to leave it out.
# The cell prints the pitch cells, for choosing `pitch_cells`.

# %%
SET_VALUES = cdo.AvalancheParams(E_over_Ec=50, Z_tot=1.4, lnLambda=14.3, c_Z=None,
                                 n_re=None, direction=+1)
RUN_VALUES = cdo.AvalancheParams.from_run(direction=+1)
ANALYTIC = [SET_VALUES]  # [SET_VALUES, RUN_VALUES] to compare with the run's values
print(f"set values: gamma0 = {SET_VALUES.gamma0:.1f}, mean energy = {SET_VALUES.gamma0 * 0.511:.1f} MeV")
xi = run.reference.read("re_xi")
if xi is not None:
    for i, x in enumerate(xi):
        print(f"runaway pitch cell {i:3d} ({i - len(xi):4d}): xi = {x:.5f}")

# %% editable=true deletable=true slideshow={"slide_type": ""}
sel = Selection(
    kinetic=KineticSelection(
        times_ms=[0, 1, 2, 5, 10, 15, 20, 25, 30, 40, 50, "end"],
        radii=[0.0, 0.5, 1.0],
        ranges=[(None, None)],
        range_unit="MeV",
        decades=12,
        pitch_cells=4,  # None, a number, "all" or e.g. [-1, -4, -7]
        pitch_angle_max_deg=20,
        analytic=ANALYTIC,
        combined_per_time=False,
        combined_ranges=[(None, 10)],
        runaway_zoom_ranges=[(None, 40)],
    ),
    only=["f_", "favg_", "dnde_", "pitch_"],
)
fn = figures.names(run, sel)
print("\n".join(fn))

# %% [markdown]
# ## Single figures
# The cell below draws the first five names of the list. One figure is drawn
# by name, with or without a selection, or by calling a plotting function
# directly.
#
# **With the selection above.** The name must be in `figures.names(run, sel)`,
# so it has passed `only` and `skip`:
# ```python
# figures.build(run, "f_combined_ra0.50_Emin-10MeV", sel)
# plt.show()
# ```
#
# **Without a selection.** `figures.build(run, name)` uses the defaults in the
# table above, all times `"auto"`, r/a 0 and 0.5, the full grid. Take the
# name from `figures.names(run)`:
# ```python
# figures.build(run, "dnde_combined_ra0.50")
# plt.show()
# ```
#
# **With a one-off selection**, for a radius, time or range not in `sel`.
# Check the snapped name first, since 0.25 can snap to `ra0.23`:
# ```python
# one = Selection(kinetic=KineticSelection(radii=[0.25], times_ms=[1, 10, "end"],
#                                          ranges=[(1, 5)]))
# print(figures.names(run, one))
# figures.build(run, "f_re_ra0.23_E1-5MeV", one)
# plt.show()
# ```
#
# **Pitch lines directly**, with the analytic curves dashed:
# ```python
# plotting.pitch_resolved_1D(run, "runaway", times_ms=[50, "end"], r_over_a=0.0,
#                            pitch_cells=4, analytic=ANALYTIC)
# plt.show()
# ```
#
# **Directly from `plotting`**, with no names and no selection. The combined
# plot takes times in ms, r/a and a momentum range in m_e c. `moment` is
# `"distribution"` (angle-averaged f, the default), `"density"` (dn/dp) or
# `"energy"` (dn/dE against kinetic energy):
# ```python
# plotting.angle_averaged_combined_1D(run, "density", times_ms=[1, 10, "end"],
#                                     r_over_a=0.5, p_range=(None, 20))
# plt.show()
# ```
# A selection's values pass straight in. `ranges` are in `range_unit`, so
# convert them to m_e c with `Range`:
# ```python
# from cdo.selection import Range
# k = sel.kinetic
# plotting.angle_averaged_combined_1D(
#     run, times_ms=k.times_ms, r_over_a=k.radii[0],
#     p_range=Range(*k.ranges[0], k.range_unit).momentum())
# ```
# `distribution_momentum_2D` and `distribution_momentum_panels` take a
# time-step index and a radial cell index:
# ```python
# from cdo.selection import snap_radii
# step = int(plotting.nearest_indices(run.timegrid_ms, [10])[0])
# cell = snap_radii(run.radialgrid, run.minor_radius, [0.5])[0].cell
# plotting.distribution_momentum_2D(run, "runaway", timestep=step, radial_cell=cell)
# ```
#
# **Editing and saving one figure.** `figures.build` and the plotting
# functions return a matplotlib figure, so `fig.axes[0].set_ylim(1e8, None)`
# and `fig.savefig("name.png", dpi=150, bbox_inches="tight")` work on it. The
# plotting functions also save a PNG and close the figure when given
# `folder=` and `savename=`.

# %%
for i in range(min(5, len(fn))):
    name = fn[i]
    figures.build(run, name, sel)
    plt.show()


# %% [markdown]
# ## The whole set
# Draws every name in `figures.names(run, sel)`. `SAVE = False` shows them
# here. `SAVE = True` saves PNGs to `CDO_FIGURE_ROOT/notebook/<run>/`, plus
# `plot_settings.json` with the requested and snapped selection. `CDO_OUT`
# overrides the folder.

# %%
SAVE = True  # True or 1 saves to CDO_FIGURE_ROOT/notebook/<run>/, False or 0 shows them here
OUT = os.environ.get("CDO_OUT") or (
    cdo.paths.figure_root() / "notebook" / Path(RUN).name if SAVE else None)
if OUT is not None or "ipykernel" in sys.modules:
    result = figures.make_figures(run, sel, out=OUT)
    print(len(result["made"]), "figures,", len(result["skipped"]), "skipped")

# %%
