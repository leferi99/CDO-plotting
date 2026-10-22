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
# The kinetic figures only: momentum-pitch maps of f and energy spectra per
# grid, and the angle-averaged f and dn/dp of both grids on one momentum axis.
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

# %%
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
# | `decades` | depth of the log colour scale of the f maps | `12` |
#
# `only` and `skip` take figure-name prefixes. The kinetic names are
# `f_hot_`, `f_re_` (momentum-pitch maps), `spectrum_hot_`, `spectrum_re_`
# (dn/dE), `f_combined_` and `dndp_combined_` (angle-averaged f and dn/dp of
# both grids on one momentum axis, the hot-runaway boundary dashed grey),
# each followed by `<r/a>[_<range>]`. `only=["f_combined_", "dndp_combined_"]`
# keeps just the combined plots; `only=["f_", "spectrum_"]` keeps every map,
# spectrum and combined f.
#
# The `<r/a>` tag is the snapped radius and depends on the run's radial grid:
# on an 11-cell grid r/a = 0 becomes `ra0.05` and 1.0 becomes `ra0.95`. Range
# tags read `Emin-10MeV`, `E1-5MeV` or `pmin-10mec`; the full grid has no
# tag. Copy names from the printed `figures.names` list.

# %% editable=true deletable=true slideshow={"slide_type": ""}
sel = Selection(
    kinetic=KineticSelection(
        times_ms=[0, 1, 2, 5, 10, 15, 20, 25, 30, 40, 50, "end"],
        radii=[0.0, 0.5, 1.0],
        ranges=[(None, 10)],
        range_unit="MeV",
        decades=12,
    ),
    only=["f_combined_"],  #  "dndp_"
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
# figures.build(run, "dndp_combined_ra0.50")
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
# **Directly from `plotting`**, with no names and no selection. The combined
# plot takes times in ms, r/a and a momentum range in m_e c. `moment` is
# `"distribution"` (angle-averaged f, the default) or `"density"` (dn/dp):
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
