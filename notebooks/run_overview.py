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

# %% [markdown] editable=true deletable=true slideshow={"slide_type": ""}
# # One run: all standard figures
#
# Pick a run, set what to look at, then show single figures or save the whole
# set. Opens as a notebook in JupyterLab through Jupytext, runs cell by cell in
# any `# %%` editor, or top to bottom with `python notebooks/run_overview.py`.
#
# Fluid and kinetic quantities have separate time and radius choices. Radii
# are r/a, snapped to the nearest cell. Times are ms, snapped to the nearest
# saved step. Titles show the snapped values.

# %%
import os
import sys
from pathlib import Path

REPO = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "cdo" / "__init__.py").exists())
sys.path.insert(0, str(REPO))

import matplotlib.pyplot as plt

import cdo
from cdo import figures
from cdo.selection import FluidSelection, KineticSelection, Selection

cdo.plotting.use_base_style(fontsize=11)

# %% [markdown]
# ## Choose the run
# Click one run in the list (Jupyter), or set `DEFAULT_RUN` by hand below.
# The line under the list shows the current choice.

# %%
picker = cdo.pick_runs(multiple=False)

# %%
DEFAULT_RUN = os.environ.get(
    "CDO_RUN", cdo.paths.run_folder("dream_runs", "iter-nrt/iter_dthmode24_rp_boltzmann-eqterm"))
RUN = picker.selected[0] if picker.selected else Path(DEFAULT_RUN)
if not picker.selected:
    print("No run selected in the picker, using DEFAULT_RUN. Select one and re-run this cell.")
run = cdo.Run.from_folder(RUN)
print(RUN)
print(run.report())

# %% [markdown]
# ## What to look at
# A `Selection` has two independent parts: `fluid` drives the fluid figures
# (currents, densities, temperature, fields, rates), `kinetic` drives the
# distribution-function figures. Every value is snapped to the stored data,
# times to the nearest saved step and radii to the nearest radial cell. Titles
# and figure names show the snapped values.
#
# | Part | Field | Values | Default |
# |---|---|---|---|
# | `FluidSelection` | `times_ms` | list of ms and `"end"`, or `"auto"` | `"auto"` |
# | | `radii` | r/a values, for traces and `electric_field_<r/a>` | `(0.0, 0.5, 0.9)` |
# | | `time_range_ms` | `(start, stop)` crop of every fluid time axis, `None` = run edge | `(None, None)` |
# | `KineticSelection` | `times_ms` | list of ms and `"end"`, or `"auto"` | `"auto"` |
# | | `radii` | r/a values | `(0.0, 0.5)` |
# | | `ranges` | list of `(min, max)`, one figure each, `None` = grid edge | `[(None, None)]` |
# | | `range_unit` | `"MeV"` (kinetic energy) or `"mec"` (momentum over m_e c) | `"MeV"` |
# | | `decades` | depth of the log colour scale of the f maps | `12` |
# | | `pitch_cells` | turns on the `pitch_` plots: a number of cells spread from the beam, `"all"`, or a list of cell indices | `None` (off) |
# | | `analytic` | `AvalancheParams`, or a list, for the dashed analytic avalanche curves | `None` (off) |
# | | `combined_per_time` | one combined figure per time, `..._t<ms>ms` | `False` |
# | `Selection` | `skip` | figure names to leave out, as prefixes or wildcard patterns | `[]` |
# | | `only` | keep only names matching one of these prefixes or patterns | `[]` |
#
# Fields left out keep their defaults, so
# `Selection(kinetic=KineticSelection(radii=[0.5]))` changes only the kinetic
# radii. `"auto"` picks a standard set of times from the run length.
#
# A plain `skip`/`only` entry matches every name starting with it:
# `skip=["rate_map"]` drops all `rate_map_<channel>` figures. An entry with
# `*`, `?` or `[...]` is a wildcard pattern over the whole name:
# `skip=["spectrum_*_Emin-10MeV"]` drops the spectra of that range at every
# radius, `only=["*_ra0.05*"]` keeps every figure at r/a 0.05, and
# `skip=["*_map"]` drops every `<quantity>_map`. Both kinds mix in one list.
#
# ### Figure names
# `figures.names(run, sel)` prints every name the run can make with `sel`.
#
# - Fluid: `currents`; `<quantity>_traces`, `<quantity>_profiles` and
#   `<quantity>_map` for `temperature`, `electron_density`,
#   `runaway_density`, `hot_density`, `ohmic_current_density`,
#   `runaway_current_density`, `hot_current_density`,
#   `total_current_density`, `radiated_power`, `effective_charge` (those the
#   run has); `electric_field_<r/a>`, `electric_field_average`,
#   `electric_field_profiles`, `critical_field_ratio_map`,
#   `generation_rates`, `rate_map_<channel>`.
# - Kinetic, per radius and range: `f_hot_`, `f_re_` (momentum-pitch maps),
#   `spectrum_hot_`, `spectrum_re_` (dn/dE), `pitch_hot_`, `pitch_re_` (f
#   against p per pitch cell, when `pitch_cells` is set), `f_combined_`,
#   `dndp_combined_` (angle-averaged f and dn/dp of both grids on one momentum
#   axis), each followed by `<r/a>[_<range>]`.
#
# ### Analytic avalanche distribution
# `ANALYTIC` holds the parameters of the analytic avalanche f of Embréus et
# al. 2018, eq. 2.17, drawn dashed on the combined and pitch figures in the
# colour of the data. The defaults are the DEMO termination case. The
# `distributions.py` notebook describes every parameter and the option to
# take E/E_c, Zeff and lnΛ from the run. `ANALYTIC = None` leaves it out.
#
# The `<r/a>` tag is the snapped radius, so it depends on the run's radial
# grid. On an 11-cell grid r/a = 0 becomes `ra0.05` and 0.9 becomes
# `ra0.86`. Range tags read `Emin-10MeV`, `E1-5MeV` or `pmin-10mec`; the
# full grid has no tag. Copy names from the `figures.names` output.

# %%
ANALYTIC = cdo.AvalancheParams(E_over_Ec=50, Z_tot=1.4, lnLambda=14.3)

# %% editable=true deletable=true slideshow={"slide_type": ""}
sel = Selection(
    fluid=FluidSelection(
        times_ms="auto",
        radii=[0.0, 0.25, 0.5, 0.75, 0.9],
        time_range_ms=(None, None),
    ),
    kinetic=KineticSelection(
        times_ms=[0, 1, 2, 5, 10, 15, 20, 25, 30, 40, 50, "end"],
        radii=[0.0, 0.25, 0.5, 0.75, 0.9],
        ranges=[(None, None), (None, 10)],  # , (None, 5)
        range_unit="MeV",
        decades=12,
        pitch_cells=None,  # 4 for the pitch_ plots
        analytic=ANALYTIC,
    ),
    skip=["dndp_", "f_re_*10MeV", "f_hot_*10MeV", "spectrum*10MeV"],
    only=[],
)

print("\n".join(figures.names(run, sel)))

# %% [markdown]
# ## Single figures
# One figure is drawn by name, with or without a selection, or by calling a
# plotting function directly.
#
# **With the selection above.** The name must be in `figures.names(run, sel)`.
# `skip` and `only` apply here too: with `only=["f_combined"]`,
# `figures.build(run, "currents", sel)` raises `KeyError`.
# ```python
# figures.build(run, "temperature_traces", sel)
# plt.show()
# ```
#
# **Without a selection.** `figures.build(run, name)` uses the defaults in the
# table above. Names that carry a radius must then come from
# `figures.names(run)`, the default list.
# ```python
# figures.build(run, "currents")
# figures.build(run, "f_combined_ra0.50")  # r/a 0.5 is a default kinetic radius
# plt.show()
# ```
#
# **With a one-off selection**, for a radius, time or range not in `sel`.
# Check the snapped name first, since 0.25 can snap to `ra0.23`:
# ```python
# one = Selection(kinetic=KineticSelection(radii=[0.25], times_ms=[1, 10, "end"],
#                                          ranges=[(1, 5)]))
# print(figures.names(run, one))
# figures.build(run, "f_combined_ra0.23_E1-5MeV", one)
# plt.show()
# ```
#
# **Directly from `cdo.plotting`**, with no names and no selection. The
# combined hot-tail and runaway plot takes times in ms, r/a and a momentum
# range in m_e c. `moment` is `"distribution"` (angle-averaged f) or
# `"density"` (dn/dp):
# ```python
# from cdo import plotting
# plotting.angle_averaged_combined_1D(run, "density", times_ms=[1, 10, "end"],
#                                     r_over_a=0.5, p_range=(None, 20))
# ```
# A selection's values pass straight in. Ranges in the selection are in
# `range_unit`, so convert them to m_e c with `Range`:
# ```python
# from cdo.selection import Range
# k = sel.kinetic
# plotting.angle_averaged_combined_1D(
#     run, times_ms=k.times_ms, r_over_a=k.radii[0],
#     p_range=Range(*k.ranges[0], k.range_unit).momentum())
# ```
# The other distribution plotters, `distribution_momentum_2D` and
# `distribution_momentum_panels`, take a time-step index and a radial cell
# index:
# ```python
# from cdo.selection import snap_radii
# step = int(plotting.nearest_indices(run.timegrid_ms, [10])[0])
# cell = snap_radii(run.radialgrid, run.minor_radius, [0.5])[0].cell
# plotting.distribution_momentum_2D(run, "runaway", timestep=step, radial_cell=cell)
# ```
#
# **Editing and saving one figure.** `figures.build` returns a matplotlib
# figure:
# ```python
# fig = figures.build(run, "temperature_traces", sel)
# fig.axes[0].set_xlim(0, 10)
# fig.savefig("temperature_traces.png", dpi=150, bbox_inches="tight")
# ```
# The `cdo.plotting` functions save a PNG and close the figure when given
# `folder=` and `savename=`, and return it open otherwise.
#
# **A few figures at once.** Put their name prefixes or patterns in `only` and call
# `figures.make_figures(run, Selection(only=["currents", "temperature_map"]))`,
# which shows them inline, or add `out=<folder>` to save them.

# %%
figures.build(run, "currents", sel)
plt.show()

# %%
figures.build(run, "generation_rates", sel)
plt.show()

# %%
figures.build(run, "electric_field_average", sel)
plt.show()

# %% [markdown]
# ## The whole set
# `SAVE = False` shows every figure here. `SAVE = True` saves PNGs to
# `CDO_FIGURE_ROOT/notebook/<run>/`, plus `plot_settings.json` with the
# requested and snapped selection. `CDO_OUT` overrides the folder.
# `pdf=True` adds PDF copies in `OUT/pdf/`.

# %%
SAVE = True  # True or 1 saves to CDO_FIGURE_ROOT, False or 0 shows the figures here
OUT = os.environ.get("CDO_OUT") or (
    cdo.paths.figure_root() / "notebook" / Path(RUN).name if SAVE else None)
if OUT is not None or "ipykernel" in sys.modules:
    result = figures.make_figures(run, sel, out=OUT, pdf=False)
    print(len(result["made"]), "figures,", len(result["skipped"]), "skipped")

# %% editable=true deletable=true slideshow={"slide_type": ""}
