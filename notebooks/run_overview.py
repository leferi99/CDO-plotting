# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
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
# Tick a run in the box (Jupyter), or set `RUN` by hand below.

# %%
picker = cdo.pick_runs()

# %%
DEFAULT_RUN = os.environ.get(
    "CDO_RUN", "/project/nr_dream002/DREAM-runs/output/iter_dthmode24_rp_boltzmann-eqterm")
RUN = picker.selected[0] if picker.selected else Path(DEFAULT_RUN)
run = cdo.Run.from_folder(RUN)
print(RUN)
print(run.report())

# %% [markdown]
# ## What to look at
# `times_ms`: a list of ms values and `"end"`, or `"auto"` for the standard
# ladder. `radii`: r/a values. `ranges`: one figure per `(min, max)` tuple,
# `None` for the grid edge, in `range_unit` `"MeV"` or `"mec"`.
# `skip` / `only`: figure-name prefixes, see `figures.names(run, sel)`.

# %%
sel = Selection(
    fluid=FluidSelection(
        times_ms="auto",
        radii=[0.0, 0.5, 0.9],
        time_range_ms=(None, None),
    ),
    kinetic=KineticSelection(
        times_ms=[1, 10, 30, "end"],
        radii=[0.0, 0.5],
        ranges=[(None, None), (None, 5)],
        range_unit="MeV",
        decades=12,
    ),
    skip=[],
    only=[],
)
print("\n".join(figures.names(run, sel)))

# %% [markdown]
# ## Single figures

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
# `OUT = None` shows every figure here. A folder saves PNGs there, plus
# `plot_settings.json` with the requested and snapped selection.
# `pdf=True` adds PDF copies in `OUT/pdf/`.

# %%
OUT = os.environ.get("CDO_OUT")  # e.g. REPO / "plots" / "notebook" / Path(RUN).name
if OUT is not None or "ipykernel" in sys.modules:
    result = figures.make_figures(run, sel, out=OUT, pdf=False)
    print(len(result["made"]), "figures,", len(result["skipped"]), "skipped")
