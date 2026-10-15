# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # Compare runs
#
# Overlays of I_p, I_RE, I_hot and the volume-averaged temperature for any set
# of runs. Tick runs in the box, or list folders in `RUNS` by hand.

# %%
import os
import sys
from pathlib import Path

REPO = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "cdo" / "__init__.py").exists())
sys.path.insert(0, str(REPO))

import cdo
from cdo import figures

cdo.plotting.use_base_style(fontsize=11)

# %%
picker = cdo.pick_runs()

# %%
DEFAULT_RUNS = os.environ.get(
    "CDO_RUNS",
    "/project/nr_dream002/DREAM-runs/output/iter_dthmode24_rp_boltzmann-eqterm:"
    "/project/nr_dream002/DREAM-runs/output/iter_dthmode24_boltzmann_boltzmann-eqterm",
).split(":")
RUNS = picker.selected or [Path(p) for p in DEFAULT_RUNS]
runs = {Path(folder).name: cdo.Run.from_folder(folder) for folder in RUNS}
for label, run in runs.items():
    print(f"{label}: {run.timegrid_ms[-1]:.4g} ms, {len(run.files)} legs")

# %% [markdown]
# ## Options
# `time_shift_ms` moves a run on the time axis, for example to JOREK time.
# `styles` sets line styles per run label. `references` adds curves or points
# to a figure: `{"compare_plasma_current": [(label, t_ms, I_MA, style)]}`.

# %%
time_shift_ms = {}
styles = {}
references = {}

OUT = os.environ.get("CDO_OUT")  # a folder to save PNGs instead of showing them
made = figures.compare(runs, out=OUT, time_shift_ms=time_shift_ms, styles=styles,
                       references=references)
print(made)
