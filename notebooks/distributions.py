# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # Distribution functions of one run
#
# The kinetic figures only: momentum-pitch maps of f, pitch-integrated dn/dp
# and energy spectra, for the hot-tail and runaway grids. Each (radius, range)
# pair is one figure, with one panel or curve per time.

# %%
import os
import sys
from pathlib import Path

REPO = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "cdo" / "__init__.py").exists())
sys.path.insert(0, str(REPO))

import matplotlib.pyplot as plt

import cdo
from cdo import figures
from cdo.selection import KineticSelection, Selection

cdo.plotting.use_base_style(fontsize=11)

# %%
picker = cdo.pick_runs()

# %%
DEFAULT_RUN = os.environ.get(
    "CDO_RUN", "/project/nr_dream002/DREAM-runs/output/iter_dthmode24_rp_boltzmann-eqterm")
RUN = picker.selected[0] if picker.selected else Path(DEFAULT_RUN)
run = cdo.Run.from_folder(RUN)
print(run.report())

# %% [markdown]
# ## Kinetic selection
# Ranges in MeV of kinetic energy (`range_unit="MeV"`) or in m_e c
# (`range_unit="mec"`). `(None, 5)` is everything up to 5 MeV, a zoomed view.

# %%
sel = Selection(
    kinetic=KineticSelection(
        times_ms=[0.01, 1, 10, 30, "end"],
        radii=[0.0, 0.3, 0.6],
        ranges=[(None, None), (None, 2)],
        range_unit="MeV",
        decades=12,
    ),
    only=["f_", "dndp_", "spectrum_"],
)
print("\n".join(figures.names(run, sel)))

# %%
name = figures.names(run, sel)[0]
figures.build(run, name, sel)
plt.show()

# %%
OUT = os.environ.get("CDO_OUT")  # a folder to save PNGs instead of showing them
if OUT is not None or "ipykernel" in sys.modules:
    result = figures.make_figures(run, sel, out=OUT)
    print(len(result["made"]), "figures,", len(result["skipped"]), "skipped")
