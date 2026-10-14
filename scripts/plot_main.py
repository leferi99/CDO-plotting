"""Main DREAM run figures, as REPL cell (# %%) script.

Run cell by cell in an editor, or top to bottom with `python scripts/plot_main.py`.
Point `DATA_FOLDER` at a run's output folder and set `OUTPUT_FOLDER` for the PNGs.
Figures use the native reader in `cdo`, so no DREAM install is needed.
"""

# %%
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import cdo
from cdo import labels, plotting

# --- configuration -------------------------------------------------------
# Edit these two paths for the run to plot.
DATA_FOLDER = os.environ.get(
    "CDO_DATA_FOLDER",
    str(Path.home() / "nr_dream002/DREAM-runs/output/iter_dthmode24_rp_commit-53d8afb"),
)
OUTPUT_FOLDER = os.environ.get("CDO_OUTPUT_FOLDER", "")  # empty shows figures instead of saving

SAVE = bool(OUTPUT_FOLDER)
plotting.use_base_style(latex=False, sansserif=True)


# %%
# Load and concatenate the run.
run = cdo.Run.from_folder(DATA_FOLDER)
print(run.report())
endtime = f"{run.timegrid_ms[-1]:.0f}ms"
folder = os.path.join(OUTPUT_FOLDER, endtime) if SAVE else None


# %%
# Currents over time, in megaamperes.
import matplotlib.pyplot as plt

with plt.rc_context(plotting.LINE_RC):
    curves = [run.plasma_current / 1e6]
    names = [r"$I_{\rm p}$"]
    styles = ["-"]

    kinetic = run.runaway_current
    if run.hot_current is not None:
        kinetic = kinetic + run.hot_current
    curves.append(kinetic / 1e6)
    names.append(r"$I_{\rm RE}$")
    styles.append("-")

    i_wall = run.field("I_wall")
    if i_wall is not None:
        curves.append(i_wall.reshape(-1) / 1e6)
        names.append(r"$I_{\rm wall}$")
        styles.append("-")

    curves.append(run.ohmic_current / 1e6)
    names.append(r"$I_{\rm ohm}$")
    styles.append("--")

    plotting.basic_1D(
        curves, run.timegrid_ms,
        labels=names, linestyles=styles,
        xlabel=labels.TIME, ylabel=labels.quantity("currents"),
        xlim=(0, None), ylim=(0, None), legendloc="upper right",
        folder=folder, savename="currents" if SAVE else None,
    )


# %%
# Runaway generation rates, volume-averaged, one curve per channel. With a
# kinetic runaway grid the total is the hot-tail grid flux alone, and the
# Dreicer and avalanche curves are fluid-model estimates outside it.
rates = cdo.derived.generation_rates(run)
if rates:
    curves = [cdo.derived.volume_average(r, run.cell_volumes) for r in rates.values()]
    peak = max(np.nanmax(np.abs(c)) for c in curves)
    with plt.rc_context(plotting.LINE_RC):
        plotting.basic_1D(
            curves, run.timegrid_ms,
            labels=[labels.RATES[key] for key in rates],
            yscale="symlog", ylinthresh=peak * 1e-8,
            yticks=plotting.symlog_ticks(peak * 1e-8, peak),
            xlabel=labels.TIME,
            ylabel=r"Volume-averaged rate [1/(s m$^3$)]",
            title=labels.quantity("re_generation_rates"),
            legendloc="best", legendfontsize="x-small",
            folder=folder, savename="generation_rates" if SAVE else None,
        )


# %%
# Electron temperature over time and radius, logarithmic colour scale.
with plt.rc_context(plotting.MESH_RC):
    plotting.basic_2D(
        run.field("T_cold"), run.radialgrid, run.timegrid_ms,
        normalization="log", logdiff=5,
        xlabel=labels.RADIUS, ylabel=labels.TIME,
        title=labels.quantity("electron_temperature"),
        cbarlabel=r"log$_{10}(T_{\rm cold}$/1eV)",
        folder=folder, savename="temperature" if SAVE else None,
    )


# %%
# Ohmic current density over time and radius.
with plt.rc_context(plotting.MESH_RC):
    plotting.basic_2D(
        np.abs(run.field("j_ohm")), run.radialgrid, run.timegrid_ms,
        normalization="log", logdiff=4,
        xlabel=labels.RADIUS, ylabel=labels.TIME,
        title=labels.quantity("ohmic_current_density"),
        cbarlabel=r"log$_{10}(|j_{\rm ohm}|$/(1Am$^{-2}$))",
        folder=folder, savename="ohmic_current_density" if SAVE else None,
    )


# %%
# Runaway momentum-pitch distribution at the final time step, inner radius.
with plt.rc_context(plotting.MESH_RC):
    plotting.distribution_momentum_2D(
        run, "runaway", timestep=-1, radial_cell=0,
        cbarlabel=labels.quantity("electron_distribution"),
        folder=folder, savename="runaway_distribution" if SAVE else None,
    )


# %%
run.close()
