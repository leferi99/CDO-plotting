"""Runaway energy-spectrum figures, as REPL cell (# %%) script.

Plots the runaway distribution as a spectrum in kinetic energy, dn/(dE dr), at
several times and radii. Point `DATA_FOLDER` at a run with a runaway grid.
"""

# %%
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

import cdo
from cdo import labels, plotting
from cdo.energy import EnergyGrid, MEC2_J

DATA_FOLDER = os.environ.get(
    "CDO_DATA_FOLDER",
    str(Path.home() / "nr_dream002/DREAM-runs/output/iter_dthmode24_rp_commit-53d8afb"),
)
OUTPUT_FOLDER = os.environ.get("CDO_OUTPUT_FOLDER", "")
SAVE = bool(OUTPUT_FOLDER)

plotting.use_base_style(latex=False, sansserif=True)


# %%
# Load the run and build the energy grid.
run = cdo.Run.from_folder(DATA_FOLDER)
energy = EnergyGrid.from_run(run)
if energy is None:
    raise SystemExit("this run has no runaway grid")

# dn/(dE dr): transform the density moment from momentum to energy.
density_spectrum_p = run.angle_average("runaway", "density")  # dn/dp per radius
density_spectrum_E = energy.to_energy(density_spectrum_p)  # dn/dE per radius
kinetic_energy_MeV = (energy.total_energy_eV - 510998.95) / 1e6

print(run.report())


# %%
# Spectrum at a chosen radius, coloured by time.
RADIAL_CELL = 0
times_ms = np.arange(0, run.timegrid_ms[-1] + 1, max(run.timegrid_ms[-1] / 6, 1))
frames = plotting.index_array(run.timegrid_ms, times_ms)
frames = np.clip(frames, 0, run.timegrid_length - 1)
line_colors = plt.cm.viridis(np.linspace(0.9, 0.0, len(frames)))

fig = plt.figure(figsize=(8, 5))
for c, i in zip(line_colors, frames):
    plt.plot(
        kinetic_energy_MeV, density_spectrum_E[i, RADIAL_CELL, :],
        color=c, label=f"{run.timegrid_ms[i]:.0f} ms",
    )
plt.yscale("log")
plt.xlabel(labels.KINETIC_ENERGY)
plt.ylabel(r"$dn/(dE\,dr)$ [1/(J m)]")
plt.title(f"At r = {run.radialgrid[RADIAL_CELL]:.2f} m")
plt.grid(True, which="both", linestyle="--", linewidth=0.3)
plt.legend(fontsize="small")
if SAVE:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    fig.savefig(os.path.join(OUTPUT_FOLDER, "spectrum_vs_time.png"), dpi=200)
    plt.close(fig)


# %%
# Runaway density per radius from the energy spectrum, as a check against n_re.
# These agree in theory and to within tens of percent in a running simulation,
# since n_re is evolved as its own fluid unknown.
n_from_spectrum = energy.integrate(density_spectrum_p)
fig = plt.figure(figsize=(8, 5))
plt.plot(run.timegrid_ms, n_from_spectrum[:, RADIAL_CELL], label="from spectrum")
plt.plot(run.timegrid_ms, run.field("n_re")[:, RADIAL_CELL], "--", label=r"$n_{\rm re}$")
plt.yscale("log")
plt.xlabel(labels.TIME)
plt.ylabel(r"Runaway density [1/m$^3$]")
plt.title(f"At r = {run.radialgrid[RADIAL_CELL]:.2f} m")
plt.legend()
if SAVE:
    fig.savefig(os.path.join(OUTPUT_FOLDER, "density_check.png"), dpi=200)
    plt.close(fig)


# %%
run.close()
