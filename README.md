# CDO-plotting

Consistent plotting of [DREAM](https://github.com/chalmersplasmatheory/DREAM) output files across DREAM versions and run setups. CDO stands for Concatenated DREAM Output.

A run is usually split across several HDF5 output files, and the datasets present and the paths they live at vary with the DREAM version and with how the run was invoked. The `cdo` package reads any of these into one concatenated series through a canonical-name layer, computes the moments and integrals natively, and plots them. Reading and the moments take no DREAM import, so the package works on output that the DREAM Python interface can no longer open.

## Requirements

Python with `numpy`, `scipy`, `h5py` and `matplotlib`.
The notebooks also use `jupyterlab`, `jupytext` and `ipywidgets`.
A DREAM install is not needed for reading, moments, or plotting.
It is used only by the optional cross-check tests.

The project venv is `.venv` in this folder, separate from the `DREAM-runs` venv the simulations use:

```bash
/opt/cray/pe/python/3.11.7/bin/python -m venv .venv
.venv/bin/python -m pip install numpy scipy h5py matplotlib pytest jupyterlab jupytext ipywidgets
```

## Layout

| Path | Contents |
|---|---|
| `cdo/` | The package: reading, physics, figures. |
| `notebooks/` | `# %%` cell scripts that open as notebooks in JupyterLab. |
| `scripts/plot_all_runs.py` | Batch command line: every run of every campaign. |
| `tests/` | `pytest` suite. |
| `docs/` | Field schema, volume and moment conventions. |
| `plots/` | Figure output, not tracked. Older figures are in `plots/archive/`. |

## Data location

All run folders sit under one data root, with the same layout on every machine:

| Folder under the root | Runs |
|---|---|
| `DREAM-runs/output/` | ITER DThmode24 and DEMO_2026 runs |
| `DEMO_data_2026/runs/` | DEMO JOREK current-quench runs |

On the cluster the root is `/project/nr_dream002`, used when nothing is set.
Elsewhere, set `CDO_DATA_ROOT` to the root of a synced copy, for example in fish:

```fish
set -Ux CDO_DATA_ROOT ~/Data/DREAM
```

The plotting reads only the committed legs, `output_NNNN.h5`, and each run's `manifest.json`.
These rsync commands copy just those, about 6 GB, and on later runs only new legs:

```bash
rsync -avH --prune-empty-dirs --exclude='archive_*/' --exclude='*.inprogress.h5' \
    --include='*/' --include='output_[0-9]*.h5' --include='manifest.json' --exclude='*' \
    komondor-2026:/project/nr_dream002/DREAM-runs/output/ ~/Data/DREAM/DREAM-runs/output/
rsync -avH --prune-empty-dirs --exclude='*.inprogress.h5' \
    --include='*/' --include='output_[0-9]*.h5' --include='manifest.json' --exclude='*' \
    komondor-2026:/project/nr_dream002/DEMO_data_2026/runs/ ~/Data/DREAM/DEMO_data_2026/runs/
```

`cdo.paths` resolves the root and the collection folders.
The run picker, the notebook defaults and the batch script all go through it.

Saved figures go under one figure root, `CDO_FIGURE_ROOT` if set, the repository's `plots/` otherwise:

| Folder under the figure root | Figures |
|---|---|
| `<group>/<run>/` | Batch script, one run |
| `<group>/_comparison/` | Batch script, cross-run overlays of a group |
| `notebook/<run>/` | `run_overview.py` and `distributions.py` with `SAVE = True` |
| `notebook/_comparison/` | `compare_runs.py` with `SAVE = True` |

## The package

| Module | Purpose |
|---|---|
| `cdo.figures` | The standard figure set of a run, by name. `make_figures`, `build`, `names`, `compare`. |
| `cdo.selection` | `Selection`: times, radii (r/a) and momentum ranges, separately for fluid and kinetic quantities. |
| `cdo.picker` | `pick_runs()`: a run chooser, checkboxes or a single-choice list in Jupyter. |
| `cdo.paths` | The data root (`CDO_DATA_ROOT`), the run collection folders, and the figure root (`CDO_FIGURE_ROOT`). |
| `cdo.derived` | Cell volumes, radial integrals and averages, ions, and `rate_channels`. |
| `cdo.schema` | Canonical field names, the path alias table, and the `Resolver` that reads them. See `docs/schema.md`. |
| `cdo.io` | Output file discovery, and `add_dream_to_path` for the optional cross-check. |
| `cdo.concat` | `Run`, a run concatenated across its files, read lazily by canonical name. |
| `cdo.moments` | Currents and angle-averaged distribution moments, computed from the HDF5 file. |
| `cdo.energy` | Energy-space transform, `dn/dp` to `dn/dE`. |
| `cdo.plotting` | Low-level plot helpers, style presets, colour-map masking, time-index helpers. |
| `cdo.labels` | Axis and quantity labels. |

Physics conventions, volumes, moments, and the momentum-to-energy transform, are in `docs/volumes-and-moments.md`.

## What the figures show

Figures are named, and `cdo.figures.names(run, selection)` lists the ones a run can make.

Fluid figures use `selection.fluid`:

- `currents`: I_p, I_ohm, I_hot, I_RE and I_wall as separate curves.
  I_hot is the hot-tail grid current above p_thr.
  It carries most of the current during a thermal quench.
  I_RE is the runaway current.
- `<quantity>_traces`: against time at the chosen radii, with the volume average in black.
- `<quantity>_profiles`: against radius at the chosen times.
- `<quantity>_map`: over time and radius.
- Quantities: temperature, electron, runaway and hot densities, ohmic, runaway, hot and total current densities, radiated power, Z_eff.
- `electric_field_<r/a>` and `electric_field_average`: |E|, E_c,eff and E_c,tot, with |E|/E_c,eff on the right axis.
- `electric_field_profiles`, `critical_field_ratio_map`.
- `generation_rates` and `rate_map_<channel>`: see below.

Kinetic figures use `selection.kinetic`, one file per grid, radius and range:

- `f_<grid>_<r/a>[_<range>]`: momentum-pitch maps at the chosen times, or f(p) curves on an isotropic grid.
- `dndp_<grid>_<r/a>[_<range>]`: pitch-integrated density per unit momentum.
- `spectrum_<grid>_<r/a>[_<range>]`: density per unit kinetic energy, in m^-3 MeV^-1.

Titles show the snapped values: r/a, r in metres, and times.
Colour maps show missing data in white.
On log scales, zero and negative values are white too, and positive values below the floor take the darkest colour.

### Runaway rate channels

`cdo.derived.rate_channels` reads the run settings and labels each channel:

| Channel | Source | Meaning |
|---|---|---|
| net | `runawayRate` | Total dn_RE/dt over the last internal step, including losses. |
| hot_flux | `gammaFhot` | Net flux out of the hot-tail grid across its p_max. In kinetic runs, Dreicer and hot-tail generation appear here together, with avalanche, Compton and tritium electrons born on the hot-tail grid. It can be negative. |
| avalanche | `GammaAva * n_re`, or `n_re n_tot C(p_max,hot)` | A term with fluid avalanche. With kinetic RP and both grids, the n_RE term rebuilt from the RP source above the hot-tail grid. |
| remainder | net minus terms | With the Boltzmann operator, the re-re knock-on source, which DREAM does not write out. Shown only when larger than 0.1% of the net rate. |
| dreicer, compton, tritium, hottail | `gamma*` | Terms when that model is on. |
| `*_estimate` | `gammaDreicer`, `GammaAva * n_re` | Fluid formulas DREAM writes regardless of the model. Not in the equations. Drawn dashed. |

## Notebooks

The notebooks are `# %%` scripts.
JupyterLab opens them as notebooks through Jupytext, and saves edits back to the `.py` file.
Outputs are not stored, so git sees only code changes.

On the cluster, inside `tmux`:

```bash
cd /project/nr_dream002/CDO-plotting
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/jupyter lab --no-browser --port 8890
```

The thread caps matter on the login nodes.
They limit each user to 500 processes and threads together, and an uncapped kernel holds about 70 threads for BLAS and OpenMP.
A few open notebooks then exhaust the limit, and processes fail with "can't start new thread" or get killed.
Kernels inherit the caps from the server.
Shut down kernels of notebooks you no longer use (Kernel > Shut Down Kernel).

On the laptop:

```bash
ssh -N -L 8890:localhost:8890 <cluster-login-host>
```

Then open the `http://localhost:8890/?token=...` link that JupyterLab prints, and open a notebook with "Open With > Jupytext Notebook".

| Notebook | Contents |
|---|---|
| `run_overview.py` | One run: choose it, set the selection, show single figures or save the whole set. |
| `distributions.py` | One run, kinetic figures only. |
| `compare_runs.py` | Overlays of currents and temperature for several runs, with optional time shifts and reference points. |
| `drag_force.py` | Drag-force schematic from `Chandrasekar.csv`. |

The saving cell of each run notebook has a `SAVE` toggle.
`SAVE = False` (or `0`) shows the figures in the notebook, `SAVE = True` (or `1`) saves them under `CDO_FIGURE_ROOT`.

The notebooks also run top to bottom with `python notebooks/<name>.py`.
`CDO_RUN`, `CDO_RUNS` and `CDO_OUT` then set the run(s) and the output folder.
`CDO_OUT` overrides the `SAVE` folder.

```python
import cdo
from cdo import figures
from cdo.selection import FluidSelection, KineticSelection, Selection

run = cdo.Run.from_folder("/path/to/run")
sel = Selection(
    fluid=FluidSelection(times_ms=[1, 10, "end"], radii=[0.0, 0.5]),
    kinetic=KineticSelection(times_ms=[5, 30], radii=[0.2],
                             ranges=[(None, None), (None, 5)], range_unit="MeV"),
)
figures.build(run, "currents", sel)
figures.make_figures(run, sel, out="plots/test")   # PNGs + plot_settings.json
```

## Batch figures

`scripts/plot_all_runs.py` saves the figure set of every run in the campaigns of its `GROUPS`, with the selection in its `SELECTION`.
It adds cross-run overlays per group and an `index.md`.

```bash
.venv/bin/python scripts/plot_all_runs.py                     # all groups -> $CDO_FIGURE_ROOT or ./plots
.venv/bin/python scripts/plot_all_runs.py --group demo_jorek  # one group
.venv/bin/python scripts/plot_all_runs.py --pdf               # PDF copies in each run's pdf/
```

PNGs are written directly in each run folder.
PDFs are off by default.
The first read of a run from the project file system can take minutes.
Later reads come from the file cache and take seconds.

## Tests

```bash
.venv/bin/python -m pytest tests/
```

The notebook and batch tests run on a reference run under the data root and skip when it is absent.
The cross-check tests compare the native moments and integrals against `DREAMOutput`.
They run when DREAM is importable and skip otherwise:

```bash
PYTHONPATH=/path/to/DREAM/py .venv/bin/python -m pytest tests/test_dream_crosscheck.py
```
