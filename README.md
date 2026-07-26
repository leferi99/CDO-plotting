# CDO-plotting

Consistent plotting of [DREAM](https://github.com/chalmersplasmatheory/DREAM) output files across DREAM versions and run setups.

A run is usually split across several HDF5 output files, and the datasets present and the paths they live at vary with the DREAM version and with how the run was invoked. The `cdo` package reads any of these into one concatenated series through a canonical-name layer, computes the moments and integrals natively, and plots them. Reading and the moments take no DREAM import, so the package works on output that the DREAM Python interface can no longer open.

## Requirements

Python with `numpy`, `scipy`, `h5py` and `matplotlib`. A DREAM install is not needed for reading, moments, or plotting. It is used only by the optional cross-check tests.

## The package

| Module | Purpose |
|---|---|
| `cdo.schema` | Canonical field names, the path alias table, and the `Resolver` that reads them. See `docs/schema.md`. |
| `cdo.io` | Output file discovery, and `add_dream_to_path` for the optional cross-check. |
| `cdo.concat` | `Run`, a run concatenated across its files, read lazily by canonical name. |
| `cdo.moments` | Currents and angle-averaged distribution moments, computed from the HDF5 file. |
| `cdo.derived` | Cell volumes, radial integrals, and ions addressed by name. |
| `cdo.energy` | Runaway energy-space transform, `dn/dp` to `dn/dE`. |
| `cdo.plotting` | `basic_1D`, `basic_2D`, style presets, distribution plots, time-index helpers. |
| `cdo.labels` | Axis and quantity labels. |

Physics conventions, volumes, moments, and the momentum-to-energy transform, are in `docs/volumes-and-moments.md`.

## Using the package

```python
import cdo

run = cdo.Run.from_folder("/path/to/run/output")
print(run.report())

run.field("T_cold")            # concatenated (time, radius) temperature
run.current("j_re")            # total runaway current in amperes
run.angle_average("runaway", "density")   # dn/dp per radius
run.radial_integral("n_re")    # total runaways over the plasma volume
run.ion("Ar").mean_charge(run.field("n_i"))
run.close()
```

`cdo.describe(path)` or `cdo.describe_run(folder)` reports which fields a file carries and why any are absent.

## Running a figure script

The scripts in `scripts/` are `# %%` cell scripts. Run them cell by cell in an editor, or top to bottom:

```bash
CDO_DATA_FOLDER=/path/to/run/output CDO_OUTPUT_FOLDER=/path/to/figures \
    python scripts/plot_main.py
```

Leave `CDO_OUTPUT_FOLDER` unset to display the figures instead of saving them. `plot_main.py` covers currents, generation rates, temperature, current density, and the runaway distribution. `plot_distribution.py` plots the runaway energy spectrum. `plot_drag_force.py` draws the drag-force schematic from `Chandrasekar.csv`.

## Tests

```bash
pytest tests/
```

The cross-check tests compare the native moments and integrals against `DREAMOutput`. They run when DREAM is importable and skip otherwise:

```bash
PYTHONPATH=/path/to/DREAM/py pytest tests/test_dream_crosscheck.py
```
