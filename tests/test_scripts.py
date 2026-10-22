"""Smoke tests for the notebooks and the batch script.

Every file is compiled to catch syntax errors. The drag-force notebook needs
only the bundled CSV and is run in full. The run-driven notebooks and the batch
script run only when the reference run folder is present, and skip otherwise,
so the suite stays portable.
"""

import os
import py_compile
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
NOTEBOOKS = REPO / "notebooks"
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(REPO))
from cdo import paths  # noqa: E402

DATA_FOLDER = paths.run_folder("demo_jorek", "tauw_inf_iso_rp")

ALL_FILES = sorted(NOTEBOOKS.glob("*.py")) + sorted(SCRIPTS.glob("*.py"))


def _run(args, env_extra):
    env = dict(os.environ, MPLBACKEND="Agg", **env_extra)
    return subprocess.run([sys.executable, *map(str, args)], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=600)


def _need_data():
    if not DATA_FOLDER.is_dir():
        pytest.skip(f"reference data folder not present: {DATA_FOLDER}")


@pytest.mark.parametrize("path", ALL_FILES, ids=lambda p: p.name)
def test_file_compiles(path):
    py_compile.compile(str(path), doraise=True)


def test_drag_force_runs_and_saves(tmp_path):
    out = tmp_path / "drag.png"
    result = _run([NOTEBOOKS / "drag_force.py"], {"CDO_OUTPUT_FILE": str(out)})
    assert result.returncode == 0, result.stderr
    assert out.exists()


@pytest.mark.parametrize("notebook", ["run_overview.py", "distributions.py"])
def test_run_notebook_saves_figures(notebook, tmp_path):
    _need_data()
    out = tmp_path / "figs"
    result = _run([NOTEBOOKS / notebook], {"CDO_RUN": str(DATA_FOLDER), "CDO_OUT": str(out)})
    assert result.returncode == 0, result.stderr
    assert any(out.glob("*.png"))
    assert (out / "plot_settings.json").exists()


def test_compare_notebook_saves_figures(tmp_path):
    _need_data()
    out = tmp_path / "cmp"
    runs = f"{DATA_FOLDER}:{DATA_FOLDER.parent / 'tauw_50ms_iso'}"
    result = _run([NOTEBOOKS / "compare_runs.py"], {"CDO_RUNS": runs, "CDO_OUT": str(out)})
    assert result.returncode == 0, result.stderr
    assert (out / "compare_plasma_current.png").exists()


def test_batch_script_one_run(tmp_path):
    _need_data()
    result = _run([SCRIPTS / "plot_all_runs.py", "--runs-root", DATA_FOLDER.parent,
                   "--glob", DATA_FOLDER.name, "--out", tmp_path, "--no-compare"], {})
    assert result.returncode == 0, result.stderr
    run_dir = tmp_path / DATA_FOLDER.parent.name / DATA_FOLDER.name
    assert (run_dir / "currents.png").exists()
    assert not (run_dir / "pdf").exists()  # PDFs are opt-in


def test_default_kinetic_names_on_the_reference_run():
    _need_data()
    import cdo
    from cdo import figures
    from cdo.selection import Selection

    run = cdo.Run.from_folder(DATA_FOLDER)
    names = figures.names(run, Selection(only=["f_", "favg_", "pitch_", "dnde_", "dndp_",
                                               "spectrum_"]))
    radius = names[0].split("_")[-1]
    combined = [n for n in names if "_combined_" in n]
    assert combined and all(n.endswith("_Emin-10MeV") for n in combined)
    assert not any(n.startswith("dndp_") for n in names)
    for kind in ("favg_re", "pitch_re"):
        assert f"{kind}_{radius}" in names and f"{kind}_{radius}_Emin-40MeV" in names
    assert not any(n.endswith("_Emin-40MeV") for n in names
                   if not n.startswith(("favg_re", "pitch_re")))
    assert not any(n.startswith("favg_hot") for n in names)
