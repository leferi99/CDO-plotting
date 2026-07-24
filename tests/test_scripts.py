"""Smoke tests for the # %% scripts.

Every script is compiled to catch syntax errors. The drag-force script needs
only the bundled CSV and is run in full. The run-driven scripts are run only when
the default data folder is present, and skipped otherwise, so the suite stays
portable.
"""

import os
import py_compile
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
DATA_FOLDER = Path.home() / "nr_dream002/DREAM-runs/output/iter_dthmode24_rp_commit-53d8afb"

ALL_SCRIPTS = ["plot_main.py", "plot_distribution.py", "plot_drag_force.py"]


def _run(script, env_extra, tmp_path):
    env = dict(os.environ, MPLBACKEND="Agg", **env_extra)
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script)],
        env=env, capture_output=True, text=True, timeout=300,
    )


@pytest.mark.parametrize("script", ALL_SCRIPTS)
def test_script_compiles(script):
    py_compile.compile(str(SCRIPTS / script), doraise=True)


def test_drag_force_runs_and_saves(tmp_path):
    out = tmp_path / "drag.png"
    result = _run("plot_drag_force.py", {"CDO_OUTPUT_FILE": str(out)}, tmp_path)
    assert result.returncode == 0, result.stderr
    assert out.exists()


@pytest.mark.parametrize("script", ["plot_main.py", "plot_distribution.py"])
def test_run_driven_script(script, tmp_path):
    if not DATA_FOLDER.is_dir():
        pytest.skip(f"reference data folder not present: {DATA_FOLDER}")
    out = tmp_path / "figs"
    result = _run(
        script,
        {"CDO_DATA_FOLDER": str(DATA_FOLDER), "CDO_OUTPUT_FOLDER": str(out)},
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert any(out.rglob("*.png"))
