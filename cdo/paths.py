"""Where the DREAM run outputs live.

All run folders sit under one data root with the same layout everywhere:

    <root>/DREAM-runs/output/<run>/        ITER and DEMO_2026 runs
    <root>/DEMO_data_2026/runs/<run>/      DEMO JOREK current-quench runs

On the cluster the root is ``/project/nr_dream002``. Elsewhere, for example on
a laptop with a synced copy, set ``CDO_DATA_ROOT`` to the root of the copy:

    export CDO_DATA_ROOT=~/Data/DREAM

The variable is read on every call, so it can also be set inside a notebook
with ``os.environ["CDO_DATA_ROOT"] = ...`` before choosing runs.

Saved figures go under one figure root, ``CDO_FIGURE_ROOT`` if set, the
repository's ``plots/`` folder otherwise:

    <figure root>/<group>/<run>/           batch script, per run
    <figure root>/<group>/_comparison/     batch script, cross-run overlays
    <figure root>/notebook/<run>/          notebooks, per run
    <figure root>/notebook/_comparison/    compare_runs notebook
"""

from __future__ import annotations

import os
from pathlib import Path

#: The data root on the cluster, used when CDO_DATA_ROOT is not set.
CLUSTER_ROOT = Path("/project/nr_dream002")

#: The figure root when CDO_FIGURE_ROOT is not set: plots/ in the repository.
REPO_FIGURE_ROOT = Path(__file__).resolve().parent.parent / "plots"

#: Run collections under the data root.
COLLECTIONS = {
    "dream_runs": Path("DREAM-runs/output"),
    "demo_jorek": Path("DEMO_data_2026/runs"),
}


def data_root() -> Path:
    """The data root: ``CDO_DATA_ROOT`` if set, the cluster path otherwise."""
    value = os.environ.get("CDO_DATA_ROOT")
    return Path(value).expanduser() if value else CLUSTER_ROOT


def figure_root() -> Path:
    """The figure root: ``CDO_FIGURE_ROOT`` if set, the repository's ``plots/`` otherwise."""
    value = os.environ.get("CDO_FIGURE_ROOT")
    return Path(value).expanduser() if value else REPO_FIGURE_ROOT


def collection(name: str) -> Path:
    """The folder holding one collection of runs, ``"dream_runs"`` or ``"demo_jorek"``."""
    return data_root() / COLLECTIONS[name]


def run_roots() -> tuple[Path, ...]:
    """Every collection folder under the data root."""
    return tuple(collection(name) for name in COLLECTIONS)


def run_folder(collection_name: str, run: str) -> Path:
    """One run folder, for example ``run_folder("dream_runs", "demo2026_boltzmann")``."""
    return collection(collection_name) / run
