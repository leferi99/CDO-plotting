"""Where the DREAM run outputs live.

All run folders sit under one data root with the same layout everywhere:

    <root>/DREAM-runs/output/<run>/        ITER and DEMO_2026 runs
    <root>/DEMO_data_2026/runs/<run>/      DEMO JOREK current-quench runs

On the cluster the root is ``/project/nr_dream002``. Elsewhere, for example on
a laptop with a synced copy, set ``CDO_DATA_ROOT`` to the root of the copy:

    export CDO_DATA_ROOT=~/Data/DREAM

The variable is read on every call, so it can also be set inside a notebook
with ``os.environ["CDO_DATA_ROOT"] = ...`` before choosing runs.
"""

from __future__ import annotations

import os
from pathlib import Path

#: The data root on the cluster, used when CDO_DATA_ROOT is not set.
CLUSTER_ROOT = Path("/project/nr_dream002")

#: Run collections under the data root.
COLLECTIONS = {
    "dream_runs": Path("DREAM-runs/output"),
    "demo_jorek": Path("DEMO_data_2026/runs"),
}


def data_root() -> Path:
    """The data root: ``CDO_DATA_ROOT`` if set, the cluster path otherwise."""
    value = os.environ.get("CDO_DATA_ROOT")
    return Path(value).expanduser() if value else CLUSTER_ROOT


def collection(name: str) -> Path:
    """The folder holding one collection of runs, ``"dream_runs"`` or ``"demo_jorek"``."""
    return data_root() / COLLECTIONS[name]


def run_roots() -> tuple[Path, ...]:
    """Every collection folder under the data root."""
    return tuple(collection(name) for name in COLLECTIONS)


def run_folder(collection_name: str, run: str) -> Path:
    """One run folder, for example ``run_folder("dream_runs", "demo2026_boltzmann")``."""
    return collection(collection_name) / run
