"""Where the DREAM run outputs live.

All run folders sit under one data root with the same layout everywhere:

    <root>/DREAM-runs/output/<campaign>/<run>/   ITER and DEMO_2026 runs
    <root>/DEMO_data_2026/runs/<run>/            DEMO JOREK runs, first build
    <root>/DEMO_data_2026/runs/<campaign>/<run>/ DEMO JOREK reruns and tests

In ``DREAM-runs/output`` the run folders directly under the collection are
older copies of campaign runs, mostly hard links of the same files, and each
``archive-<date>-<name>`` folder has an ``archive_<date>_<name>`` twin with the
same files. :data:`RUN_GLOBS` and :data:`RUN_EXCLUDE` leave both out, so each
run is listed once.

The root is ``CDO_DATA_ROOT`` if set, the repository's ``data/`` otherwise.
``data/`` is not tracked and can be a symlink to the data, for example
``ln -s /path/to/data data`` on the cluster. On a laptop with a synced copy:

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

#: The data root when CDO_DATA_ROOT is not set: data/ in the repository,
#: a folder or a symlink, not tracked.
REPO_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

#: The figure root when CDO_FIGURE_ROOT is not set: plots/ in the repository.
REPO_FIGURE_ROOT = Path(__file__).resolve().parent.parent / "plots"

#: Run collections under the data root.
COLLECTIONS = {
    "dream_runs": Path("DREAM-runs/output"),
    "demo_jorek": Path("DEMO_data_2026/runs"),
}

#: Globs of the run folders inside each collection, relative to it.
RUN_GLOBS = {
    "dream_runs": ("*/*",),
    "demo_jorek": ("*", "*/*"),
}

#: Globs of run folders to leave out, relative to the collection.
RUN_EXCLUDE = {
    "dream_runs": ("archive_*/*",),
    "demo_jorek": (),
}


def data_root() -> Path:
    """The data root: ``CDO_DATA_ROOT`` if set, the repository's ``data/`` otherwise."""
    value = os.environ.get("CDO_DATA_ROOT")
    return Path(value).expanduser() if value else REPO_DATA_ROOT


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


def run_layout() -> tuple[tuple[Path, tuple, tuple], ...]:
    """``(collection folder, run globs, exclude globs)`` for every collection."""
    return tuple((collection(name), RUN_GLOBS[name], RUN_EXCLUDE[name])
                 for name in COLLECTIONS)


def run_folder(collection_name: str, run: str) -> Path:
    """One run folder, for example ``run_folder("dream_runs", "iter-nrt/iter_dthmode24_rp_commit-53d8afb")``."""
    return collection(collection_name) / run
