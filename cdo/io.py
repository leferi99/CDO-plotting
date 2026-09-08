"""Locating DREAM output files and making the DREAM Python package importable.

A DREAM run is usually split across several output files written in sequence,
named with an index that has to be sorted numerically rather than
lexicographically. Naming has varied between runs, so discovery matches several
patterns and orders by the numbers embedded in the name.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import h5py

from .schema import Resolver, decode_string, survey

#: Output file names seen across runs.
OUTPUT_PATTERN = "output*.h5"

#: Files from the initialisation and current-matching phase. These carry only a
#: couple of time steps and do not evolve the temperature or the wall circuit,
#: so they are excluded unless asked for.
INIT_PREFIX = "output_init"

_NUMBER = re.compile(r"\d+")


def add_dream_to_path(checkout: str | os.PathLike) -> Path:
    """Put a DREAM checkout's ``py`` directory on ``sys.path``.

    ``checkout`` is either the root of a DREAM source tree or its ``py``
    subdirectory. The DREAM Python package is not installed into the environment
    on Komondor, so this has to run before ``import DREAM``. Which checkout is
    used matters: the ``DREAMOutput`` methods this package relies on, such as
    ``j_re.current()`` and ``f_re.angleAveraged()``, differ between versions.
    """
    root = Path(checkout).expanduser().resolve()
    py_dir = root if root.name == "py" else root / "py"

    if not (py_dir / "DREAM").is_dir():
        raise FileNotFoundError(f"no DREAM package under {py_dir}")

    path = str(py_dir)
    if path not in sys.path:
        sys.path.insert(0, path)
    return py_dir


def natural_key(path: str | os.PathLike):
    """Sort key ordering embedded numbers numerically.

    Keeps ``output_2_.h5`` before ``output_10_.h5``.
    """
    name = Path(path).name
    parts = _NUMBER.split(name)
    numbers = [int(n) for n in _NUMBER.findall(name)]
    return (parts[0], numbers, name)


def find_outputs(
    folder: str | os.PathLike,
    pattern: str | None = None,
    prefix: str | None = None,
    init: bool = False,
) -> list[Path]:
    """List a run's output files in time order.

    Names have varied between runs, so one glob matches them all and ordering is
    done on the embedded numbers. Settings files are never returned.

    Initialisation files are excluded unless ``init`` is true. They hold the
    current-matching phase, a couple of time steps with the temperature and wall
    circuit not evolved, and concatenating them with the production files would
    misreport the start of the simulation.

    ``prefix`` restricts the match further, for use when one folder holds
    several runs.
    """
    folder = Path(folder).expanduser()
    if not folder.is_dir():
        raise NotADirectoryError(folder)

    matches = [
        p
        for p in folder.glob(pattern or OUTPUT_PATTERN)
        if not p.name.startswith("settings")
        and (init or not p.name.startswith(INIT_PREFIX))
        and (prefix is None or p.name.startswith(prefix))
    ]
    return sorted(matches, key=natural_key)


def newest_output(folder: str | os.PathLike, **kwargs) -> Path | None:
    """The last output file of a run, or ``None`` if the folder holds none."""
    files = find_outputs(folder, **kwargs)
    return files[-1] if files else None


def open_output(path: str | os.PathLike) -> h5py.File:
    return h5py.File(str(path), "r")


def describe(path: str | os.PathLike) -> str:
    """Summarise which known fields one output file carries."""
    with open_output(path) as handle:
        return survey(handle, name=Path(path).name).report()


def describe_run(folder: str | os.PathLike, **kwargs) -> str:
    """Summarise every output file of a run.

    Worth reading before plotting a run for the first time. Files written by
    different DREAM versions, or with different grids enabled, show up here as
    differing reports.
    """
    files = find_outputs(folder, **kwargs)
    if not files:
        return f"{folder}: no output files found"
    return "\n".join(describe(path) for path in files)


def code_version(path: str | os.PathLike) -> str:
    """The DREAM commit hash a file was written by, or an empty string."""
    with open_output(path) as handle:
        resolver = Resolver(handle, name=Path(path).name)
        located = resolver.locate("commit")
        return decode_string(handle[located]) if located else ""
