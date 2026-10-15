"""Choose DREAM run folders interactively.

:func:`pick_runs` lists the run folders under one or more roots, with the
number of committed legs, the simulated end time, the DREAM commit and the
avalanche mode read from each run's last output file. In Jupyter it shows a
filterable multi-select box. Elsewhere it prints a numbered table, and
:meth:`RunPicker.choose` takes the numbers.

    picker = cdo.pick_runs()          # tick runs in the box
    folders = picker.selected         # list of Path
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py

from .io import find_outputs
from .schema import decode_string

#: Where the runs of this project live.
DEFAULT_ROOTS = (
    Path("/project/nr_dream002/DREAM-runs/output"),
    Path("/project/nr_dream002/DEMO_data_2026/runs"),
)

#: Avalanche modes of DREAM's RunawayElectrons settings.
AVALANCHE = {1: "none", 2: "fluid", 3: "fluid-hesslow", 4: "kinetic-RP",
             5: "boltzmann"}


@dataclass
class RunInfo:
    path: Path
    legs: int
    t_end_ms: float
    commit: str
    avalanche: str
    unfinished_leg: bool

    @property
    def label(self) -> str:
        flag = "  [last leg failed]" if self.unfinished_leg else ""
        return (f"{self.path.name}  |  {self.legs} legs, {self.t_end_ms:.4g} ms, "
                f"{self.avalanche}, DREAM {self.commit}{flag}")


def _manifest_end_time(folder: Path, nfiles: int) -> float | None:
    """Simulated end time in s from the run driver's manifest.json.

    Opening every output file is slow on the project file system, so the
    manifest is used when its leg count matches the committed outputs.
    """
    try:
        legs = json.loads((folder / "manifest.json").read_text())["legs"]
        if len(legs) == nfiles:
            return float(legs[-1]["cumulative_sim_time_s"])
    except (OSError, KeyError, ValueError, TypeError, IndexError):
        pass
    return None


def _info(folder: Path) -> RunInfo | None:
    files = find_outputs(folder)
    if not files:
        return None
    t_end = _manifest_end_time(folder, len(files))
    if t_end is None:
        t_end = 0.0
        for f in files:
            with h5py.File(f, "r") as h:
                t_end += float(h["grid/t"][-1])
    with h5py.File(files[-1], "r") as h:
        commit = decode_string(h["code/commit"])[:8] if "code/commit" in h else "?"
        mode = h.get("settings/eqsys/n_re/avalanche")
        avalanche = (AVALANCHE.get(int(mode[()].ravel()[0]), "?")
                     if mode is not None else "?")
    unfinished = any(folder.glob("output*.inprogress.h5"))
    return RunInfo(folder, len(files), t_end * 1e3, commit, avalanche, unfinished)


def survey_runs(roots=DEFAULT_ROOTS, globs=("*",)) -> list[RunInfo]:
    """Run folders with at least one committed output, sorted by root then name."""
    infos = []
    for root in map(Path, roots):
        folders = sorted({f for g in globs for f in root.glob(g) if f.is_dir()})
        for folder in folders:
            info = _info(folder)
            if info is not None:
                infos.append(info)
    return infos


class RunPicker:
    """A list of runs to choose from, with a widget when ipywidgets is usable."""

    def __init__(self, infos: list[RunInfo]):
        self.runs = infos
        self._chosen: list[Path] = []
        self._widget = None

    @property
    def selected(self) -> list[Path]:
        if self._widget is not None:
            return list(self._widget.value)
        return list(self._chosen)

    def choose(self, *numbers: int) -> list[Path]:
        """Pick runs by their number in the printed table."""
        self._chosen = [self.runs[n].path for n in numbers]
        return self._chosen

    def table(self) -> str:
        return "\n".join(f"{n:3d}  {info.label}" for n, info in enumerate(self.runs))

    def show(self):
        try:
            import ipywidgets as w
            from IPython import get_ipython
            from IPython.display import display
        except ImportError:
            import sys

            print(f"ipywidgets is not importable in this kernel ({sys.executable}).\n"
                  "Switch to the CDO-plotting kernel, 'Python 3 (ipykernel)', for the "
                  "selection box, or pick by number with .choose(...).\n")
            print(self.table())
            return self
        shell = get_ipython()
        if shell is None or type(shell).__name__ == "TerminalInteractiveShell":
            print(self.table())
            return self

        options = [(info.label, info.path) for info in self.runs]
        select = w.SelectMultiple(options=options, rows=min(20, max(len(options), 2)),
                                  layout=w.Layout(width="100%"))
        filt = w.Text(placeholder="filter by name, e.g. tauw or boltzmann",
                      layout=w.Layout(width="50%"))

        def refilter(change):
            keep = [o for o in options if change["new"].lower() in o[0].lower()]
            chosen = [v for v in select.value if v in {p for _, p in keep}]
            select.options = keep
            select.value = chosen

        filt.observe(refilter, names="value")
        self._widget = select
        display(w.VBox([filt, select]))
        return self


def pick_runs(roots=DEFAULT_ROOTS, globs=("*",)) -> RunPicker:
    """Show the runs under ``roots`` for choosing. Read the choice from ``.selected``."""
    return RunPicker(survey_runs(roots, globs)).show()
