"""Choose DREAM run folders interactively.

:func:`pick_runs` lists the run folders under one or more roots, with the
number of committed legs, the simulated end time, the DREAM commit and the
avalanche mode read from each run's last output file. In Jupyter it shows a
checkbox per run (``multiple=True``) or a single-choice list, with a filter
field and a line showing the current choice. Elsewhere it prints a numbered
table, and :meth:`RunPicker.choose` takes the numbers.

    picker = cdo.pick_runs()                  # tick any number of runs
    picker = cdo.pick_runs(multiple=False)    # select exactly one run
    folders = picker.selected                 # list of Path, in list order
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

import h5py

from .io import find_outputs
from . import paths
from .schema import decode_string

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
    #: Path relative to its collection, e.g. ``iter-nrt/<run>``. Run names repeat
    #: across campaigns, so labels show this. Empty means the folder name.
    name: str = ""

    @property
    def label(self) -> str:
        flag = "  [last leg failed]" if self.unfinished_leg else ""
        return (f"{self.name or self.path.name}  |  {self.legs} legs, {self.t_end_ms:.4g} ms, "
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


def _info(folder: Path, name: str = "") -> RunInfo | None:
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
    return RunInfo(folder, len(files), t_end * 1e3, commit, avalanche, unfinished, name)


def survey_runs(roots=None, globs=None, exclude=()) -> list[RunInfo]:
    """Run folders with at least one committed output, sorted by root then path.

    Without ``roots``, the collections under the data root with their own run
    globs and exclusions, see :func:`cdo.paths.run_layout`. With ``roots``,
    ``globs`` default to ``("*", "*/*")``, runs directly under a root and one
    campaign level down. ``exclude`` takes globs relative to the root. Roots
    that do not exist are skipped.
    """
    if roots is None:
        layout = paths.run_layout()
    else:
        layout = [(Path(r), tuple(globs or ("*", "*/*")), tuple(exclude)) for r in roots]
    infos = []
    for root, run_globs, skip in layout:
        if not root.is_dir():
            continue
        folders = sorted({f for g in run_globs for f in root.glob(g) if f.is_dir()})
        for folder in folders:
            name = folder.relative_to(root).as_posix()
            if any(fnmatch(name, pattern) for pattern in skip):
                continue
            info = _info(folder, name)
            if info is not None:
                infos.append(info)
    return infos


class RunPicker:
    """A list of runs to choose from, with a widget when ipywidgets is usable.

    With ``multiple`` every run has a checkbox, so any set can be ticked
    without Ctrl+click. Without it, a list where exactly one run is selected.
    The filter field only hides rows, it never changes the choice, and the line
    under the box always shows what :attr:`selected` returns.
    """

    def __init__(self, infos: list[RunInfo], multiple: bool = True):
        self.runs = infos
        self.multiple = multiple
        self._chosen: list[Path] = []
        self._checks = None  # multiple: one Checkbox per run
        self._select = None  # single: a Select

    @property
    def selected(self) -> list[Path]:
        """The chosen run folders, in list order."""
        if self._checks is not None:
            return [info.path for info, box in zip(self.runs, self._checks) if box.value]
        if self._select is not None:
            return [] if self._select.value is None else [self._select.value]
        return list(self._chosen)

    def choose(self, *numbers: int) -> list[Path]:
        """Pick runs by their number in the printed table."""
        if not self.multiple and len(numbers) > 1:
            raise ValueError("this picker takes one run")
        self._chosen = [self.runs[n].path for n in numbers]
        return self._chosen

    def table(self) -> str:
        return "\n".join(f"{n:3d}  {info.label}" for n, info in enumerate(self.runs))

    def widget(self):
        """Build the selection widget. :meth:`show` displays it."""
        import ipywidgets as w

        status = w.HTML()
        shown = {info.path: info.name or info.path.name for info in self.runs}
        filt = w.Text(placeholder="filter by name, e.g. tauw or boltzmann",
                      layout=w.Layout(width="50%"))

        def update_status(*_):
            chosen = self.selected
            names = ", ".join(shown.get(p, p.name) for p in chosen) or "nothing"
            status.value = f"<b>Selected ({len(chosen)}):</b> {names}"

        if self.multiple:
            self._checks = [w.Checkbox(value=False, description=info.label, indent=False,
                                       layout=w.Layout(width="100%"))
                            for info in self.runs]
            for box in self._checks:
                box.observe(update_status, names="value")
            rows = w.VBox(self._checks, layout=w.Layout(max_height="420px", overflow="auto"))
            none = w.Button(description="untick all", layout=w.Layout(width="auto"))
            none.on_click(lambda _: [setattr(b, "value", False) for b in self._checks])

            def refilter(change):
                text = change["new"].lower()
                for info, box in zip(self.runs, self._checks):
                    box.layout.display = "" if text in info.label.lower() else "none"

            body = [w.HBox([filt, none]), rows, status]
        else:
            options = [(info.label, info.path) for info in self.runs]
            self._select = w.Select(options=options, value=None,
                                    rows=min(20, max(len(options), 2)),
                                    layout=w.Layout(width="100%"))
            self._select.observe(update_status, names="value")

            def refilter(change):
                text = change["new"].lower()
                current = self._select.value
                keep = [o for o in options if text in o[0].lower() or o[1] == current]
                self._select.options = keep
                self._select.value = current

            body = [filt, self._select, status]

        filt.observe(refilter, names="value")
        update_status()
        return w.VBox(body)

    def show(self):
        try:
            import ipywidgets  # noqa: F401
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
        display(self.widget())
        return self


def pick_runs(roots=None, globs=None, exclude=(), *, multiple: bool = True) -> RunPicker:
    """Show the runs under ``roots`` for choosing. Read the choice from ``.selected``.

    ``roots`` defaults to the collections under the data root (``CDO_DATA_ROOT``,
    see :mod:`cdo.paths`); ``globs`` and ``exclude`` are as in
    :func:`survey_runs`. ``multiple=False`` allows exactly one run, for the
    single-run notebooks.
    """
    return RunPicker(survey_runs(roots, globs, exclude), multiple=multiple).show()
