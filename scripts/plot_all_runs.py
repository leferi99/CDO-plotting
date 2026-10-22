"""Batch figures for every DREAM run of the campaigns listed in `GROUPS`.

For each run folder of each group, draws the standard figure set of
`cdo.figures` with `SELECTION` (edit it below, or work in a notebook for one
run). PNGs go directly into `<out>/<group>/<run>/`, with `plot_settings.json`
recording the requested and snapped times and radii. `--pdf` adds PDF copies in
`<out>/<group>/<run>/pdf/`. Cross-run overlays go to `<out>/<group>/_comparison/`
and a summary table to `<out>/index.md`.

Usage:

    python scripts/plot_all_runs.py                      # all groups -> $CDO_FIGURE_ROOT or ./plots
    python scripts/plot_all_runs.py --group demo_jorek   # one group
    python scripts/plot_all_runs.py --only ar0p3pct      # substring filter
    python scripts/plot_all_runs.py --runs-root /path/runs --glob "tauw_*"
    python scripts/plot_all_runs.py --runs-root /path/output --glob "iter-nrt/*"
    python scripts/plot_all_runs.py --pdf                # PDF copies as well
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")  # headless: this driver only saves.
import numpy as np

import cdo
from cdo import figures
from cdo.concat import InconsistentRunWarning
from cdo.selection import FluidSelection, KineticSelection, Selection

#: What every run's figures show. Fluid and kinetic quantities have separate
#: times (ms, "end" or "auto") and radii (r/a). Each kinetic range is one file.
SELECTION = Selection(
    fluid=FluidSelection(times_ms="auto", radii=[0.0, 0.5, 0.9]),
    kinetic=KineticSelection(times_ms="auto", radii=[0.0, 0.5],
                             ranges=[(None, None)], range_unit="MeV"),
)

# Run collections under the data root: /project/nr_dream002 on the cluster,
# CDO_DATA_ROOT elsewhere (see cdo/paths.py).
DREAM_RUNS_OUTPUT = cdo.paths.collection("dream_runs")
DEMO_JOREK_RUNS = cdo.paths.collection("demo_jorek")

#: Campaigns swept by default: name, output root, folder globs, tag prefix.
#: Runs sit one campaign folder down, ``<collection>/<campaign>/<run>``, except
#: the first-build DEMO JOREK runs, kept as their own group because the
#: bounce-orbit reruns carry the same names.
GROUPS = [
    ("iter_nonregression", DREAM_RUNS_OUTPUT,
     ("iter-nrt/*", "iter-abstol/*"), "iter_dthmode24_"),
    ("iter_numerics", DREAM_RUNS_OUTPUT, ("iter-numerics/*",), "iter_dthmode24_"),
    ("demo2026_vde", DREAM_RUNS_OUTPUT, ("demo2026/*",), "demo2026_"),
    ("demo_jorek", DEMO_JOREK_RUNS, ("fix-wrapped-bounce-orbits/tauw_*",), ""),
    ("demo_jorek_first_build", DEMO_JOREK_RUNS, ("tauw_*",), ""),
]

#: JOREK time of the DREAM start in the JOREK benchmark: the time of
#: CQ_prof.dat, `T0_JOREK` in DEMO_JOREK_2026/run_demo_cq.py.
JOREK_T0_MS = 5.762537968682188
#: JOREK plasma current values known without a time trace, as
#: (JOREK time [ms], I_p [MA]): the eqdsk equilibrium, the CQ_prof.dat state
#: DREAM starts from, and the 2D RE plateau (I_p = I_RE) from the provider's
#: slides. See DEMO_JOREK_2026/README.md.
JOREK_IP_POINTS = [(0.0, 18.79), (JOREK_T0_MS, 18.74), (267.0, 16.1)]
JOREK_PLATEAU_MA = 16.1
JOREK_MARKER = dict(ls="none", marker="o", ms=8, mfc="none", mec="k", mew=2, zorder=5)

#: Line styles for the ITER non-regression overlay, by run tag. The three main
#: runs are wide solid lines. Each variant is a thinner patterned line drawn
#: over the run it differs from by one factor, so both stay visible where they
#: agree.
NONREGRESSION_STYLES = {
    "rp_commit-53d8afb": dict(color="tab:blue", ls="-", lw=5.0),
    "rp_boltzmann-eqterm": dict(color="tab:orange", ls=(0, (4, 2)), lw=2.0),
    "boltzmann_boltzmann-eqterm": dict(color="tab:green", ls="-", lw=5.0),
    "boltzmann_boltzmann-eqterm-ar0p3pct": dict(color="tab:brown", ls=(0, (6, 2, 1, 2)), lw=2.0),
    "tol_abstol1e-4_boltzmann-eqterm": dict(color="tab:red", ls=(0, (3, 2)), lw=2.0),
    "tol_abstol1e0_boltzmann-eqterm": dict(color="tab:purple", ls=(0, (1, 1.5)), lw=2.5),
}


def load_jorek_trace(path):
    """A JOREK I_p trace from a two-column text file, JOREK time [ms] and I_p [MA]."""
    data = np.loadtxt(path, delimiter="," if str(path).endswith(".csv") else None,
                      comments="#")
    return data[:, 0], data[:, 1]


def discover(runs_root: Path, globs, only: str | None):
    runs = []
    for pattern in globs:
        for folder in sorted(runs_root.glob(pattern)):
            if folder.is_dir() and (not only or only in folder.name) and cdo.find_outputs(folder):
                runs.append(folder)
    return runs


def summary(run, folder: Path, tag: str) -> dict:
    """One row of the index: size, end time, final I_p and peak currents."""
    def peak(current):
        return None if current is None else float(np.nanmax(np.abs(current)) / 1e6)

    return {
        "tag": tag,
        "folder": str(folder),
        "legs": len(run.files),
        "t_end_ms": float(run.timegrid_ms[-1]),
        "unfinished": any(folder.glob("output*.inprogress.h5")),
        "Ip_end_MA": float(run.plasma_current[-1] / 1e6),
        "Ire_max_MA": peak(run.runaway_current),
        "Ihot_max_MA": peak(run.hot_current),
    }


def write_index(sections, out_root: Path):
    fmt = lambda v: "-" if v is None else f"{v:.3g}"
    lines = [
        "# DREAM run figures",
        "",
        "Each run folder holds its figures as PNGs and `plot_settings.json`.",
        "I_RE is the runaway-electron current. I_hot is the hot-tail grid current above "
        "p_thr, which carries most of the current during a thermal quench.",
        "Cross-run overlays are under `<group>/_comparison/`.",
    ]
    for group, runs_root, records, compare_made in sections:
        lines += [
            "", f"## {group}", "", f"Source: `{runs_root}`", "",
            "| run | legs | t_end [ms] | last leg failed | I_p end [MA] | I_RE max [MA] "
            "| I_hot max [MA] | figures |",
            "|---|--:|--:|---|--:|--:|--:|--:|",
        ]
        for r in records:
            lines.append(
                f"| {r['tag']} | {r.get('legs', '-')} | {fmt(r.get('t_end_ms'))} | "
                f"{'yes' if r.get('unfinished') else ''} | {fmt(r.get('Ip_end_MA'))} | "
                f"{fmt(r.get('Ire_max_MA'))} | {fmt(r.get('Ihot_max_MA'))} | "
                f"{len(r.get('figures', []))} |")
        if compare_made:
            lines += ["", "Comparison figures: " + ", ".join(f"`{n}`" for n in compare_made)]
    (out_root / "index.md").write_text("\n".join(lines) + "\n")


def sweep_group(group, runs_root, globs, prefix, args):
    """Plot every run of one group and its comparison overlays."""
    folders = discover(runs_root, globs, args.only)
    print(f"== group {group}: {len(folders)} run(s) under {runs_root}")
    out_dir = args.out / group
    records, runs = [], {}
    for folder in folders:
        tag = folder.name.removeprefix(prefix) if prefix else folder.name
        print(f"-- {tag}")
        try:
            run = cdo.Run.from_folder(folder)
            record = summary(run, folder, tag)
            if not args.no_per_run:
                result = figures.make_figures(run, SELECTION, out=out_dir / tag,
                                              pdf=args.pdf, dpi=args.dpi)
                record["figures"] = result["made"]
            run.drop_cache()  # keep only what the comparison re-reads
            print(f"    {record['legs']} legs, to {record['t_end_ms']:.4g} ms, "
                  f"{len(record.get('figures', []))} figures")
            records.append(record)
            runs[tag] = run
        except Exception as exc:
            print(f"    [error] {type(exc).__name__}: {exc}")
            records.append({"tag": tag, "folder": str(folder)})

    compare_made = []
    if not args.no_compare and runs:
        shift, references, styles = {}, {}, {}
        if group.startswith("demo_jorek"):
            shift = {tag: JOREK_T0_MS for tag in runs}
            t_ref, i_ref = zip(*JOREK_IP_POINTS)
            references["compare_plasma_current"] = [("JOREK", t_ref, i_ref, JOREK_MARKER)]
            references["compare_runaway_current"] = [
                ("JOREK plateau", [267.0], [JOREK_PLATEAU_MA], JOREK_MARKER)]
            if args.jorek_csv:
                t_tr, i_tr = load_jorek_trace(args.jorek_csv)
                references["compare_plasma_current"].append(
                    ("JOREK trace", t_tr, i_tr, dict(color="k", lw=2.5)))
        if group == "iter_nonregression":
            styles = NONREGRESSION_STYLES
        options = dict(out=out_dir / "_comparison", time_shift_ms=shift,
                       references=references, styles=styles, pdf=args.pdf, dpi=args.dpi)
        compare_made = figures.compare(runs, **options)
        # Runs much shorter than the longest are squeezed at the left edge, so
        # a zoomed copy covers the span of the longest of those short runs.
        ends = {tag: float(run.timegrid_ms[-1]) + shift.get(tag, 0.0)
                for tag, run in runs.items()}
        short = [e for e in ends.values() if e < 0.5 * max(ends.values())]
        if short:
            compare_made += figures.compare(runs, time_range_ms=(None, max(short)),
                                            suffix="_zoom", **options)
        print(f"   comparison: {', '.join(compare_made)}")
    for run in runs.values():
        run.close()
    return group, runs_root, records, compare_made


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-root", type=Path, default=None,
                        help="sweep this root alone instead of the default groups")
    parser.add_argument("--glob", default="*", help="run folder glob under --runs-root")
    parser.add_argument("--group", default=None, help="only the default group of this name")
    parser.add_argument("--out", type=Path, default=cdo.paths.figure_root())
    parser.add_argument("--only", default=None,
                        help="only runs whose folder name contains this substring")
    parser.add_argument("--jorek-csv", type=Path, default=None,
                        help="JOREK I_p trace for the demo_jorek overlay: "
                             "columns JOREK time [ms], I_p [MA]")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--pdf", action="store_true",
                        help="also save each figure as a PDF in a pdf/ subfolder")
    parser.add_argument("--no-per-run", action="store_true",
                        help="skip per-run figures, draw only the comparisons")
    parser.add_argument("--no-compare", action="store_true", help="skip the cross-run overlays")
    args = parser.parse_args(argv)

    warnings.simplefilter("ignore", InconsistentRunWarning)
    cdo.plotting.use_base_style(fontsize=11)

    if args.runs_root is not None:
        groups = [(args.runs_root.name, args.runs_root, (args.glob,), "")]
    else:
        groups = [g for g in GROUPS if args.group in (None, g[0])]
    if not groups:
        raise SystemExit(f"no group named {args.group}")
    args.out.mkdir(parents=True, exist_ok=True)

    sections = [sweep_group(*g, args) for g in groups]
    if not any(records for _, _, records, _ in sections):
        raise SystemExit("no runs found")

    print(f"\nfigures under {args.out}" + ("  (PDF copies in pdf/)" if args.pdf else ""))
    if args.no_per_run or args.group or args.only or args.runs_root:
        return  # a partial sweep would overwrite the full index
    write_index(sections, args.out)
    print(f"index: {args.out / 'index.md'}")


if __name__ == "__main__":
    main()
