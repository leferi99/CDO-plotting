"""Batch figures for every DREAM run of the campaigns listed in `GROUPS`.

Sweeps the run folders of each group, builds a standard figure set for each run,
and draws cross-run overlays within the group. Every figure is guarded, so a run
missing a field or a grid is skipped for that figure without aborting the sweep.

Each figure is written as a PNG directly in `<out>/<group>/<tag>/`, for quick
previewing and side-by-side comparison. With `--pdf` a PDF copy also goes into
`<out>/<group>/<tag>/pdf/`. Cross-run overlays go under
`<out>/<group>/_comparison/`. A progress table is written to `<out>/index.md`.

Snapshot times come from `plotting.time_marks`, picked by run length and closed
with the final step. Line plots use 0, 1, 10, 20, 30, 50, 100, 200, 300 ms past
100 ms, 0, 1, 5, 10, 20, 30, 40, 50 ms for 20-100 ms, and 0, 0.1, 0.5, 1, 2,
4 ms below 20 ms. Distribution panels use a fuller set and skip its first two
times, where the runaway population is still negligible.

Usage:

    python scripts/plot_all_runs.py                      # all groups -> ./plots
    python scripts/plot_all_runs.py --group demo_jorek   # one group
    python scripts/plot_all_runs.py --only ar0p3pct      # substring filter
    python scripts/plot_all_runs.py --out /path/figures
    python scripts/plot_all_runs.py --runs-root /path/output --glob "tauw_*"
    python scripts/plot_all_runs.py --pdf                # PDF copies as well

Figures use the native reader in `cdo`, so no DREAM install is needed.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")  # headless: this driver only ever saves.
import matplotlib.pyplot as plt
import numpy as np

import cdo
from cdo import labels, plotting
from cdo.concat import InconsistentRunWarning

DREAM_RUNS_OUTPUT = Path.home() / "nr_dream002/DREAM-runs/output"
DEMO_JOREK_RUNS = Path("/project/nr_dream002/DEMO_data_2026/runs")
MEC2_EV = 510998.95  # electron rest energy, eV

#: Campaigns swept by default: name, output root, folder globs, tag prefix.
GROUPS = [
    ("iter_nonregression", DREAM_RUNS_OUTPUT,
     ("iter_dthmode24_*", "tol_*"), "iter_dthmode24_"),
    ("demo2026_vde", DREAM_RUNS_OUTPUT, ("demo2026_*",), "demo2026_"),
    ("demo_jorek", DEMO_JOREK_RUNS, ("tauw_*",), ""),
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

# --- figure emitter -------------------------------------------------------


class Emitter:
    """Save a returned figure as a PNG in ``base``, and a PDF in ``base/pdf``.

    The PNGs sit together at the top of the run folder for previewing. PDFs are
    written only when ``pdf`` is true. The `cdo.plotting` builders return the
    open figure when given no `savename`, and saving happens here.
    """

    def __init__(self, base: Path, dpi: int = 150, pdf: bool = False):
        self.base = base
        self.pdf_dir = base / "pdf" if pdf else None
        self.dpi = dpi
        self.base.mkdir(parents=True, exist_ok=True)
        if self.pdf_dir is not None:
            self.pdf_dir.mkdir(exist_ok=True)

    def __call__(self, fig, name: str):
        if fig is None:
            return
        fig.savefig(self.base / f"{name}.png", dpi=self.dpi)
        if self.pdf_dir is not None:
            fig.savefig(self.pdf_dir / f"{name}.pdf")
        plt.close(fig)


def _kinetic_current(run):
    """Runaway plus hot-tail current, in amperes, or None if neither exists."""
    kinetic = run.runaway_current
    if kinetic is None:
        return None
    if run.hot_current is not None:
        kinetic = kinetic + run.hot_current
    return kinetic


# --- per-run figure set ---------------------------------------------------


def figure_set(run, emit):
    """Build the standard per-run figures, skipping any the run cannot supply.

    Returns a list of the figure names written, for the index.
    """
    made = []

    def attempt(name, fn):
        try:
            fig = fn()
        except Exception as exc:  # a preliminary run may lack a clean field
            print(f"    [skip] {name}: {exc}")
            plt.close("all")
            return
        if fig is not None:
            emit(fig, name)
            made.append(name)

    t_ms = run.timegrid_ms
    # The first two panel times, 0 and 1 ms or 0 and 0.1 ms, show no runaway
    # population yet, so the distribution panels start from the third.
    panel_frames = plotting.time_marks(t_ms)[2:]
    line_frames = plotting.time_marks(t_ms, lines=True)

    # Currents over time, in megaamperes.
    def currents():
        with plt.rc_context(plotting.LINE_RC):
            curves = [run.plasma_current / 1e6]
            names = [r"$I_{\rm p}$"]
            styles = ["-"]
            kinetic = _kinetic_current(run)
            if kinetic is not None:
                curves.append(kinetic / 1e6)
                names.append(r"$I_{\rm RE}$")
                styles.append("-")
            i_wall = run.field("I_wall")
            if i_wall is not None:
                curves.append(i_wall.reshape(-1) / 1e6)
                names.append(r"$I_{\rm wall}$")
                styles.append("-")
            if run.ohmic_current is not None:
                curves.append(run.ohmic_current / 1e6)
                names.append(r"$I_{\rm ohm}$")
                styles.append("--")
            return plotting.basic_1D(
                curves, t_ms, labels=names, linestyles=styles,
                xlabel=labels.TIME, ylabel=labels.quantity("currents"),
                xlim=(0, None), legendloc="upper right",
            )

    attempt("currents", currents)

    # Negligible channels, such as tritium and Compton with those sources off,
    # are already dropped here.
    rates = cdo.derived.generation_rates(run)

    # Runaway generation rates, volume-averaged over the plasma.
    def generation_rates():
        if not rates:
            return None
        curves = [cdo.derived.volume_average(r, run.cell_volumes)
                  for r in rates.values()]
        peak = max(np.nanmax(np.abs(c)) for c in curves)
        linthresh = peak * 1e-8
        with plt.rc_context(plotting.LINE_RC):
            return plotting.basic_1D(
                curves, t_ms, labels=[labels.RATES[key] for key in rates],
                yscale="symlog", ylinthresh=linthresh,
                yticks=plotting.symlog_ticks(linthresh, peak),
                xlabel=labels.TIME,
                ylabel=r"Volume-averaged rate [1/(s m$^3$)]",
                title=labels.quantity("re_generation_rates"),
                legendloc="best", legendfontsize="x-small",
            )

    attempt("generation_rates", generation_rates)

    # Each generation rate over time and radius.
    def rate_map(label, rate):
        def build():
            peak = float(np.nanmax(np.abs(rate)))
            signed = np.nanmin(rate) < -1e-3 * peak
            with plt.rc_context(plotting.MESH_RC):
                return plotting.basic_2D(
                    rate, run.radialgrid, t_ms, mesh=True,
                    normalization="symlog" if signed else "log",
                    logdiff=8, linthresh=peak * 1e-6,
                    datamin=-peak if signed else None,
                    datamax=peak if signed else None,
                    xlabel=labels.RADIUS, ylabel=labels.TIME,
                    ylim=(t_ms[0], t_ms[-1]),
                    title=label,
                    cbarlabel=(r"Rate [1/(s m$^3$)]" if signed
                               else r"log$_{10}$(rate / 1 s$^{-1}$m$^{-3}$)"),
                )
        return build

    for key, rate in rates.items():
        attempt(f"rate_map_{key}", rate_map(labels.RATES[key], rate))

    # Electron temperature over time and radius.
    def temperature():
        t_cold = run.field("T_cold")
        if t_cold is None:
            return None
        with plt.rc_context(plotting.MESH_RC):
            return plotting.basic_2D(
                t_cold, run.radialgrid, t_ms,
                normalization="log", logdiff=5,
                xlabel=labels.RADIUS, ylabel=labels.TIME,
                title=labels.quantity("temperature"),
                cbarlabel=r"log$_{10}(T_{\rm cold}$/1eV)",
            )

    attempt("temperature", temperature)

    # Ohmic current density over time and radius.
    def ohmic_density():
        j_ohm = run.field("j_ohm")
        if j_ohm is None:
            return None
        with plt.rc_context(plotting.MESH_RC):
            return plotting.basic_2D(
                np.abs(j_ohm), run.radialgrid, t_ms,
                normalization="log", logdiff=4,
                xlabel=labels.RADIUS, ylabel=labels.TIME,
                title=labels.quantity("ohmic_current_density"),
                cbarlabel=r"log$_{10}(|j_{\rm ohm}|$/(1Am$^{-2}$))",
            )

    attempt("ohmic_current_density", ohmic_density)

    # Radiated power density over time and radius. A proper thermal collapse
    # shows up here as a strong, radially broad radiation front.
    def radiated_power():
        rad = run.field("Tcold_radiation")
        if rad is None:
            return None
        with plt.rc_context(plotting.MESH_RC):
            return plotting.basic_2D(
                np.abs(rad), run.radialgrid, t_ms,
                normalization="log", logdiff=6,
                xlabel=labels.RADIUS, ylabel=labels.TIME,
                title=labels.quantity("radiated_power"),
                cbarlabel=r"log$_{10}(P_{\rm rad}$/(1Wm$^{-3}$))",
            )

    attempt("radiated_power", radiated_power)

    # Where the field exceeds the effective critical field: the region and time
    # runaways can be generated. Warm (log10 ratio > 0) is E > Eceff.
    def critical_field_ratio():
        E = run.field("E_field")
        eceff = run.field("Eceff")
        if E is None or eceff is None:
            return None
        with plt.rc_context(plotting.MESH_RC):
            return plotting.critical_field_ratio_2D(
                E, eceff, run.radialgrid, t_ms,
                xlabel=labels.RADIUS, ylabel=labels.TIME,
                title=labels.quantity("critical_field_ratio"),
            )

    attempt("critical_field_ratio", critical_field_ratio)

    # Runaway momentum-pitch distribution, final step, inner radius.
    def distribution():
        if run.field("f_re") is None:
            return None
        with plt.rc_context(plotting.MESH_RC):
            return plotting.distribution_momentum_2D(
                run, "runaway", timestep=-1, radial_cell=0,
                cbarlabel=labels.quantity("electron_distribution"),
            )

    attempt("runaway_distribution", distribution)

    # Momentum-pitch distributions at the snapshot times, at the magnetic axis
    # and at mid-radius. An isotropic hot-tail grid has one pitch cell, so it
    # gets f(p) curves instead.
    radial_cells = sorted({0, len(run.radialgrid) // 2})

    def distribution_panels(grid, radial_cell):
        def build():
            if run.field("f_hot" if grid == "hottail" else "f_re") is None:
                return None
            with plt.rc_context(plotting.MESH_RC):
                return plotting.distribution_momentum_panels(
                    run, grid, timesteps=panel_frames, radial_cell=radial_cell,
                    cbarlabel=labels.quantity("electron_distribution"),
                )
        return build

    def isotropic_hottail(radial_cell):
        def build():
            f = run.field("f_hot")
            if f is None:
                return None
            p = run.reference.read("hot_p")
            colours = plt.cm.viridis(np.linspace(0.9, 0.0, len(line_frames)))
            with plt.rc_context(plotting.LINE_RC):
                fig = plt.figure(figsize=(8, 5))
                for c, i in zip(colours, line_frames):
                    plt.plot(p, f[i, radial_cell, 0, :], color=c,
                             label=plotting.time_label(t_ms, i))
                plt.yscale("log")
                plt.xlabel(labels.MOMENTUM)
                plt.ylabel(labels.quantity("electron_distribution"))
                plt.title(f"Hot tail, r = {run.radialgrid[radial_cell]:.2f} m")
                plt.legend(fontsize="x-small")
                return fig
        return build

    hot_xi = run.reference.read("hot_xi") if run.reference.grids.get("hottail") else None
    for rc in radial_cells:
        attempt(f"f_re_snapshots_r{rc}", distribution_panels("runaway", rc))
        if hot_xi is None:
            continue
        if len(hot_xi) > 1:
            attempt(f"f_hot_snapshots_r{rc}", distribution_panels("hottail", rc))
        else:
            attempt(f"f_hot_isotropic_r{rc}", isotropic_hottail(rc))

    # Runaway energy spectrum at the inner radius, coloured by time.
    def energy_spectrum():
        from cdo.energy import EnergyGrid

        grid = EnergyGrid.from_run(run)
        if grid is None:
            return None
        dndp = run.angle_average("runaway", "density")
        if dndp is None:
            return None
        dnde = grid.to_energy(dndp, per="MeV")  # dE in MeV, matching the x-axis
        kinetic_MeV = (grid.total_energy_eV - MEC2_EV) / 1e6
        radial_cell = 0
        colours = plt.cm.viridis(np.linspace(0.9, 0.0, len(line_frames)))
        with plt.rc_context(plotting.LINE_RC):
            fig = plt.figure(figsize=(8, 5))
            for c, i in zip(colours, line_frames):
                plt.plot(kinetic_MeV, dnde[i, radial_cell, :],
                         color=c, label=plotting.time_label(t_ms, i))
            plt.yscale("log")
            plt.xlabel(labels.KINETIC_ENERGY)
            plt.ylabel(r"$dn/(dE\,dr)$ [1/(MeV m)]")
            plt.title(f"At r = {run.radialgrid[radial_cell]:.2f} m")
            plt.legend(fontsize="small")
            return fig

    attempt("energy_spectrum", energy_spectrum)

    # Electron temperature radial profiles, colored by time.
    def radial_electron_temperature_profiles():
        colours = plt.cm.viridis(np.linspace(0.9, 0.0, len(line_frames)))
        with plt.rc_context(plotting.LINE_RC):
            fig = plt.figure(figsize=(8, 5))
            for c, i in zip(colours, line_frames):
                plt.plot(run.radialgrid, run.field("T_cold")[i, :],
                         color=c, label=plotting.time_label(t_ms, i))
            plt.yscale("log")
            plt.xlabel(labels.RADIUS)
            plt.ylabel(labels.QUANTITIES["temperature_eV"])
            plt.title(labels.QUANTITIES["temperature"])
            plt.legend(fontsize="small")
            return fig

    attempt(
        "radial_electron_temperature_profiles",
        radial_electron_temperature_profiles
    )

    return made


# --- cross-run comparison -------------------------------------------------


def load_jorek_trace(path):
    """A JOREK I_p trace from a two-column text file, JOREK time [ms] and I_p [MA]."""
    data = np.loadtxt(path, delimiter="," if str(path).endswith(".csv") else None,
                      comments="#")
    return data[:, 0], data[:, 1]


def comparison_figures(series, emit, jorek=False, jorek_trace=None):
    """Overlay the plasma and runaway currents of every run on shared axes.

    With ``jorek`` the time axis is JOREK time, DREAM time shifted by
    :data:`JOREK_T0_MS`, and the known JOREK current values are overdrawn in
    black: :data:`JOREK_IP_POINTS` on the plasma current, the RE plateau on the
    runaway current, and ``jorek_trace`` (times, I_p in MA) when given.
    """
    if not series:
        return []
    made = []
    colours = plt.cm.turbo(np.linspace(0.05, 0.95, len(series)))
    offset = JOREK_T0_MS if jorek else 0.0

    def overlay(key, ylabel, absval=False, reference=()):
        with plt.rc_context(plotting.LINE_RC):
            fig = plt.figure(figsize=(11, 5))
            drawn = False
            for c, s in zip(colours, series):
                y = s.get(key)
                if y is None:
                    continue
                y = np.abs(y) if absval else y
                plt.plot(s["t_ms"] + offset, y / 1e6, color=c, label=s["tag"])
                drawn = True
            if not drawn:
                plt.close(fig)
                return None
            if jorek and key == "Ip" and jorek_trace is not None:
                plt.plot(*jorek_trace, color="k", lw=2.5, label="JOREK")
            if reference:
                t_ref, i_ref = zip(*reference)
                plt.plot(t_ref, i_ref, ls="none", marker="o", ms=8, mfc="none",
                         mec="k", mew=2, zorder=5,
                         label="JOREK" if jorek_trace is None or key != "Ip" else None)
            axes = plt.gca()
            axes.set_xlabel("JOREK time [ms]" if jorek else labels.TIME)
            axes.set_ylabel(ylabel)
            axes.set_xlim(0, None)
            # Outside the axes, so no curve or reference marker is hidden.
            axes.legend(fontsize="x-small", ncol=1, loc="upper left",
                        bbox_to_anchor=(1.01, 1.0))
            return fig

    fig = overlay("Ip", r"Plasma current $I_{\rm p}$ [MA]",
                  reference=JOREK_IP_POINTS if jorek else ())
    if fig is not None:
        emit(fig, "compare_plasma_current")
        made.append("compare_plasma_current")

    fig = overlay("Ire", r"Runaway current $I_{\rm RE}$ [MA]", absval=True,
                  reference=[(267.0, JOREK_PLATEAU_MA)] if jorek else ())
    if fig is not None:
        emit(fig, "compare_runaway_current")
        made.append("compare_runaway_current")

    return made


# --- driver ---------------------------------------------------------------


def discover(runs_root: Path, globs, only: str | None):
    runs = []
    for pattern in globs:
        for folder in sorted(runs_root.glob(pattern)):
            if not folder.is_dir():
                continue
            if only and only not in folder.name:
                continue
            if not cdo.find_outputs(folder):
                continue
            runs.append(folder)
    return runs


def process(folder: Path, tag: str, out_dir: Path, dpi: int, per_run: bool = True,
            pdf: bool = False):
    """Load one run, write its figures, and return a progress record."""
    record = {"tag": tag, "folder": str(folder)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentRunWarning)
        run = cdo.Run.from_folder(folder)
        try:
            t_ms = run.timegrid_ms
            record.update(
                nfiles=len(run.files),
                nsteps=run.timegrid_length,
                t_end_ms=float(t_ms[-1]),
                grids=", ".join(k for k, v in run.reference.grids.items() if v) or "none",
                Ip_end_MA=float(run.plasma_current[-1] / 1e6),
            )
            kinetic = _kinetic_current(run)
            record["Ire_max_MA"] = (
                float(np.nanmax(np.abs(kinetic)) / 1e6) if kinetic is not None else None)
            comp = {
                "tag": tag,
                "t_ms": t_ms,
                "Ip": run.plasma_current.copy(),
                "Ire": None if kinetic is None else np.asarray(kinetic).copy(),
            }
            if per_run:
                emit = Emitter(out_dir / tag, dpi=dpi, pdf=pdf)
                record["figures"] = figure_set(run, emit)
            else:
                record["figures"] = []
        finally:
            run.close()
    return record, comp


def write_index(sections, out_root: Path):
    lines = [
        "# DREAM run figures",
        "",
        "Each run folder holds its figures as PNGs, with PDF copies in `pdf/` "
        "when the sweep ran with `--pdf`.",
        "Cross-run overlays are under `<group>/_comparison/`.",
    ]
    for group, runs_root, records, compare_made in sections:
        lines += [
            "",
            f"## {group}",
            "",
            f"Source: `{runs_root}`",
            "",
            "| run | files | steps | t_end [ms] | I_p end [MA] | I_RE max [MA] | grids | figures |",
            "|---|--:|--:|--:|--:|--:|---|--:|",
        ]
        for r in records:
            ire = "-" if r.get("Ire_max_MA") is None else f"{r['Ire_max_MA']:.3g}"
            lines.append(
                "| {tag} | {nfiles} | {nsteps} | {t_end:.4g} | {ip:.4g} | {ire} | {grids} | {nfig} |".format(
                    tag=r["tag"], nfiles=r.get("nfiles", "-"), nsteps=r.get("nsteps", "-"),
                    t_end=r.get("t_end_ms", float("nan")), ip=r.get("Ip_end_MA", float("nan")),
                    ire=ire, grids=r.get("grids", "-"), nfig=len(r.get("figures", [])),
                )
            )
        if compare_made:
            lines += ["", "Comparison figures: " + ", ".join(f"`{n}`" for n in compare_made)]
    (out_root / "index.md").write_text("\n".join(lines) + "\n")


def sweep_group(group, runs_root, globs, prefix, args):
    """Plot every run of one group and its comparison overlays."""
    runs = discover(runs_root, globs, args.only)
    print(f"== group {group}: {len(runs)} run(s) under {runs_root}")
    out_dir = args.out / group
    records, series = [], []
    for folder in runs:
        tag = folder.name.removeprefix(prefix) if prefix else folder.name
        print(f"-- {tag}")
        try:
            record, comp = process(folder, tag, out_dir, args.dpi,
                                   per_run=not args.no_per_run, pdf=args.pdf)
        except Exception as exc:
            print(f"    [error] {exc}")
            records.append({"tag": tag, "folder": str(folder)})
            continue
        ire = record.get("Ire_max_MA")
        print("    {nfiles} files, {nsteps} steps, to {t:.4g} ms, "
              "I_p={ip:.3g} MA, I_RE_max={ire} MA, figs={nf}".format(
                  nfiles=record.get("nfiles"), nsteps=record.get("nsteps"),
                  t=record.get("t_end_ms", float("nan")), ip=record.get("Ip_end_MA", float("nan")),
                  ire="-" if ire is None else f"{ire:.3g}", nf=len(record.get("figures", []))))
        records.append(record)
        series.append(comp)

    compare_made = []
    if not args.no_compare and series:
        emit = Emitter(out_dir / "_comparison", dpi=args.dpi, pdf=args.pdf)
        jorek = group == "demo_jorek"
        trace = load_jorek_trace(args.jorek_csv) if jorek and args.jorek_csv else None
        compare_made = comparison_figures(series, emit, jorek=jorek, jorek_trace=trace)
        if compare_made:
            print(f"   comparison: {', '.join(compare_made)}")
    return group, runs_root, records, compare_made


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-root", type=Path, default=None,
                        help="sweep this root alone instead of the default groups")
    parser.add_argument("--glob", default="*",
                        help="run folder glob under --runs-root")
    parser.add_argument("--group", default=None,
                        help="only the default group of this name")
    parser.add_argument("--out", type=Path,
                        default=Path(os.environ.get(
                            "CDO_FIGURE_ROOT", Path(__file__).resolve().parent.parent / "plots")))
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
    parser.add_argument("--no-compare", action="store_true",
                        help="skip the cross-run overlays")
    args = parser.parse_args(argv)

    plotting.use_base_style(latex=False, sansserif=True)

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
    if args.no_per_run or args.group or args.only:
        return  # a partial sweep would overwrite the full index
    write_index(sections, args.out)
    print(f"index: {args.out / 'index.md'}")


if __name__ == "__main__":
    main()
