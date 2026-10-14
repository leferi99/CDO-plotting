"""Batch figures for every DREAM run of the campaigns listed in `GROUPS`.

Sweeps the run folders of each group, builds a standard figure set for each run,
and draws cross-run overlays within the group. Every figure is guarded, so a run
missing a field or a grid is skipped for that figure without aborting the sweep.

Each figure is written twice, as a PNG under `<out>/<group>/<tag>/png/` and as a
PDF under `<out>/<group>/<tag>/pdf/`. Cross-run overlays go under
`<out>/<group>/_comparison/`. A progress table is written to `<out>/index.md`.

Snapshot times for profiles, spectra and distributions come from
`plotting.time_marks`: 0, 1, 10, 25, 50, 75, 100, 150 ... ms for long runs, a
finer ladder for runs shorter than 20 ms, always closing with the final step.

Usage:

    python scripts/plot_all_runs.py                      # all groups -> ./plots
    python scripts/plot_all_runs.py --group demo_jorek   # one group
    python scripts/plot_all_runs.py --only ar0p3pct      # substring filter
    python scripts/plot_all_runs.py --out /path/figures
    python scripts/plot_all_runs.py --runs-root /path/output --glob "tauw_*"

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

#: Runaway generation channels: canonical field, file-name key, legend label.
#: With a kinetic runaway grid, `runawayRate` is the flux from the hot-tail grid
#: into the runaway grid, and the Dreicer and avalanche rates are fluid-model
#: estimates that do not enter it. Avalanche acts on `f_re` directly.
RATE_CHANNELS = [
    ("runawayRate", "total", r"$\gamma_{\rm tot}$"),
    ("gammaHottail", "hottail_flux", r"$\gamma_{\rm hot}$"),
    ("gammaDreicer", "dreicer", r"$\gamma_{\rm D}$"),
    ("avalanche", "avalanche", r"$\Gamma_{\rm ava} n_{\rm RE}$"),
    ("gammaCompton", "compton", r"$\gamma_{\rm C}$"),
    ("gammaTritium", "tritium", r"$\gamma_{\rm T}$"),
]


# --- figure emitter -------------------------------------------------------


class Emitter:
    """Save a returned figure as PNG and PDF in separate subdirectories.

    The `cdo.plotting` builders return the open figure when given no `savename`.
    Saving both formats is done here rather than in the package, which writes
    PNG only.
    """

    def __init__(self, base: Path, dpi: int = 150):
        self.png_dir = base / "png"
        self.pdf_dir = base / "pdf"
        self.dpi = dpi
        self.png_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, fig, name: str):
        if fig is None:
            return
        fig.savefig(self.png_dir / f"{name}.png", dpi=self.dpi)
        fig.savefig(self.pdf_dir / f"{name}.pdf")
        plt.close(fig)


def _rate(run, name):
    """A generation rate in 1/(s m^3) over (time, radius), or None if absent."""
    if name == "avalanche":
        gamma, n_re = run.field("GammaAva"), run.field("n_re")
        return None if gamma is None or n_re is None else gamma * n_re
    return run.field(name)


def _volume_mean(run, field):
    """Volume-weighted radial mean of a (time, radius) field."""
    dV = run.cell_volumes
    return np.nansum(field * dV, axis=1) / np.sum(dV)


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
    mesh_frames = plotting.time_marks(t_ms)
    line_frames = plotting.time_marks(t_ms, sparse=True)

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

    rates = [(key, label, _rate(run, name)) for name, key, label in RATE_CHANNELS]
    rates = [(k, lab, r) for k, lab, r in rates if r is not None]
    # Drop channels that stay negligible, such as tritium and Compton when those
    # sources are off, or a fluid Dreicer estimate many decades below the rest.
    strongest = max((np.nanmax(np.abs(r)) for _, _, r in rates), default=0.0)
    rates = [(k, lab, r) for k, lab, r in rates
             if np.nanmax(np.abs(r)) > 1e-12 * strongest]

    # Runaway generation rates, volume-averaged over the plasma.
    def generation_rates():
        if not rates:
            return None
        curves = [_volume_mean(run, r) for _, _, r in rates]
        peak = max(np.nanmax(np.abs(c)) for c in curves)
        linthresh = peak * 1e-8
        with plt.rc_context(plotting.LINE_RC):
            fig = plotting.basic_1D(
                curves, t_ms, labels=[lab for _, lab, _ in rates],
                yscale="symlog", ylinthresh=linthresh,
                xlabel=labels.TIME,
                ylabel=r"Volume-averaged rate [1/(s m$^3$)]",
                title=labels.quantity("re_generation_rates"),
                legendloc="best", legendfontsize="x-small",
            )
            fig.axes[0].set_yticks(plotting.symlog_ticks(linthresh, peak))
            return fig

    attempt("generation_rates", generation_rates)

    # Each generation rate over time and radius, on linear and log time axes.
    # The log axis resolves the thermal quench in the first millisecond.
    def rate_map(label, rate, yscale):
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
                    xlabel=labels.RADIUS, ylabel=labels.TIME, yscale=yscale,
                    ylim=(t_ms[0], t_ms[-1]),
                    title=label,
                    cbarlabel=(r"Rate [1/(s m$^3$)]" if signed
                               else r"log$_{10}$(rate / 1 s$^{-1}$m$^{-3}$)"),
                )
        return build

    for key, label, rate in rates:
        attempt(f"rate_map_{key}", rate_map(label, rate, "linear"))
        attempt(f"rate_map_{key}_logtime", rate_map(label, rate, "log"))

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
                    run, grid, timesteps=mesh_frames, radial_cell=radial_cell,
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
                             label=f"{t_ms[i]:.3g} ms")
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
                         color=c, label=f"{t_ms[i]:.3g} ms")
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
                         color=c, label=f"{t_ms[i]:.3g} ms")
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


def comparison_figures(series, emit):
    """Overlay the plasma and runaway currents of every run on shared axes."""
    if not series:
        return []
    made = []
    colours = plt.cm.turbo(np.linspace(0.05, 0.95, len(series)))

    def overlay(key, ylabel, absval=False):
        with plt.rc_context(plotting.LINE_RC):
            fig = plt.figure(figsize=(8, 5))
            drawn = False
            for c, s in zip(colours, series):
                y = s.get(key)
                if y is None:
                    continue
                y = np.abs(y) if absval else y
                plt.plot(s["t_ms"], y / 1e6, color=c, label=s["tag"])
                drawn = True
            if not drawn:
                plt.close(fig)
                return None
            axes = plt.gca()
            axes.set_xlabel(labels.TIME)
            axes.set_ylabel(ylabel)
            axes.set_xlim(0, None)
            axes.legend(fontsize="x-small", ncol=1, loc="best")
            return fig

    fig = overlay("Ip", r"Plasma current $I_{\rm p}$ [MA]")
    if fig is not None:
        emit(fig, "compare_plasma_current")
        made.append("compare_plasma_current")

    fig = overlay("Ire", r"Runaway current $I_{\rm RE}$ [MA]", absval=True)
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


def process(folder: Path, tag: str, out_dir: Path, dpi: int, per_run: bool = True):
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
                emit = Emitter(out_dir / tag, dpi=dpi)
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
        "Each run folder holds `png/` and `pdf/` subdirectories with the same figures.",
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
                                   per_run=not args.no_per_run)
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
        emit = Emitter(out_dir / "_comparison", dpi=args.dpi)
        compare_made = comparison_figures(series, emit)
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
    parser.add_argument("--dpi", type=int, default=150)
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

    write_index(sections, args.out)
    print(f"\nfigures under {args.out}  (per run: png/ and pdf/)")
    print(f"index: {args.out / 'index.md'}")


if __name__ == "__main__":
    main()
