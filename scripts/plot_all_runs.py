"""Batch figures for every DREAM run under an output root.

Sweeps all `iter_dthmode24_*` run folders, builds a standard figure set for
each, and draws cross-run overlays comparing the versions and variants. Meant
for preliminary looks at runs still on the cluster, so every figure is guarded
and a run that is missing a field or a grid is skipped for that figure rather
than aborting the sweep.

Each figure is written twice, as a PNG under `<out>/<tag>/png/` and as a PDF
under `<out>/<tag>/pdf/`. Cross-run overlays go under `<out>/_comparison/`.
A progress table is written to `<out>/index.md`.

Usage:

    python scripts/plot_all_runs.py                      # all runs -> ./plots
    python scripts/plot_all_runs.py --only ar0p3pct      # substring filter
    python scripts/plot_all_runs.py --out /path/figures
    python scripts/plot_all_runs.py --runs-root /path/output

Figures use the native reader in `cdo`, so no DREAM install is needed. This is a
driver on top of `cdo`; it changes none of the package or the # %% scripts.
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

DEFAULT_RUNS_ROOT = Path.home() / "nr_dream002/DREAM-runs/output"
RUN_GLOB = "iter_dthmode24_*"
MEC2_EV = 510998.95  # electron rest energy, eV


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


def _radial_mean(field):
    """Radial mean of a (time, radius) field, ignoring NaN stretches."""
    return np.nanmean(field, axis=1)


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
            return
        if fig is not None:
            emit(fig, name)
            made.append(name)

    t_ms = run.timegrid_ms

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

    # Runaway generation rates, radially averaged.
    def generation_rates():
        rate = run.field("runawayRate")
        if rate is None:
            return None
        flux = cdo.derived.flux_to_re(
            rate, run.field("n_re"), run.field("GammaAva"),
            run.field("gammaTritium"), run.field("gammaCompton"),
        )
        with plt.rc_context(plotting.LINE_RC):
            return plotting.basic_1D(
                [
                    _radial_mean(run.field("GammaAva") * run.field("n_re")),
                    _radial_mean(run.field("gammaCompton")),
                    _radial_mean(run.field("gammaTritium")),
                    _radial_mean(flux),
                ],
                t_ms,
                labels=[r"$\gamma_{\rm ava}$", r"$\gamma_{\rm C}$",
                        r"$\gamma_{\rm T}$", r"$\gamma_{\rm D+ht}$"],
                yscale="symlog", ylinthresh=1.0,
                xlabel=labels.TIME,
                ylabel=labels.quantity("re_generation_rates"),
                legendloc="upper left",
            )

    attempt("generation_rates", generation_rates)

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
                title=labels.quantity("electron_temperature"),
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
        end = t_ms[-1]
        marks = np.arange(0, end + 1, max(end / 6, 1))
        frames = np.clip(
            plotting.index_array(t_ms, marks), 0, run.timegrid_length - 1)
        colours = plt.cm.viridis(np.linspace(0.9, 0.0, len(frames)))
        with plt.rc_context(plotting.LINE_RC):
            fig = plt.figure(figsize=(8, 5))
            for c, i in zip(colours, frames):
                plt.plot(kinetic_MeV, dnde[i, radial_cell, :],
                         color=c, label=f"{t_ms[i]:.0f} ms")
            plt.yscale("log")
            plt.xlabel(labels.KINETIC_ENERGY)
            plt.ylabel(r"$dn/(dE\,dr)$ [1/(MeV m)]")
            plt.title(f"At r = {run.radialgrid[radial_cell]:.2f} m")
            plt.legend(fontsize="small")
            return fig

    attempt("energy_spectrum", energy_spectrum)

    return made


# --- cross-run comparison -------------------------------------------------


def comparison_figures(series, emit):
    """Overlay the plasma and runaway currents of every run on shared axes."""
    if not series:
        return []
    made = []
    colours = plt.cm.turbo(np.linspace(0.05, 0.95, len(series)))

    def overlay(key, ylabel, name, absval=False):
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

    fig = overlay("Ip", r"Plasma current $I_{\rm p}$ [MA]", "compare_plasma_current")
    if fig is not None:
        emit(fig, "compare_plasma_current")
        made.append("compare_plasma_current")

    fig = overlay("Ire", r"Runaway current $I_{\rm RE}$ [MA]",
                  "compare_runaway_current", absval=True)
    if fig is not None:
        emit(fig, "compare_runaway_current")
        made.append("compare_runaway_current")

    return made


# --- driver ---------------------------------------------------------------


def discover(runs_root: Path, only: str | None):
    runs = []
    for folder in sorted(runs_root.glob(RUN_GLOB)):
        if not folder.is_dir():
            continue
        if only and only not in folder.name:
            continue
        if not cdo.find_outputs(folder):
            continue
        runs.append(folder)
    return runs


def process(folder: Path, out_root: Path, dpi: int, per_run: bool = True):
    """Load one run, write its figures, and return a progress record."""
    tag = folder.name.replace("iter_dthmode24_", "")
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
                emit = Emitter(out_root / tag, dpi=dpi)
                record["figures"] = figure_set(run, emit)
            else:
                record["figures"] = []
        finally:
            run.close()
    return record, comp


def write_index(records, out_root: Path, runs_root: Path, compare_made):
    lines = [
        "# DREAM run figures",
        "",
        f"Source: `{runs_root}`",
        "",
        "Each run folder holds `png/` and `pdf/` subdirectories with the same figures.",
        "Cross-run overlays are under `_comparison/`. Wall-time and leg progress come",
        "from `non_regression_testing_2026/run_files/leg_stats.py`.",
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-root", type=Path,
                        default=Path(os.environ.get("CDO_RUNS_ROOT", DEFAULT_RUNS_ROOT)))
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

    runs = discover(args.runs_root, args.only)
    if not runs:
        raise SystemExit(f"no runs matching under {args.runs_root}")
    print(f"{len(runs)} run(s) under {args.runs_root}")
    args.out.mkdir(parents=True, exist_ok=True)

    records, series = [], []
    for folder in runs:
        tag = folder.name.replace("iter_dthmode24_", "")
        print(f"== {tag}")
        try:
            record, comp = process(folder, args.out, args.dpi, per_run=not args.no_per_run)
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
    if not args.no_compare:
        emit = Emitter(args.out / "_comparison", dpi=args.dpi)
        compare_made = comparison_figures(series, emit)
        if compare_made:
            print(f"comparison: {', '.join(compare_made)}")

    write_index(records, args.out, args.runs_root, compare_made)
    print(f"\nfigures under {args.out}  (per run: png/ and pdf/)")
    print(f"index: {args.out / 'index.md'}")


if __name__ == "__main__":
    main()
