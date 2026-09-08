"""Concatenate a DREAM run split across several output files into one series.

A run is written as a sequence of files, each starting at its own local time
zero and sharing its first time step with the previous file's last. Stitching
them means dropping that shared initial step from every ``eqsys`` dataset, while
``other`` datasets are already one step shorter and need no trimming. The
per-field time alignment recorded in :mod:`cdo.schema` drives this, so the
trimming happens once here rather than as a scattered ``[1:]`` at every read,
which is where the retired ``CDOconcat`` accumulated silent mistakes.

Fields are concatenated lazily by canonical name and cached. Grid and geometry
come from one reference file. Before stitching, the files are checked for
agreement in grid shape and in which fields they carry, and any disagreement is
reported rather than being hidden and zero-filled.
"""

from __future__ import annotations

import warnings

import numpy as np

from . import derived
from .io import find_outputs, open_output
from .moments import MomentumMoments, current
from .schema import FIELDS, Presence, Resolver, TimeAxis


class InconsistentRunWarning(UserWarning):
    """The files of one run disagree in structure."""


class Run:
    """A concatenated DREAM run.

    Open a run from a folder with :meth:`from_folder`, or pass an explicit list
    of files. Access raw quantities by canonical name through :meth:`field`, or
    the named convenience properties. Momentum-grid moments come from
    :meth:`current` and :meth:`angle_average`.
    """

    def __init__(self, files, start_time: float = 0.0, reference: int = -1):
        if not files:
            raise ValueError("a run needs at least one output file")

        self.files = [str(f) for f in files]
        self.start_time = start_time
        self._handles = [open_output(f) for f in self.files]
        self._resolvers = [
            Resolver(h, name=f) for h, f in zip(self._handles, self.files)
        ]
        self.reference = self._resolvers[reference]

        # Time steps contributed by each file, the initial step dropped.
        self._nt = [len(r.read("time")) - 1 for r in self._resolvers]
        self.timegrid_length = sum(self._nt)

        self._cache: dict[str, np.ndarray] = {}
        self._moments: dict[str, MomentumMoments] = {}

        self._build_time_grid()
        self._check_consistency()

    @classmethod
    def from_folder(cls, folder, *, init: bool = False, **kwargs) -> "Run":
        files = find_outputs(folder, init=init)
        if not files:
            raise FileNotFoundError(f"no output files in {folder}")
        return cls(files, **kwargs)

    def close(self):
        for handle in self._handles:
            handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # --- time grid ---------------------------------------------------------

    def _build_time_grid(self):
        time = np.zeros(self.timegrid_length)
        ti = 0
        offset = self.start_time
        for resolver, nt in zip(self._resolvers, self._nt):
            local = resolver.read("time")[1:]
            time[ti : ti + nt] = local + offset
            offset += local[-1]
            ti += nt
        self.timegrid = time
        self.timegrid_ms = time * 1000.0

    # --- consistency -------------------------------------------------------

    def _check_consistency(self):
        """Report structural disagreements between the run's files."""
        ref = self.reference
        problems = []

        ref_shape = ref.read("radius").shape
        for resolver in self._resolvers:
            if resolver is ref:
                continue
            if resolver.read("radius").shape != ref_shape:
                problems.append(
                    f"{resolver.name}: radial grid {resolver.read('radius').shape} "
                    f"differs from reference {ref_shape}"
                )
            for grid in ("hottail", "runaway"):
                if resolver.grids.get(grid) != ref.grids.get(grid):
                    problems.append(
                        f"{resolver.name}: {grid} grid enabled="
                        f"{resolver.grids.get(grid)}, reference="
                        f"{ref.grids.get(grid)}"
                    )

        if problems:
            warnings.warn(
                "run files disagree in structure:\n  " + "\n  ".join(problems),
                InconsistentRunWarning,
                stacklevel=2,
            )
        self.problems = problems

    # --- field access ------------------------------------------------------

    def field(self, name: str) -> np.ndarray | None:
        """Concatenated time series for a canonical field, or ``None`` if absent.

        Non-time fields (grids, geometry, scalars) come from the reference file
        unchanged. Time fields are stitched across all files. A field present in
        some files but not others is filled with NaN over the missing stretches
        and reported once, so the gap shows in a plot instead of reading as zero.
        """
        if name in self._cache:
            return self._cache[name]

        spec = FIELDS[name]
        if spec.time is TimeAxis.NONE:
            value = self.reference.read(name)
            self._cache[name] = value
            return value

        value = self._concatenate(name, spec)
        self._cache[name] = value
        return value

    def _concatenate(self, name: str, spec) -> np.ndarray | None:
        reads = [r.read(name) for r in self._resolvers]
        present = [a is not None for a in reads]

        if not any(present):
            return None

        sample = next(a for a in reads if a is not None)
        out = np.full(
            (self.timegrid_length, *sample.shape[1:]), np.nan, dtype=float
        )

        ti = 0
        for resolver, nt, arr in zip(self._resolvers, self._nt, reads):
            if arr is not None:
                trimmed = arr[1:] if spec.time is TimeAxis.FULL else arr
                out[ti : ti + nt] = trimmed
            ti += nt

        if not all(present):
            missing = [
                r.name for r, ok in zip(self._resolvers, present) if not ok
            ]
            warnings.warn(
                f"'{name}' is absent from {', '.join(missing)}; "
                f"those stretches are NaN",
                InconsistentRunWarning,
                stacklevel=3,
            )
        return out

    # --- named quantities --------------------------------------------------

    @property
    def radialgrid(self) -> np.ndarray:
        return self.reference.read("radius")

    @property
    def major_radius(self) -> float:
        return float(np.asarray(self.reference.read("major_radius")).ravel()[0])

    @property
    def minor_radius(self) -> float:
        return float(np.asarray(self.reference.read("minor_radius")).ravel()[0])

    @property
    def cell_volumes(self) -> np.ndarray:
        return derived.cell_volumes(
            self.reference.read("VpVol"),
            self.reference.read("radial_step"),
            self.reference.read("major_radius"),
        )

    def radial_integral(self, name: str) -> np.ndarray:
        """Integrate a per-volume field over radius, ``(t,)``."""
        return derived.radial_integral(self.field(name), self.cell_volumes)

    @property
    def plasma_current(self) -> np.ndarray:
        """Total plasma current, ``(t,)``. DREAM stores ``I_p`` as ``(t, 1)``."""
        return self.field("I_p").reshape(self.timegrid_length)

    @property
    def runaway_current(self) -> np.ndarray | None:
        return self.current("j_re")

    @property
    def hot_current(self) -> np.ndarray | None:
        return self.current("j_hot")

    @property
    def ohmic_current(self) -> np.ndarray | None:
        return self.current("j_ohm")

    # --- momentum-grid moments ---------------------------------------------

    def moments(self, grid: str) -> MomentumMoments | None:
        if grid not in self._moments:
            self._moments[grid] = MomentumMoments.from_resolver(
                self.reference, grid
            )
        return self._moments[grid]

    def current(self, name: str) -> np.ndarray | None:
        """Total current from a current density field, ``(t,)``.

        ``name`` is the density, for example ``"j_re"``. Returns ``None`` when
        the density is absent.
        """
        j = self.field(name)
        if j is None:
            return None
        return current(
            j,
            self.reference.read("VpVol"),
            self.reference.read("radial_step"),
            self.reference.read("GR0"),
            self.reference.read("B_min"),
            self.reference.read("FSA_R02OverR2"),
        )

    def angle_average(
        self, grid: str, moment: str = "distribution"
    ) -> np.ndarray | None:
        """Angle-averaged distribution moment for a momentum grid.

        ``grid`` is ``"hottail"`` or ``"runaway"``. ``moment`` is
        ``"distribution"``, ``"density"`` or ``"current"``. Returns ``None`` when
        the grid is disabled.
        """
        mm = self.moments(grid)
        if mm is None:
            return None
        f = self.field("f_hot" if grid == "hottail" else "f_re")
        if f is None:
            return None
        return {
            "distribution": mm.distribution,
            "density": mm.density,
            "current": mm.current_density,
        }[moment](f)

    # --- ions --------------------------------------------------------------

    @property
    def ions(self) -> list[derived.IonSpecies]:
        if not hasattr(self, "_ions"):
            self._ions = derived.parse_ions(
                self.reference.read_string_list("ion_names"),
                self.reference.read("ion_Z"),
            )
        return self._ions

    def ion(self, name: str) -> derived.IonSpecies:
        for species in self.ions:
            if species.name == name:
                return species
        raise KeyError(f"no ion species '{name}'. Have: {[s.name for s in self.ions]}")

    # --- reporting ---------------------------------------------------------

    def report(self) -> str:
        lines = [
            f"Run of {len(self.files)} file(s), "
            f"{self.timegrid_length} time steps, "
            f"{self.timegrid_ms[0]:.4g} to {self.timegrid_ms[-1]:.4g} ms",
            f"  radius: {len(self.radialgrid)} cells, "
            f"minor {self.minor_radius:.4g} m, major {self.major_radius:.4g} m",
            f"  ions: {', '.join(f'{s.name}(Z={s.Z})' for s in self.ions)}",
        ]
        grids = ", ".join(k for k, v in self.reference.grids.items() if v) or "none"
        lines.append(f"  momentum grids: {grids}")
        if self.problems:
            lines.append("  structural problems:")
            lines += [f"    {p}" for p in self.problems]
        return "\n".join(lines)
