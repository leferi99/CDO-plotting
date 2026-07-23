"""Cross-check native moments against DREAM's own reader.

Skipped unless DREAM is importable and the reference output files are present.
This is the regression guard for the formulas in :mod:`cdo.moments`: if a future
DREAM changes a convention, this fails while the synthetic tests still pass.

Point the environment at a DREAM checkout before running, for example
``PYTHONPATH=~/nr_dream002/DREAM-runs/commit-53d8afb/py``.
"""

import os

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("DREAM")

from DREAM.DREAMOutput import DREAMOutput  # noqa: E402

from cdo.concat import Run  # noqa: E402

REFERENCE_FILES = [
    os.path.join(os.path.dirname(__file__), "..", "output_7_.h5"),
    "/home/nr_drlf/nr_dream002/DREAM-runs/output/"
    "iter_dthmode24_rp_commit-53d8afb/output_0000.h5",
]

AVAILABLE = [p for p in REFERENCE_FILES if os.path.exists(p)]


def reldiff(a, b):
    return np.nanmax(np.abs(a - b) / (np.abs(b) + 1e-300))


@pytest.fixture(params=AVAILABLE, ids=[os.path.basename(p) for p in AVAILABLE])
def reference(request):
    if not AVAILABLE:
        pytest.skip("no reference output files present")
    path = request.param
    run = Run([path])
    do = DREAMOutput(path, loadsettings=False)
    yield run, do
    run.close()
    do.close()


@pytest.mark.parametrize("density", ["j_re", "j_ohm", "j_hot"])
def test_current_matches_dream(reference, density):
    run, do = reference
    theirs = np.asarray(getattr(do.eqsys, density).current())[1:]
    assert reldiff(run.current(density), theirs) < 1e-12


@pytest.mark.parametrize("grid,field", [("runaway", "f_re"), ("hottail", "f_hot")])
@pytest.mark.parametrize("moment", ["distribution", "density", "current"])
def test_angle_average_matches_dream(reference, grid, field, moment):
    run, do = reference
    theirs = np.asarray(getattr(do.eqsys, field).angleAveraged(moment=moment))[1:]
    assert reldiff(run.angle_average(grid, moment), theirs) < 1e-12


@pytest.mark.parametrize("quantity", ["n_re", "n_cold", "T_cold"])
def test_radial_integral_is_r0_times_dream_flux_label_integral(reference, quantity):
    """The physical radial integral is R0 times DREAM's own integral().

    DREAM's Grid.integrate uses the bare VpVol*dr, a flux-label integral, since
    VpVol comes from a Jacobian normalised to R/R0. The physical volume carries
    the R0 back, so our total is larger by exactly the major radius.
    """
    run, do = reference
    flux_label = np.asarray(getattr(do.eqsys, quantity).integral())[1:]
    assert reldiff(run.radial_integral(quantity), run.major_radius * flux_label) < 1e-10


def _meaningful(theirs, mine, threshold=1e-3):
    theirs = np.asarray(theirs)
    keep = np.abs(theirs) > np.abs(theirs).max() * threshold
    return np.max(np.abs(mine[keep] - theirs[keep]) / np.abs(theirs[keep]))


def test_density_moment_integrates_to_roughly_n_re(reference):
    """The density spectrum integrated over dp is the runaway density.

    This is exact in theory (theory.tex line 501) but only approximate in a
    running simulation: n_re is evolved as its own fluid unknown and drifts from
    the kinetic integral by up to tens of percent, most at early times and the
    edge. The loose bound here catches a wrong Jacobian or volume weight, which
    would miss by orders of magnitude, without over-claiming exactness. The tight
    validation of the moment is against DREAM's own angleAveraged, above.
    """
    run, do = reference
    dp = run.reference.read("re_dp")
    spectrum = run.angle_average("runaway", "density")
    integrated = np.tensordot(spectrum, dp, axes=([-1], [0]))
    assert _meaningful(run.field("n_re"), integrated) < 0.5


def test_energy_space_current_recovers_j_re(reference):
    """The current moment transformed to energy and integrated over dE gives j_re,
    up to the midpoint discretisation. Guards against the alternate_current bug of
    weighting the density spectrum by speed."""
    from cdo.energy import EnergyGrid

    run, do = reference
    eg = EnergyGrid.from_run(run)
    dj_dE = eg.integrate(run.angle_average("runaway", "current"))
    assert _meaningful(run.field("j_re"), dj_dE) < 2e-3
