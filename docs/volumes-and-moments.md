# Volume and moment conventions

Reference for the two volume-like quantities in DREAM output and how densities, currents, and distribution moments are formed from them. The values below are verified against DREAM's own `DREAMOutput`, the theory notes at `DREAM-runs/commit-53d8afb/doc/notes/theory.tex`, and the kernel. All momenta are normalised to `m_e c`, so `gamma = sqrt(1 + p^2)`.

## Real-space cell volume

The physical volume of a radial cell is `VpVol * dr * R0`, in cubic metres.

`VpVol` is stored from a Jacobian normalised to `R/R0` rather than `R`. In the kernel, `AnalyticBRadialGridGenerator::JacobianAtTheta` returns `r * (R/R0) * normalizedJacobian`, and `normalizedJacobian` is normalised to `r*R`. `VpVol` is therefore the physical spatial Jacobian divided by the major radius, and the physical volume element carries one factor of `R0` back.

The magnitude confirms this. Summed over the analytic-geometry reference file, `VpVol * dr * R0` gives 2361 cubic metres, matching the elongation-corrected plasma volume `2 pi^2 R0 a^2 <kappa>` of about 2357 cubic metres for its mean elongation near 1.53. On the ITER case it gives 790 cubic metres, close to the known ITER plasma volume. The bare `VpVol * dr` gives values several times too small for a machine of that size, because it omits the `R0`.

Use `cdo.derived.cell_volumes(VpVol, dr, R0)` or the `Run.cell_volumes` property. A radial total of a density is `cdo.derived.radial_integral(density, cell_volumes)` or `Run.radial_integral(name)`.

## The flux-label integral in DREAM

DREAM's own volume integral is not the physical volume integral. `Grid.integrate` and `FluidQuantity.integral` in the DREAM Python package both reduce to `sum(VpVol * dr * data)`, using the bare `VpVol` without the `R0`. This is a flux-label integral, smaller than the physical integral by exactly `R0`.

A physical radial total from this package therefore equals `R0` times `DREAMOutput`'s `integral()`. This relation is checked in `tests/test_dream_crosscheck.py`. When comparing a total against a number produced by `DREAMOutput.integral()`, account for the `R0` factor.

## Phase-space volume and distribution moments

The distribution functions `f_hot` and `f_re` live on a `(radius, pitch, momentum)` grid. Their moments use the momentum-space Jacobian `Vprime`, stored per grid as `grid/<name>/Vprime` with shape `(radius, pitch, momentum)`. The combination `Vprime / VpVol` is the weight that turns a distribution into a density spectrum.

Three moments are available through `Run.angle_average(grid, moment)`, each summing over the pitch axis.

- `distribution`: the pitch mean `sum(f / 2 * dxi)`, an average over pitch cosine from minus one to one.
- `density`: `sum(f * Vprime/VpVol * dxi)`, the density spectrum `dn/dp` per radius. Integrated over `dp` it gives the runaway or hot-electron density.
- `current`: the parallel-current spectrum `dj/dp` per radius. Integrated over `dp` it gives the current density.

The density spectrum integrates over momentum to the fluid density `n_re` exactly in theory (theory.tex, the runaway distribution normalisation), and to within tens of percent in a running simulation. The gap is expected, since `n_re` is evolved as its own fluid unknown and drifts from the kinetic integral, most at early times and the plasma edge.

## Current from the distribution

The parallel current is `j = integral of e v xi f` over momentum, weighted by the pitch `xi`. The current moment above uses the bounce-averaged parallel velocity, so trapped pitch cells, those straddling the trapped-passing boundary at `grid/geometry/xi0TrappedBoundary`, contribute nothing. The `sigma` sum over co- and counter-passing electrons cancels them. Integrated over momentum, the current moment matches `eqsys/j_re` to nine significant figures.

Do not form a current by multiplying the density spectrum by speed and integrating, `integral of e v (dn/dp) dp`. That expression drops the pitch weighting and treats every electron as fully parallel, overcounting by about a factor of twelve on the reference file. DREAM's simpler `currentDensity`, which ignores trapping, misses `eqsys/j_re` by about half and is also not used.

A current density from a fluid current field, for example `j_re`, is integrated to a total current with `Run.current("j_re")`. This uses the flux-surface weight `VpVol * dr * GR0/Bmin * FSA_R02OverR2` and a factor of `1/(2 pi)`, reproducing DREAM's `current()` and giving the physical current in amperes. No `R0` enters here, since the geometry weight and the `2 pi` factor already carry it.

## Momentum to energy

Total electron energy is `E = gamma m_e c^2`. Kinetic energy is `E - m_e c^2`. The runaway momentum grid is uniform in normalised momentum, not in energy, so a spectrum in energy needs the change of variable `dn/dE = (dn/dp) dp/dE`.

The Jacobian is `dp/dE = gamma / (m_e c^2 p)`. In energy variables this is `E / (m_e c^2 sqrt(E^2 - (m_e c^2)^2))`, which is identical since `sqrt(E^2 - (m_e c^2)^2) = m_e c^2 p` for momentum normalised to `m_e c`. The Jacobian does not depend on which moment it transforms, so it applies to the density spectrum and the current spectrum alike.

`cdo.energy.EnergyGrid` holds the energy grid and this transform. `to_energy` applies the Jacobian to any per-momentum moment, and `integrate` sums a per-momentum moment over the energy grid. Integrating over energy recovers the momentum integral up to the midpoint error between spectra evaluated at cell centres and energy widths taken at cell edges, which falls with grid resolution.
