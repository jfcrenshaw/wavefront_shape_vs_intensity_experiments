# Shape-vs-intensity wavefront experiments

Code to reproduce the analyses in the *"Investigations with Algorithmic
Wavefront Estimation"* section of the shape-vs-intensity paper, using the
[danish](https://github.com/jmeyers314/danish) geometric donut engine for both
simulation and fitting.

The guiding idea: a defocused donut carries wavefront information in two places
— its **shape** (outline) and its **intensity** (surface-brightness pattern).
By fitting the same donuts with a normal danish model and with a *shape-only*
model, we measure how much of each Zernike mode lives in shape vs intensity.

## What's here

```text
analysis/
  config.py            all physical constants and experiment knobs
  sim.py               the simulate-one-donut-and-fit-it-back engine
  plotutils.py         shared plotting helpers (all-terms heatmap)
  plot_donuts.py       -> figures/example_donuts.png (gallery of simulated donuts)
  study_sparsity.py    -> figures/zk_estimates.pdf, shape_degeneracy.pdf
  study_obscuration.py -> figures/coma_vs_obscuration.png, spherical_vs_obscuration.png,
                          all_terms_vs_obscuration.png
  study_vignetting.py  -> figures/vignetting.png, all_terms_vs_vignetting.png  (new study)
danish/                danish as a git submodule, with our shape-only patch
data/                  cached simulation outputs (*.npz; see "Running" below)
figures/               output figures
```

Everything uses one deliberately simple toy optical model: a single Rubin-sized
paraboloid, defocused, observed at the **center** of the field of view.  At
field center a paraboloid is aberration-free apart from defocus, so the
reference wavefront is pure defocus (Noll Z4); we inject random Zernike
perturbations, simulate a donut, and fit them back.  No `batoid` needed.

## The danish patch (shape-only rendering)

We use danish **triangle mode** (`DonutTriangleFactory`) throughout: it is a
forward mesh projection with no focal-to-pupil inversion, so it stays stable
near caustics, and it is API-compatible with danish's fitter.

danish does not ship a shape-only renderer, so we added a `surface_brightness`
flag (default `True`) to the factories.  When `False`, every fully illuminated
pixel gets the same value and edge pixels are weighted by the fraction of the
pixel covered by the pupil — a flat-topped donut with all intensity structure
removed.  In triangle mode this is achieved by depositing each triangle's
*projected* area (so flux/area = 1) instead of its pupil area.

The patch lives on the `shape-only-patch` branch of the `danish/` submodule
(one commit, ~10 lines in `danish/factory.py`).  The submodule is pinned to
danish release `v1.2.0` (commit `a4b680a`) plus that patch.

## Setup

```bash
git clone --recurse-submodules <this repo>
mamba activate shape_vs_intensity          # env with numpy/scipy/matplotlib/galsim
pip install -e ./danish --no-deps          # builds the C++ extension; batoid not needed
```

## Running

Run from the `analysis/` directory:

```bash
cd analysis
python study_sparsity.py      # add --quick for a fast, low-statistics preview
python study_obscuration.py
python study_vignetting.py
python plot_donuts.py         # instant; renders the donut gallery
```

Simulation and plotting are separated so you can tweak a figure without paying
for the Monte-Carlo again.  Each study script (`study_sparsity.py`,
`study_obscuration.py`, `study_vignetting.py`) runs the simulation, caches the
results to `data/<study>.npz`, and then draws the figures.  Once that cache
exists, re-run with `--plot-only` to skip straight to the plotting:

```bash
python study_obscuration.py             # simulate, cache to data/, then plot
python study_obscuration.py --plot-only # instant: re-draw from data/obscuration.npz
```

(`plot_donuts.py` renders directly and needs no cache — it does no Monte-Carlo.)

Each study script accepts:

- `--plot-only` -- skip the simulation and re-draw the figures from the cached
  `data/<study>.npz` (errors out if the cache does not exist yet).
- `--n-mc N` -- Monte-Carlo realizations per condition (raise for
  publication-quality figures; the `config.py` default is a compromise).
- `--jobs J` -- number of worker processes.  The Monte-Carlo trials are
  embarrassingly parallel, so this scales nearly linearly.  The default is the
  number of performance cores (8 on an Apple M1 Max); pass `--jobs 1` to force
  serial.  Results are identical regardless of `--jobs` (the random draws are
  fixed up front), so parallelism only changes wall-clock time.
- `--quick` -- fast, low-statistics preview.

A full run at `--n-mc 200 --jobs 8` takes a few minutes per study.

Because multiprocessing uses the "spawn" start method on macOS, always run these
as scripts (they guard the entry point with `if __name__ == "__main__"`); do not
call `monte_carlo(..., n_jobs>1)` from an unguarded top-level context.

## Key methodological choices (see `config.py`)

- **Fair comparison.** Shape-only and full renderers normalize to different
  total flux, so `sim.fit_one` rescales the flux to a fixed total (`FLUX`).
  This keeps the effective signal-to-noise identical between the two modes — the
  comparison is about *information*, not photon budget.
- **Noise.** High flux + tiny sky, so photon noise is negligible; the
  Monte-Carlo spread we plot comes from drawing different true perturbations,
  not from noise. The small noise only keeps the optimizer well behaved.
- **Injection.** Every fitted mode is drawn `N(0, INJECT_SIGMA)` with the same
  sigma, so residuals are comparable across modes and the plotted "relative
  error" is `residual / INJECT_SIGMA`.
- **Sparse mode set.** `SPARSE_TERMS` in `config.py` lists the lowest-nu
  detectable mode per azimuthal order. **This list should be confirmed against
  the nu-classification tables in Sections 3-5 of the paper.**
