"""Shared configuration for the shape-vs-intensity wavefront experiments.

Every physical constant and experiment knob lives here so the study scripts
read as plain top-to-bottom recipes.  All wavefront quantities are in meters
of optical path (danish's native unit); we convert to waves only for display.

The optical model is a deliberately simple toy: a single Rubin-sized
paraboloid, defocused, observed at the center of the field of view.  At field
center a paraboloid is aberration-free apart from defocus, so the reference
wavefront is pure defocus (Noll Z4).  We then inject random Zernike
perturbations, simulate a donut, and fit it back -- the whole experiment.
"""

from pathlib import Path

import numpy as np

# Where study scripts write their figures (repo-root ``figures/``).
FIGDIR = Path(__file__).resolve().parent.parent / "figures"

# Where study scripts cache their simulation outputs (repo-root ``data/``).
# Each study saves its Monte-Carlo results here so the plots can be re-drawn
# with ``--plot-only`` without re-running the (slow) simulations.
DATADIR = Path(__file__).resolve().parent.parent / "data"

# --- Telescope geometry (Rubin-like) --------------------------------------
# Entrance-pupil outer radius and the nominal central obscuration.
R_OUTER = 4.18  # meters (Rubin primary mirror rim)
EPS_RUBIN = 0.61  # Rubin central-obscuration fraction
R_INNER = EPS_RUBIN * R_OUTER
FOCAL_LENGTH = 10.31  # meters
PIXEL_SCALE = 10e-6  # meters per pixel (10 micron)

# --- Image + rendering ----------------------------------------------------
NPIX = 181  # odd; Rubin-nominal donut is ~133 px across
NRAD = 35  # radial rings in the triangle mesh (accuracy vs speed)
DEFOCUS_Z4 = 24.3e-6  # meters of Z4 at EPS_RUBIN; sets the donut size
FWHM = 0.7  # atmospheric seeing FWHM, arcsec

# --- Reference wavelength (for reporting only) ----------------------------
WAVELENGTH = 750e-9  # meters (r-band-ish); used to quote errors in waves

# --- Noise --------------------------------------------------------------
# High flux + tiny sky => noise is negligible for the fit.  It exists only to
# keep the optimizer well behaved; the Monte-Carlo spread we plot comes from
# drawing different true perturbations, not from photon noise.
FLUX = 1e7  # total donut flux (photons); used by fixed-total-flux mode
SKY_LEVEL = 10.0  # sky variance per pixel

# Target mean counts per illuminated pixel, used when flux is normalized to hold
# per-pixel signal-to-noise constant across pupil geometries (the obscuration and
# vignetting sweeps).  Removing pupil area shrinks the donut footprint; at fixed
# total flux that would raise the per-pixel SNR and artificially lower every
# mode's error.  Fixing counts-per-pixel instead isolates the genuine
# information change from that trivial brightness effect.
COUNTS_PER_PIX = 2000.0

# --- Monte-Carlo perturbations -------------------------------------------
# Each fitted Zernike mode is drawn from N(0, INJECT_SIGMA).  The same sigma
# is used for every mode so residuals are directly comparable across modes,
# and the "relative error" we plot is residual / INJECT_SIGMA.
INJECT_SIGMA = 50e-9  # meters RMS per mode (~0.067 waves at 750 nm)
N_MC = 50  # realizations per condition (raise for final figures)
SEED = 20260704

# --- Uncertainty-aware heatmap display -----------------------------------
# All-mode heatmaps color the physical effect size, log10(error / fiducial).
# Cells are muted unless the bootstrap confidence interval excludes zero and
# the effect is at least this large.  A ratio of 1.20 means "show only changes
# of roughly 20% or more" in either direction on the log scale.
HEATMAP_BOOTSTRAP_CONFIDENCE = 0.95
HEATMAP_MIN_ERROR_RATIO = 1.20
HEATMAP_BOOTSTRAP_SAMPLES = 1000
HEATMAP_BOOTSTRAP_SEED = SEED + 1

# Line-plot error bars show bootstrap intervals on error / fiducial error.
# A 68% interval is close to a 1-sigma visual error bar and keeps the figure
# readable; raise to 0.95 for publication-style confidence intervals.
LINE_PLOT_BOOTSTRAP_CONFIDENCE = 0.68
LINE_PLOT_BOOTSTRAP_SAMPLES = 1000
LINE_PLOT_BOOTSTRAP_SEED = SEED + 2

# --- Highest Noll index we ever touch (defines z_ref padding length) ------
JMAX = 22

# --- Zernike mode sets for the sparsity study -----------------------------
# Dense: every coefficient Noll 4..22.
DENSE_TERMS = list(range(4, JMAX + 1))

# Sparse: the lowest-nu detectable mode for each azimuthal order m, per the
# paper's annular-Zernike classification.  Defocus (Z4) is always included; for
# m=0 and m=1 the nu=0 modes (piston/tilt) are undetectable so we take the nu=1
# modes (primary spherical Z11, coma Z7/Z8); for m>=2 we take the nu=0 m-foils.
#   m=1 coma        Z7, Z8
#   m=2 astigmatism Z5, Z6
#   m=3 trefoil     Z9, Z10
#   m=0 spherical   Z11
#   m=4 tetrafoil   Z14, Z15
#   m=5 pentafoil   Z20, Z21
SPARSE_TERMS = [4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 20, 21]

# --- Modes highlighted in the obscuration / vignetting studies ------------
COMA_TERMS = [7, 8, 16, 17]  # primary (7,8) and secondary (16,17) coma
SPHERICAL_TERMS = [11, 22]  # primary (11) and secondary (22) spherical


def waves(x_meters):
    """Convert a wavefront quantity from meters to waves at ``WAVELENGTH``.

    Parameters
    ----------
    x_meters : array_like
        Wavefront value(s) in meters.

    Returns
    -------
    ndarray
        The same value(s) expressed in waves.
    """
    return np.asarray(x_meters) / WAVELENGTH
