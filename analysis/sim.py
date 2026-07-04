"""Core simulate-and-fit engine shared by every study.

We always use danish's *triangle* renderer (``DonutTriangleFactory``).  It is a
forward mesh projection with no focal-to-pupil inversion, so it stays stable
even near caustics, and it is API-compatible with the fitter.  The same factory
generates the "truth" donut and serves as the fit model, so truth and model use
one identical forward operator -- exactly the comparison the paper makes.

The single knob that separates the two experiments is ``surface_brightness``:

* ``True``  -> full shape + intensity information (physical surface brightness).
* ``False`` -> shape only (flat-topped donut, edge pixels weighted by coverage).

Both truth and model use the *same* value, so a shape-only run answers
"what can donut shape alone determine?".
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import least_squares
import danish

import config as C


def default_jobs():
    """Suggested worker count: use the performance cores, spare the rest.

    On this Apple M1 Max (8 performance + 2 efficiency cores) this returns 8.
    """
    return max(1, (os.cpu_count() or 1) - 2)


def make_factory(
    *,
    surface_brightness,
    zk_r_inner=C.R_INNER,
    pupil_r_outer=C.R_OUTER,
    pupil_r_inner=None,
    nrad=C.NRAD,
):
    """Build a triangle donut factory for the toy Rubin paraboloid.

    Parameters
    ----------
    surface_brightness : bool
        If True, render full shape+intensity donuts; if False, shape-only.
    zk_r_inner : float, optional
        Inner radius of the annular-Zernike normalization, in meters.  Equal to
        the physical obscuration for the obscuration study; held fixed at the
        Rubin value for the vignetting study.
    pupil_r_outer : float, optional
        Outer edge of the illuminated pupil, in meters.  Shrinking this below
        ``R_OUTER`` (while keeping the Zernike normalization fixed) is how the
        vignetting study removes outer-pupil light.
    pupil_r_inner : float, optional
        Inner edge of the illuminated pupil, in meters.  Defaults to
        ``zk_r_inner`` (the central obscuration).
    nrad : int, optional
        Number of radial rings in the mesh.

    Returns
    -------
    danish.DonutTriangleFactory
        Configured factory.
    """
    if pupil_r_inner is None:
        pupil_r_inner = zk_r_inner
    return danish.DonutTriangleFactory(
        R_outer=C.R_OUTER,          # Zernike normalization: full aperture
        R_inner=zk_r_inner,
        pupil_R_outer=pupil_r_outer,
        pupil_R_inner=pupil_r_inner,
        focal_length=C.FOCAL_LENGTH,
        pixel_scale=C.PIXEL_SCALE,
        nrad=nrad,
        surface_brightness=surface_brightness,
    )


def make_reference(defocus=C.DEFOCUS_Z4):
    """Return the reference Zernike vector: pure defocus, padded to ``JMAX``.

    Parameters
    ----------
    defocus : float, optional
        Z4 coefficient in meters.

    Returns
    -------
    ndarray
        Reference coefficients, length ``JMAX + 1`` (Noll-indexed, [0] unused).
    """
    z_ref = np.zeros(C.JMAX + 1)
    z_ref[4] = defocus
    return z_ref


def fit_one(z_true, z_terms, factory, *, z_ref=None, seed=0, return_images=False):
    """Simulate one donut with known aberrations and fit them back.

    Parameters
    ----------
    z_true : array_like
        True perturbation for each mode in ``z_terms`` (meters), same length
        and order as ``z_terms``.
    z_terms : sequence of int
        Noll indices to fit.
    factory : danish.DonutTriangleFactory
        Renderer used for both the truth image and the fit model.
    z_ref : ndarray, optional
        Reference Zernikes (defaults to pure defocus).
    seed : int, optional
        Seed for the (tiny) photon noise added to the truth image.
    return_images : bool, optional
        If True, also return the truth image and best-fit model image.

    Returns
    -------
    residual : ndarray
        ``z_fit - z_true`` for each fitted mode, in meters.
    z_fit : ndarray
        Best-fit perturbations, in meters.
    images : tuple of ndarray, optional
        ``(truth, model)`` images, only if ``return_images`` is True.
    """
    z_true = np.asarray(z_true, dtype=float)
    if z_ref is None:
        z_ref = make_reference()

    model = danish.SingleDonutModel(
        factory, z_ref=z_ref, z_terms=list(z_terms),
        thx=0.0, thy=0.0, bkg_order=-1, seed=seed,
    )

    # Equalize total flux across rendering modes.  The shape-only and full
    # renderers normalize to different total sums, so we rescale the flux so
    # that every truth donut carries the same total counts (C.FLUX).  This keeps
    # the effective signal-to-noise identical between shape-only and full fits,
    # which is essential for a fair comparison.
    base = model.model(1.0, 0.0, 0.0, C.FWHM, z_true, sky_level=None)
    flux = C.FLUX / base.sum()

    # Truth image: reference + injected perturbations, with tiny photon noise.
    truth = model.model(flux, 0.0, 0.0, C.FWHM, z_true, sky_level=C.SKY_LEVEL)

    guess = [truth.sum(), 0.0, 0.0, C.FWHM] + [0.0] * len(z_terms)
    # The optimizer may briefly probe a negative flux, making the model image
    # negative and the sqrt-based error undefined; least_squares rejects those
    # steps, so we quietly ignore the transient warning.
    with np.errstate(invalid="ignore"):
        result = least_squares(
            model.chi, guess, jac=model.jac,
            args=(truth, C.SKY_LEVEL),
            x_scale="jac", ftol=1e-5, xtol=1e-5, gtol=1e-5, max_nfev=100,
        )
    unpacked = model.unpack_params(result.x)
    z_fit = np.asarray(unpacked["z_fit"], dtype=float)
    residual = z_fit - z_true

    if return_images:
        model_img = model.model(**unpacked)
        return residual, z_fit, (truth, model_img)
    return residual, z_fit


def _mc_chunk(payload):
    """Run a contiguous chunk of Monte-Carlo trials (one worker's share).

    Builds the factory once from ``factory_kwargs`` so the triangle mesh is
    cached and reused across this chunk's trials, then fits each pre-drawn
    perturbation.  Defined at module level so it is picklable for
    multiprocessing.
    """
    factory_kwargs, z_terms, z_trues, seeds = payload
    factory = make_factory(**factory_kwargs)
    z_ref = make_reference()
    out = np.empty((len(z_trues), len(z_terms)))
    for i, (z_true, sd) in enumerate(zip(z_trues, seeds)):
        out[i], _ = fit_one(z_true, z_terms, factory, z_ref=z_ref, seed=int(sd))
    return out


def monte_carlo(z_terms, factory_kwargs, *, n_mc=C.N_MC, seed=C.SEED,
                inject_sigma=C.INJECT_SIGMA, n_jobs=1):
    """Run many random-perturbation trials and collect the fit residuals.

    Each trial draws an independent perturbation ``N(0, inject_sigma)`` for
    every mode in ``z_terms``, simulates the donut, and fits it back.  All
    perturbations and noise seeds are drawn up front from ``seed``, so results
    are identical regardless of ``n_jobs`` (parallelism does not change the
    answer, only the wall-clock time).

    Parameters
    ----------
    z_terms : sequence of int
        Noll indices to inject and fit.
    factory_kwargs : dict
        Keyword arguments for :func:`make_factory`.  Passed (not the factory
        object) so each worker process can build its own factory and mesh.
    n_mc : int, optional
        Number of realizations.
    seed : int, optional
        Base random seed.
    inject_sigma : float, optional
        Per-mode injection sigma, in meters.
    n_jobs : int, optional
        Number of worker processes.  1 (default) runs serially.

    Returns
    -------
    residuals : ndarray, shape (n_mc, len(z_terms))
        ``z_fit - z_true`` for each trial and mode, in meters.
    """
    z_terms = list(z_terms)
    rng = np.random.default_rng(seed)
    z_trues = rng.normal(0.0, inject_sigma, size=(n_mc, len(z_terms)))
    seeds = rng.integers(1 << 30, size=n_mc)

    if n_jobs <= 1:
        return _mc_chunk((factory_kwargs, z_terms, z_trues, seeds))

    # Split the trials into contiguous chunks, one per worker.  Contiguous
    # chunks + in-order concatenation preserve the trial ordering.
    n_jobs = min(n_jobs, n_mc)
    chunks = np.array_split(np.arange(n_mc), n_jobs)
    payloads = [(factory_kwargs, z_terms, z_trues[c], seeds[c]) for c in chunks]
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        results = list(ex.map(_mc_chunk, payloads))
    return np.concatenate(results, axis=0)
