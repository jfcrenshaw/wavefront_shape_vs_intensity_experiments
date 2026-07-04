"""Core simulate-and-fit engine shared by every study.

We always use danish's *triangle* renderer (``DonutTriangleFactory``).  It is a
forward mesh projection with no focal-to-pupil inversion, so it stays stable
even near caustics, and it is API-compatible with the fitter.  The same factory
generates the "truth" donut and serves as the fit model, so truth and model use
one identical forward operator -- exactly the comparison the paper makes.

The shape-vs-intensity comparison is controlled by ``surface_brightness``:

* ``True``  -> full shape + intensity information (physical surface brightness).
* ``False`` -> shape only (flat-topped donut, edge pixels weighted by coverage).

Both truth and model use the *same* value, so a shape-only run answers
"what can donut shape alone determine?".
"""

import os
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import least_squares
import danish

from shape_vs_intensity import config as C


class VignettedTriangleFactory(danish.DonutTriangleFactory):
    """Triangle factory with an extra straight-edge vignetting cut.

    Real vignetting is *asymmetric*: it eats into the pupil from one side and
    leaves a nearly straight edge, not a uniform shrink of the outer radius.
    We keep the half-plane ``u <= vignette_x_edge`` illuminated and remove the
    rest of the pupil.

    The Zernike normalization and nominal pupil radii are left untouched (the
    outer wavefront still physically exists and sets the flux normalization);
    only illuminated triangles are removed.

    Parameters
    ----------
    vignette_x_edge : float
        Location of the vertical vignetting edge in pupil ``u`` coordinates,
        in meters.  Points with ``u > vignette_x_edge`` are removed.
    **kwargs
        Forwarded to :class:`danish.DonutTriangleFactory`.
    """

    def __init__(self, *, vignette_x_edge, **kwargs):
        super().__init__(**kwargs)
        self._vig_x_edge = float(vignette_x_edge)
        self._vig_cache = OrderedDict()

    def _get_mesh(self, thx, thy):
        """Return the annulus mesh with the vignetting cut applied, cached.

        The base annulus mesh (and any obscuration clipping) is built by the
        parent; here we additionally clip it against the vignetting edge and
        cache the result per field angle, mirroring the parent's own cache.
        """
        base = super()._get_mesh(thx, thy)
        key = (round(thx, 6), round(thy, 6))
        if key not in self._vig_cache:
            if len(self._vig_cache) >= 10:
                self._vig_cache.popitem(last=False)
            self._vig_cache[key] = self._clip_to_vignette(base)
        return self._vig_cache[key]

    def _clip_to_vignette(self, mesh):
        """Clip a triangle mesh to the illuminated half-plane ``u <= x_edge``."""
        x_edge = self._vig_x_edge
        tri = mesh["vertices"][mesh["triangles"]]  # (M, 3, 2)
        u = tri[..., 0]

        keep_full = tri[u.max(axis=1) <= x_edge]
        band = tri[(u.min(axis=1) <= x_edge) & (u.max(axis=1) > x_edge)]
        triangles = list(keep_full)
        for t in band:
            triangles.extend(self._clip_triangle_to_vignette(t, x_edge))
        return self._mesh_from_triangles(triangles)

    @staticmethod
    def _clip_triangle_to_vignette(tri, x_edge):
        """Clip one triangle to ``u <= x_edge`` and fan-triangulate the result."""
        out = []
        for i in range(3):
            s = tri[i]
            p = tri[(i + 1) % 3]
            s_in = s[0] <= x_edge
            p_in = p[0] <= x_edge

            if s_in and p_in:
                out.append(p)
            elif s_in and not p_in:
                t = (x_edge - s[0]) / (p[0] - s[0])
                out.append(s + t * (p - s))
            elif not s_in and p_in:
                t = (x_edge - s[0]) / (p[0] - s[0])
                out.append(s + t * (p - s))
                out.append(p)

        if len(out) < 3:
            return []

        clipped = []
        for i in range(1, len(out) - 1):
            t = np.array([out[0], out[i], out[i + 1]], dtype=float)
            area = 0.5 * abs(
                (t[1, 0] - t[0, 0]) * (t[2, 1] - t[0, 1])
                - (t[1, 1] - t[0, 1]) * (t[2, 0] - t[0, 0])
            )
            if area > 1e-30:
                clipped.append(t)
        return clipped


def default_jobs():
    """Suggested worker count: use the performance cores, spare the rest.

    On this Apple M1 Max (8 performance + 2 efficiency cores) this returns 8.
    """
    return max(1, (os.cpu_count() or 1) - 2)


def make_factory(
    *,
    surface_brightness,
    zk_r_inner=C.R_INNER,
    pupil_r_inner=None,
    nrad=C.NRAD,
    vignette_x_edge=None,
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
    pupil_r_inner : float, optional
        Inner edge of the illuminated pupil, in meters.  Defaults to
        ``zk_r_inner`` (the central obscuration).
    nrad : int, optional
        Number of radial rings in the mesh.
    vignette_x_edge : float, optional
        If given, an additional straight-edge cut is applied to model
        *asymmetric* vignetting: pixels with ``u > vignette_x_edge`` are
        removed (see :class:`VignettedTriangleFactory`).  Used by the
        vignetting study.

    Returns
    -------
    danish.DonutTriangleFactory
        Configured factory.
    """
    if pupil_r_inner is None:
        pupil_r_inner = zk_r_inner
    kwargs = dict(
        R_outer=C.R_OUTER,  # Zernike normalization: full aperture
        R_inner=zk_r_inner,
        pupil_R_inner=pupil_r_inner,
        focal_length=C.FOCAL_LENGTH,
        pixel_scale=C.PIXEL_SCALE,
        nrad=nrad,
        surface_brightness=surface_brightness,
    )
    if vignette_x_edge is not None:
        return VignettedTriangleFactory(
            vignette_x_edge=vignette_x_edge,
            **kwargs,
        )
    return danish.DonutTriangleFactory(**kwargs)


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


def fixed_diameter_defocus(
    zk_r_inner,
    *,
    reference_r_inner=C.R_INNER,
    reference_defocus=C.DEFOCUS_Z4,
    r_outer=C.R_OUTER,
):
    """Return the Z4 coefficient that keeps the outer donut diameter fixed.

    The annular-Zernike Z4 slope is proportional to
    ``defocus / (1 - epsilon**2)``.  When the Zernike inner radius changes, a
    fixed numerical Z4 coefficient therefore changes the physical defocus
    slope and hence the donut diameter.  Scaling by the annulus area factor
    keeps the outer-edge slope fixed relative to the reference annulus.
    """
    eps = float(zk_r_inner) / r_outer
    eps_ref = float(reference_r_inner) / r_outer
    if not (0.0 <= eps < 1.0):
        raise ValueError("zk_r_inner must satisfy 0 <= zk_r_inner < r_outer")
    if not (0.0 <= eps_ref < 1.0):
        raise ValueError(
            "reference_r_inner must satisfy 0 <= reference_r_inner < r_outer"
        )
    return reference_defocus * (1.0 - eps**2) / (1.0 - eps_ref**2)


def vignette_fraction(x_edge, n_rho=400, n_theta=1440):
    """Return the fraction of the annular pupil removed by a straight-edge cut.

    The vignetted region is ``u > x_edge``.  The integral is a simple polar
    quadrature over the nominal Rubin annulus; the ``rho`` weight is the area
    element, and constant factors cancel in the ratio.
    """
    rho = np.linspace(C.R_INNER, C.R_OUTER, n_rho)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    rr, th = np.meshgrid(rho, theta, indexing="ij")
    u = rr * np.cos(th)
    outside = u > x_edge
    return float((outside * rr).sum() / rr.sum())


def fit_one(
    z_true,
    z_terms,
    factory,
    *,
    z_ref=None,
    seed=0,
    flux_norm="total",
    return_images=False,
):
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
        factory,
        z_ref=z_ref,
        z_terms=list(z_terms),
        thx=0.0,
        thy=0.0,
        bkg_order=-1,
        seed=seed,
    )

    # Render a unit-flux donut so we can choose the flux scale explicitly.
    base = model.model(1.0, 0.0, 0.0, C.FWHM, z_true, sky_level=None)
    if flux_norm == "total":
        # Fixed total photons.  Both rendering modes carry the same total counts
        # (C.FLUX), keeping shape-only vs full comparable at fixed geometry.
        flux = C.FLUX / base.sum()
    elif flux_norm == "per_pixel":
        # Fixed mean counts per illuminated pixel, so the per-pixel SNR is the
        # same regardless of how much pupil (and thus donut footprint) is left.
        # a_eff is the participation-ratio pixel count (= geometric footprint
        # for a flat-topped donut), a threshold-free measure of donut area.
        a_eff = base.sum() ** 2 / np.square(base).sum()
        flux = C.COUNTS_PER_PIX * a_eff / base.sum()
    else:
        raise ValueError(f"unknown flux_norm {flux_norm!r}")

    # Truth image: reference + injected perturbations, with tiny photon noise.
    truth = model.model(flux, 0.0, 0.0, C.FWHM, z_true, sky_level=C.SKY_LEVEL)

    guess = [truth.sum(), 0.0, 0.0, C.FWHM] + [0.0] * len(z_terms)
    # The optimizer may briefly probe a negative flux, making the model image
    # negative and the sqrt-based error undefined; least_squares rejects those
    # steps, so we quietly ignore the transient warning.
    with np.errstate(invalid="ignore"):
        result = least_squares(
            model.chi,
            guess,
            jac=model.jac,
            args=(truth, C.SKY_LEVEL),
            x_scale="jac",
            ftol=1e-5,
            xtol=1e-5,
            gtol=1e-5,
            max_nfev=100,
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
    factory_kwargs, z_terms, z_trues, seeds, flux_norm, z_ref = payload
    factory = make_factory(**factory_kwargs)
    if z_ref is None:
        z_ref = make_reference()
    out = np.empty((len(z_trues), len(z_terms)))
    for i, (z_true, sd) in enumerate(zip(z_trues, seeds)):
        out[i], _ = fit_one(
            z_true,
            z_terms,
            factory,
            z_ref=z_ref,
            seed=int(sd),
            flux_norm=flux_norm,
        )
    return out


def monte_carlo(
    z_terms,
    factory_kwargs,
    *,
    n_mc=C.N_MC,
    seed=C.SEED,
    inject_sigma=C.INJECT_SIGMA,
    n_jobs=1,
    flux_norm="total",
    z_ref=None,
):
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
    flux_norm : {"total", "per_pixel"}, optional
        Flux normalization (see :func:`fit_one`).  Use "total" at fixed geometry
        (e.g. the sparsity study) and "per_pixel" when sweeping pupil geometry
        (obscuration, vignetting) so per-pixel SNR stays constant.
    z_ref : ndarray, optional
        Reference Zernike vector.  Defaults to :func:`make_reference`.  The
        obscuration study passes an adjusted Z4 reference here so the simulated
        donuts keep a fixed outer diameter while the annular normalization
        radius changes.

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
        return _mc_chunk((factory_kwargs, z_terms, z_trues, seeds, flux_norm, z_ref))

    # Split the trials into contiguous chunks, one per worker.  Contiguous
    # chunks + in-order concatenation preserve the trial ordering.
    n_jobs = min(n_jobs, n_mc)
    chunks = np.array_split(np.arange(n_mc), n_jobs)
    payloads = [
        (factory_kwargs, z_terms, z_trues[c], seeds[c], flux_norm, z_ref)
        for c in chunks
    ]
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        results = list(ex.map(_mc_chunk, payloads))
    return np.concatenate(results, axis=0)


def simulate_donut(defocus=C.DEFOCUS_Z4, zernikes=None, dx=0.0, dy=0.0, seed=1):
    """Simulate one extra-focal Rubin donut with the repo's danish simulator.

    Uses the same triangle-mesh forward model as the experiments: a defocused
    toy-Rubin paraboloid observed at field center, blurred by ``C.FWHM`` seeing
    and given a little photon noise.  Extra Zernike aberrations imprint their
    characteristic shape on the donut, and a lateral shift ``(dx, dy)`` moves it
    off-centre (the image-plane signature of wavefront tilt).

    The donut is rendered *extra*-focal (negative Z4).  An extra-focal image is
    upright with respect to the pupil, so the pupil-plane schematics drawn by
    ``map_circles`` line up with the donuts directly; the intra-focal image is
    the same donut rotated 180 degrees, which would flip the odd-m modes.

    Parameters
    ----------
    defocus : float, optional
        Magnitude of the Z4 defocus, in meters (sets the donut diameter).
    zernikes : dict, optional
        Extra aberrations to add, as ``{noll_index: coefficient_in_meters}``
        (e.g. ``{5: 7e-6}`` for astigmatism).
    dx, dy : float, optional
        Centroid offset in arcseconds (the tilt signature).
    seed : int, optional
        Photon-noise seed.

    Returns
    -------
    ndarray
        The simulated donut image, shape ``(C.NPIX, C.NPIX)``.
    """
    zernikes = zernikes or {}
    jmax = max([C.JMAX, *zernikes.keys()])
    z_ref = np.zeros(jmax + 1)
    z_ref[4] = -defocus  # negative Z4 -> extra-focal, upright pupil mapping
    for noll, coeff in zernikes.items():
        z_ref[noll] += coeff

    model = danish.SingleDonutModel(
        make_factory(surface_brightness=True),
        z_ref=z_ref,
        z_terms=[],
        thx=0.0,
        thy=0.0,
        bkg_order=-1,
        seed=seed,
        npix=C.NPIX,
    )
    # Normalize to a fixed total flux, then render with seeing + sky noise.
    base = model.model(1.0, 0.0, 0.0, C.FWHM, [], sky_level=None)
    flux = C.FLUX / base.sum()
    return model.model(flux, dx, dy, C.FWHM, [], sky_level=C.SKY_LEVEL)
