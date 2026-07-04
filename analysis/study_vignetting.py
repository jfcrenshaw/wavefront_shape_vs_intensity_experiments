"""Study 3 (new): impact of vignetting on wavefront estimation.

Vignetting removes light from part of the pupil.  Crucially it is *asymmetric*:
a real vignette eats into the aperture from one side, leaving a nearly straight
edge -- not a uniform shrink of the outer radius.  We model the vignette as a
vertical half-plane cut in pupil coordinates: the region ``u <= x_edge`` is
illuminated, and the part of the pupil with larger ``u`` is removed.  As
``x_edge`` moves left, the straight edge sweeps across the pupil and removes an
asymmetric crescent.

The full-aperture annular-Zernike normalization stays fixed (the outer
wavefront still physically exists), the donut stays at the center of the field
of view, and the only thing changing is how much (and which side) of the pupil
is illuminated -- pure vignetting, isolated from field angle and from the
central obscuration.  We expect estimation of modes with power near the outer
edge (spherical, high-order modes) to degrade as vignetting increases, and the
broken azimuthal symmetry to couple modes that a symmetric cut would leave
untouched.  Errors are normalized to the no-vignetting case per mode.

The simulation and the plotting are separate: a normal run simulates, caches
the results to ``data/vignetting.npz``, and then plots; ``--plot-only`` skips
the simulation and re-draws the figures from that cache (for quick cosmetic
tweaks).

Run:  python study_vignetting.py [--n-mc N] [--quick] [--plot-only]
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

import config as C
import sim
import plotutils


def sweep(x_edges, n_mc, seed, n_jobs):
    """Return per-mode 1-sigma error at each vignette cut position (dense fit).

    Parameters
    ----------
    x_edges : ndarray
        Straight-edge cut positions to sweep, in meters.
    n_mc : int
        Monte-Carlo realizations per position.
    seed : int
        Base random seed (shared across positions).
    n_jobs : int
        Number of worker processes.

    Returns
    -------
    sig : ndarray, shape (len(x_edges), len(DENSE_TERMS))
        Standard deviation of ``(z_fit - z_true) / INJECT_SIGMA`` per mode.
    residuals : ndarray, shape (len(x_edges), n_mc, len(DENSE_TERMS))
        Raw Monte-Carlo residuals, cached so heatmaps can mask noisy cells.
    """
    sig = np.empty((len(x_edges), len(C.DENSE_TERMS)))
    residuals = np.empty((len(x_edges), n_mc, len(C.DENSE_TERMS)))
    for i, xe in enumerate(x_edges):
        # Zernike normalization and pupil radii stay on the full aperture; only
        # the illuminated triangles left of the straight vignette edge survive.
        kwargs = dict(surface_brightness=True, zk_r_inner=C.R_INNER,
                      vignette_x_edge=xe)
        res = sim.monte_carlo(C.DENSE_TERMS, kwargs, n_mc=n_mc, seed=seed,
                              n_jobs=n_jobs, flux_norm="per_pixel")
        residuals[i] = res
        sig[i] = res.std(axis=0) / C.INJECT_SIGMA
        print(f"  vignette fraction={sim.vignette_fraction(xe):.2f} done")
    return sig, residuals


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-mc", type=int, default=C.N_MC)
    p.add_argument("--jobs", type=int, default=sim.default_jobs(),
                   help="worker processes (default: performance-core count)")
    p.add_argument("--quick", action="store_true", help="fast, low-stats run")
    p.add_argument("--plot-only", action="store_true",
                   help="skip the simulation; re-draw plots from saved data")
    args = p.parse_args()

    C.FIGDIR.mkdir(exist_ok=True)
    C.DATADIR.mkdir(exist_ok=True)
    datafile = C.DATADIR / "vignetting.npz"

    # Simulate and cache, unless we were asked to only re-draw the plots.
    if args.plot_only:
        if not datafile.exists():
            raise SystemExit(f"no saved data at {datafile}; run without "
                             "--plot-only first")
    else:
        n_mc = 8 if args.quick else args.n_mc
        if n_mc < 2:
            raise SystemExit("--n-mc must be at least 2 for bootstrap error bars")
        print(f"running with n_mc={n_mc}, jobs={args.jobs}")
        # Slide the vignette edge inward from the pupil rim.  The first entry
        # (x_edge = R_OUTER) leaves the edge tangent to the rim -- zero
        # vignetting -- and is the reference used for per-mode normalization.
        x_edges = np.linspace(C.R_OUTER, -0.4 * C.R_OUTER, 8)
        frac = np.array([sim.vignette_fraction(xe) for xe in x_edges])
        sig, residuals = sweep(x_edges, n_mc, C.SEED, args.jobs)
        np.savez(datafile, frac=frac, sig=sig, residuals=residuals)
        print(f"wrote {datafile}")

    data = np.load(datafile)
    frac, sig = data["frac"], data["sig"]
    residuals = data["residuals"]
    if residuals.shape[1] < 2:
        raise SystemExit(f"{datafile} has fewer than 2 Monte-Carlo trials; "
                         "rerun without --plot-only")

    # Highlight coma and spherical (same families as the obscuration study), plus
    # the median over all dense modes as a summary curve.
    styles = [(7, "Z7 (primary coma)", "C0", "-"),
              (16, "Z16 (secondary coma)", "C0", "--"),
              (11, "Z11 (primary spherical)", "C1", "-"),
              (22, "Z22 (secondary spherical)", "C1", "--")]

    fig, ax = plt.subplots(figsize=(6, 5))
    for term, label, color, ls in styles:
        j = C.DENSE_TERMS.index(term)
        curve = sig[:, j] / sig[0, j]
        kwargs = dict(marker="o", color=color, linestyle=ls, label=label)
        lo, hi = plotutils.relative_error_interval(residuals, j)
        ax.errorbar(frac, curve,
                    yerr=plotutils.yerr_from_interval(curve, lo, hi),
                    capsize=2.0, capthick=0.8, elinewidth=0.8,
                    ecolor=color, **kwargs)
    med = np.median(sig / sig[0], axis=1)
    ax.plot(frac, med, color="k", lw=2, alpha=0.5, label="median (all modes)")
    ax.axhline(1.0, color="gray", lw=0.6)
    ax.set_yscale("log")   # Z11 spikes by ~100x, so log keeps everything legible
    ax.set_xlabel("vignetted fraction of pupil area")
    ax.set_ylabel("relative error (normalized to no vignetting)")
    ax.set_title("Wavefront estimation vs asymmetric vignetting (toy, field center)")
    ax.legend()
    fig.tight_layout()
    out = C.FIGDIR / "vignetting.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    plt.close(fig)

    plotutils.all_terms_heatmap(
        frac, sig,
        xlabel="vignetted fraction of pupil area",
        title="All modes vs asymmetric vignetting (toy, field center)",
        out=C.FIGDIR / "all_terms_vs_vignetting.png",
        residuals=residuals,
    )


if __name__ == "__main__":
    main()
