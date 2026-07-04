"""Study 3 (new): impact of vignetting on wavefront estimation.

Vignetting removes light from the *outer* part of the pupil.  We model it in
the simplest controlled way: keep the full-aperture annular-Zernike
normalization fixed (the outer wavefront still physically exists), keep the
donut at the center of the field of view, and progressively shrink the
illuminated outer pupil radius ``pupil_R_outer``.  The only thing changing is
how much outer-pupil light reaches the detector -- pure vignetting, isolated
from field angle and from the central obscuration.

This is the outer-pupil analogue of the central-obscuration study: there we ate
into the pupil from the inside, here from the outside.  We expect estimation of
modes with power near the outer edge (spherical, high-order modes) to degrade as
vignetting increases.  Errors are normalized to the no-vignetting case per mode.

Run:  python study_vignetting.py [--n-mc N] [--quick]
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

import config as C
import sim
import plotutils


def vignette_fraction(pupil_r_outer):
    """Fraction of the (annular) pupil area removed by vignetting.

    Parameters
    ----------
    pupil_r_outer : float
        Illuminated outer pupil radius, in meters.

    Returns
    -------
    float
        Removed annulus area divided by the full annulus area.
    """
    full = C.R_OUTER**2 - C.R_INNER**2
    kept = pupil_r_outer**2 - C.R_INNER**2
    return (full - kept) / full


def sweep(pupil_radii, n_mc, seed, n_jobs):
    """Return per-mode 1-sigma error at each outer pupil radius (dense fit).

    Parameters
    ----------
    pupil_radii : ndarray
        Illuminated outer pupil radii to sweep, in meters.
    n_mc : int
        Monte-Carlo realizations per radius.
    seed : int
        Base random seed (shared across radii).
    n_jobs : int
        Number of worker processes.

    Returns
    -------
    sig : ndarray, shape (len(pupil_radii), len(DENSE_TERMS))
        Standard deviation of ``(z_fit - z_true) / INJECT_SIGMA`` per mode.
    """
    sig = np.empty((len(pupil_radii), len(C.DENSE_TERMS)))
    for i, pro in enumerate(pupil_radii):
        # Zernike normalization stays on the full aperture; only the
        # illuminated outer edge shrinks.  Central obscuration held at Rubin.
        kwargs = dict(surface_brightness=True, zk_r_inner=C.R_INNER,
                      pupil_r_outer=pro, pupil_r_inner=C.R_INNER)
        res = sim.monte_carlo(C.DENSE_TERMS, kwargs, n_mc=n_mc, seed=seed, n_jobs=n_jobs)
        sig[i] = res.std(axis=0) / C.INJECT_SIGMA
        print(f"  vignette fraction={vignette_fraction(pro):.2f} done")
    return sig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-mc", type=int, default=C.N_MC)
    p.add_argument("--jobs", type=int, default=sim.default_jobs(),
                   help="worker processes (default: performance-core count)")
    p.add_argument("--quick", action="store_true", help="fast, low-stats run")
    args = p.parse_args()
    n_mc = 8 if args.quick else args.n_mc
    print(f"running with n_mc={n_mc}, jobs={args.jobs}")

    C.FIGDIR.mkdir(exist_ok=True)
    # Sweep the illuminated outer radius from the full aperture inward.  The
    # first entry (R_OUTER) is the no-vignetting reference used for normalization.
    pupil_radii = np.linspace(C.R_OUTER, 0.75 * C.R_OUTER, 8)
    frac = np.array([vignette_fraction(r) for r in pupil_radii])
    sig = sweep(pupil_radii, n_mc, C.SEED, args.jobs)

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
        ax.plot(frac, curve, marker="o", color=color, ls=ls, label=label)
    med = np.median(sig / sig[0], axis=1)
    ax.plot(frac, med, color="k", lw=2, alpha=0.5, label="median (all modes)")
    ax.axhline(1.0, color="gray", lw=0.6)
    ax.set_yscale("log")   # Z11 spikes by ~100x, so log keeps everything legible
    ax.set_xlabel("vignetted fraction of pupil area")
    ax.set_ylabel("relative error (normalized to no vignetting)")
    ax.set_title("Wavefront estimation vs vignetting (toy, field center)")
    ax.legend()
    fig.tight_layout()
    out = C.FIGDIR / "vignetting.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    plt.close(fig)

    plotutils.all_terms_heatmap(
        frac, sig,
        xlabel="vignetted fraction of pupil area",
        title="All modes vs vignetting (toy, field center)",
        out=C.FIGDIR / "all_terms_vs_vignetting.png",
    )


if __name__ == "__main__":
    main()
