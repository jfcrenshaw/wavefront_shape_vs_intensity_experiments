"""Study 2: impact of the central obscuration on wavefront estimation.

Using the toy Rubin paraboloid at field center, we sweep the central
obscuration fraction epsilon = R_inner / R_outer from 0 to 0.7 and measure how
the (full-information) wavefront-estimation error changes for each mode.  The
annular-Zernike normalization tracks epsilon, and the physical pupil hole is
set to the same radius.  Errors are normalized to the epsilon = 0 case per mode.

Reproduces ``coma_vs_obscuration.png`` and ``spherical_vs_obscuration.png``:
increasing the obscuration degrades coma (especially secondary coma) but
improves primary spherical up to epsilon ~ 0.5.

Run:  python study_obscuration.py [--n-mc N] [--quick]
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

import config as C
import sim
import plotutils


def sweep(eps_values, n_mc, seed, n_jobs):
    """Return per-mode 1-sigma error at each obscuration, for all dense modes.

    Parameters
    ----------
    eps_values : ndarray
        Central-obscuration fractions to sweep.
    n_mc : int
        Monte-Carlo realizations per obscuration.
    seed : int
        Base random seed (shared across obscurations).
    n_jobs : int
        Number of worker processes.

    Returns
    -------
    sig : ndarray, shape (len(eps_values), len(DENSE_TERMS))
        Standard deviation of ``(z_fit - z_true) / INJECT_SIGMA`` per mode.
    """
    sig = np.empty((len(eps_values), len(C.DENSE_TERMS)))
    for i, eps in enumerate(eps_values):
        r_in = eps * C.R_OUTER
        kwargs = dict(surface_brightness=True, zk_r_inner=r_in, pupil_r_inner=r_in)
        res = sim.monte_carlo(C.DENSE_TERMS, kwargs, n_mc=n_mc, seed=seed,
                              n_jobs=n_jobs, flux_norm="per_pixel")
        sig[i] = res.std(axis=0) / C.INJECT_SIGMA
        print(f"  eps={eps:.2f} done")
    return sig


def _curve(sig, eps_values, term):
    """Error vs epsilon for one mode, normalized to its epsilon=0 value."""
    j = C.DENSE_TERMS.index(term)
    c = sig[:, j]
    return c / c[0]


def plot_family(eps_values, sig, styles, title, fname):
    """Plot normalized error vs obscuration for a family of modes.

    ``styles`` is a list of ``(term, label, color, filled)`` tuples.  Color
    distinguishes primary from secondary order; a filled vs hollow marker
    distinguishes the two parities of a paired mode, matching the paper.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    for term, label, color, filled in styles:
        ax.plot(eps_values, _curve(sig, eps_values, term),
                marker="o", color=color,
                mfc=color if filled else "white", label=label)
    ax.axvline(C.EPS_RUBIN, color="k", ls=":", lw=1)
    ax.text(C.EPS_RUBIN, ax.get_ylim()[1], " Rubin", va="top", fontsize=9)
    ax.axhline(1.0, color="gray", lw=0.6)
    ax.set_xlabel(r"central obscuration $\varepsilon = R_{\rm inner}/R_{\rm outer}$")
    ax.set_ylabel("relative error (normalized to $\\varepsilon=0$)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out = C.FIGDIR / fname
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    plt.close(fig)


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
    eps_values = np.linspace(0.0, 0.7, 8)
    sig = sweep(eps_values, n_mc, C.SEED, args.jobs)

    plot_family(
        eps_values, sig,
        styles=[(7, "Z7 (primary coma)", "C0", True),
                (8, "Z8 (primary coma)", "C0", False),
                (16, "Z16 (secondary coma)", "C1", True),
                (17, "Z17 (secondary coma)", "C1", False)],
        title="Coma estimation vs central obscuration",
        fname="coma_vs_obscuration.png",
    )
    plot_family(
        eps_values, sig,
        styles=[(11, "Z11 (primary spherical)", "C0", True),
                (22, "Z22 (secondary spherical)", "C1", False)],
        title="Spherical estimation vs central obscuration",
        fname="spherical_vs_obscuration.png",
    )
    plotutils.all_terms_heatmap(
        eps_values, sig,
        xlabel=r"central obscuration $\varepsilon = R_{\rm inner}/R_{\rm outer}$",
        title="All modes vs central obscuration",
        out=C.FIGDIR / "all_terms_vs_obscuration.png",
    )


if __name__ == "__main__":
    main()
