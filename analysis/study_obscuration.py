"""Study 2: impact of the central obscuration on wavefront estimation.

Using the toy Rubin paraboloid at field center, we sweep the central
obscuration fraction epsilon = R_inner / R_outer from 0 to 0.7 and measure how
the (full-information) wavefront-estimation error changes for each mode.  The
annular-Zernike normalization tracks epsilon, and the physical pupil hole is
set to the same radius.  The reference Z4 coefficient is rescaled with annulus
area so the outer donut diameter stays fixed; otherwise changing the annular
normalization would also change the physical defocus in this toy model.  Errors
are normalized to the epsilon = 0 case per mode.

Reproduces ``coma_vs_obscuration.png`` and ``spherical_vs_obscuration.png``:
increasing the obscuration degrades coma (especially secondary coma) but
improves primary spherical up to epsilon ~ 0.5.

The simulation and the plotting are separate: a normal run simulates, caches
the results to ``data/obscuration.npz``, and then plots; ``--plot-only`` skips
the simulation and re-draws the figures from that cache (for quick cosmetic
tweaks).

Run:  python study_obscuration.py [--n-mc N] [--quick] [--plot-only]
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
    residuals : ndarray, shape (len(eps_values), n_mc, len(DENSE_TERMS))
        Raw Monte-Carlo residuals, cached so heatmaps can mask noisy cells.
    """
    sig = np.empty((len(eps_values), len(C.DENSE_TERMS)))
    residuals = np.empty((len(eps_values), n_mc, len(C.DENSE_TERMS)))
    for i, eps in enumerate(eps_values):
        r_in = eps * C.R_OUTER
        kwargs = dict(surface_brightness=True, zk_r_inner=r_in, pupil_r_inner=r_in)
        z_ref = sim.make_reference(defocus=sim.fixed_diameter_defocus(r_in))
        res = sim.monte_carlo(
            C.DENSE_TERMS,
            kwargs,
            n_mc=n_mc,
            seed=seed,
            n_jobs=n_jobs,
            flux_norm="per_pixel",
            z_ref=z_ref,
        )
        residuals[i] = res
        sig[i] = res.std(axis=0) / C.INJECT_SIGMA
        print(f"  eps={eps:.2f}, Z4={z_ref[4] * 1e6:.1f} um done")
    return sig, residuals


def _curve(sig, eps_values, term):
    """Error vs epsilon for one mode, normalized to its epsilon=0 value."""
    j = C.DENSE_TERMS.index(term)
    c = sig[:, j]
    return c / c[0]


def plot_family(eps_values, sig, residuals, styles, title, fname):
    """Plot normalized error vs obscuration for a family of modes.

    ``styles`` is a list of ``(term, label, color, filled)`` tuples.  Color
    distinguishes primary from secondary order; a filled vs hollow marker
    distinguishes the two parities of a paired mode, matching the paper.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    for term, label, color, filled in styles:
        j = C.DENSE_TERMS.index(term)
        curve = _curve(sig, eps_values, term)
        kwargs = dict(
            marker="o",
            color=color,
            linestyle="-",
            markerfacecolor=color if filled else "white",
            label=label,
        )
        lo, hi = plotutils.relative_error_interval(residuals, j)
        ax.errorbar(
            eps_values,
            curve,
            yerr=plotutils.yerr_from_interval(curve, lo, hi),
            capsize=2.0,
            capthick=0.8,
            elinewidth=0.8,
            ecolor=color,
            **kwargs,
        )
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
    p.add_argument(
        "--jobs",
        type=int,
        default=sim.default_jobs(),
        help="worker processes (default: performance-core count)",
    )
    p.add_argument("--quick", action="store_true", help="fast, low-stats run")
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="skip the simulation; re-draw plots from saved data",
    )
    args = p.parse_args()

    C.FIGDIR.mkdir(exist_ok=True)
    C.DATADIR.mkdir(exist_ok=True)
    datafile = C.DATADIR / "obscuration.npz"

    # Simulate and cache, unless we were asked to only re-draw the plots.
    if args.plot_only:
        if not datafile.exists():
            raise SystemExit(
                f"no saved data at {datafile}; run without --plot-only first"
            )
    else:
        n_mc = 8 if args.quick else args.n_mc
        if n_mc < 2:
            raise SystemExit("--n-mc must be at least 2 for bootstrap error bars")
        print(f"running with n_mc={n_mc}, jobs={args.jobs}")
        eps_values = np.linspace(0.0, 0.7, 8)
        sig, residuals = sweep(eps_values, n_mc, C.SEED, args.jobs)
        defocus_z4 = np.array(
            [sim.fixed_diameter_defocus(eps * C.R_OUTER) for eps in eps_values]
        )
        np.savez(
            datafile,
            eps_values=eps_values,
            sig=sig,
            residuals=residuals,
            defocus_z4=defocus_z4,
        )
        print(f"wrote {datafile}")

    data = np.load(datafile)
    eps_values, sig = data["eps_values"], data["sig"]
    residuals = data["residuals"]
    if residuals.shape[1] < 2:
        raise SystemExit(
            f"{datafile} has fewer than 2 Monte-Carlo trials; rerun without --plot-only"
        )

    plot_family(
        eps_values,
        sig,
        residuals,
        styles=[
            (7, "Z7 (primary coma)", "C0", True),
            (8, "Z8 (primary coma)", "C0", False),
            (16, "Z16 (secondary coma)", "C1", True),
            (17, "Z17 (secondary coma)", "C1", False),
        ],
        title="Coma estimation vs central obscuration",
        fname="coma_vs_obscuration.png",
    )
    plot_family(
        eps_values,
        sig,
        residuals,
        styles=[
            (11, "Z11 (primary spherical)", "C0", True),
            (22, "Z22 (secondary spherical)", "C1", False),
        ],
        title="Spherical estimation vs central obscuration",
        fname="spherical_vs_obscuration.png",
    )
    plotutils.all_terms_heatmap(
        eps_values,
        sig,
        xlabel=r"central obscuration $\varepsilon$",
        title="All modes vs central obscuration",
        out=C.FIGDIR / "all_terms_vs_obscuration.png",
        residuals=residuals,
    )


if __name__ == "__main__":
    main()
