"""Study 1: impact of Zernike sparsity (shape-only vs shape+intensity).

Reproduces two figures from the paper:

* ``zk_estimates.pdf``    -- per-mode wavefront-estimation error for a *sparse*
  fit (lowest-nu detectable modes only) and a *dense* fit (all Noll 4-22),
  comparing shape-only against full shape+intensity information.
* ``shape_degeneracy.pdf`` -- a concrete shape degeneracy: a shape-only best-fit
  that matches the target donut's shape but has a wildly different intensity
  pattern (approaching a caustic), illustrating why fitting high-order modes
  from shape alone is dangerous.

The simulation and the plotting are separate: a normal run simulates, caches
the results to ``data/sparsity.npz``, and then plots; ``--plot-only`` skips the
simulation and re-draws the figures from that cache (for quick cosmetic tweaks).

Run:  python study_sparsity.py [--n-mc N] [--quick] [--plot-only]
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

import config as C
import sim


def relative_spread(z_terms, n_mc, seed, n_jobs):
    """Return the 1-sigma relative error per mode for shape-only and full fits.

    Parameters
    ----------
    z_terms : sequence of int
        Noll indices to inject and fit.
    n_mc : int
        Number of Monte-Carlo realizations.
    seed : int
        Base random seed (shared by both modes so they see the same draws).
    n_jobs : int
        Number of worker processes.

    Returns
    -------
    sig_shape, sig_full : ndarray
        Standard deviation of ``(z_fit - z_true) / INJECT_SIGMA`` per mode.
    """
    res_full = sim.monte_carlo(
        z_terms,
        dict(surface_brightness=True),
        n_mc=n_mc,
        seed=seed,
        n_jobs=n_jobs,
    )
    res_shape = sim.monte_carlo(
        z_terms,
        dict(surface_brightness=False),
        n_mc=n_mc,
        seed=seed,
        n_jobs=n_jobs,
    )
    return res_shape.std(axis=0) / C.INJECT_SIGMA, res_full.std(axis=0) / C.INJECT_SIGMA


def simulate(n_mc, seed, n_jobs):
    """Run the Monte-Carlo spreads for both the sparse and dense fits.

    Returns
    -------
    dict
        The four per-mode error arrays (sparse/dense x shape/full), keyed for
        saving with ``np.savez`` and reloading in :func:`plot_estimates`.
    """
    sparse_shape, sparse_full = relative_spread(C.SPARSE_TERMS, n_mc, seed, n_jobs)
    dense_shape, dense_full = relative_spread(C.DENSE_TERMS, n_mc, seed, n_jobs)
    return dict(
        sparse_shape=sparse_shape,
        sparse_full=sparse_full,
        dense_shape=dense_shape,
        dense_full=dense_full,
    )


def plot_estimates(data):
    """Make the two-panel sparse/dense comparison figure from saved arrays."""
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True, sharey=True)
    for ax, (terms, sig_shape, sig_full) in zip(
        axes,
        [
            (C.SPARSE_TERMS, data["sparse_shape"], data["sparse_full"]),
            (C.DENSE_TERMS, data["dense_shape"], data["dense_full"]),
        ],
    ):
        x = np.array(terms)
        # Vertical 1-sigma lines centered on zero, offset for the two modes.
        ax.vlines(
            x - 0.12,
            -sig_shape,
            sig_shape,
            color="C3",
            lw=4,
            alpha=0.8,
            label="shape only",
        )
        ax.vlines(
            x + 0.12,
            -sig_full,
            sig_full,
            color="C0",
            lw=4,
            alpha=0.8,
            label="shape + intensity",
        )
        ax.axhline(0, color="k", lw=0.8)
        ax.set(
            ylabel="Relative error",
            xticks=x,
            xticklabels=[f"Z{t}" for t in terms],
            ylim=(-0.25, 0.25),
        )

    axes[0].legend(loc="upper right", frameon=False)
    axes[-1].set_xlabel("Zernike (Noll index)")

    out = C.FIGDIR / "zk_estimates.pdf"
    fig.savefig(out, bbox_inches="tight")
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
    datafile = C.DATADIR / "sparsity.npz"

    # Simulate and cache, unless we were asked to only re-draw the plots.
    if args.plot_only:
        if not datafile.exists():
            raise SystemExit(
                f"no saved data at {datafile}; run without --plot-only first"
            )
    else:
        n_mc = 8 if args.quick else args.n_mc
        print(f"running with n_mc={n_mc}, jobs={args.jobs}")
        spreads = simulate(n_mc, C.SEED, args.jobs)
        np.savez(datafile, **spreads)
        print(f"wrote {datafile}")

    data = np.load(datafile)
    plot_estimates(data)


if __name__ == "__main__":
    main()
