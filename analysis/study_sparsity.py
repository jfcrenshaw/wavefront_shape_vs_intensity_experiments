"""Study 1: impact of Zernike sparsity (shape-only vs shape+intensity).

Reproduces two figures from the paper:

* ``zk_estimates.pdf``    -- per-mode wavefront-estimation error for a *sparse*
  fit (lowest-nu detectable modes only) and a *dense* fit (all Noll 4-22),
  comparing shape-only against full shape+intensity information.
* ``shape_degeneracy.pdf`` -- a concrete shape degeneracy: a shape-only best-fit
  that matches the target donut's shape but has a wildly different intensity
  pattern (approaching a caustic), illustrating why fitting high-order modes
  from shape alone is dangerous.

Run:  python study_sparsity.py [--n-mc N] [--quick]
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

import config as C
import sim


def relative_spread(z_terms, n_mc, seed):
    """Return the 1-sigma relative error per mode for shape-only and full fits.

    Parameters
    ----------
    z_terms : sequence of int
        Noll indices to inject and fit.
    n_mc : int
        Number of Monte-Carlo realizations.
    seed : int
        Base random seed (shared by both modes so they see the same draws).

    Returns
    -------
    sig_shape, sig_full : ndarray
        Standard deviation of ``(z_fit - z_true) / INJECT_SIGMA`` per mode.
    """
    fac_full = sim.make_factory(surface_brightness=True)
    fac_shape = sim.make_factory(surface_brightness=False)
    res_full = sim.monte_carlo(z_terms, fac_full, n_mc=n_mc, seed=seed)
    res_shape = sim.monte_carlo(z_terms, fac_shape, n_mc=n_mc, seed=seed)
    return res_shape.std(axis=0) / C.INJECT_SIGMA, res_full.std(axis=0) / C.INJECT_SIGMA


def plot_estimates(n_mc, seed):
    """Make the two-panel sparse/dense comparison figure."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharey=True)
    for ax, (terms, title) in zip(
        axes,
        [(C.SPARSE_TERMS, "Sparse fit (lowest-$\\nu$ detectable modes)"),
         (C.DENSE_TERMS, "Dense fit (all Noll indices 4-22)")],
    ):
        sig_shape, sig_full = relative_spread(terms, n_mc, seed)
        x = np.arange(len(terms))
        # Vertical 1-sigma lines centered on zero, offset for the two modes.
        ax.vlines(x - 0.12, -sig_shape, sig_shape, color="C3", lw=4,
                  alpha=0.8, label="shape only")
        ax.vlines(x + 0.12, -sig_full, sig_full, color="C0", lw=4,
                  alpha=0.8, label="shape + intensity")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Z{t}" for t in terms])
        ax.set_ylabel("relative error\n$(z_{\\rm fit}-z_{\\rm true})/\\sigma_{\\rm inj}$")
        ax.set_title(title)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Zernike (Noll index)")
    # Clip to show detail; grossly unconstrained shape-only modes run off-panel.
    axes[0].set_ylim(-0.6, 0.6)
    axes[1].set_ylim(-1.5, 1.5)
    fig.tight_layout()
    out = C.FIGDIR / "zk_estimates.pdf"
    fig.savefig(out)
    print(f"wrote {out}")
    plt.close(fig)


def plot_degeneracy(seed):
    """Find and plot a shape degeneracy from a dense shape-only fit.

    We fit several random targets with the shape-only model, then render each
    best-fit wavefront through the *full* (intensity) renderer.  The case whose
    full render differs most from the target -- while the shapes still match --
    is the clearest illustration of a shape/intensity degeneracy.
    """
    fac_shape = sim.make_factory(surface_brightness=False)
    fac_full = sim.make_factory(surface_brightness=True)
    z_ref = sim.make_reference()
    rng = np.random.default_rng(seed)

    best = None
    for _ in range(30):
        z_true = rng.normal(0.0, C.INJECT_SIGMA, size=len(C.DENSE_TERMS))
        _, z_fit = sim.fit_one(z_true, C.DENSE_TERMS, fac_shape,
                               seed=int(rng.integers(1 << 30)))
        # Render truth and shape-only fit through the full-intensity renderer.
        ab_true = z_ref.copy(); ab_true[C.DENSE_TERMS] += z_true
        ab_fit = z_ref.copy(); ab_fit[C.DENSE_TERMS] += z_fit
        img_true = fac_full.image(aberrations=ab_true, npix=C.NPIX)
        img_fit = fac_full.image(aberrations=ab_fit, npix=C.NPIX)
        # Score: large intensity difference is the interesting degeneracy.
        norm = lambda a: a / a.sum()
        score = np.abs(norm(img_true) - norm(img_fit)).max()
        if best is None or score > best[0]:
            best = (score, img_true, img_fit)

    _, img_true, img_fit = best
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, img, title in zip(axes, [img_true, img_fit],
                              ["Target (shape + intensity)",
                               "Shape-only fit, rendered with intensity"]):
        ax.imshow(img, origin="lower", cmap="gray")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    out = C.FIGDIR / "shape_degeneracy.pdf"
    fig.savefig(out)
    print(f"wrote {out}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-mc", type=int, default=C.N_MC)
    p.add_argument("--quick", action="store_true", help="fast, low-stats run")
    args = p.parse_args()
    n_mc = 8 if args.quick else args.n_mc

    C.FIGDIR.mkdir(exist_ok=True)
    plot_estimates(n_mc, C.SEED)
    plot_degeneracy(C.SEED + 1)


if __name__ == "__main__":
    main()
