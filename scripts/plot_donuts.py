"""Show what the simulated donuts look like.

Renders a gallery of toy Rubin donuts, each a random draw of the dense-mode
perturbations on top of the pure-defocus reference (full shape + intensity
rendering).  This illustrates the typical Monte-Carlo variation at the
injection sigma used in the experiments.

Also renders a compact gallery showing how the donut changes across the
central-obscuration and asymmetric-vignetting sweeps.

Run:  python plot_donuts.py
"""

import numpy as np
import matplotlib.pyplot as plt

import config as C
import sim

# Layout of the random-donut gallery.
N_DONUT_ROWS = 3
N_DONUT_COLS = 4

N_GEOMETRY_COLS = 5
GEOMETRY_CROP_HALF = C.NPIX // 2


def crop(img, half=70):
    """Center-crop a square image to +/- ``half`` pixels for display."""
    c = img.shape[0] // 2
    return img[c - half : c + half + 1, c - half : c + half + 1]


def plot_geometry_sweep():
    """Plot example donuts across the obscuration and vignetting sweeps."""
    eps_values = np.linspace(0.0, 0.7, N_GEOMETRY_COLS)
    x_edges = np.linspace(C.R_OUTER, -0.4 * C.R_OUTER, N_GEOMETRY_COLS)
    z_ref = sim.make_reference()

    fig, axes = plt.subplots(2, N_GEOMETRY_COLS, figsize=(7, 3.4))
    for col, eps in enumerate(eps_values):
        r_in = eps * C.R_OUTER
        z_obsc = sim.make_reference(defocus=sim.fixed_diameter_defocus(r_in))
        fac = sim.make_factory(
            surface_brightness=True,
            zk_r_inner=r_in,
            pupil_r_inner=r_in,
        )
        ax = axes[0, col]
        ax.imshow(
            crop(fac.image(aberrations=z_obsc, npix=C.NPIX), half=GEOMETRY_CROP_HALF),
            origin="lower",
            cmap="magma",
        )
        ax.set_title(f"$\\varepsilon$={eps:.2f}")

    for col, x_edge in enumerate(x_edges):
        frac = sim.vignette_fraction(x_edge)
        fac = sim.make_factory(surface_brightness=True, vignette_x_edge=x_edge)
        ax = axes[1, col]
        ax.imshow(
            crop(fac.image(aberrations=z_ref, npix=C.NPIX), half=GEOMETRY_CROP_HALF),
            origin="lower",
            cmap="magma",
        )
        ax.set_title(f"$f_{{\\rm vig}}$={frac:.2f}")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 0].set_ylabel("central\nobscuration")
    axes[1, 0].set_ylabel("vignetting\nfraction")
    out = C.FIGDIR / "geometry_sweep_donuts.pdf"
    fig.savefig(out, dpi=500, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


def plot_random_donuts():
    """Plot a grid of donuts, each a random draw of the dense-mode perturbations."""
    C.FIGDIR.mkdir(exist_ok=True)
    fac_full = sim.make_factory(surface_brightness=True)
    z_ref = sim.make_reference()
    rng = np.random.default_rng(C.SEED)

    fig, axes = plt.subplots(
        N_DONUT_ROWS, N_DONUT_COLS, figsize=(2.3 * N_DONUT_COLS, 2.3 * N_DONUT_ROWS)
    )
    for ax in axes.flat:
        z = z_ref.copy()
        z[C.DENSE_TERMS] += rng.normal(0.0, C.INJECT_SIGMA, size=len(C.DENSE_TERMS))
        img = crop(fac_full.image(aberrations=z, npix=C.NPIX))
        ax.imshow(img, origin="lower")  # per-image autoscale
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"Toy Rubin donuts, random dense draws "
        f"(defocus $Z_4$={C.DEFOCUS_Z4 * 1e6:.0f} $\\mu$m, "
        f"$\\sigma$={C.INJECT_SIGMA * 1e9:.0f} nm RMS/mode)",
        fontsize=12,
    )
    out = C.FIGDIR / "example_donuts.pdf"
    fig.savefig(out, dpi=500, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    plot_random_donuts()
    plot_geometry_sweep()
