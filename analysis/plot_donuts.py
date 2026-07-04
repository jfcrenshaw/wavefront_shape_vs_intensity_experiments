"""Show what the simulated donuts look like.

Renders a gallery of donuts for the toy Rubin paraboloid: the pure-defocus
reference plus a few individual aberrations and one random dense draw, each in
both full (shape + intensity) and shape-only rendering.  This is purely
illustrative -- the single-mode amplitudes here are larger than the Monte-Carlo
injection sigma so the shape/intensity effects are visible by eye.

Also renders a compact gallery showing how the donut changes across the
central-obscuration and asymmetric-vignetting sweeps.

Run:  python plot_donuts.py
"""

import numpy as np
import matplotlib.pyplot as plt

import config as C
import sim

# Illustrative single-mode amplitude (meters of wavefront), larger than the
# Monte-Carlo INJECT_SIGMA so the effect on the donut is clearly visible.
AMP = 0.4e-6

# (label, Noll index to add on top of defocus).  None => defocus only.
CASES = [
    ("Defocus only", None),
    ("+ Astigmatism (Z5)", 5),
    ("+ Coma (Z7)", 7),
    ("+ Trefoil (Z9)", 9),
    ("+ Spherical (Z11)", 11),
    ("+ 2nd coma (Z16)", 16),
]

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
    fig.tight_layout()
    out = C.FIGDIR / "geometry_sweep_donuts.png"
    fig.savefig(out, dpi=500, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


def main():
    C.FIGDIR.mkdir(exist_ok=True)
    fac_full = sim.make_factory(surface_brightness=True)
    fac_shape = sim.make_factory(surface_brightness=False)
    z_ref = sim.make_reference()

    # One random dense draw, shown as the last row.
    rng = np.random.default_rng(C.SEED)
    z_rand = z_ref.copy()
    z_rand[C.DENSE_TERMS] += rng.normal(0.0, C.INJECT_SIGMA, size=len(C.DENSE_TERMS))
    cases = [
        (label, z_ref if idx is None else _add(z_ref, idx, AMP)) for label, idx in CASES
    ]
    cases.append(("Random dense draw", z_rand))

    nrow = len(cases)
    fig, axes = plt.subplots(nrow, 2, figsize=(5, 2.3 * nrow))
    for row, (label, ab) in enumerate(cases):
        for col, fac in enumerate([fac_full, fac_shape]):
            img = crop(fac.image(aberrations=ab, npix=C.NPIX))
            ax = axes[row, col]
            ax.imshow(img, origin="lower", cmap="magma")  # per-image autoscale
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(
                    "full (shape + intensity)" if col == 0 else "shape only",
                    fontsize=11,
                )
            if col == 0:
                ax.set_ylabel(label, fontsize=10)
    fig.suptitle(
        f"Toy Rubin donuts (defocus $Z_4$={C.DEFOCUS_Z4 * 1e6:.0f} $\\mu$m, "
        f"aberration amp={AMP * 1e9:.0f} nm)",
        fontsize=12,
    )
    fig.tight_layout()
    out = C.FIGDIR / "example_donuts.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    plt.close(fig)
    plot_geometry_sweep()


def _add(z_ref, idx, amp):
    """Return a copy of ``z_ref`` with ``amp`` added to Noll index ``idx``."""
    z = z_ref.copy()
    z[idx] += amp
    return z


if __name__ == "__main__":
    plot_geometry_sweep()
