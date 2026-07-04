"""Small shared plotting helpers for the studies."""

import numpy as np
import matplotlib.pyplot as plt

import config as C


def all_terms_heatmap(param_values, sig, xlabel, title, out, xtick_fmt="{:.2f}"):
    """Heatmap of normalized error for every dense Zernike vs a swept parameter.

    Color is ``log10(error / error_at_first_param)`` on a diverging scale, so
    white means "unaffected", red means "degraded", and blue means "improved"
    relative to the first (baseline) parameter value.  This makes it easy to see
    at a glance which modes are and are not sensitive to the pupil geometry.

    Parameters
    ----------
    param_values : array_like
        Swept parameter values (columns), length ``P``.  The first is the
        normalization baseline.
    sig : ndarray, shape (P, len(DENSE_TERMS))
        1-sigma relative error per mode at each parameter value.
    xlabel : str
        Axis label for the swept parameter.
    title : str
        Plot title.
    out : pathlib.Path
        Output file path.
    xtick_fmt : str, optional
        Format string for the x tick labels.
    """
    ratio = sig / sig[0]                      # normalize to baseline per mode
    data = np.log10(ratio).T                  # rows = terms, cols = param
    vmax = np.nanmax(np.abs(data))

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(data, aspect="auto", origin="lower", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(C.DENSE_TERMS)))
    ax.set_yticklabels([f"Z{t}" for t in C.DENSE_TERMS])
    ax.set_xticks(range(len(param_values)))
    ax.set_xticklabels([xtick_fmt.format(v) for v in param_values], rotation=45)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Zernike (Noll index)")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("$\\log_{10}$(error / baseline)   "
                   "[red = worse, blue = better]")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    plt.close(fig)
