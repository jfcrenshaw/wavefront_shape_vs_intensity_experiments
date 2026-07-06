"""Small shared plotting helpers for the studies."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from shape_vs_intensity import config as C

# The shared Computer-Modern figure style, shipped inside the package so it is
# found by absolute path regardless of the working directory.
STYLE_FILE = Path(__file__).resolve().parent / "shape_vs_intensity.mplstyle"


def use_style():
    """Apply the repo's shared matplotlib style to the current session.

    Loads the ``shape_vs_intensity.mplstyle`` shipped with the installed
    package, so every study script and notebook gets the same Computer-Modern
    figure style with no ``matplotlibrc``-in-the-working-directory dependency.
    """
    plt.style.use(STYLE_FILE)


def _confidence_quantiles(confidence, name):
    """Return lower/upper quantiles for a central confidence interval."""
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"{name} must be between 0 and 1")
    alpha = 1.0 - confidence
    return [0.5 * alpha, 1.0 - 0.5 * alpha]


def _paired_bootstrap_sample(n_mc, n_bootstrap, seed):
    """Trial indices for paired bootstrap resampling across sweep positions."""
    if n_mc < 2:
        raise ValueError("at least two Monte-Carlo samples are needed")
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_mc, size=(n_bootstrap, n_mc))


def relative_error_interval(residuals, term_index):
    """Bootstrap interval for ``error / fiducial error`` for one mode.

    Parameters
    ----------
    residuals : ndarray, shape (P, n_mc, n_terms)
        Raw Monte-Carlo residuals saved by a sweep.
    term_index : int
        Column index of the Zernike mode in ``residuals``.

    Returns
    -------
    lo, hi : ndarray
        Lower and upper bootstrap curves, each with shape ``(P,)``.
    """
    residuals = np.asarray(residuals)
    _, n_mc, _ = residuals.shape
    sample = _paired_bootstrap_sample(
        n_mc, C.LINE_PLOT_BOOTSTRAP_SAMPLES, C.LINE_PLOT_BOOTSTRAP_SEED
    )
    boot_sig = residuals[:, sample, term_index].std(axis=2)

    with np.errstate(divide="ignore", invalid="ignore"):
        boot_ratio = boot_sig / boot_sig[0:1]

    q = _confidence_quantiles(
        C.LINE_PLOT_BOOTSTRAP_CONFIDENCE, "LINE_PLOT_BOOTSTRAP_CONFIDENCE"
    )
    return np.nanquantile(boot_ratio, q, axis=1)


def yerr_from_interval(y, lo, hi):
    """Convert lower/upper curves to Matplotlib's asymmetric ``yerr`` format."""
    return np.clip(np.vstack([y - lo, hi - y]), 0.0, np.inf)


# Zernike mode families shared by the pupil-geometry studies, so the same
# families are plotted the same way in every study.  Each entry is
# ``(stem, family, lines)``: ``stem`` names the output file, ``family`` is the
# title prefix, and ``lines`` is the per-curve spec passed to ``plot_family``.
# Color marks primary (C0) vs secondary (C1) order; a hollow marker
# (``filled=False``) marks the second parity of a paired mode, matching the paper.
FAMILIES = [
    (
        "spherical",
        "Defocus & Spherical",
        [
            dict(term=4, label="Z4", color="C0"),
            dict(term=11, label="Z11", color="C1"),
            dict(term=22, label="Z22", color="C2"),
        ],
    ),
    (
        "astigmatism",
        "Astigmatism",
        [
            dict(term=5, label="Z5", color="C0"),
            dict(term=6, label="Z6", color="C0", filled=False),
            dict(term=12, label="Z12", color="C1", filled=False),
            dict(term=13, label="Z13", color="C1"),
        ],
    ),
    (
        "coma",
        "Coma",
        [
            dict(term=7, label="Z7", color="C0"),
            dict(term=8, label="Z8", color="C0", filled=False),
            dict(term=16, label="Z16", color="C1", filled=False),
            dict(term=17, label="Z17", color="C1"),
        ],
    ),
    (
        "trefoil",
        "Trefoil",
        [
            dict(term=9, label="Z9", color="C0"),
            dict(term=10, label="Z10", color="C0", filled=False),
            dict(term=18, label="Z18", color="C1", filled=False),
            dict(term=19, label="Z19", color="C1"),
        ],
    ),
    (
        "quadpenta",
        "Quadra/Pentafoil",
        [
            dict(term=14, label="Z14", color="C0", filled=False),
            dict(term=15, label="Z15", color="C0"),
            dict(term=20, label="Z20", color="C1", filled=False),
            dict(term=21, label="Z21", color="C1"),
        ],
    ),
]


def plot_family(
    param_values,
    sig,
    residuals,
    lines,
    *,
    xlabel,
    ylabel,
    title,
    out,
    rubin_line=False,
    logy=False,
):
    """Plot normalized error vs a swept parameter for a family of modes.

    Each curve is the per-mode error normalized to its value at the first
    (baseline) parameter, with a bootstrap confidence interval.

    Parameters
    ----------
    param_values : array_like
        Swept parameter values (x axis), length ``P``.  The first is the
        normalization baseline.
    sig : ndarray, shape (P, len(DENSE_TERMS))
        1-sigma relative error per mode at each parameter value.
    residuals : ndarray, shape (P, n_mc, len(DENSE_TERMS))
        Raw Monte-Carlo residuals, used for the bootstrap error bars.
    lines : list of dict
        One dict per curve.  Required keys ``term`` (Zernike Noll index) and
        ``label``; optional ``color`` (matplotlib color), ``filled`` (bool,
        default ``True``; ``False`` draws a hollow marker), and ``linestyle``
        (default ``"-"``).
    xlabel, ylabel, title : str
        Axis labels and title.
    out : pathlib.Path
        Output file path.
    rubin_line : bool, optional
        Draw a dotted vertical line at ``C.EPS`` labelled "Rubin".
    logy : bool, optional
        Use a logarithmic y axis.
    """
    fig, ax = plt.subplots(figsize=(3.3, 2.8))
    for line in lines:
        j = C.DENSE_TERMS.index(line["term"])
        curve = sig[:, j] / sig[0, j]
        color = line.get("color")
        lo, hi = relative_error_interval(residuals, j)
        ax.errorbar(
            param_values,
            curve,
            yerr=yerr_from_interval(curve, lo, hi),
            marker="o",
            color=color,
            linestyle=line.get("linestyle", "-"),
            markerfacecolor=color if line.get("filled", True) else "white",
            label=line["label"],
            capsize=2.0,
            capthick=0.8,
            elinewidth=0.8,
            ecolor=color,
        )

    if logy:
        ax.set_yscale("log")
    if rubin_line:
        ax.axvline(C.EPS, color="k", ls=":", lw=1)
    ax.axhline(1.0, color="gray", lw=0.6)
    ax.set(
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
    )
    ax.legend(frameon=False, handlelength=1.5)

    fig.savefig(out, dpi=500, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


def all_terms_heatmap(
    param_values,
    sig,
    xlabel,
    out,
    xtick_fmt="{:.2f}",
):
    """Heatmap of normalized error for every dense Zernike vs a swept parameter.

    Color is ``log10(error / error_at_first_param)`` on a diverging scale, so
    white means "unaffected", red means "degraded", and blue means "improved"
    relative to the first (baseline) parameter value.  This makes it easy to see
    at a glance which modes are and are not sensitive to the pupil geometry.
    Cells are muted unless the bootstrap confidence interval excludes zero and
    the effect exceeds the configured practical threshold.

    Parameters
    ----------
    param_values : array_like
        Swept parameter values (columns), length ``P``.  The first is the
        normalization baseline.
    sig : ndarray, shape (P, len(DENSE_TERMS))
        1-sigma relative error per mode at each parameter value.
    xlabel : str
        Axis label for the swept parameter.
    out : pathlib.Path
        Output file path.
    xtick_fmt : str, optional
        Format string for the x tick labels.
    """
    ratio = sig / sig[0]  # normalize to baseline per mode
    data = np.log10(ratio).T  # rows = terms, cols = param
    vmax = np.nanmax(np.abs(data))

    fig, ax = plt.subplots(figsize=(3, 4))
    im = ax.imshow(
        data,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set(
        xlabel=xlabel,
        ylabel="Zernike Noll index",
        xticks=range(len(param_values)),
        yticks=range(len(C.DENSE_TERMS)),
        yticklabels=[f"Z{t}" for t in C.DENSE_TERMS],
    )
    ax.set_xticklabels([xtick_fmt.format(v) for v in param_values], rotation=45)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(
        "$\\log_{10}$(error / baseline)\n"
        "[blue = better, red = worse]\n"
    )

    fig.savefig(out, dpi=500, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)
