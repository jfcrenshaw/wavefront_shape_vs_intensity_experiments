"""Small shared plotting helpers for the studies."""

import numpy as np
import matplotlib.pyplot as plt

import config as C


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


def _bootstrap_log_ratio_interval(residuals):
    """Bootstrap the uncertainty on ``log10(error / fiducial error)``.

    The same resampled trial indices are used at every swept parameter, matching
    the paired Monte-Carlo setup in the sweeps.
    """
    residuals = np.asarray(residuals)
    _, n_mc, _ = residuals.shape
    sample = _paired_bootstrap_sample(
        n_mc, C.HEATMAP_BOOTSTRAP_SAMPLES, C.HEATMAP_BOOTSTRAP_SEED
    )
    boot_sig = residuals[:, sample, :].std(axis=2)

    with np.errstate(divide="ignore", invalid="ignore"):
        boot_log_ratio = np.log10(boot_sig / boot_sig[0:1])

    q = _confidence_quantiles(
        C.HEATMAP_BOOTSTRAP_CONFIDENCE, "HEATMAP_BOOTSTRAP_CONFIDENCE"
    )
    return np.nanquantile(boot_log_ratio, q, axis=1)


def _significant_effect_mask(data, residuals):
    """Return cells whose effect is both significant and practically large."""
    if C.HEATMAP_MIN_ERROR_RATIO <= 1.0:
        raise ValueError("HEATMAP_MIN_ERROR_RATIO must be greater than 1")

    lo, hi = _bootstrap_log_ratio_interval(residuals)
    ci_excludes_zero = (lo > 0.0) | (hi < 0.0)
    large_enough = np.abs(data.T) >= np.log10(C.HEATMAP_MIN_ERROR_RATIO)
    return (ci_excludes_zero & large_enough).T


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


def all_terms_heatmap(
    param_values,
    sig,
    xlabel,
    title,
    out,
    residuals,
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
    title : str
        Plot title.
    out : pathlib.Path
        Output file path.
    xtick_fmt : str, optional
        Format string for the x tick labels.
    residuals : ndarray, shape (P, n_mc, len(DENSE_TERMS))
        Raw Monte-Carlo residuals used for uncertainty masking.
    """
    ratio = sig / sig[0]  # normalize to baseline per mode
    data = np.log10(ratio).T  # rows = terms, cols = param
    vmax = np.nanmax(np.abs(data))
    reliable = _significant_effect_mask(data, residuals)
    shown = np.ma.masked_where(~reliable, data)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#f0f0f0")

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        shown,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_yticks(range(len(C.DENSE_TERMS)))
    ax.set_yticklabels([f"Z{t}" for t in C.DENSE_TERMS])
    ax.set_xticks(range(len(param_values)))
    ax.set_xticklabels([xtick_fmt.format(v) for v in param_values], rotation=45)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Zernike (Noll index)")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(
        "$\\log_{10}$(error / baseline)   "
        "[red = worse, blue = better]\n"
        "muted = not significant or too small"
    )
    muted = reliable.size - int(np.count_nonzero(reliable))
    print(f"  muted {muted}/{reliable.size} heatmap cells by MC thresholds")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    plt.close(fig)
