"""Statistics helpers for the `ax.utils.stats` subset, on Mojo kernels.

Every function keeps the argument shape and return type of its
`ax.utils.stats` counterpart, so the two can be compared directly. The
arithmetic over arrays lives in `src/kernels.mojo`; argument validation and
the scalar branches stay here because they are policy, not compute.

Deliberate difference from upstream: `relativize` and `unrelativize` require a
scalar control (`mean_c`, `sem_c`). Upstream also accepts arrays there and lets
NumPy broadcast, which would mean a strided control array inside the kernel.
"""

from __future__ import annotations

import logging

import numpy as np

from . import _lib

logger = logging.getLogger(__name__)

# Minimum absolute value for a control mean to be considered non-zero for
# relativization via the delta method. Same constant as ax.
MEAN_CONTROL_EPSILON: float = 1e-10

_NOISELESS_OPTIONS = {"warn", "ignore", "raise"}
_CONFLICT = "Multiple observations zero variance but different means."


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _is_scalar(x) -> bool:
    return np.ndim(x) == 0


def _broadcast_cov(cov_means, shape) -> np.ndarray:
    cov = _arr(cov_means)
    return np.ascontiguousarray(
        np.broadcast_to(cov, np.broadcast_shapes(shape, cov.shape))
    )


def inverse_variance_weight(
    means, variances, conflicting_noiseless: str = "warn"
) -> tuple[float, float]:
    """Inverse-variance weighted mean and variance.

    Mirrors `ax.utils.stats.statstools.inverse_variance_weight`, including the
    zero-variance branch: when any observation has zero variance the estimator
    degenerates to the plain mean of the noiseless observations and a variance
    of 0.
    """
    if conflicting_noiseless not in _NOISELESS_OPTIONS:
        raise ValueError(
            f"Unsupported option `{conflicting_noiseless}` for "
            f"conflicting_noiseless."
        )
    means_a = _arr(means)
    variances_a = _arr(variances)
    if means_a.shape != variances_a.shape:
        raise ValueError("Means and variances must be of the same length.")

    parts = _lib.ivw_partials(means_a.ravel(), variances_a.ravel())
    sum_inv_var, dot_inv_mean, zero_count, _, zero_sumsq, zero_mean = parts

    if zero_count > 0.0:
        # Population variance of the noiseless subset, in the same
        # mean-of-squared-deviations form np.var uses.
        var_z = zero_sumsq / zero_count
        if var_z > 0.0:
            if conflicting_noiseless == "warn":
                logger.warning(_CONFLICT)
            elif conflicting_noiseless == "raise":
                raise ValueError(_CONFLICT)
        return float(zero_mean), 0.0

    return float(dot_inv_mean / sum_inv_var), float(1.0 / sum_inv_var)


def total_variance(means, variances, sample_sizes) -> float:
    """`ax.utils.stats.statstools.total_variance`."""
    means_a = _arr(means).ravel()
    variances_a = _arr(variances).ravel()
    sizes_a = _arr(sample_sizes).ravel()
    sum_size, wsum_sq, wsum_var = _lib.total_variance_parts(
        means_a, variances_a, sizes_a
    )
    return float((wsum_sq / sum_size + wsum_var / sum_size) / sum_size)


def positive_part_james_stein(means, sems) -> tuple[np.ndarray, np.ndarray]:
    """`ax.utils.stats.statstools.positive_part_james_stein`."""
    means_a = _arr(means).ravel()
    sems_a = _arr(sems).ravel()
    if sems_a.size < 4:
        raise ValueError(
            "Less than 4 measurements passed to positive_part_james_stein. "
            + "Returning raw estimates."
        )
    if sems_a.min() < 0:
        raise ValueError("sems cannot be negative.")
    mu, sig = _lib.positive_part_james_stein(means_a, sems_a)
    return mu, sig


def agresti_coull_sem(
    n_numer, n_denom, prior_successes: int = 2, prior_failures: int = 2
):
    """`ax.utils.stats.statstools.agresti_coull_sem`."""
    numer = _arr(n_numer)
    denom = _arr(n_denom)
    shape = np.broadcast_shapes(numer.shape, denom.shape)
    sem = _lib.agresti_coull_sem(
        np.ascontiguousarray(np.broadcast_to(numer, shape)),
        np.ascontiguousarray(np.broadcast_to(denom, shape)),
        prior_successes,
        prior_failures,
    )
    if _is_scalar(n_numer) and _is_scalar(n_denom):
        return float(sem[0])
    return sem


def relativize(
    means_t,
    sems_t,
    mean_c,
    sem_c,
    bias_correction: bool = True,
    cov_means=0.0,
    as_percent: bool = False,
    control_as_constant: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """`ax.utils.stats.math_utils.relativize` with a scalar control."""
    if not _is_scalar(mean_c) or not _is_scalar(sem_c):
        raise ValueError(
            "mojo_ax_platform.relativize requires a scalar mean_c and sem_c; "
            "use ax.utils.stats.math_utils.relativize for array controls."
        )
    if np.any(np.abs(_arr(mean_c)) < MEAN_CONTROL_EPSILON):
        raise ValueError(
            f"mean_control ({mean_c} +/- {sem_c}) is smaller than 1 in 10 "
            f"billion, which is too small to reliably analyze ratios using "
            f"the delta method."
        )
    m_t = _arr(means_t)
    s_t = _arr(sems_t)
    cov = _broadcast_cov(cov_means, np.broadcast_shapes(m_t.shape, s_t.shape))
    # Upstream skips the bias correction when the control sem is NaN.
    effective_bias = bool(bias_correction) and not np.all(np.isnan(sem_c))
    r_hat, s_hat = _lib.relativize(
        np.ascontiguousarray(m_t), np.ascontiguousarray(s_t), float(mean_c),
        float(sem_c), effective_bias, cov, control_as_constant,
    )
    if as_percent:
        return r_hat * 100.0, s_hat * 100.0
    return r_hat, s_hat


def unrelativize(
    means_t,
    sems_t,
    mean_c,
    sem_c,
    bias_correction: bool = True,
    cov_means=0.0,
    as_percent: bool = False,
    control_as_constant: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """`ax.utils.stats.math_utils.unrelativize` with a scalar control.

    Mirrors upstream exactly on the `as_percent` rescaling: it divides the
    inputs by 100 and does not scale the outputs back.
    """
    if not _is_scalar(mean_c) or not _is_scalar(sem_c):
        raise ValueError(
            "mojo_ax_platform.unrelativize requires a scalar mean_c and sem_c; "
            "use ax.utils.stats.math_utils.unrelativize for array controls."
        )
    m_t = _arr(means_t)
    s_t = _arr(sems_t)
    if as_percent:
        m_t = m_t / 100.0
        s_t = s_t / 100.0
    cov = _broadcast_cov(cov_means, np.broadcast_shapes(m_t.shape, s_t.shape))
    m, s = _lib.unrelativize(
        np.ascontiguousarray(m_t), np.ascontiguousarray(s_t), float(mean_c),
        float(sem_c), bool(bias_correction), cov, control_as_constant,
    )
    return m, s
