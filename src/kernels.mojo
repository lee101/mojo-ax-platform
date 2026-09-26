"""Experiment-statistics kernels for the `ax.utils.stats` subset.

Every exported symbol takes buffer addresses as plain `Int` values and rebuilds
the pointer inside the body, because `@export` rejects parametric functions and
an inferred pointer origin would make the symbol parametric.

The loops here are the numeric bodies of `ax.utils.stats.statstools` and
`ax.utils.stats.math_utils`. Policy (argument validation, the "conflicting
noiseless" branch, the control-epsilon bail-out) stays in the Python shim;
these kernels only do the arithmetic.
"""

from std.math import fma, sqrt

comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


@export("ax_ivw_partials")
def ax_ivw_partials(means_addr: Int, var_addr: Int, n: Int, out_addr: Int) abi("C"):
    """Partial sums for `ax.utils.stats.statstools.inverse_variance_weight`.

    One pass accumulates the inverse-variance weight sum and the weighted mean
    numerator over strictly-positive variances; a second pass computes the
    population variance of the zero-variance subset with the same
    mean-of-squared-deviations form NumPy uses, so the caller can reproduce the
    `np.var(means_z) > 0` conflict test without re-scanning the array.

    out layout (Float64, 6 slots):
        0 sum(1 / var_i)          over var_i != 0
        1 sum(mean_i / var_i)     over var_i != 0
        2 count of var_i == 0
        3 sum(mean_i)             over var_i == 0
        4 sum((mean_i - mean_z)^2) over var_i == 0
        5 mean_z (0.0 when the subset is empty)
    """
    var means = fp(means_addr)
    var variances = fp(var_addr)
    var out = fp(out_addr)

    var s_inv = Float64(0.0)
    var s_dot = Float64(0.0)
    var zero_count = Float64(0.0)
    var zero_sum = Float64(0.0)
    for i in range(n):
        var m = means[unsafe_offset=i]
        var v = variances[unsafe_offset=i]
        if v == 0.0:
            zero_count += 1.0
            zero_sum += m
        else:
            var iv = 1.0 / v
            s_inv += iv
            s_dot = fma(iv, m, s_dot)

    var zero_mean = Float64(0.0)
    var zero_sumsq = Float64(0.0)
    if zero_count > 0.0:
        zero_mean = zero_sum / zero_count
        for i in range(n):
            if variances[unsafe_offset=i] == 0.0:
                var d = means[unsafe_offset=i] - zero_mean
                zero_sumsq = fma(d, d, zero_sumsq)

    out[unsafe_offset=0] = s_inv
    out[unsafe_offset=1] = s_dot
    out[unsafe_offset=2] = zero_count
    out[unsafe_offset=3] = zero_sum
    out[unsafe_offset=4] = zero_sumsq
    out[unsafe_offset=5] = zero_mean


@export("ax_total_variance")
def ax_total_variance(
    means_addr: Int, var_addr: Int, size_addr: Int, n: Int, out_addr: Int
) abi("C"):
    """`ax.utils.stats.statstools.total_variance`.

    out layout (Float64, 4 slots):
        0 sum(sample_sizes)
        1 sum(sample_sizes * (mean - mean_bar)^2)
        2 sum(sample_sizes * (variance * sample_size))
    """
    var means = fp(means_addr)
    var variances = fp(var_addr)
    var sizes = fp(size_addr)
    var out = fp(out_addr)

    var mean_bar = Float64(0.0)
    for i in range(n):
        mean_bar += means[unsafe_offset=i]
    mean_bar /= Float64(n)

    var sum_size = Float64(0.0)
    var wsum_sq = Float64(0.0)
    var wsum_var = Float64(0.0)
    for i in range(n):
        var m = means[unsafe_offset=i]
        var s = sizes[unsafe_offset=i]
        var d = m - mean_bar
        sum_size += s
        wsum_sq = fma(s, d * d, wsum_sq)
        wsum_var = fma(s, variances[unsafe_offset=i] * s, wsum_var)

    out[unsafe_offset=0] = sum_size
    out[unsafe_offset=1] = wsum_sq
    out[unsafe_offset=2] = wsum_var


@export("ax_james_stein")
def ax_james_stein(
    means_addr: Int, sems_addr: Int, k: Int, mu_addr: Int, sig_addr: Int
) abi("C"):
    """`ax.utils.stats.statstools.positive_part_james_stein`.

    Two passes: the first fixes the grand mean and the ddof=3 variance, the
    second forms the shrinkage factor and the posterior mean/sem for each arm.
    """
    var means = fp(means_addr)
    var sems = fp(sems_addr)
    var mu = fp(mu_addr)
    var sig = fp(sig_addr)

    var kf = Float64(k)
    var ybar = Float64(0.0)
    for i in range(k):
        ybar += means[unsafe_offset=i]
    ybar /= kf

    var s2 = Float64(0.0)
    for i in range(k):
        var d = means[unsafe_offset=i] - ybar
        s2 = fma(d, d, s2)
    s2 /= Float64(k - 3)

    for i in range(k):
        var y = means[unsafe_offset=i]
        var sigma2 = sems[unsafe_offset=i] * sems[unsafe_offset=i]
        var phi = Float64(1.0)
        if s2 != 0.0:
            phi = min(1.0, sigma2 / s2)
        var dev = y - ybar
        mu[unsafe_offset=i] = y + phi * (ybar - y)
        var acc = (1.0 - phi) * sigma2 + phi * sigma2 / kf
        acc += 2.0 * phi * phi * dev * dev / Float64(k - 3)
        sig[unsafe_offset=i] = sqrt(acc)


@export("ax_agresti_coull_sem")
def ax_agresti_coull_sem(
    numer_addr: Int, denom_addr: Int, n: Int, prior_succ: Float64,
    prior_fail: Float64, out_addr: Int
) abi("C"):
    """`ax.utils.stats.statstools.agresti_coull_sem`, elementwise."""
    var numer = fp(numer_addr)
    var denom = fp(denom_addr)
    var out = fp(out_addr)
    var total_prior = prior_succ + prior_fail
    for i in range(n):
        var p = (numer[unsafe_offset=i] + prior_succ) / (
            denom[unsafe_offset=i] + total_prior
        )
        out[unsafe_offset=i] = sqrt(p * (1.0 - p) / denom[unsafe_offset=i])



@export("ax_relativize")
def ax_relativize(
    means_t_addr: Int, sems_t_addr: Int, cov_addr: Int, n: Int, mean_c: Float64,
    sem_c: Float64, bias_correction: Int, control_as_constant: Int,
    out_r_addr: Int, out_s_addr: Int
) abi("C"):
    """`ax.utils.stats.math_utils.relativize`, elementwise over the arms.

    `control_as_constant` and `bias_correction` are passed as Int flags because
    an exported `bool` is awkward across the C ABI; any nonzero value selects
    the corresponding branch. The caller resolves `np.all(np.isnan(sem_c))`
    and the `MEAN_CONTROL_EPSILON` bail-out, which are scalar policy.
    """
    var means_t = fp(means_t_addr)
    var sems_t = fp(sems_t_addr)
    var cov_t = fp(cov_addr)
    var out_r = fp(out_r_addr)
    var out_s = fp(out_s_addr)

    var abs_mean_c = abs_value(mean_c)
    for i in range(n):
        var m = means_t[unsafe_offset=i]
        var s = sems_t[unsafe_offset=i]
        var r = (m - mean_c) / abs_mean_c
        var var_t: Float64
        if control_as_constant != 0:
            var_t = (s / abs_mean_c) ** 2
        else:
            if bias_correction != 0:
                r = r - m * sem_c * sem_c / (abs_mean_c * abs_mean_c * abs_mean_c)
            if (m == mean_c) and (s == sem_c):
                r = 0.0
            var c = m / mean_c
            var cs = sem_c * sem_c
            var_t = ((s * s) - 2.0 * c * cov_t[unsafe_offset=i] + (c * c) * cs) / (
                mean_c * mean_c
            )
        out_r[unsafe_offset=i] = r
        out_s[unsafe_offset=i] = sqrt(var_t)

@export("ax_unrelativize")
def ax_unrelativize(
    means_t_addr: Int, sems_t_addr: Int, cov_addr: Int, n: Int, mean_c: Float64,
    sem_c: Float64, bias_correction: Int, control_as_constant: Int,
    out_m_addr: Int, out_s_addr: Int
) abi("C"):
    """`ax.utils.stats.math_utils.unrelativize`, elementwise over the arms.

    Includes the two pieces of upstream policy that are elementwise rather than
    scalar: the `clip(min=0)` on the recovered variance and the exact
    `means_t == 0.0` short circuit back to the control mean and sem.
    """
    var means_t = fp(means_t_addr)
    var sems_t = fp(sems_t_addr)
    var cov_t = fp(cov_addr)
    var out_m = fp(out_m_addr)
    var out_s = fp(out_s_addr)

    var abs_mean_c = abs_value(mean_c)
    for i in range(n):
        var mt = means_t[unsafe_offset=i]
        var st = sems_t[unsafe_offset=i]
        var m = mt * abs_mean_c + mean_c
        var s: Float64
        if control_as_constant != 0:
            s = st * abs_mean_c
        else:
            if bias_correction != 0:
                var ratio = sem_c / abs_mean_c
                m = m / (1.0 - ratio * ratio)
            var c = m / mean_c
            var s_t2 = st * st * (mean_c * mean_c) + 2.0 * c * cov_t[
                unsafe_offset=i
            ] - (c * c) * (sem_c * sem_c)
            s = sqrt(max(s_t2, 0.0))

        if mt == 0.0:
            out_m[unsafe_offset=i] = mean_c
            out_s[unsafe_offset=i] = sem_c
        else:
            out_m[unsafe_offset=i] = m
            out_s[unsafe_offset=i] = s

def abs_value(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x
