"""Correctness-gated benchmark for mojo-ax-platform.

Every case checks numerical agreement with the vectorised NumPy reference
before timing, so a regression in the Mojo kernels shows up as a correctness
failure rather than a suspiciously good number.

The baselines are the fastest reasonable NumPy formulations of the same
arithmetic, which is also roughly what `ax.utils.stats` runs: upstream is
already vectorised, so the honest expectation for the two-pass reduction
kernels is parity or a small win, not a landslide.
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import mojo_ax_platform as max_  # noqa: E402

RTOL = 1e-11
# Mojo sums ybar sequentially where NumPy sums pairwise, so at k=1e6 the grand
# mean itself differs in the last ~1e-14; anything derived from it inherits
# that absolute floor.
ATOL = 1e-12


def _time(fn, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _ivw_numpy(means, variances):
    inv = 1.0 / variances
    s_inv = inv.sum()
    return float(np.inner(inv, means) / s_inv), float(1.0 / s_inv)


def bench_ivw(k: int = 1 << 20):
    rng = np.random.default_rng(0)
    means = rng.normal(10.0, 3.0, size=k)
    variances = rng.uniform(0.05, 5.0, size=k)

    ref_mean, ref_var = _ivw_numpy(means, variances)
    got_mean, got_var = max_.inverse_variance_weight(means, variances)
    np.testing.assert_allclose(got_mean, ref_mean, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(got_var, ref_var, rtol=RTOL, atol=ATOL)

    return (
        f"inverse_variance_weight k={k}",
        _time(lambda: _ivw_numpy(means, variances)),
        _time(lambda: max_.inverse_variance_weight(means, variances)),
    )


def _total_variance_numpy(means, variances, sizes):
    variances = variances * sizes
    w1 = np.average((means - means.mean()) ** 2, weights=sizes)
    w2 = np.average(variances, weights=sizes)
    return float((w1 + w2) / sizes.sum())


def bench_total_variance(k: int = 1 << 20):
    rng = np.random.default_rng(1)
    means = rng.normal(0.0, 1.0, size=k)
    variances = rng.uniform(0.01, 0.5, size=k)
    sizes = rng.integers(5, 200, size=k).astype(np.float64)

    np.testing.assert_allclose(
        max_.total_variance(means, variances, sizes),
        _total_variance_numpy(means, variances, sizes),
        rtol=RTOL,
    )

    return (
        f"total_variance k={k}",
        _time(lambda: _total_variance_numpy(means, variances, sizes)),
        _time(lambda: max_.total_variance(means, variances, sizes)),
    )


def _james_stein_numpy(means, sems):
    sigma2 = np.power(sems, 2)
    ybar = np.mean(means)
    s2 = np.var(means - ybar, ddof=3)
    phi = np.ones_like(sigma2) if s2 == 0 else np.minimum(1, sigma2 / s2)
    mu = means + phi * np.subtract(ybar, means)
    sig = np.sqrt(
        np.subtract(1.0, phi) * sigma2
        + phi * sigma2 / len(means)
        + np.multiply(2, phi**2) * (means - ybar) ** 2 / (len(means) - 3)
    )
    return mu, sig


def bench_james_stein(k: int = 1 << 20):
    rng = np.random.default_rng(2)
    means = rng.normal(1.0, 1.0, size=k)
    sems = rng.uniform(0.05, 0.6, size=k)

    want_mu, want_sig = _james_stein_numpy(means, sems)
    mu, sig = max_.positive_part_james_stein(means, sems)
    np.testing.assert_allclose(mu, want_mu, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(sig, want_sig, rtol=RTOL, atol=ATOL)

    return (
        f"positive_part_james_stein k={k}",
        _time(lambda: _james_stein_numpy(means, sems)),
        _time(lambda: max_.positive_part_james_stein(means, sems)),
    )


def _relativize_numpy(m_t, s_t, mean_c, sem_c, cov):
    abs_mean_c = abs(mean_c)
    r = (m_t - mean_c) / abs_mean_c - m_t * sem_c**2 / abs_mean_c**3
    c = m_t / mean_c
    var = ((s_t**2) - 2 * c * cov + (c**2) * (sem_c**2)) / (mean_c**2)
    return r, np.sqrt(var)


def bench_relativize(k: int = 1 << 20):
    rng = np.random.default_rng(3)
    m_t = rng.uniform(1.0, 20.0, size=k)
    s_t = rng.uniform(0.01, 0.5, size=k)
    cov = rng.normal(0.0, 0.001, size=k)
    mean_c, sem_c = 12.0, 0.25

    want_r, want_s = _relativize_numpy(m_t, s_t, mean_c, sem_c, cov)
    assert np.all(want_s > 0.0), "reference produced a non-positive variance"
    got_r, got_s = max_.relativize(m_t, s_t, mean_c, sem_c, cov_means=cov)
    np.testing.assert_allclose(got_r, want_r, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(got_s, want_s, rtol=RTOL, atol=ATOL)

    return (
        f"relativize k={k}",
        _time(lambda: _relativize_numpy(m_t, s_t, mean_c, sem_c, cov)),
        _time(lambda: max_.relativize(m_t, s_t, mean_c, sem_c, cov_means=cov)),
    )


def bench_agresti(k: int = 1 << 20):
    rng = np.random.default_rng(4)
    numer = rng.integers(0, 500, size=k).astype(np.float64)
    denom = rng.integers(500, 1000, size=k).astype(np.float64)

    def numpy_sem():
        p = (numer + 2.0) / (denom + 4.0)
        return np.sqrt(p * (1.0 - p) / denom)

    np.testing.assert_allclose(
        max_.agresti_coull_sem(numer, denom), numpy_sem(), rtol=RTOL
    )
    return (
        f"agresti_coull_sem k={k}",
        _time(numpy_sem),
        _time(lambda: max_.agresti_coull_sem(numer, denom)),
    )


def main():
    print(f"{'case':<36}{'numpy':>12}{'mojo-ax-platform':>20}{'ratio':>10}")
    print("-" * 78)
    for fn in (
        bench_ivw,
        bench_total_variance,
        bench_james_stein,
        bench_relativize,
        bench_agresti,
    ):
        label, ref, got = fn()
        ratio = ref / got if got else float("nan")
        print(f"{label:<36}{ref*1e3:>10.2f}ms{got*1e3:>18.2f}ms{ratio:>9.2f}x")


if __name__ == "__main__":
    main()
