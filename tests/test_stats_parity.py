"""Parity tests against the real `ax.utils.stats`, module by module.

Tolerances: every one of these functions is a sum of float64 products, and
Mojo emits FMA where NumPy multiplies and adds separately, so bit-exact
agreement is impossible by construction. rtol=1e-12 is many orders of magnitude
tighter than the accumulated roundoff at these magnitudes, and still ~1e4x
looser than the observed difference.
"""

import numpy as np
import pytest

import mojo_ax_platform as max_

ax_stats = pytest.importorskip("ax.utils.stats.statstools")
ax_math = pytest.importorskip("ax.utils.stats.math_utils")

RTOL = 1e-12
ATOL = 1e-12


# ------------------------------------------------------------------ ivw


def test_ivw_matches_ax_on_unequal_variances():
    rng = np.random.default_rng(11)
    means = rng.normal(5.0, 2.0, size=37)
    variances = rng.uniform(0.1, 4.0, size=37)
    got = max_.inverse_variance_weight(means, variances)
    want = ax_stats.inverse_variance_weight(means, variances)
    np.testing.assert_allclose(got[0], want[0], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(got[1], want[1], rtol=RTOL, atol=ATOL)


def test_ivw_weighting_is_not_the_plain_mean():
    """A dropped 1/var weight would silently make this equal to np.mean."""
    means = np.array([1.0, 2.0, 30.0])
    variances = np.array([1.0, 1.0, 1e-8])
    got, _ = max_.inverse_variance_weight(means, variances)
    want, _ = ax_stats.inverse_variance_weight(means, variances)
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL)
    assert abs(got - means.mean()) > 1.0


def test_ivw_zero_variance_branch_matches_ax():
    means = np.array([4.0, 5.0, 100.0])
    variances = np.array([0.0, 2.0, 0.0])
    got = max_.inverse_variance_weight(means, variances, "ignore")
    want = ax_stats.inverse_variance_weight(means, variances, "ignore")
    np.testing.assert_allclose(got[0], want[0], rtol=RTOL, atol=ATOL)
    assert got[1] == 0.0


def test_ivw_zero_variance_branch_averages_only_noiseless():
    means = np.array([4.0, 500.0, 100.0])
    variances = np.array([0.0, 2.0, 0.0])
    got, got_var = max_.inverse_variance_weight(means, variances, "ignore")
    assert got == pytest.approx(52.0, rel=1e-12)
    assert got_var == 0.0


def test_ivw_identical_noiseless_means_do_not_raise():
    means = np.array([7.0, 9.0, 7.0])
    variances = np.array([0.0, 1.0, 0.0])
    got = max_.inverse_variance_weight(means, variances, "raise")
    want = ax_stats.inverse_variance_weight(means, variances, "raise")
    np.testing.assert_allclose(got[0], want[0], rtol=RTOL, atol=ATOL)


def test_ivw_conflicting_noiseless_raises():
    means = np.array([7.0, 8.0, 3.0])
    variances = np.array([0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="zero variance"):
        max_.inverse_variance_weight(means, variances, "raise")


def test_ivw_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        max_.inverse_variance_weight(np.zeros(3), np.ones(4))


def test_ivw_rejects_unknown_noiseless_option():
    with pytest.raises(ValueError, match="Unsupported option"):
        max_.inverse_variance_weight(np.zeros(2), np.ones(2), "explode")


# -------------------------------------------------------- total variance


def test_total_variance_matches_ax():
    rng = np.random.default_rng(3)
    means = rng.normal(0.0, 1.0, size=23)
    variances = rng.uniform(0.01, 0.5, size=23)
    sizes = rng.integers(5, 200, size=23).astype(np.float64)
    got = max_.total_variance(means, variances, sizes)
    want = ax_stats.total_variance(means, variances, sizes)
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL)


def test_total_variance_detects_a_missing_sample_size_weight():
    """Dropping the size weights or the 1/sum(n) normalisation shows up here."""
    means = np.array([1.0, 2.0, 3.0, 10.0])
    variances = np.array([0.1, 0.2, 0.3, 0.4])
    sizes = np.array([10.0, 10.0, 10.0, 100.0])
    got = max_.total_variance(means, variances, sizes)
    want = ax_stats.total_variance(means, variances, sizes)
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL)
    unweighted = np.mean((means - means.mean()) ** 2) + np.mean(variances)
    assert abs(got - unweighted) > 1e-3


def test_total_variance_zero_variance_all():
    means = np.full(5, 2.0)
    variances = np.zeros(5)
    sizes = np.full(5, 7.0)
    got = max_.total_variance(means, variances, sizes)
    want = ax_stats.total_variance(means, variances, sizes)
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL)
    assert got == 0.0


# ------------------------------------------------------------ james-stein


def test_james_stein_matches_ax():
    rng = np.random.default_rng(7)
    means = rng.normal(1.0, 1.0, size=9)
    sems = rng.uniform(0.05, 0.6, size=9)
    mu, sig = max_.positive_part_james_stein(means, sems)
    want_mu, want_sig = ax_stats.positive_part_james_stein(means, sems)
    np.testing.assert_allclose(mu, want_mu, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(sig, want_sig, rtol=RTOL, atol=ATOL)


def test_james_stein_shrinks_hard_when_sems_dominate_spread():
    """phi = min(1, sigma2_i / s2); a large sem against a tight cluster of
    means drives phi to 1, so every arm collapses onto the grand mean."""
    rng = np.random.default_rng(8)
    means = 5.0 + rng.normal(0.0, 1e-3, size=12)
    sems = np.full(12, 1.0)
    mu, _ = max_.positive_part_james_stein(means, sems)
    assert np.allclose(mu, means.mean(), atol=1e-9)
    want_mu, _ = ax_stats.positive_part_james_stein(means, sems)
    np.testing.assert_allclose(mu, want_mu, rtol=RTOL, atol=ATOL)


def test_james_stein_shrinks_partially_in_between():
    """phi strictly between 0 and 1 leaves mu between y_i and the grand mean."""
    rng = np.random.default_rng(41)
    means = 5.0 + rng.normal(0.0, 0.5, size=10)
    sems = np.full(10, 0.25)
    mu, _ = max_.positive_part_james_stein(means, sems)
    ybar = means.mean()
    assert np.all((mu - ybar) * (means - ybar) >= 0.0)
    assert not np.allclose(mu, means)
    assert not np.allclose(mu, ybar)


def test_james_stein_no_shrinkage_when_arms_differ():
    """phi -> 0 for a precise arm in a spread-out set, so mu stays at y_i; a
    hardcoded phi = 1 would collapse everything onto the grand mean."""
    means = np.array([0.0, 20.0, 40.0, 60.0, 80.0, 100.0])
    sems = np.array([1e-6] * 6)
    mu, _ = max_.positive_part_james_stein(means, sems)
    np.testing.assert_allclose(mu, means, rtol=1e-6, atol=1e-9)
    want_mu, _ = ax_stats.positive_part_james_stein(means, sems)
    np.testing.assert_allclose(mu, want_mu, rtol=RTOL, atol=ATOL)


def test_james_stein_posterior_sem_shrinks():
    """With phi = 1 the posterior sem collapses to sigma / sqrt(K)."""
    means = 5.0 + np.linspace(-0.001, 0.001, 6)
    sems = np.full(6, 2.0)
    _, sig = max_.positive_part_james_stein(means, sems)
    assert np.all(sig < 0.5 * sems)
    want_mu, want_sig = ax_stats.positive_part_james_stein(means, sems)
    np.testing.assert_allclose(sig, want_sig, rtol=RTOL, atol=ATOL)

def test_james_stein_rejects_few_arms():
    with pytest.raises(ValueError, match="Less than 4"):
        max_.positive_part_james_stein(np.zeros(3), np.ones(3))


def test_james_stein_rejects_negative_sems():
    with pytest.raises(ValueError, match="negative"):
        max_.positive_part_james_stein(
            np.zeros(5), np.array([1.0, 1.0, 1.0, 1.0, -1.0])
        )


# ---------------------------------------------------------- agresti-coull


def test_agresti_coull_matches_ax_array():
    numer = np.array([10, 30, 0, 97, 5])
    denom = np.array([100, 100, 100, 100, 20])
    got = max_.agresti_coull_sem(numer, denom)
    want = ax_stats.agresti_coull_sem(numer, denom)
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL)


def test_agresti_coull_matches_ax_scalar():
    got = max_.agresti_coull_sem(37, 41)
    want = ax_stats.agresti_coull_sem(37, 41)
    assert isinstance(got, float)
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL)


def test_agresti_coull_respects_priors():
    """A hardcoded 2+2 prior would make this identical to the default call."""
    numer = np.array([10, 30])
    denom = np.array([100, 100])
    got = max_.agresti_coull_sem(numer, denom, prior_successes=5,
                                 prior_failures=7)
    want = ax_stats.agresti_coull_sem(numer, denom, prior_successes=5,
                                      prior_failures=7)
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL)
    assert not np.allclose(got, max_.agresti_coull_sem(numer, denom))


# ------------------------------------------------------------- relativize


def _relativize_case(n=6, seed=5):
    rng = np.random.default_rng(seed)
    m_t = rng.uniform(1.0, 20.0, size=n)
    s_t = rng.uniform(0.01, 0.5, size=n)
    cov = rng.normal(0.0, 0.01, size=n)
    return m_t, s_t, 12.0, 0.25, cov


@pytest.mark.parametrize("as_percent", [False, True])
@pytest.mark.parametrize("constant", [False, True])
def test_relativize_matches_ax(as_percent, constant):
    m, s, mc, sc, cov = _relativize_case()
    got = max_.relativize(m, s, mc, sc, cov_means=cov, as_percent=as_percent,
                          control_as_constant=constant)
    want = ax_math.relativize(m, s, mc, sc, cov_means=cov,
                              as_percent=as_percent,
                              control_as_constant=constant)
    np.testing.assert_allclose(got[0], want[0], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(got[1], want[1], rtol=RTOL, atol=ATOL)


def test_relativize_equals_the_well_known_ratio():
    m, s, mc, sc, cov = _relativize_case(seed=17)
    r, sr = max_.relativize(m, s, mc, sc, bias_correction=False, cov_means=cov)
    np.testing.assert_allclose(r, (m - mc) / mc, rtol=RTOL, atol=ATOL)
    assert np.all(sr > 0.0)


def test_relativize_bias_correction_differs_from_none():
    """The second-order bias term is a real term; dropping it is a real bug."""
    m, s, mc, sc, cov = _relativize_case(seed=19)
    with_bias, _ = max_.relativize(m, s, mc, sc, cov_means=cov,
                                   bias_correction=True)
    without, _ = max_.relativize(m, s, mc, sc, cov_means=cov,
                                 bias_correction=False)
    assert not np.allclose(with_bias, without)
    want_with, _ = ax_math.relativize(m, s, mc, sc, cov_means=cov,
                                      bias_correction=True)
    np.testing.assert_allclose(with_bias, want_with, rtol=RTOL, atol=ATOL)


def test_relativize_zeroes_identical_arm():
    m, s, mc, sc, _ = _relativize_case(seed=23)
    m = m.copy()
    s = s.copy()
    m[2] = mc
    s[2] = sc
    zero_cov = np.zeros_like(m)
    r, _ = max_.relativize(m, s, mc, sc, cov_means=zero_cov)
    want_r, _ = ax_math.relativize(m, s, mc, sc, cov_means=zero_cov)
    assert r[2] == 0.0
    np.testing.assert_allclose(r, want_r, rtol=RTOL, atol=ATOL)


def test_relativize_negative_control_uses_absolute_value():
    m, s, mc, sc, cov = _relativize_case(seed=29)
    r, _ = max_.relativize(m, s, -mc, sc, cov_means=cov, bias_correction=False)
    np.testing.assert_allclose(r, (m + mc) / mc, rtol=RTOL, atol=ATOL)


def test_relativize_rejects_tiny_control():
    with pytest.raises(ValueError, match="1 in 10 billion"):
        max_.relativize(np.array([1.0]), np.array([0.1]), 1e-12, 0.01)


def test_relativize_rejects_array_control():
    with pytest.raises(ValueError, match="scalar mean_c"):
        max_.relativize(np.array([1.0, 2.0]), np.array([0.1, 0.2]),
                        np.array([1.0, 1.0]), 0.1)


# ---------------------------------------------------------- unrelativize


def test_unrelativize_matches_ax():
    m, s, mc, sc, cov = _relativize_case(seed=31)
    m_t = m / mc - 1.0
    s_t = s / mc
    got = max_.unrelativize(m_t, s_t, mc, sc, cov_means=cov)
    want = ax_math.unrelativize(m_t, s_t, mc, sc, cov_means=cov)
    np.testing.assert_allclose(got[0], want[0], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(got[1], want[1], rtol=RTOL, atol=ATOL)


def test_relativize_unrelativize_round_trip():
    m, s, mc, sc, _ = _relativize_case(seed=37)
    r, sr = max_.relativize(m, s, mc, sc, cov_means=np.zeros_like(m),
                            bias_correction=False)
    back_m, back_s = max_.unrelativize(r, sr, mc, sc, bias_correction=False)
    np.testing.assert_allclose(back_m, m, rtol=1e-9, atol=1e-9)
    assert np.all(back_s > 0.0)


def test_unrelativize_zero_ratio_returns_control():
    m_t = np.array([0.0, 0.1, -0.2])
    s_t = np.array([0.01, 0.02, 0.03])
    m, s = max_.unrelativize(m_t, s_t, 5.0, 0.4)
    assert m[0] == 5.0
    assert s[0] == 0.4
    want_m, want_s = ax_math.unrelativize(m_t, s_t, 5.0, 0.4)
    np.testing.assert_allclose(m, want_m, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(s, want_s, rtol=RTOL, atol=ATOL)


def test_unrelativize_clips_negative_variance():
    """Upstream clips the recovered variance at 0; without it we would get NaN."""
    m_t = np.array([0.001])
    s_t = np.array([1e-9])
    m, s = max_.unrelativize(m_t, s_t, 1.0, 0.5)
    assert np.all(np.isfinite(s))
    assert s[0] >= 0.0
    want_m, want_s = ax_math.unrelativize(m_t, s_t, 1.0, 0.5)
    np.testing.assert_allclose(m, want_m, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(s, want_s, rtol=RTOL, atol=ATOL)


def test_unrelativize_control_as_constant_ignores_sem_c():
    m_t = np.array([0.2, -0.1])
    s_t = np.array([0.01, 0.02])
    m, s = max_.unrelativize(m_t, s_t, 8.0, 0.9, control_as_constant=True)
    np.testing.assert_allclose(s, s_t * 8.0, rtol=RTOL, atol=ATOL)
    want_m, want_s = ax_math.unrelativize(m_t, s_t, 8.0, 0.9,
                                          control_as_constant=True)
    np.testing.assert_allclose(m, want_m, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(s, want_s, rtol=RTOL, atol=ATOL)


def test_unrelativize_as_percent_divides_inputs_only():
    """Upstream divides by 100 on input and does not scale the outputs back."""
    m_t = np.array([20.0, -10.0])
    s_t = np.array([1.0, 2.0])
    m_pct, _ = max_.unrelativize(m_t, s_t, 5.0, 0.4, as_percent=True)
    m_raw, _ = max_.unrelativize(m_t / 100.0, s_t / 100.0, 5.0, 0.4)
    np.testing.assert_allclose(m_pct, m_raw, rtol=RTOL, atol=ATOL)
    want_m, _ = ax_math.unrelativize(m_t, s_t, 5.0, 0.4, as_percent=True)
    np.testing.assert_allclose(m_pct, want_m, rtol=RTOL, atol=ATOL)
