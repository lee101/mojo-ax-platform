# mojo-ax-platform

`mojo-ax-platform` is the compute-oriented subset of
[ax-platform](https://ax.dev/) (Meta's Bayesian optimization library) with the
array arithmetic of `ax.utils.stats` implemented in Mojo and callable from
Python.

The Python package is named `mojo_ax_platform`, so it installs alongside the
real `ax_platform` and the tests compare the two directly, function by
function, against `ax.utils.stats.statstools` and `ax.utils.stats.math_utils`.

```python
import numpy as np
import mojo_ax_platform as max_

rng = np.random.default_rng(0)
means = rng.normal(10.0, 3.0, size=8)
variances = rng.uniform(0.1, 4.0, size=8)

max_.inverse_variance_weight(means, variances)   # (10.7777..., 0.061387...)
max_.total_variance(means, variances, np.full(8, 100.0))  # 0.265438...
max_.positive_part_james_stein(means, np.sqrt(variances / 100.0))
max_.relativize(means, np.full(8, 0.1), 10.0, 0.2)  # (0.037304, -0.040016, ...)
```

## Why this subset

`ax-platform` is mostly orchestration: trial scheduling, storage backends,
`Client`/`Experiment` lifecycle, SQLAlchemy models, plotting. None of that is
numeric work. The genuinely compute-oriented part of the public API is
`ax.utils.stats`, which is where every experiment-analysis number Ax reports
comes from: inverse-variance pooling across repeated measurements, pooled
variance across sample sizes, the positive-part James-Stein empirical-Bayes
shrinkage, the Agresti-Coull binomial sem, and the delta-method
relativize/unrelativize pair. Those are all reductions and elementwise passes
over vectors of arms, which is exactly the shape that a compiled inner loop
serves well, so that is what is ported.

## Covered subset

| upstream function | Mojo kernel | what the kernel does |
| --- | --- | --- |
| `statstools.inverse_variance_weight` | `ax_ivw_partials` | one pass for `sum(1/var)` and `sum(mean/var)`, a second pass for the population variance of the zero-variance subset |
| `statstools.total_variance` | `ax_total_variance` | two passes: the mean, then the two size-weighted sums |
| `statstools.positive_part_james_stein` | `ax_james_stein` | two passes: `ybar` and the ddof=3 `s2`, then `phi_i`, `mu_hat_i`, `sigma_hat_i` |
| `statstools.agresti_coull_sem` | `ax_agresti_coull_sem` | elementwise over the successes array |
| `math_utils.relativize` | `ax_relativize` | elementwise: ratio, delta-method variance, second-order bias term, identical-arm zeroing |
| `math_utils.unrelativize` | `ax_unrelativize` | elementwise: inverse transform, `clip(min=0)` on the recovered variance, exact-zero short circuit back to the control |

### Not implemented

- `statstools.marginal_effects` — it is a `pandas` reshape/regression driver
  that happens to end in a formula call; there is no array loop to move.
- Array-valued control arguments to `relativize` / `unrelativize`. Upstream
  broadcasts `mean_c` and `sem_c` across arms; the kernel takes them as
  scalars, so the port raises `ValueError` for that case rather than silently
  narrowing. Use the real `ax.utils.stats.math_utils` there.
- Everything outside `ax.utils.stats`: `Client`, `Experiment`, `OptimizationConfig`,
  storages, `Analysis`, model fitting, runners, and the whole service layer.
  Those are control flow and IO.

The `conflicting_noiseless` warning/error branch, the `K < 4` guard, the
`MEAN_CONTROL_EPSILON` bail-out and the `np.all(np.isnan(sem_c))` check stay in
Python: they are scalar policy, not arithmetic, and moving them would not make
anything faster.

## Install

The repository pins its own Mojo toolchain:

```bash
pixi run build
pixi run test
```

`bash build/build.sh` compiles `src/kernels.mojo` with
`mojo build --emit shared-lib` into `dist/libmojo-ax-platform.so`. Set
`PYTHONPATH=python` when using the package outside a Pixi task.

## Performance

Best-of-five wall clock in one process, against vectorised NumPy running the
same arithmetic. Every case asserts numerical agreement first, so a kernel
regression surfaces as a failed assertion rather than a fast number. This box
is shared, so absolute times move between runs by a factor of two; the ratios
below are from a loaded run and are the conservative end of what was observed
across three runs.

| case | NumPy | mojo-ax-platform | result |
| --- | ---: | ---: | ---: |
| `inverse_variance_weight` k=1048576 | 72.71 ms | 12.47 ms | 5.83x |
| `total_variance` k=1048576 | 19.70 ms | 5.91 ms | 3.33x |
| `positive_part_james_stein` k=1048576 | 97.33 ms | 18.96 ms | 5.13x |
| `relativize` k=1048576 | 78.40 ms | 36.89 ms | 2.13x |
| `agresti_coull_sem` k=1048576 | 16.70 ms | 6.79 ms | 2.46x |

Upstream `ax` is itself already vectorised NumPy, so the baseline here is not
a strawman. The wins come from doing the whole reduction in one pass with no
intermediate arrays: `inverse_variance_weight` in NumPy materialises
`1/variances` (8 MB at k=2^20) and then runs two separate reductions over it,
while the kernel keeps everything in registers. The elementwise cases
(`relativize`, `agresti_coull_sem`) show the smaller, more honest margin that
you get when the operation is already bandwidth-bound.

Reproduce with:

```bash
pixi run bench
```

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit, because shared
library build cost is largely fixed. Buffers cross the C ABI as 64-bit
addresses and are reconstructed in Mojo as
`Pointer[Float64, AnyOrigin[mut=True]]`, which keeps the exported functions
non-parametric. `python/mojo_ax_platform` owns every array, normalises inputs
to contiguous `float64`, and makes one call per function.

Every loop here is a plain serial loop. 1.2.0 cannot pass pointers into a
`parallelize` body, and these kernels are memory-bound reductions, where
threading measured slower than serial on this box. There is no `ThreadPoolExecutor`
fan-out in this port because there is nothing to fan out.

Mojo emits FMA, so results match NumPy and `ax` only to a tolerance, never
bit-for-bit. On top of that, Mojo sums `ybar` sequentially where NumPy sums
pairwise, so at k=2^20 the grand mean itself differs in the last ~1e-14 and
anything derived from it inherits that absolute floor. The parity tests use
`rtol=1e-12, atol=1e-12`; exact equality is asserted only where the operation
is genuinely exact (`got[1] == 0.0` for the noiseless branch, the `clip` at
zero, the identical-arm zeroing).

## Tests

37 parity tests in `tests/test_stats_parity.py` run against the real
`ax_platform` 1.3.1 from the test venv. Beyond the direct parity assertions
they pin the behaviour that distinguishes the correct formula from a plausible
near-miss: that the inverse-variance weighting is not a plain mean, that
`phi_i` collapses the estimate to the grand mean when the sem dominates the
spread and leaves it alone when the arm is precise, that the bias term changes
the result, that the recovered variance is clipped instead of going NaN, and
that the zero-ratio short circuit returns the control rather than zero.

## License

MIT
