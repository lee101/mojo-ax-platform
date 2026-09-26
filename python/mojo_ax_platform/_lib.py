"""ctypes bridge to the compiled Mojo statistics kernels.

The shared library owns no memory. Every buffer crosses the C ABI as a 64-bit
address, so the argtypes below must stay `c_int64` for addresses; `c_int`
truncates them and segfaults.
"""

import ctypes
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-ax-platform.so"

_I64 = ctypes.c_int64
_F64 = ctypes.c_double


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.ax_ivw_partials.restype = None
    lib.ax_ivw_partials.argtypes = [_I64, _I64, _I64, _I64]
    lib.ax_total_variance.restype = None
    lib.ax_total_variance.argtypes = [_I64, _I64, _I64, _I64, _I64]
    lib.ax_james_stein.restype = None
    lib.ax_james_stein.argtypes = [_I64, _I64, _I64, _I64, _I64]
    lib.ax_agresti_coull_sem.restype = None
    lib.ax_agresti_coull_sem.argtypes = [_I64, _I64, _I64, _F64, _F64, _I64]
    lib.ax_relativize.restype = None
    lib.ax_relativize.argtypes = [
        _I64, _I64, _I64, _I64, _F64, _F64, _I64, _I64, _I64, _I64,
    ]
    lib.ax_unrelativize.restype = None
    lib.ax_unrelativize.argtypes = [
        _I64, _I64, _I64, _I64, _F64, _F64, _I64, _I64, _I64, _I64,
    ]
    return lib


lib = _load()


def _f(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64)


def _addr(a: np.ndarray) -> int:
    return a.ctypes.data


def ivw_partials(means, variances) -> np.ndarray:
    """Six partial sums used by `inverse_variance_weight`; see the kernel."""
    means = _f(means)
    variances = _f(variances)
    out = np.empty(6, dtype=np.float64)
    lib.ax_ivw_partials(
        _addr(means), _addr(variances), means.size, _addr(out)
    )
    return out


def total_variance_parts(means, variances, sample_sizes) -> np.ndarray:
    means = _f(means)
    variances = _f(variances)
    sample_sizes = _f(sample_sizes)
    out = np.empty(3, dtype=np.float64)
    lib.ax_total_variance(
        _addr(means), _addr(variances), _addr(sample_sizes), means.size,
        _addr(out),
    )
    return out


def positive_part_james_stein(means, sems):
    means = _f(means)
    sems = _f(sems)
    mu = np.empty_like(means)
    sig = np.empty_like(means)
    lib.ax_james_stein(
        _addr(means), _addr(sems), means.size, _addr(mu), _addr(sig)
    )
    return mu, sig


def agresti_coull_sem(n_numer, n_denom, prior_successes, prior_failures):
    numer = _f(n_numer)
    denom = _f(n_denom)
    out = np.empty_like(numer)
    lib.ax_agresti_coull_sem(
        _addr(numer), _addr(denom), numer.size, _F64(prior_successes),
        _F64(prior_failures), _addr(out),
    )
    return out


def relativize(means_t, sems_t, mean_c, sem_c, bias_correction, cov_means,
               control_as_constant):
    means_t = _f(means_t)
    sems_t = _f(sems_t)
    cov = _f(cov_means)
    r = np.empty_like(means_t)
    s = np.empty_like(means_t)
    lib.ax_relativize(
        _addr(means_t), _addr(sems_t), _addr(cov), means_t.size,
        _F64(mean_c), _F64(sem_c), int(bool(bias_correction)),
        int(bool(control_as_constant)), _addr(r), _addr(s),
    )
    return r, s


def unrelativize(means_t, sems_t, mean_c, sem_c, bias_correction, cov_means,
                 control_as_constant):
    means_t = _f(means_t)
    sems_t = _f(sems_t)
    cov = _f(cov_means)
    m = np.empty_like(means_t)
    s = np.empty_like(means_t)
    lib.ax_unrelativize(
        _addr(means_t), _addr(sems_t), _addr(cov), means_t.size,
        _F64(mean_c), _F64(sem_c), int(bool(bias_correction)),
        int(bool(control_as_constant)), _addr(m), _addr(s),
    )
    return m, s
