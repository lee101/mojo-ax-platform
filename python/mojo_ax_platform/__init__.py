"""mojo-ax-platform: Mojo kernels for the `ax.utils.stats` numeric subset.

Installable alongside the real `ax-platform` package, which the tests compare
against for parity.
"""

from .core import (
    MEAN_CONTROL_EPSILON,
    agresti_coull_sem,
    inverse_variance_weight,
    positive_part_james_stein,
    relativize,
    total_variance,
    unrelativize,
)

__all__ = [
    "MEAN_CONTROL_EPSILON",
    "agresti_coull_sem",
    "inverse_variance_weight",
    "positive_part_james_stein",
    "relativize",
    "total_variance",
    "unrelativize",
]
__version__ = "0.1.0"
