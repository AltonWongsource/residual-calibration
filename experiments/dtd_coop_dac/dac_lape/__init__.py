"""Official-formula DAC and LAPE bridge for the DTD base-to-new rebuttal."""

from .dac import (
    DACResult,
    apply_dac,
    compute_textual_deviation,
    fit_temperature,
    softmax_np,
)

__all__ = [
    "DACResult",
    "apply_dac",
    "compute_textual_deviation",
    "fit_temperature",
    "softmax_np",
]
