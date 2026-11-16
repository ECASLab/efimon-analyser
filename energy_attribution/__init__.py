"""Public library interface for NVML/CUPTI energy attribution analyses."""

from energy_attribution.api import (
    analyze_multisession,
    analyze_single_session,
    run_multisession,
    run_single_session,
)
from energy_attribution.config import (
    ModelConfig,
    MultiSessionConfig,
    SingleSessionConfig,
)

__all__ = [
    "ModelConfig",
    "MultiSessionConfig",
    "SingleSessionConfig",
    "analyze_multisession",
    "analyze_single_session",
    "run_multisession",
    "run_single_session",
]
