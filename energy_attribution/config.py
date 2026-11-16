"""Typed configuration contracts for Bayesian energy analyses.

The dataclass field defaults below are the single source of default values:
the command-line parsers in :mod:`energy_attribution.cli` read them from here.
"""

from dataclasses import dataclass
from pathlib import Path

# "model": split each window's measured dynamic energy in proportion to the
# fitted terms b0 + sum(beta_f * a_f), so the intercept keeps its share.
# "active-fraction": legacy rule that scales energy by the kernel-busy
# fraction and splits all of it among families, ignoring b0.
ATTRIBUTION_MODES = ("model", "active-fraction")
DEFAULT_MIN_WINDOW_FRACTION = 0.9
DEFAULT_MIN_WINDOWS_PER_PARAMETER = 3.0


@dataclass(frozen=True, kw_only=True)
class ModelConfig:  # pylint: disable=too-many-instance-attributes
    """Settings shared by every kernel-energy model."""

    window_ms: float = 200.0
    idle_power_mw: float = 0.0
    max_families: int = 64
    prior_scale_w: float = 250.0
    noise_prior_scale_w: float = 20.0
    burn_in: int = 1000
    posterior_draws: int = 2000
    thin: int = 2
    seed: int = 2026
    attribution: str = "model"
    min_windows_per_parameter: float = DEFAULT_MIN_WINDOWS_PER_PARAMETER
    allow_underdetermined: bool = False
    min_window_fraction: float = DEFAULT_MIN_WINDOW_FRACTION

    @property
    def window_ns(self) -> int:
        """Nominal analysis-window duration in nanoseconds."""
        return int(self.window_ms * 1_000_000)

    def validate(self, *, minimum_families: int = 2) -> None:
        """Reject invalid model settings before loading large input files.

        Args:
          minimum_families: Smallest accepted ``max_families``.

        Raises:
          ValueError: If any setting is out of range.
        """
        if self.window_ms <= 0.0:
            raise ValueError("window_ms must be greater than zero")
        if self.idle_power_mw < 0.0:
            raise ValueError("idle_power_mw cannot be negative")
        if self.max_families < minimum_families:
            raise ValueError(
                f"max_families must be at least {minimum_families}"
            )
        if self.prior_scale_w <= 0.0 or self.noise_prior_scale_w <= 0.0:
            raise ValueError("prior scales must be greater than zero")
        if self.burn_in < 0 or self.posterior_draws < 100 or self.thin < 1:
            raise ValueError("invalid MCMC settings")
        if self.min_windows_per_parameter <= 0.0:
            raise ValueError(
                "min_windows_per_parameter must be greater than zero"
            )
        if not 0.0 <= self.min_window_fraction <= 1.0:
            raise ValueError("min_window_fraction must be in [0, 1]")
        if self.attribution not in ATTRIBUTION_MODES:
            raise ValueError(f"unknown attribution mode: {self.attribution}")


def _validate_train_fraction(train_fraction: float) -> None:
    if not 0.5 <= train_fraction < 1.0:
        raise ValueError("train_fraction must be in [0.5, 1.0)")


@dataclass(frozen=True, kw_only=True)
class SingleSessionConfig(ModelConfig):
    """Inputs and output policy for one profiling session."""

    telemetry: Path
    kernels: Path
    output_dir: Path = Path("bayesian-kernel-energy-analysis")
    train_fraction: float = 0.8
    plot: bool = False
    skip_event_output: bool = False

    def validate(self, *, minimum_families: int = 2) -> None:
        super().validate(minimum_families=minimum_families)
        _validate_train_fraction(self.train_fraction)


@dataclass(frozen=True, kw_only=True)
class PooledModelConfig(ModelConfig):
    """Model settings of the workflows that pool several sessions."""

    temperature_feature: bool = False
    temperature_reference_c: float = 25.0

    def validate(self, *, minimum_families: int = 2) -> None:
        super().validate(minimum_families=minimum_families)
        if not -50.0 <= self.temperature_reference_c <= 150.0:
            raise ValueError("temperature_reference_c must be in [-50, 150]")


@dataclass(frozen=True, kw_only=True)
class MultiSessionConfig(PooledModelConfig):
    """Inputs and output policy for a pooled multi-session model."""

    sessions_root: Path
    output_dir: Path
    train_fraction: float = 0.8
    plot: bool = False

    def validate(self, *, minimum_families: int = 2) -> None:
        super().validate(minimum_families=minimum_families)
        _validate_train_fraction(self.train_fraction)
