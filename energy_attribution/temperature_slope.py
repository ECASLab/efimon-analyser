"""Estimate the GPU temperature power slope from constant-load phases.

Fitted together with the kernel families, the temperature coefficient is
confounded with load: the hottest training sessions are also the busiest.
Inside one benchmark's measurement phase the load is constant, so the power
change that follows the temperature there is the thermal (leakage) slope.

The estimator demeans power and temperature within each phase and pools the
phases: ``slope = sum(dT * dP) / sum(dT * dT)``. Phases that heat up the
most carry the most weight. Sessions are resampled to give a bootstrap
interval.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from energy_attribution.io import KernelTable

logger = logging.getLogger(__name__)

# Marker kernels launched by the benchmarks at each phase boundary. Kernel
# names are mangled, so they are matched as substrings.
MEASUREMENT_START_MARKER = "benchmark_measurement_start_marker"
COOLDOWN_START_MARKER = "benchmark_cooldown_start_marker"
MIN_TEMPERATURE_RANGE_C = 2.0
MIN_PHASE_WINDOWS = 10
BOOTSTRAP_DRAWS = 2000


@dataclass(frozen=True)
class TemperatureSlopeEstimate:
    """Pooled constant-load temperature slope and its bootstrap interval."""

    coefficient_w_per_c: float
    lower_95_w_per_c: float
    upper_95_w_per_c: float
    sessions: list
    windows: int

    def summary(self) -> dict:
        """JSON-ready description of the estimate."""
        return {
            "method": "pooled within-phase slope of benchmark measurement "
            "phases",
            "coefficient_w_per_c": self.coefficient_w_per_c,
            "bootstrap_lower_95_w_per_c": self.lower_95_w_per_c,
            "bootstrap_upper_95_w_per_c": self.upper_95_w_per_c,
            "sessions": self.sessions,
            "windows": self.windows,
            "min_temperature_range_c": MIN_TEMPERATURE_RANGE_C,
        }


def _first_start_ns(table: KernelTable, marker: str) -> Optional[int]:
    """Start of the first event whose kernel name contains ``marker``."""
    codes = [index for index, name in enumerate(table.names) if marker in name]
    if not codes:
        return None
    starts = table.start_ns[np.isin(table.name_codes, codes)]
    return int(starts.min())


def measurement_phase_ns(table: KernelTable) -> Optional[tuple[int, int]]:
    """Return the ``(start, end)`` of the measurement phase, else ``None``.

    The phase runs from the measurement start marker to the cooldown start
    marker. Traces without both markers have no known constant-load phase.
    """
    start = _first_start_ns(table, MEASUREMENT_START_MARKER)
    end = _first_start_ns(table, COOLDOWN_START_MARKER)
    if start is None or end is None or end <= start:
        return None
    return start, end


def _phase_moments(
    rows: list[dict], target_power_w: np.ndarray, phases: dict
) -> list[tuple[str, int, float, float]]:
    """Return ``(name, windows, sum dT*dP, sum dT*dT)`` per usable phase.

    A phase is usable with at least ``MIN_PHASE_WINDOWS`` windows and a
    temperature range of at least ``MIN_TEMPERATURE_RANGE_C``. Temperature
    and power are centred within each phase, so the load level of each
    benchmark does not enter the slope.

    Args:
      rows: Window records with ``session``, ``start_ns``, ``end_ns`` and
        ``temperature_c``.
      target_power_w: Measured power of each window in watts.
      phases: Measurement phase of each session, or ``None``.

    Returns:
      The within-phase moments of every usable phase.
    """
    sessions = np.asarray([row["session"] for row in rows])
    starts = np.asarray([row["start_ns"] for row in rows])
    ends = np.asarray([row["end_ns"] for row in rows])
    temperature = np.asarray([row["temperature_c"] for row in rows], float)
    moments = []
    for name, phase in phases.items():
        if phase is None:
            continue
        inside = (sessions == name) & (starts >= phase[0]) & (ends <= phase[1])
        phase_temperature = temperature[inside]
        if (
            inside.sum() < MIN_PHASE_WINDOWS
            or np.ptp(phase_temperature) < MIN_TEMPERATURE_RANGE_C
        ):
            continue
        delta_t = phase_temperature - phase_temperature.mean()
        power = target_power_w[inside]
        delta_p = power - power.mean()
        moments.append(
            (
                name,
                int(inside.sum()),
                float(delta_t @ delta_p),
                float(delta_t @ delta_t),
            )
        )
    return moments


def estimate_temperature_slope(
    rows: list[dict],
    target_power_w: np.ndarray,
    phases: dict,
    seed: int,
) -> Optional[TemperatureSlopeEstimate]:
    """Pool the measurement phases of the training sessions.

    The slope is ``sum(dT * dP) / sum(dT * dT)`` over the usable phases; its
    95% interval comes from resampling phases with replacement.

    Args:
      rows: Training windows carrying ``session``, ``start_ns``, ``end_ns``
        and ``temperature_c``.
      target_power_w: Measured power of each window in watts.
      phases: Measurement phase of each session name, or ``None``.
      seed: Seed of the bootstrap.

    Returns:
      The estimate, or ``None`` when no phase heats up by
      ``MIN_TEMPERATURE_RANGE_C`` or the pooled slope is not positive.
    """
    moments = _phase_moments(rows, np.asarray(target_power_w), phases)
    if not moments:
        return None
    cross = np.asarray([moment[2] for moment in moments])
    variance = np.asarray([moment[3] for moment in moments])
    coefficient = float(cross.sum() / variance.sum())
    if coefficient <= 0.0:
        return None
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(moments), size=(BOOTSTRAP_DRAWS, len(moments)))
    bootstrap = cross[draws].sum(axis=1) / variance[draws].sum(axis=1)
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    return TemperatureSlopeEstimate(
        coefficient_w_per_c=coefficient,
        lower_95_w_per_c=float(lower),
        upper_95_w_per_c=float(upper),
        sessions=[moment[0] for moment in moments],
        windows=sum(moment[1] for moment in moments),
    )
