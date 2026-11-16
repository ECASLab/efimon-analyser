"""Conjugate Bayesian regression of window power on NVML/CUPTI features.

Unlike the kernel-family model, this model explains power with device
counters (utilization, clocks, temperature) and aggregate kernel activity.
"""

import csv
import logging
import math
from pathlib import Path

import numpy as np

from energy_attribution.io import require_fields
from energy_attribution.prediction import interval_coverage, regression_metrics
from energy_attribution.units import NS_PER_SECOND

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "gpu_utilization_pct",
    "memory_utilization_pct",
    "graphics_clock_mhz",
    "temperature_c",
    "kernel_active_pct",
    "log_kernel_count",
    "kernel_concurrency",
    "gpu_clock_interaction",
    "memory_clock_interaction",
]


def load_telemetry(
    path: Path, window_ns: int
) -> tuple[int, int, np.ndarray, dict[str, np.ndarray]]:
    """Sum NVML samples into fixed windows starting at the first sample.

    Args:
      path: ``gpu_telemetry.csv`` with power, temperature, utilization and
        graphics clock.
      window_ns: Window length in nanoseconds.

    Returns:
      ``(origin_ns, window_count, counts, sums)``: the first timestamp, the
      number of windows, the samples per window and the per-window sums of
      each quantity.

    Raises:
      ValueError: If columns are missing or fewer than three valid samples
        span more than one timestamp.
    """
    required = {
        "timestamp_ns",
        "power_mw",
        "temperature_c",
        "gpu_utilization_pct",
        "memory_utilization_pct",
        "graphics_clock_mhz",
    }
    samples = []
    skipped = 0
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_fields(reader.fieldnames, required, "telemetry")
        for row in reader:
            try:
                sample = {
                    "timestamp_ns": int(row["timestamp_ns"]),
                    "power_w": float(row["power_mw"]) / 1000.0,
                    "temperature_c": float(row["temperature_c"]),
                    "gpu_utilization_pct": float(row["gpu_utilization_pct"]),
                    "memory_utilization_pct": float(
                        row["memory_utilization_pct"]
                    ),
                    "graphics_clock_mhz": float(row["graphics_clock_mhz"]),
                }
            except (TypeError, ValueError):
                skipped += 1
                continue
            if not all(math.isfinite(value) for value in sample.values()):
                skipped += 1
                continue
            samples.append(sample)

    if skipped:
        logger.warning("Skipped %d invalid telemetry rows.", skipped)
    if len(samples) < 3:
        raise ValueError("At least three valid telemetry samples are required")
    samples.sort(key=lambda sample: sample["timestamp_ns"])
    if samples[0]["timestamp_ns"] == samples[-1]["timestamp_ns"]:
        raise ValueError("Telemetry must cover more than one timestamp")

    origin_ns = samples[0]["timestamp_ns"]
    window_count = (samples[-1]["timestamp_ns"] - origin_ns) // window_ns + 1
    sums = {
        key: np.zeros(window_count, dtype=np.float64)
        for key in (
            "power_w",
            "temperature_c",
            "gpu_utilization_pct",
            "memory_utilization_pct",
            "graphics_clock_mhz",
        )
    }
    counts = np.zeros(window_count, dtype=np.int64)

    for sample in samples:
        index = (sample["timestamp_ns"] - origin_ns) // window_ns
        counts[index] += 1
        for key in sums:
            sums[key][index] += sample[key]

    return origin_ns, window_count, counts, sums


def aggregate_kernels(
    path: Path, origin_ns: int, window_count: int, window_ns: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stream kernel events into launch counts and busy time per window.

    Args:
      path: ``kernel_activity.csv`` sorted by ``start_ns``.
      origin_ns: Start of the first window.
      window_count: Number of windows.
      window_ns: Window length in nanoseconds.

    Returns:
      Kernels launched per window, summed kernel duration (concurrency
      counted once per kernel) and kernel-union busy time, in ns.

    Raises:
      ValueError: If columns are missing or rows are not sorted.
    """
    required = {"start_ns", "end_ns"}
    analysis_end_ns = origin_ns + window_count * window_ns
    kernel_count = np.zeros(window_count, dtype=np.int64)
    duration_sum_ns = np.zeros(window_count, dtype=np.float64)
    active_ns = np.zeros(window_count, dtype=np.float64)
    union_end_ns = np.full(window_count, -1, dtype=np.int64)
    valid = 0
    skipped = 0
    previous_start = None

    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_fields(reader.fieldnames, required, "kernel activity")
        for row in reader:
            try:
                start_ns = int(row["start_ns"])
                end_ns = int(row["end_ns"])
            except (TypeError, ValueError):
                skipped += 1
                continue
            if end_ns <= start_ns:
                skipped += 1
                continue
            if previous_start is not None and start_ns < previous_start:
                raise ValueError(
                    "Kernel activity rows must be sorted by start_ns for "
                    "streaming aggregation"
                )
            previous_start = start_ns
            valid += 1

            if origin_ns <= start_ns < analysis_end_ns:
                start_index = (start_ns - origin_ns) // window_ns
                kernel_count[start_index] += 1

            clipped_start = max(start_ns, origin_ns)
            clipped_end = min(end_ns, analysis_end_ns)
            if clipped_end <= clipped_start:
                continue

            first_index = (clipped_start - origin_ns) // window_ns
            last_index = (clipped_end - 1 - origin_ns) // window_ns
            for index in range(first_index, last_index + 1):
                window_start = origin_ns + index * window_ns
                window_end = window_start + window_ns
                overlap_start = max(clipped_start, window_start)
                overlap_end = min(clipped_end, window_end)
                duration_sum_ns[index] += overlap_end - overlap_start

                previous_end = union_end_ns[index]
                if overlap_end > previous_end:
                    uncovered_start = max(overlap_start, previous_end)
                    active_ns[index] += overlap_end - uncovered_start
                    union_end_ns[index] = overlap_end

    if skipped:
        logger.warning("Skipped %d invalid kernel rows.", skipped)
    logger.info("Aggregated %d kernel events.", valid)
    return kernel_count, duration_sum_ns, active_ns


def build_dataset(
    origin_ns: int,
    window_count: int,
    window_ns: int,
    telemetry_counts: np.ndarray,
    telemetry_sums: dict[str, np.ndarray],
    kernel_count: np.ndarray,
    duration_sum_ns: np.ndarray,
    active_ns: np.ndarray,
    min_telemetry_samples: int,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Build the feature matrix of :data:`FEATURE_NAMES` per window.

    Args:
      origin_ns: Start of the first window.
      window_count: Number of windows.
      window_ns: Window length in nanoseconds.
      telemetry_counts: Telemetry samples per window.
      telemetry_sums: Per-window sums of each telemetry quantity.
      kernel_count: Kernels launched per window.
      duration_sum_ns: Summed kernel duration per window.
      active_ns: Kernel-union busy time per window.
      min_telemetry_samples: Windows with fewer samples are dropped.

    Returns:
      The window records, the (windows, features) matrix and the mean
      measured power of each window in watts.

    Raises:
      ValueError: If fewer than ten windows remain.
    """
    rows = []
    features = []
    targets = []

    for index in range(window_count):
        count = int(telemetry_counts[index])
        if count < min_telemetry_samples:
            continue
        averages = {
            key: float(values[index] / count)
            for key, values in telemetry_sums.items()
        }
        kernel_active_pct = min(100.0, active_ns[index] / window_ns * 100.0)
        concurrency = duration_sum_ns[index] / window_ns
        log_kernel_count = math.log1p(kernel_count[index])
        clock_ghz = averages["graphics_clock_mhz"] / 1000.0
        feature_values = [
            averages["gpu_utilization_pct"],
            averages["memory_utilization_pct"],
            averages["graphics_clock_mhz"],
            averages["temperature_c"],
            kernel_active_pct,
            log_kernel_count,
            concurrency,
            averages["gpu_utilization_pct"] * clock_ghz,
            averages["memory_utilization_pct"] * clock_ghz,
        ]
        window_start_ns = origin_ns + index * window_ns
        window_end_ns = window_start_ns + window_ns
        rows.append(
            {
                "window_index": index,
                "window_start_ns": window_start_ns,
                "window_end_ns": window_end_ns,
                "telemetry_samples": count,
                "measured_power_w": averages["power_w"],
                "measured_energy_j": averages["power_w"]
                * window_ns
                / NS_PER_SECOND,
                "temperature_c": averages["temperature_c"],
                "gpu_utilization_pct": averages["gpu_utilization_pct"],
                "memory_utilization_pct": averages["memory_utilization_pct"],
                "graphics_clock_mhz": averages["graphics_clock_mhz"],
                "kernel_active_pct": kernel_active_pct,
                "kernel_count": int(kernel_count[index]),
                "log_kernel_count": log_kernel_count,
                "kernel_concurrency": concurrency,
                "gpu_clock_interaction": feature_values[-2],
                "memory_clock_interaction": feature_values[-1],
            }
        )
        features.append(feature_values)
        targets.append(averages["power_w"])

    if len(rows) < 10:
        raise ValueError(
            "At least ten complete windows are required after telemetry "
            "filtering"
        )
    return rows, np.asarray(features), np.asarray(targets)


def fit_bayesian_regression(
    features: np.ndarray,
    targets: np.ndarray,
    train_count: int,
    prior_scale: float,
    posterior_draws: int,
    seed: int,
) -> dict:
    """Sample a conjugate normal/inverse-gamma linear regression.

    Features and target are standardized with the training statistics. The
    coefficients have independent normal priors (variance 25 for the
    intercept, ``prior_scale**2`` for the slopes) scaled by the noise
    variance, which has an inverse-gamma(2, 1) prior; draws come from the
    exact posterior.

    Args:
      features: (windows, features) matrix; the first ``train_count`` rows
        are fitted.
      targets: Measured power or energy per window.
      train_count: Number of leading training rows.
      prior_scale: Prior standard deviation of the standardized slopes.
      posterior_draws: Number of draws.
      seed: Random seed.

    Returns:
      The standardization statistics, coefficient draws and the mean and
      predictive draws of the target for every window.

    Raises:
      ValueError: If the training targets are constant or the posterior
        variance is not positive.
    """
    train_features = features[:train_count]
    train_targets = targets[:train_count]
    feature_mean = train_features.mean(axis=0)
    feature_scale = train_features.std(axis=0, ddof=1)
    feature_scale[feature_scale < 1e-12] = 1.0
    target_mean = float(train_targets.mean())
    target_scale = float(train_targets.std(ddof=1))
    if target_scale < 1e-12:
        raise ValueError("Training power measurements have no variation")

    standardized = (features - feature_mean) / feature_scale
    design = np.column_stack([np.ones(len(features)), standardized])
    train_design = design[:train_count]
    train_target = (train_targets - target_mean) / target_scale

    parameter_count = train_design.shape[1]
    prior_variance = np.full(parameter_count, prior_scale**2)
    prior_variance[0] = 25.0
    prior_precision = np.diag(1.0 / prior_variance)

    posterior_precision = prior_precision + train_design.T @ train_design
    posterior_covariance = np.linalg.inv(posterior_precision)
    posterior_mean = posterior_covariance @ (train_design.T @ train_target)
    shape = 2.0 + train_count / 2.0
    rate = 1.0 + 0.5 * (
        train_target @ train_target
        - posterior_mean @ posterior_precision @ posterior_mean
    )
    if rate <= 0.0:
        raise ValueError("Bayesian variance posterior is not positive")

    rng = np.random.default_rng(seed)
    noise_variance = 1.0 / rng.gamma(
        shape, scale=1.0 / rate, size=posterior_draws
    )
    covariance_root = np.linalg.cholesky(posterior_covariance)
    standard_normal = rng.normal(size=(posterior_draws, parameter_count))
    coefficient_draws = (
        posterior_mean
        + (standard_normal @ covariance_root.T)
        * np.sqrt(noise_variance)[:, None]
    )

    mean_draws_standardized = coefficient_draws @ design.T
    mean_draws_w = target_mean + target_scale * mean_draws_standardized
    predictive_draws_w = mean_draws_w + (
        rng.normal(size=mean_draws_w.shape)
        * np.sqrt(noise_variance)[:, None]
        * target_scale
    )

    return {
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "target_mean": target_mean,
        "target_scale": target_scale,
        "coefficient_draws": coefficient_draws,
        "mean_draws_w": mean_draws_w,
        "predictive_draws_w": predictive_draws_w,
    }


def interval_regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict:
    """Point-error metrics and the share of ``actual`` in [lower, upper]."""
    metrics = regression_metrics(actual, predicted)
    metrics["predictive_interval_95_coverage"] = interval_coverage(
        actual, lower, upper
    )
    return metrics


def coefficient_rows(
    fit: dict, intercept_unit: str, slope_unit: str
) -> list[dict]:
    """Posterior coefficients in target units (intercept at feature means).

    Args:
      fit: Output of :func:`fit_bayesian_regression`.
      intercept_unit: Unit label of the intercept.
      slope_unit: Unit label of the slopes.

    Returns:
      One row per term with its posterior mean, 95% interval and the
      training mean and standard deviation of its feature.
    """
    rows = []
    for index, name in enumerate(
        ["intercept_at_training_mean"] + FEATURE_NAMES
    ):
        values = fit["coefficient_draws"][:, index] * fit["target_scale"]
        if index == 0:
            values = values + fit["target_mean"]
        lower, upper = np.quantile(values, [0.025, 0.975])
        rows.append(
            {
                "term": name,
                "posterior_mean": float(values.mean()),
                "credible_lower_95": float(lower),
                "credible_upper_95": float(upper),
                "unit": intercept_unit if index == 0 else slope_unit,
                "training_feature_mean": (
                    "" if index == 0 else fit["feature_mean"][index - 1]
                ),
                "training_feature_stddev": (
                    "" if index == 0 else fit["feature_scale"][index - 1]
                ),
            }
        )
    return rows


def posterior_intervals(
    fit: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mean plus credible and predictive 95% bounds of the target per window."""
    mean_draws = fit["mean_draws_w"]
    return (
        mean_draws.mean(axis=0),
        *np.quantile(mean_draws, [0.025, 0.975], axis=0),
        *np.quantile(fit["predictive_draws_w"], [0.025, 0.975], axis=0),
    )
