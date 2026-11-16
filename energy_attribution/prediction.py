"""Posterior predictions, credible/predictive intervals and holdout metrics."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from energy_attribution.units import NS_PER_SECOND


@dataclass
class PosteriorPredictions:
    """Per-window posterior mean power with 95% credible and predictive bounds.

    Credible bounds cover coefficient uncertainty only; predictive bounds add
    observation noise and are the ones expected to contain measurements.
    """

    mean: np.ndarray
    credible_lower: np.ndarray
    credible_upper: np.ndarray
    predictive_lower: np.ndarray
    predictive_upper: np.ndarray
    prediction_draws: np.ndarray
    predictive_draws: np.ndarray


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Return count, MAE, RMSE and R-squared of a power prediction.

    Args:
      actual: Measured power in watts.
      predicted: Predicted power in watts, aligned with ``actual``.

    Returns:
      A dictionary with ``count``, ``mae_w``, ``rmse_w`` and ``r_squared``;
      ``r_squared`` is ``None`` when the measured power is constant.
    """
    residual = actual - predicted
    total_variation = float(np.sum((actual - actual.mean()) ** 2))
    squared_error = float(residual @ residual)
    return {
        "count": int(len(actual)),
        "mae_w": float(np.mean(np.abs(residual))),
        "rmse_w": float(np.sqrt(np.mean(residual * residual))),
        "r_squared": (
            1.0 - squared_error / total_variation
            if total_variation > 0.0
            else None
        ),
    }


def interval_coverage(
    actual: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> Optional[float]:
    """Return the share of ``actual`` inside ``[lower, upper]``, or ``None``."""
    if len(actual) == 0:
        return None
    return float(np.mean((actual >= lower) & (actual <= upper)))


def summarize_predictions(
    prediction_draws: np.ndarray, noise_draws: np.ndarray, seed: int
) -> PosteriorPredictions:
    """Posterior mean plus credible and predictive 95% intervals per window.

    ``prediction_draws`` reflect only coefficient uncertainty (credible). A
    measured window also carries observation noise sigma, so intervals meant
    to contain measurements add N(0, sigma^2) from the same draw (predictive).
    Power is non-negative, so predictive draws are clipped at zero.

    Args:
      prediction_draws: (draws, windows) latent mean power in watts.
      noise_draws: Observation variance of each draw (W^2).
      seed: Seed of the predictive noise.

    Returns:
      The mean, credible and predictive bounds and both sets of draws.
    """
    rng = np.random.default_rng(seed)
    predictive_draws = np.maximum(
        0.0,
        prediction_draws
        + rng.normal(size=prediction_draws.shape)
        * np.sqrt(noise_draws)[:, None],
    )
    credible_lower, credible_upper = np.quantile(
        prediction_draws, [0.025, 0.975], axis=0
    )
    predictive_lower, predictive_upper = np.quantile(
        predictive_draws, [0.025, 0.975], axis=0
    )
    return PosteriorPredictions(
        mean=prediction_draws.mean(axis=0),
        credible_lower=credible_lower,
        credible_upper=credible_upper,
        predictive_lower=predictive_lower,
        predictive_upper=predictive_upper,
        prediction_draws=prediction_draws,
        predictive_draws=predictive_draws,
    )


def add_prediction_columns(
    window_rows: list[dict], predictions: PosteriorPredictions
) -> None:
    """Add the predicted power mean and 95% bounds to each window row.

    Args:
      window_rows: Window records, updated in place.
      predictions: Posterior predictions of the same windows.
    """
    for index, row in enumerate(window_rows):
        row.update(
            {
                "predicted_dynamic_power_mean_w": predictions.mean[index],
                "predicted_dynamic_power_lower_95_w": (
                    predictions.credible_lower[index]
                ),
                "predicted_dynamic_power_upper_95_w": (
                    predictions.credible_upper[index]
                ),
                "predicted_dynamic_power_predictive_lower_95_w": (
                    predictions.predictive_lower[index]
                ),
                "predicted_dynamic_power_predictive_upper_95_w": (
                    predictions.predictive_upper[index]
                ),
            }
        )


def prediction_metrics(
    target_power_w: np.ndarray,
    predictions: PosteriorPredictions,
    mask: np.ndarray,
) -> dict:
    """Point-error metrics plus 95% interval coverage on the masked windows.

    Args:
      target_power_w: Measured power of every window in watts.
      predictions: Posterior predictions of the same windows.
      mask: Boolean selection of the evaluated windows.

    Returns:
      :func:`regression_metrics` plus credible and predictive coverage.
    """
    actual = target_power_w[mask]
    metrics = regression_metrics(actual, predictions.mean[mask])
    metrics["credible_interval_95_coverage"] = interval_coverage(
        actual,
        predictions.credible_lower[mask],
        predictions.credible_upper[mask],
    )
    metrics["predictive_interval_95_coverage"] = interval_coverage(
        actual,
        predictions.predictive_lower[mask],
        predictions.predictive_upper[mask],
    )
    return metrics


def session_holdout_energy(
    window_rows: list[dict], predictions: PosteriorPredictions
) -> tuple[list[dict], dict]:
    """Per-session totals over holdout windows: measured vs predicted energy.

    Intervals sum each posterior draw over the session's windows before taking
    quantiles, so they describe the total rather than adding per-window
    bounds. The predictive total treats window noise as independent; NVML
    samples are autocorrelated, so it can still be too narrow. Power ranges
    skip truncated windows (shorter than half the session's longest).

    Args:
      window_rows: Window records with ``split``, ``session``,
        ``duration_ns`` and ``dynamic_energy_mj``.
      predictions: Posterior predictions of the same windows.

    Returns:
      One row per session and the metrics across sessions.
    """
    indices_by_session = {}
    for index, row in enumerate(window_rows):
        if row["split"] == "test":
            indices_by_session.setdefault(
                row.get("session", "session"), []
            ).append(index)

    session_rows = []
    for name, indices in indices_by_session.items():
        seconds = (
            np.asarray([window_rows[index]["duration_ns"] for index in indices])
            / NS_PER_SECOND
        )
        measured_j = (
            np.asarray(
                [window_rows[index]["dynamic_energy_mj"] for index in indices]
            )
            / 1000.0
        )
        window_draws = predictions.prediction_draws[:, indices]
        total_draws = window_draws @ seconds
        lower, upper = np.quantile(total_draws, [0.025, 0.975])
        predictive_lower, predictive_upper = np.quantile(
            predictions.predictive_draws[:, indices] @ seconds, [0.025, 0.975]
        )
        measured_power = measured_j / seconds
        predicted_power = window_draws.mean(axis=0)
        full = seconds >= 0.5 * seconds.max()
        measured_total = float(measured_j.sum())
        predicted_total = float(total_draws.mean())
        session_rows.append(
            {
                "session": name,
                "test_windows": len(indices),
                "test_seconds": float(seconds.sum()),
                "measured_energy_j": measured_total,
                "predicted_energy_mean_j": predicted_total,
                "predicted_energy_lower_95_j": float(lower),
                "predicted_energy_upper_95_j": float(upper),
                "predicted_energy_predictive_lower_95_j": float(
                    predictive_lower
                ),
                "predicted_energy_predictive_upper_95_j": float(
                    predictive_upper
                ),
                "energy_error_pct": (
                    100.0 * (predicted_total - measured_total) / measured_total
                    if measured_total > 0.0
                    else None
                ),
                "measured_power_mean_w": measured_total / float(seconds.sum()),
                "measured_power_min_w": float(measured_power[full].min()),
                "measured_power_max_w": float(measured_power[full].max()),
                "predicted_power_mean_w": predicted_total
                / float(seconds.sum()),
                "predicted_power_min_w": float(predicted_power[full].min()),
                "predicted_power_max_w": float(predicted_power[full].max()),
            }
        )

    measured = np.asarray([row["measured_energy_j"] for row in session_rows])
    predicted = np.asarray(
        [row["predicted_energy_mean_j"] for row in session_rows]
    )
    errors = [
        abs(row["energy_error_pct"])
        for row in session_rows
        if row["energy_error_pct"] is not None
    ]

    def coverage(prefix: str) -> float:
        """Share of sessions whose measured energy is inside the bounds."""
        return float(
            np.mean(
                [
                    row[f"{prefix}lower_95_j"]
                    <= row["measured_energy_j"]
                    <= row[f"{prefix}upper_95_j"]
                    for row in session_rows
                ]
            )
        )

    total_variation = float(np.sum((measured - measured.mean()) ** 2))
    session_metrics = {
        "sessions": len(session_rows),
        "energy_r_squared": (
            1.0 - float(np.sum((measured - predicted) ** 2)) / total_variation
            if len(session_rows) > 1 and total_variation > 0.0
            else None
        ),
        "mean_absolute_energy_error_pct": (
            float(np.mean(errors)) if errors else None
        ),
        "max_absolute_energy_error_pct": float(max(errors)) if errors else None,
        "total_energy_error_pct": (
            100.0
            * float(predicted.sum() - measured.sum())
            / float(measured.sum())
        ),
        "credible_interval_95_coverage": coverage("predicted_energy_"),
        "predictive_interval_95_coverage": coverage(
            "predicted_energy_predictive_"
        ),
    }
    return session_rows, session_metrics
