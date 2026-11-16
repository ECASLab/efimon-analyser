"""Non-negative Bayesian linear regression sampled with a Gibbs sampler."""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class PosteriorSamples:
    """Posterior draws of a non-negative linear power model.

    ``coefficient_draws`` is (draws, parameters) in watts per unit of the
    design column, ``noise_draws`` the matching observation variance (W^2)
    and ``prediction_draws`` the latent mean power (draws, windows) for every
    row of the design, train and holdout alike.
    """

    coefficient_draws: np.ndarray
    noise_draws: np.ndarray
    prediction_draws: np.ndarray
    column_scale: np.ndarray


def sample_lower_truncated_normal(
    mean: float, standard_deviation: float, rng: np.random.Generator
) -> float:
    """Draw from N(mean, sd) truncated to [0, infinity).

    Uses plain rejection when the mean is non-negative and Robert's
    exponential proposal otherwise.

    Args:
      mean: Mean of the untruncated normal.
      standard_deviation: Its standard deviation; zero returns
        ``max(0, mean)``.
      rng: Random generator.

    Returns:
      One non-negative draw.
    """
    if standard_deviation <= 0.0:
        return max(0.0, mean)
    lower = -mean / standard_deviation
    if lower <= 0.0:
        while True:
            value = rng.normal()
            if value >= lower:
                return mean + standard_deviation * value
    alpha = 0.5 * (lower + math.sqrt(lower * lower + 4.0))
    while True:
        value = lower + rng.exponential(1.0 / alpha)
        if rng.random() <= math.exp(-0.5 * (value - alpha) ** 2):
            return mean + standard_deviation * value


def fit_nonnegative_bayesian_regression(
    design: np.ndarray,
    target: np.ndarray,
    train_count: int,
    prior_scale_w: float,
    noise_prior_scale_w: float,
    burn_in: int,
    posterior_draws: int,
    thin: int,
    seed: int,
) -> PosteriorSamples:
    """Sample a linear power model with non-negative coefficients.

    The model is ``target = design @ beta + noise`` with
    ``noise ~ N(0, sigma2)``. Columns are divided by their training RMS
    before sampling; each scaled coefficient has a normal prior with standard
    deviation ``prior_scale_w`` truncated to non-negative values, and
    ``sigma2`` an inverse-gamma prior with shape 2 and rate
    ``noise_prior_scale_w**2``. Gibbs sampling updates one coefficient at a
    time from its truncated-normal conditional, then ``sigma2``.

    Args:
      design: (windows, parameters) design matrix; only the first
        ``train_count`` rows are fitted.
      target: Measured power per window in watts.
      train_count: Number of leading training rows.
      prior_scale_w: Prior standard deviation of the scaled coefficients.
      noise_prior_scale_w: Square root of the inverse-gamma prior rate.
      burn_in: Discarded initial iterations.
      posterior_draws: Number of saved draws.
      thin: Iterations between saved draws.
      seed: Random seed.

    Returns:
      The posterior draws, in watts per unit of each unscaled column, and
      the latent mean prediction for every design row.
    """
    train_design = design[:train_count]
    train_target = target[:train_count]
    column_scale = np.sqrt(np.mean(train_design * train_design, axis=0))
    column_scale[column_scale < 1e-12] = 1.0
    scaled_design = design / column_scale
    train_scaled = scaled_design[:train_count]

    beta = np.maximum(
        0.0, np.linalg.lstsq(train_scaled, train_target, rcond=None)[0]
    )
    residual = train_target - train_scaled @ beta
    sigma2 = max(float(np.mean(residual * residual)), 1e-6)
    prior_variance = prior_scale_w**2
    noise_shape = 2.0
    noise_rate = noise_prior_scale_w**2
    rng = np.random.default_rng(seed)
    draws = np.empty((posterior_draws, design.shape[1]), dtype=np.float64)
    noise_draws = np.empty(posterior_draws, dtype=np.float64)
    total_iterations = burn_in + posterior_draws * thin
    saved = 0

    for iteration in range(total_iterations):
        for column in range(train_scaled.shape[1]):
            values = train_scaled[:, column]
            residual += values * beta[column]
            precision = float(values @ values) / sigma2 + 1.0 / prior_variance
            variance = 1.0 / precision
            mean = variance * float(values @ residual) / sigma2
            beta[column] = sample_lower_truncated_normal(
                mean, math.sqrt(variance), rng
            )
            residual -= values * beta[column]

        posterior_shape = noise_shape + train_count / 2.0
        posterior_rate = noise_rate + 0.5 * float(residual @ residual)
        sigma2 = 1.0 / rng.gamma(posterior_shape, 1.0 / posterior_rate)

        if iteration >= burn_in and (iteration - burn_in) % thin == 0:
            draws[saved] = beta / column_scale
            noise_draws[saved] = sigma2
            saved += 1

    return PosteriorSamples(draws, noise_draws, draws @ design.T, column_scale)


def posterior_summary(values: np.ndarray) -> tuple[float, float, float, float]:
    """Return (mean, median, lower 2.5%, upper 97.5%) of 1-D draws as floats."""
    lower, upper = np.quantile(values, [0.025, 0.975])
    return (
        float(values.mean()),
        float(np.median(values)),
        float(lower),
        float(upper),
    )


def coefficient_summary_rows(
    terms: list[str],
    coefficient_draws: np.ndarray,
    column_scale: np.ndarray,
    *,
    prefix: str = "posterior_power_",
    suffix: str = "_w",
    include_median: bool = False,
) -> list[dict]:
    """One CSV row per model term with posterior mean and 95% interval.

    Column names are ``{prefix}{mean|median|lower_95|upper_95}{suffix}`` plus
    ``activity_rms_scale``; the kernel and transfer CSVs differ only there.

    Args:
      terms: Name of each design column.
      coefficient_draws: (draws, terms) posterior coefficients.
      column_scale: Training RMS of each design column.
      prefix: Leading part of the summary column names.
      suffix: Trailing part of the summary column names.
      include_median: Also report the posterior median.

    Returns:
      One dictionary per term.
    """
    rows = []
    for index, term in enumerate(terms):
        mean, median, lower, upper = posterior_summary(
            coefficient_draws[:, index]
        )
        row = {"term": term, f"{prefix}mean{suffix}": mean}
        if include_median:
            row[f"{prefix}median{suffix}"] = median
        row[f"{prefix}lower_95{suffix}"] = lower
        row[f"{prefix}upper_95{suffix}"] = upper
        row["activity_rms_scale"] = float(column_scale[index])
        rows.append(row)
    return rows
