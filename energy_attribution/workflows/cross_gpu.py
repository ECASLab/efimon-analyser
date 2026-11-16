"""Cross-GPU transfer of the Bayesian kernel-energy model (computation only).

A pooled model is trained on the source GPU's sessions, then evaluated on one
external target-GPU session twice: zero-shot, and after a non-negative
offset/scale calibration fitted on the target session's first windows.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from energy_attribution.allocation import (
    EnergyAllocation,
    attribution_pool_mj,
    energy_accounting,
    family_energy_rows,
    posterior_energy_allocation,
)
from energy_attribution.config import (
    TEMPERATURE_COEFFICIENT_AUTO,
    CrossGpuConfig,
)
from energy_attribution.dataset import activity_design, split_mask, window_durations_ns
from energy_attribution.diagnostics import check_identifiability
from energy_attribution.families import (
    OTHER_FAMILY,
    TEMPERATURE_FAMILY,
    KernelScan,
    scan_kernel_families,
)
from energy_attribution.io import load_kernels
from energy_attribution.prediction import (
    PosteriorPredictions,
    interval_coverage,
    regression_metrics,
    summarize_predictions,
)
from energy_attribution.regression import (
    PosteriorSamples,
    coefficient_summary_rows,
    fit_nonnegative_bayesian_regression,
)
from energy_attribution.sessions import (
    SessionInput,
    SessionScan,
    SessionWindows,
    build_session_windows,
    discover_session_paths,
    scan_sessions,
    session_files,
)
from energy_attribution.temperature_slope import (
    TemperatureSlopeEstimate,
    estimate_temperature_slope,
)
from energy_attribution.units import NS_PER_SECOND

logger = logging.getLogger(__name__)

ZERO_SHOT_SEED_OFFSET = 10
CALIBRATION_SEED_OFFSET = 1
CALIBRATED_SEED_OFFSET = 11
METHODS = ("zero_shot", "calibrated")
PREDICTION_DESCRIPTION = (
    "calibrated: source-trained model plus a non-negative offset and scale "
    "fitted on the first calibration_fraction of the target session"
)


@dataclass
class CrossGpuResult:  # pylint: disable=too-many-instance-attributes
    """Everything computed for one transfer evaluation."""

    config: CrossGpuConfig
    families: list
    training_sessions: list
    train_count: int
    calibration_count: int
    test_rows: list
    base: PosteriorSamples
    calibration: PosteriorSamples
    predictions: dict[str, PosteriorPredictions]
    allocation: EnergyAllocation
    identifiability: dict
    base_coefficient_rows: list
    calibration_rows: list
    family_rows: list
    metrics: dict
    coverage: dict
    accounting: dict
    summary: dict


def training_family_basis(scan: SessionScan) -> tuple[list[str], set]:
    """Return a train-only family basis with a learned fallback column.

    When every training kernel has its own family, the least active one is
    folded into ``__other__`` so that kernels unseen in training still map
    to a learned coefficient.

    Args:
      scan: The training sessions and their family basis.

    Returns:
      The families, ending in ``__other__``, and the selected kernel names.
    """
    families = list(scan.families)
    selected = set(scan.selected)
    if OTHER_FAMILY not in families:
        fallback_source = min(
            families, key=lambda name: scan.duration_by_name[name]
        )
        families.remove(fallback_source)
        families.append(OTHER_FAMILY)
        selected.remove(fallback_source)
        logger.info(
            "Reserving __other__ using the least-active training family "
            "(%s) so unseen test kernels have a learned fallback coefficient.",
            fallback_source,
        )
    return families, selected


def complete_windows(rows: list[dict], config: CrossGpuConfig) -> np.ndarray:
    """Return a mask of the windows at least ``min_window_fraction`` long."""
    return (
        window_durations_ns(rows)
        >= config.min_window_fraction * config.window_ns
    )


def training_windows(
    config: CrossGpuConfig,
    scan: SessionScan,
    selected: set,
    family_index: dict,
) -> tuple[list[dict], np.ndarray, np.ndarray, list[str], int, dict]:
    """Pool every source session, dropping each truncated final window.

    Args:
      config: Window and temperature settings.
      scan: The training sessions.
      selected: Kernel names with their own family.
      family_index: Column of each family.

    Returns:
      ``(rows, activity_ns, target_power_w, session_names, excluded,
      phases)``, where ``excluded`` counts the dropped windows and
      ``phases`` maps each session to its measurement phase.
    """
    sessions = [
        build_session_windows(
            session,
            selected,
            family_index,
            config.window_ns,
            config.idle_power_mw,
            config.temperature_reference_c,
        )
        for session in scan.sessions
    ]
    rows = [row for session in sessions for row in session.rows]
    activity = np.vstack([session.activity_ns for session in sessions])
    target = np.concatenate([session.target_power_w for session in sessions])
    complete = complete_windows(rows, config)
    return (
        [row for row, keep in zip(rows, complete) if keep],
        activity[complete],
        target[complete],
        [session.name for session in sessions],
        int((~complete).sum()),
        {session.name: session.measurement_phase_ns for session in sessions},
    )


def external_session(
    config: CrossGpuConfig, selected: set, family_index: dict
) -> tuple[KernelScan, SessionWindows]:
    """Aggregate the target session with the source family basis.

    Args:
      config: The target session and window settings.
      selected: Kernel names with their own family in training.
      family_index: Column of each family.

    Returns:
      The target session's kernel statistics and windows.
    """
    telemetry, kernels = session_files(config.external_test_session)
    scan = scan_kernel_families(load_kernels(kernels), max_families=2)
    session = SessionInput(
        config.external_test_session.name, telemetry, kernels, scan
    )
    windows = build_session_windows(
        session,
        selected,
        family_index,
        config.window_ns,
        config.idle_power_mw,
        config.temperature_reference_c,
    )
    return scan, windows


def calibration_prefix(
    complete: np.ndarray, calibration_fraction: float
) -> int:
    """Return how many leading target windows calibrate the prediction.

    Args:
      complete: Mask of the complete target windows.
      calibration_fraction: Requested share of the complete windows.

    Returns:
      The prefix length, keeping at least five calibration and two
      evaluation windows.

    Raises:
      ValueError: If a truncated window falls inside the prefix.
    """
    complete_count = int(complete.sum())
    count = int(complete_count * calibration_fraction)
    count = min(max(count, 5), complete_count - 2)
    if not complete[:count].all():
        raise ValueError(
            "A truncated window falls inside the calibration prefix"
        )
    return count


def calibrate(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    config: CrossGpuConfig,
    zero_latent: np.ndarray,
    zero_mean: np.ndarray,
    test_target: np.ndarray,
    calibration_count: int,
    fixed_w: np.ndarray,
) -> tuple[PosteriorSamples, PosteriorPredictions]:
    """Fit power_target = offset + scale * zero_shot on the calibration set.

    ``fixed_w`` is the per-window power of a term with a given coefficient
    (the fixed temperature slope, else zeros). It is kept out of the offset
    and scale, so calibration rescales only the fitted part of the model.

    Args:
      config: Prior, sampler and label settings.
      zero_latent: (draws, windows) zero-shot latent power of the target.
      zero_mean: Posterior mean of ``zero_latent``.
      test_target: Measured target power in watts.
      calibration_count: Number of leading calibration windows.
      fixed_w: Per-window power kept outside the offset and scale.

    Returns:
      The (offset, scale) posterior and the calibrated predictions.
    """
    logger.info(
        "Sampling %s offset/scale calibration with %d prefix windows...",
        config.target_label,
        calibration_count,
    )
    calibration = fit_nonnegative_bayesian_regression(
        np.column_stack([np.ones(len(zero_mean)), zero_mean - fixed_w]),
        test_target - fixed_w,
        calibration_count,
        config.prior_scale_w,
        config.noise_prior_scale_w,
        config.burn_in,
        config.posterior_draws,
        config.thin,
        config.seed + CALIBRATION_SEED_OFFSET,
    )
    draws = calibration.coefficient_draws
    calibrated = summarize_predictions(
        draws[:, 0, None]
        + draws[:, 1, None] * (zero_latent - fixed_w)
        + fixed_w,
        calibration.noise_draws,
        config.seed + CALIBRATED_SEED_OFFSET,
    )
    return calibration, calibrated


def temperature_coefficient(
    config: CrossGpuConfig,
    train_rows: list[dict],
    train_target: np.ndarray,
    phases: dict,
) -> tuple[Optional[float], Optional[str], Optional[TemperatureSlopeEstimate]]:
    """Resolve the temperature slope that stays out of the fit.

    Args:
      config: The temperature settings.
      train_rows: Training window records with ``temperature_c``.
      train_target: Measured training power in watts.
      phases: Measurement phase of each training session.

    Returns:
      ``(coefficient, source, estimate)``. ``coefficient`` is ``None`` when
      the slope is fitted with the kernel families; ``source`` is
      ``"given"``, ``"auto"``, ``"fit"`` or ``"fit_fallback"``.
    """
    if not config.temperature_feature:
        return None, None, None
    given = config.given_temperature_coefficient_w_per_c
    if given is not None:
        return given, "given", None
    if config.temperature_coefficient != TEMPERATURE_COEFFICIENT_AUTO:
        return None, "fit", None
    estimate = estimate_temperature_slope(
        train_rows, train_target, phases, config.seed
    )
    if estimate is None:
        logger.warning(
            "No training measurement phase heats up enough to estimate the "
            "temperature slope; fitting it with the kernel families instead."
        )
        return None, "fit_fallback", None
    logger.info(
        "Temperature slope from %d constant-load phases: %.4f W/C "
        "(95%% bootstrap %.4f to %.4f).",
        len(estimate.sessions),
        estimate.coefficient_w_per_c,
        estimate.lower_95_w_per_c,
        estimate.upper_95_w_per_c,
    )
    return estimate.coefficient_w_per_c, "auto", estimate


def with_fixed_coefficient(
    draws: np.ndarray, coefficient: Union[None, float, np.ndarray]
) -> np.ndarray:
    """Append a last column holding the fixed coefficient.

    ``coefficient`` is one value for every draw or an array with one per draw.
    """
    if coefficient is None:
        return draws
    column = np.broadcast_to(np.asarray(coefficient, dtype=float), len(draws))
    return np.column_stack([draws, column])


def calibrated_coefficients(
    base_draws: np.ndarray, calibration_draws: np.ndarray
) -> np.ndarray:
    """Map source-model draws through each draw's offset and scale."""
    calibrated = base_draws.copy()
    calibrated[:, 0] = (
        calibration_draws[:, 0] + calibration_draws[:, 1] * base_draws[:, 0]
    )
    calibrated[:, 1:] *= calibration_draws[:, 1, None]
    return calibrated


def add_transfer_columns(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    rows: list[dict],
    actual_power: np.ndarray,
    calibration_count: int,
    predictions: dict[str, PosteriorPredictions],
    allocation: EnergyAllocation,
    outputs: dict[str, str],
) -> None:
    """Add measured and predicted values to target-session rows.

    Args:
      rows: Target window records, updated in place.
      actual_power: Measured power of each window in watts.
      calibration_count: Number of leading calibration windows.
      predictions: Predictions of each method.
      allocation: Posterior energy split of the target windows.
      outputs: Column-name prefix of each reported method.
    """
    for index, row in enumerate(rows):
        seconds = row["duration_ns"] / NS_PER_SECOND
        row["split"] = "calibration" if index < calibration_count else "test"
        row["measured_dynamic_power_w"] = float(actual_power[index])
        for method, name in outputs.items():
            prediction = predictions[method]
            row[f"{name}_power_mean_w"] = float(prediction.mean[index])
            row[f"{name}_power_lower_95_w"] = float(
                prediction.predictive_lower[index]
            )
            row[f"{name}_power_upper_95_w"] = float(
                prediction.predictive_upper[index]
            )
        row["measured_dynamic_energy_j"] = float(actual_power[index] * seconds)
        for method, name in outputs.items():
            prediction = predictions[method]
            row[f"{name}_energy_mean_j"] = float(
                prediction.mean[index] * seconds
            )
            row[f"{name}_energy_lower_95_j"] = float(
                prediction.predictive_lower[index] * seconds
            )
            row[f"{name}_energy_upper_95_j"] = float(
                prediction.predictive_upper[index] * seconds
            )
        row["bayesian_attributed_energy_mean_mj"] = float(
            allocation.mean[index].sum()
        )
        row["baseline_energy_mean_mj"] = float(allocation.baseline_mean[index])


def evaluate(
    test_rows: list[dict],
    test_target: np.ndarray,
    predictions: dict[str, PosteriorPredictions],
    outputs: dict[str, str],
) -> tuple[np.ndarray, dict, dict]:
    """Point metrics and predictive coverage on the untouched target suffix.

    Args:
      test_rows: Target window records with ``split``.
      test_target: Measured target power in watts.
      predictions: Predictions of each method.
      outputs: Output name of each reported method.

    Returns:
      The evaluation mask, and metrics and coverage keyed by output name.
    """
    evaluation = split_mask(test_rows, "test")
    actual = test_target[evaluation]
    metrics = {
        name: regression_metrics(actual, predictions[method].mean[evaluation])
        for method, name in outputs.items()
    }
    coverage = {
        name: interval_coverage(
            actual,
            predictions[method].predictive_lower[evaluation],
            predictions[method].predictive_upper[evaluation],
        )
        for method, name in outputs.items()
    }
    return evaluation, metrics, coverage


def prediction_summary(
    config: CrossGpuConfig, metrics: dict, coverage: dict
) -> dict:
    """Evaluation keys of the summary JSON for the reported predictions.

    Args:
      config: Whether the zero-shot prediction is also reported.
      metrics: Power metrics keyed by output name.
      coverage: Predictive coverage keyed by output name.

    Returns:
      The summary entries.
    """
    if not config.compare_zero_shot:
        return {
            "prediction": PREDICTION_DESCRIPTION,
            "prediction_evaluation_power_metrics": metrics["predicted"],
            "prediction_predictive_interval_coverage": coverage["predicted"],
        }
    return {
        "zero_shot_evaluation_power_metrics": metrics["zero_shot"],
        "calibrated_evaluation_power_metrics": metrics["calibrated"],
        "calibrated_rmse_reduction_fraction": (
            1.0
            - metrics["calibrated"]["rmse_w"] / metrics["zero_shot"]["rmse_w"]
        ),
        "zero_shot_predictive_interval_coverage": coverage["zero_shot"],
        "calibrated_predictive_interval_coverage": coverage["calibrated"],
    }


def explicit_family_coverage(
    scan: KernelScan, selected: set
) -> tuple[float, float]:
    """Share of target events and busy time covered by explicit families."""
    names = set(scan.duration_by_name) & selected
    events = sum(scan.event_count_by_name[name] for name in names)
    duration = sum(scan.duration_by_name[name] for name in names)
    return (
        events / scan.valid_events,
        duration / sum(scan.duration_by_name.values()),
    )


def analyze(config: CrossGpuConfig) -> CrossGpuResult:  # pylint: disable=too-many-locals
    """Train on the source GPU, calibrate and evaluate on the target session.

    Args:
      config: Training root, target session and model settings.

    Returns:
      The source posterior, the calibration, the predictions, metrics and
      JSON summary, and the target energy split.

    Raises:
      ValueError: If the inputs are invalid or too short.
    """
    source, target = config.source_label, config.target_label
    train_scan = scan_sessions(
        discover_session_paths(config.train_sessions_root), config.max_families
    )
    families, selected = training_family_basis(train_scan)
    if config.temperature_feature:
        families.append(TEMPERATURE_FAMILY)
    family_index = {family: index for index, family in enumerate(families)}
    (
        train_rows,
        train_activity,
        train_target,
        session_names,
        excluded_training,
        phases,
    ) = training_windows(config, train_scan, selected, family_index)
    test_scan, external = external_session(config, selected, family_index)
    test_rows, test_target = external.rows, external.target_power_w
    test_complete = complete_windows(test_rows, config)
    calibration_count = calibration_prefix(
        test_complete, config.calibration_fraction
    )

    all_design = activity_design(
        np.vstack([train_activity, external.activity_ns]),
        train_rows + test_rows,
    )
    train_count = len(train_rows)
    # A fixed temperature slope leaves the design: its power is subtracted
    # from the target before fitting and added back to every prediction.
    fixed_coefficient, coefficient_source, slope_estimate = (
        temperature_coefficient(config, train_rows, train_target, phases)
    )
    fixed_w = np.zeros(len(all_design))
    fit_design, fit_terms = all_design, families
    if fixed_coefficient is not None:
        fixed_w = fixed_coefficient * all_design[:, -1]
        fit_design, fit_terms = all_design[:, :-1], families[:-1]
    identifiability = check_identifiability(
        fit_design,
        train_count,
        ["unattributed_baseline"] + fit_terms,
        config.min_windows_per_parameter,
        config.allow_underdetermined,
    )
    logger.info(
        "\nSampling %s posterior with %d training windows...",
        source,
        train_count,
    )
    base = fit_nonnegative_bayesian_regression(
        fit_design,
        np.concatenate([train_target, test_target]) - fixed_w,
        train_count,
        config.prior_scale_w,
        config.noise_prior_scale_w,
        config.burn_in,
        config.posterior_draws,
        config.thin,
        config.seed,
    )
    test_fixed_w = fixed_w[train_count:]
    zero_latent = base.prediction_draws[:, train_count:] + test_fixed_w
    zero_shot = summarize_predictions(
        zero_latent, base.noise_draws, config.seed + ZERO_SHOT_SEED_OFFSET
    )
    # On another GPU model the source slope does not hold, so the whole
    # prediction, temperature term included, is rescaled by calibration.
    calibration_fixed_w = (
        np.zeros_like(test_fixed_w)
        if config.target_gpu_differs
        else test_fixed_w
    )
    calibration, calibrated = calibrate(
        config,
        zero_latent,
        zero_shot.mean,
        test_target,
        calibration_count,
        calibration_fixed_w,
    )
    calibrated_temperature = None
    if fixed_coefficient is not None:
        calibrated_temperature = fixed_coefficient * (
            calibration.coefficient_draws[:, 1]
            if config.target_gpu_differs
            else np.ones(len(calibration.coefficient_draws))
        )
    predictions = {"zero_shot": zero_shot, "calibrated": calibrated}
    base_draws = with_fixed_coefficient(
        base.coefficient_draws, fixed_coefficient
    )
    base_scale = base.column_scale
    if fixed_coefficient is not None:
        temperature = all_design[:train_count, -1]
        base_scale = np.append(
            base_scale, float(np.sqrt(np.mean(temperature * temperature)))
        )

    allocation = posterior_energy_allocation(
        with_fixed_coefficient(
            calibrated_coefficients(
                base.coefficient_draws, calibration.coefficient_draws
            ),
            calibrated_temperature,
        ),
        external.activity_ns,
        attribution_pool_mj(test_rows, config.attribution),
        window_durations_ns(test_rows),
        config.attribution,
    )
    add_transfer_columns(
        test_rows,
        test_target,
        calibration_count,
        predictions,
        allocation,
        config.outputs,
    )
    for row, keep in zip(test_rows, test_complete):
        if not keep:
            row["split"] = "excluded"
    evaluation, metrics, coverage = evaluate(
        test_rows, test_target, predictions, config.outputs
    )

    calibration_rows = coefficient_summary_rows(
        [f"{target.lower()}_offset_w", f"{source.lower()}_prediction_scale"],
        calibration.coefficient_draws,
        calibration.column_scale,
        prefix="posterior_",
        suffix="",
        include_median=True,
    )
    event_coverage, duration_coverage = explicit_family_coverage(
        test_scan, selected
    )
    accounting = energy_accounting(test_rows, allocation)
    summary = {
        "model": "cross_gpu_bayesian_kernel_family_transfer",
        "attribution_mode": config.attribution,
        "training_sessions_root": str(config.train_sessions_root),
        "training_sessions": session_names,
        "excluded_training_sessions": train_scan.excluded_sessions,
        "external_test_session": str(config.external_test_session),
        "validation": (
            f"{source}-only training; first {target} prefix reserved for "
            "offset/scale calibration; both methods compared on the untouched "
            f"{target} suffix"
        ),
        "window_ms": config.window_ms,
        **(
            {"temperature_reference_c": config.temperature_reference_c}
            if config.temperature_feature
            else {}
        ),
        **(
            {"temperature_coefficient_source": coefficient_source}
            if config.temperature_feature
            else {}
        ),
        **(
            {"fixed_temperature_coefficient_w_per_c": fixed_coefficient}
            if fixed_coefficient is not None
            else {}
        ),
        **(
            {"temperature_slope_estimate": slope_estimate.summary()}
            if slope_estimate is not None
            else {}
        ),
        **(
            {
                "target_gpu_differs": True,
                "calibrated_temperature_coefficient_w_per_c": {
                    "mean": float(calibrated_temperature.mean()),
                    "lower_95": float(
                        np.quantile(calibrated_temperature, 0.025)
                    ),
                    "upper_95": float(
                        np.quantile(calibrated_temperature, 0.975)
                    ),
                },
            }
            if config.target_gpu_differs and calibrated_temperature is not None
            else {}
        ),
        "calibration_fraction": config.calibration_fraction,
        "training_windows": train_count,
        "v100_calibration_windows": calibration_count,
        "v100_evaluation_windows": int(evaluation.sum()),
        "excluded_short_windows": {
            "training": excluded_training,
            "v100": int((~test_complete).sum()),
        },
        "min_window_fraction": config.min_window_fraction,
        "kernel_families": len(families) - int(config.temperature_feature),
        "training_design_rank": identifiability["design_matrix_rank"],
        "training_design_parameters": int(fit_design.shape[1]),
        "identifiability": identifiability,
        "v100_explicit_family_event_coverage": event_coverage,
        "v100_explicit_family_duration_coverage": duration_coverage,
        **prediction_summary(config, metrics, coverage),
        "calibration_posterior": {
            row["term"]: {
                key: value for key, value in row.items() if key != "term"
            }
            for row in calibration_rows
        },
        "v100_measured_dynamic_energy_mj": accounting[
            "measured_dynamic_energy_mj"
        ],
        "v100_assignable_dynamic_energy_mj": accounting[
            "assignable_dynamic_energy_mj"
        ],
        "v100_attributed_posterior_mean_mj": accounting[
            "attributed_posterior_mean_mj"
        ],
        "v100_baseline_posterior_mean_mj": accounting[
            "baseline_posterior_mean_mj"
        ],
        "conservation_error_mj": accounting["conservation_error_mj"],
        "important_limitation": (
            "A global offset and scale calibrate device-level power but do not "
            "prove that every kernel-family coefficient transfers unchanged."
        ),
    }
    return CrossGpuResult(
        config=config,
        families=families,
        training_sessions=session_names,
        train_count=train_count,
        calibration_count=calibration_count,
        test_rows=test_rows,
        base=base,
        calibration=calibration,
        predictions=predictions,
        allocation=allocation,
        identifiability=identifiability,
        base_coefficient_rows=coefficient_summary_rows(
            ["unattributed_baseline"] + families,
            base_draws,
            base_scale,
            prefix="posterior_",
            suffix="",
            include_median=True,
        ),
        calibration_rows=calibration_rows,
        family_rows=family_energy_rows(
            family_index,
            test_scan.duration_by_name,
            test_scan.event_count_by_name,
            selected,
            allocation.family_total_draws,
        ),
        metrics=metrics,
        coverage=coverage,
        accounting=accounting,
        summary=summary,
    )
