"""Matplotlib diagnostics for the kernel-energy analyses.

Matplotlib is imported lazily inside ``create_plots`` so the analyses
run without it unless ``--plot``.
"""

from pathlib import Path
from types import ModuleType
from typing import Optional

import numpy as np

from energy_attribution.units import NS_PER_SECOND
from energy_attribution.reporting.text import (
    format_optional,
    short_kernel_name,
    short_session_name,
)


def plot_kernel_family_energy(
    plt: ModuleType, output_dir: Path, family_rows: list[dict]
) -> None:
    """Plot the 15 largest family energies with 95% intervals.

    Args:
      plt: ``matplotlib.pyplot``.
      output_dir: Directory of ``bayesian_kernel_family_energy.png``.
      family_rows: Family energy rows, largest first.
    """
    top_families = family_rows[: min(15, len(family_rows))]
    labels = [short_kernel_name(row["kernel_family"]) for row in top_families]
    medians = np.asarray(
        [row["energy_posterior_median_mj"] / 1000.0 for row in top_families]
    )
    lowers = np.asarray(
        [row["energy_posterior_lower_95_mj"] / 1000.0 for row in top_families]
    )
    uppers = np.asarray(
        [row["energy_posterior_upper_95_mj"] / 1000.0 for row in top_families]
    )
    positions = np.arange(len(top_families))[::-1]
    figure, axis = plt.subplots(figsize=(11, 7))
    axis.barh(positions, medians, color="tab:blue", alpha=0.8)
    axis.errorbar(
        medians,
        positions,
        xerr=np.vstack([medians - lowers, uppers - medians]),
        fmt="none",
        ecolor="black",
        capsize=3,
        linewidth=1,
        label="95% credible interval",
    )
    axis.set_yticks(positions, labels)
    axis.set_xlabel("Attributed energy posterior median (J)")
    axis.set_title("Bayesian energy attribution by kernel family")
    axis.grid(axis="x", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "bayesian_kernel_family_energy.png", dpi=180)
    plt.close(figure)


def build_plot_series(window_rows: list[dict]) -> dict:
    """Collect the measured and predicted series that the plots share.

    Args:
      window_rows: Non-excluded window records with predictions.

    Returns:
      Elapsed time, measured and predicted power and energy with their 95%
      bounds, and ``test_start``, the index of the first test window.
    """
    elapsed_seconds = np.asarray(
        [
            (row["start_ns"] - window_rows[0]["start_ns"]) / NS_PER_SECOND
            for row in window_rows
        ]
    )
    measured_power = np.asarray(
        [
            row["dynamic_energy_mj"]
            / (row["duration_ns"] / NS_PER_SECOND)
            / 1000.0
            for row in window_rows
        ]
    )
    predicted_power = np.asarray(
        [row["predicted_dynamic_power_mean_w"] for row in window_rows]
    )
    predicted_power_lower = np.asarray(
        [row["predicted_dynamic_power_lower_95_w"] for row in window_rows]
    )
    predicted_power_upper = np.asarray(
        [row["predicted_dynamic_power_upper_95_w"] for row in window_rows]
    )
    predictive_power_lower = np.asarray(
        [
            row["predicted_dynamic_power_predictive_lower_95_w"]
            for row in window_rows
        ]
    )
    predictive_power_upper = np.asarray(
        [
            row["predicted_dynamic_power_predictive_upper_95_w"]
            for row in window_rows
        ]
    )
    window_seconds = np.asarray(
        [row["duration_ns"] / NS_PER_SECOND for row in window_rows]
    )
    measured_energy = measured_power * window_seconds
    predicted_energy = predicted_power * window_seconds
    predicted_energy_lower = predicted_power_lower * window_seconds
    predicted_energy_upper = predicted_power_upper * window_seconds
    test_start = next(
        (
            index
            for index, row in enumerate(window_rows)
            if row["split"] == "test"
        ),
        len(window_rows),
    )
    return {
        "elapsed_seconds": elapsed_seconds,
        "measured_power": measured_power,
        "predicted_power": predicted_power,
        "predicted_power_lower": predicted_power_lower,
        "predicted_power_upper": predicted_power_upper,
        "predictive_power_lower": predictive_power_lower,
        "predictive_power_upper": predictive_power_upper,
        "predictive_energy_lower": predictive_power_lower * window_seconds,
        "predictive_energy_upper": predictive_power_upper * window_seconds,
        "measured_energy": measured_energy,
        "predicted_energy": predicted_energy,
        "predicted_energy_lower": predicted_energy_lower,
        "predicted_energy_upper": predicted_energy_upper,
        "test_start": test_start,
    }


def session_row_groups(window_rows: list[dict]) -> list[tuple[str, list]]:
    """Group window rows by session, each group in window order.

    Args:
      window_rows: Window records, optionally labelled with ``session``.

    Returns:
      ``(session, rows)`` pairs in order of first appearance.
    """
    names = list(
        dict.fromkeys(row.get("session", "session") for row in window_rows)
    )
    groups = []
    for name in names:
        rows = [
            row for row in window_rows if row.get("session", "session") == name
        ]
        rows.sort(
            key=lambda row: row.get("session_window_index", row["window_index"])
        )
        groups.append((name, rows))
    return groups


def plot_kernel_power_fit(
    plt: ModuleType, output_dir: Path, window_rows: list[dict]
) -> None:
    """Plot measured and predicted power over time, one panel per session.

    Args:
      plt: ``matplotlib.pyplot``.
      output_dir: Directory of ``bayesian_kernel_power_fit.png``.
      window_rows: Non-excluded window records with predictions.
    """
    groups = session_row_groups(window_rows)
    figure, axes = plt.subplots(
        len(groups),
        1,
        figsize=(12, max(5, 2.3 * len(groups))),
        squeeze=False,
        constrained_layout=True,
    )
    for group_index, (name, rows) in enumerate(groups):
        axis = axes[group_index, 0]
        series = build_plot_series(rows)
        axis.plot(
            series["elapsed_seconds"],
            series["measured_power"],
            label="Measured NVML dynamic power",
            linewidth=1.1,
        )
        axis.plot(
            series["elapsed_seconds"],
            series["predicted_power"],
            label="Posterior mean",
            linewidth=1.1,
        )
        axis.fill_between(
            series["elapsed_seconds"],
            series["predictive_power_lower"],
            series["predictive_power_upper"],
            alpha=0.12,
            color="tab:orange",
            linewidth=0,
            label="95% predictive interval",
        )
        axis.fill_between(
            series["elapsed_seconds"],
            series["predicted_power_lower"],
            series["predicted_power_upper"],
            alpha=0.3,
            color="tab:orange",
            linewidth=0,
            label="95% credible interval",
        )
        test_start = series["test_start"]
        if test_start < len(series["elapsed_seconds"]):
            axis.axvspan(
                series["elapsed_seconds"][test_start],
                series["elapsed_seconds"][-1],
                color="grey",
                alpha=0.12,
                label="Temporal holdout",
            )
        axis.set_ylabel("Power (W)")
        axis.set_title(
            name if len(groups) > 1 else "Kernel-family Bayesian power fit"
        )
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8, ncol=4, loc="best")
    axes[-1, 0].set_xlabel("Elapsed time (s)")
    if len(groups) > 1:
        figure.suptitle("Multi-session kernel-family Bayesian power fit")
    figure.savefig(output_dir / "bayesian_kernel_power_fit.png", dpi=180)
    plt.close(figure)


def plot_kernel_energy_test_scatter(
    plt: ModuleType, output_dir: Path, window_rows: list[dict], series: dict
) -> None:
    """Plot measured against predicted energy of the test windows.

    Args:
      plt: ``matplotlib.pyplot``.
      output_dir: Directory of ``bayesian_kernel_energy_test_scatter.png``.
      window_rows: Non-excluded window records with predictions.
      series: Output of :func:`build_plot_series`.
    """
    test_start = series["test_start"]
    actual = series["measured_energy"][test_start:]
    expected = series["predicted_energy"][test_start:]
    lower = series["predictive_energy_lower"][test_start:]
    upper = series["predictive_energy_upper"][test_start:]
    if len(actual) == 0:
        raise ValueError("Temporal holdout contains no windows to plot")

    figure, (fit_axis, residual_axis) = plt.subplots(
        2,
        1,
        figsize=(9, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )
    test_rows = window_rows[test_start:]
    session_names = list(
        dict.fromkeys(row.get("session", "session") for row in test_rows)
    )
    colors = plt.get_cmap("tab20", len(session_names))
    for session_index, name in enumerate(session_names):
        indices = np.asarray(
            [
                index
                for index, row in enumerate(test_rows)
                if row.get("session", "session") == name
            ]
        )
        color = colors(session_index)
        fit_axis.vlines(
            actual[indices],
            lower[indices],
            upper[indices],
            color=color,
            alpha=0.18,
            linewidth=0.8,
        )
        fit_axis.scatter(
            actual[indices],
            expected[indices],
            color=color,
            alpha=0.7,
            label=name
            if len(session_names) > 1
            else "Temporal holdout windows",
        )
        residual_axis.scatter(
            actual[indices],
            expected[indices] - actual[indices],
            color=color,
            alpha=0.7,
        )
    minimum = float(min(actual.min(), expected.min(), lower.min()))
    maximum = float(max(actual.max(), expected.max(), upper.max()))
    fit_axis.plot(
        [minimum, maximum],
        [minimum, maximum],
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="Central line: predicted = measured",
    )
    fit_axis.set_ylabel("Posterior predicted energy (J/window)")
    fit_axis.set_title(
        "Kernel-energy temporal holdout predictions "
        "(vertical bars: 95% predictive interval)"
    )
    fit_axis.grid(alpha=0.2)
    fit_axis.legend()
    residual_axis.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    residual_axis.set_xlabel("Measured dynamic energy (J/window)")
    residual_axis.set_ylabel("Distance from central line (J/window)")
    residual_axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(
        output_dir / "bayesian_kernel_energy_test_scatter.png", dpi=180
    )
    plt.close(figure)


MEASURED_COLOR = "#52514e"


PREDICTED_COLOR = "#2a78d6"


REFERENCE_COLOR = "#8a8984"


def plot_session_energy(
    plt: ModuleType,
    output_dir: Path,
    session_rows: list[dict],
    session_metrics: dict,
    test_metrics: dict,
) -> None:
    """Plot measured against predicted holdout energy and power per session.

    Args:
      plt: ``matplotlib.pyplot``.
      output_dir: Directory of ``bayesian_session_energy.png``.
      session_rows: Holdout energy per session.
      session_metrics: Metrics across sessions.
      test_metrics: Window power metrics of the holdout.
    """
    # Numbered markers (shared with the power panel's ticks) stay legible when
    # sessions with similar totals cluster; full names would collide.
    rows = sorted(
        session_rows, key=lambda row: row["measured_power_mean_w"], reverse=True
    )
    numbers = [str(index + 1) for index in range(len(rows))]
    labels = [
        number + "  " + short_session_name(row["session"])
        for number, row in zip(numbers, rows)
    ]
    measured = np.asarray([row["measured_energy_j"] for row in rows])
    predicted = np.asarray([row["predicted_energy_mean_j"] for row in rows])
    lower = np.asarray([row["predicted_energy_lower_95_j"] for row in rows])
    upper = np.asarray([row["predicted_energy_upper_95_j"] for row in rows])
    predictive_lower = np.asarray(
        [row["predicted_energy_predictive_lower_95_j"] for row in rows]
    )
    predictive_upper = np.asarray(
        [row["predicted_energy_predictive_upper_95_j"] for row in rows]
    )

    figure, (energy_axis, power_axis) = plt.subplots(
        1,
        2,
        figsize=(15, max(6.0, 0.5 * len(rows) + 3.0)),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )

    limit_low = 0.0
    limit_high = 1.05 * float(
        max(measured.max(), upper.max(), predictive_upper.max())
    )
    energy_axis.plot(
        [limit_low, limit_high],
        [limit_low, limit_high],
        color=REFERENCE_COLOR,
        linestyle="--",
        linewidth=1.2,
        label="Predicted = measured",
    )
    energy_axis.errorbar(
        measured,
        predicted,
        yerr=np.vstack(
            [predicted - predictive_lower, predictive_upper - predicted]
        ),
        fmt="none",
        ecolor=PREDICTED_COLOR,
        alpha=0.45,
        capsize=4,
        elinewidth=1.0,
        label="95% predictive interval",
    )
    energy_axis.errorbar(
        measured,
        predicted,
        yerr=np.vstack([predicted - lower, upper - predicted]),
        fmt="o",
        color=PREDICTED_COLOR,
        ecolor=PREDICTED_COLOR,
        markersize=7,
        capsize=0,
        elinewidth=3.0,
        markeredgecolor="white",
        markeredgewidth=1.0,
        label="Posterior mean ± 95% credible interval",
    )
    for index, (number, x_value, y_value) in enumerate(
        zip(numbers, measured, predicted)
    ):
        energy_axis.annotate(
            number,
            (x_value, y_value),
            xytext=(-12, 2) if index % 2 else (6, -10),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color=MEASURED_COLOR,
        )
    energy_axis.set_xlim(limit_low, limit_high)
    energy_axis.set_ylim(limit_low, limit_high)
    energy_axis.set_aspect("equal", adjustable="box")
    energy_axis.set_xlabel("Measured dynamic energy in holdout (J)")
    energy_axis.set_ylabel("Predicted dynamic energy in holdout (J)")
    energy_axis.set_title("Total holdout energy per session")
    energy_axis.grid(alpha=0.2)
    energy_axis.legend(loc="lower right", fontsize=8)
    session_credible = session_metrics["credible_interval_95_coverage"]
    session_predictive = session_metrics["predictive_interval_95_coverage"]
    energy_axis.text(
        0.03,
        0.97,
        "\n".join(
            [
                "Per session (energy totals):",
                "  R² = "
                + format_optional(
                    session_metrics["energy_r_squared"], "{:.4f}"
                ),
                "  Mean |error| = "
                + format_optional(
                    session_metrics["mean_absolute_energy_error_pct"], "{:.1f}%"
                ),
                "  All sessions: "
                + format_optional(
                    session_metrics["total_energy_error_pct"], "{:+.1f}%"
                ),
                "  95% coverage: credible "
                + f"{100.0 * session_credible:.0f}%"
                + ", predictive "
                + f"{100.0 * session_predictive:.0f}%",
                "Per window (power):",
                "  R² = "
                + format_optional(test_metrics["r_squared"], "{:.4f}"),
                "  RMSE = "
                + format_optional(test_metrics["rmse_w"], "{:.3f} W"),
                "  95% coverage: credible "
                + format_optional(
                    test_metrics["credible_interval_95_coverage"], "{:.0%}"
                )
                + ", predictive "
                + format_optional(
                    test_metrics["predictive_interval_95_coverage"], "{:.0%}"
                ),
            ]
        ),
        transform=energy_axis.transAxes,
        va="top",
        fontsize=9,
        family="monospace",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#d6d5d0"},
    )

    positions = np.arange(len(rows))[::-1]
    offset = 0.17
    power_axis.hlines(
        positions + offset,
        [row["measured_power_min_w"] for row in rows],
        [row["measured_power_max_w"] for row in rows],
        color=MEASURED_COLOR,
        linewidth=2.0,
        label="Measured min–max",
    )
    power_axis.plot(
        [row["measured_power_mean_w"] for row in rows],
        positions + offset,
        "o",
        color=MEASURED_COLOR,
        markersize=7,
        markeredgecolor="white",
        label="Measured mean",
    )
    power_axis.hlines(
        positions - offset,
        [row["predicted_power_min_w"] for row in rows],
        [row["predicted_power_max_w"] for row in rows],
        color=PREDICTED_COLOR,
        linewidth=2.0,
        label="Predicted min–max",
    )
    power_axis.plot(
        [row["predicted_power_mean_w"] for row in rows],
        positions - offset,
        "o",
        color=PREDICTED_COLOR,
        markersize=7,
        markeredgecolor="white",
        label="Predicted mean",
    )
    power_axis.set_yticks(positions, labels, fontsize=8)
    power_axis.set_xlabel("Dynamic power per window in holdout (W)")
    power_axis.set_title("Power range per session")
    power_axis.grid(axis="x", alpha=0.2)
    power_axis.legend(loc="lower right", fontsize=8)

    figure.suptitle("Temporal holdout accuracy per session")
    figure.tight_layout()
    figure.savefig(output_dir / "bayesian_session_energy.png", dpi=180)
    plt.close(figure)


def create_plots(
    output_dir: Path,
    window_rows: list[dict],
    family_rows: list[dict],
    session_rows: Optional[list[dict]] = None,
    session_metrics: Optional[dict] = None,
    test_metrics: Optional[dict] = None,
) -> None:
    """Write the kernel-energy diagnostic PNG files.

    Args:
      output_dir: Directory of the PNG files.
      window_rows: Window records with predictions; excluded rows are
        skipped.
      family_rows: Family energy rows, largest first.
      session_rows: Holdout energy per session; ``None`` skips that plot.
      session_metrics: Metrics across sessions.
      test_metrics: Window power metrics of the holdout.

    Raises:
      RuntimeError: If matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    except ImportError as error:
        raise RuntimeError("--plot requires matplotlib") from error

    for obsolete_name in (
        "bayesian_energy_partition.png",
        "bayesian_kernel_energy_temporal_holdout.png",
    ):
        (output_dir / obsolete_name).unlink(missing_ok=True)

    window_rows = [row for row in window_rows if row["split"] != "excluded"]
    plot_kernel_family_energy(plt, output_dir, family_rows)
    series = build_plot_series(window_rows)
    plot_kernel_power_fit(plt, output_dir, window_rows)
    plot_kernel_energy_test_scatter(plt, output_dir, window_rows, series)
    if session_rows:
        plot_session_energy(
            plt, output_dir, session_rows, session_metrics, test_metrics
        )
