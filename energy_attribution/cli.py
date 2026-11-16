"""Build command-line parsers from the typed configuration dataclasses.

Every field of a config dataclass becomes one ``--kebab-case`` option with
the dataclass default, so the CLI and the library API cannot drift apart.
"""

import argparse
import logging
import sys
from dataclasses import MISSING, fields
from typing import Any, Callable, Optional, Sequence, TypeVar

from energy_attribution.config import ATTRIBUTION_MODES

ConfigT = TypeVar("ConfigT")

HELP = {
    "window_ms": "Analysis window length in milliseconds.",
    "idle_power_mw": (
        "Power (mW) subtracted from every telemetry sample before "
        "attribution; 0 attributes the whole device power."
    ),
    "train_fraction": (
        "Leading share of each session's windows used for fitting; the rest "
        "is the temporal holdout."
    ),
    "prior_scale_w": (
        "Prior standard deviation (W) of each coefficient after scaling its "
        "column to unit RMS."
    ),
    "noise_prior_scale_w": (
        "Scale (W) of the inverse-gamma prior on the observation noise."
    ),
    "burn_in": "Gibbs iterations discarded before saving draws.",
    "posterior_draws": "Posterior draws kept.",
    "thin": "Gibbs iterations between saved draws.",
    "seed": "Random seed; the same seed reproduces the same results.",
    "telemetry": "NVML telemetry CSV (gpu_telemetry.csv).",
    "kernels": "CUPTI kernel activity CSV (kernel_activity.csv).",
    "sessions_root": "Directory whose subdirectories are profiling sessions.",
    "output_dir": "Directory for CSV, JSON and plot outputs.",
    "max_families": (
        "Maximum family columns, including __other__; rarer kernels are "
        "pooled into __other__."
    ),
    "attribution": (
        "'model' (default) keeps the intercept's share of each window as "
        "baseline energy; 'active-fraction' reproduces the legacy rule."
    ),
    "min_windows_per_parameter": "Minimum training windows per regression "
    "parameter.",
    "allow_underdetermined": (
        "Fit even when training windows per parameter is below the minimum."
    ),
    "min_window_fraction": (
        "Windows shorter than this fraction of --window-ms (the truncated "
        "last window of a session) are excluded from fitting, metrics and "
        "plots; their energy is still attributed. 0 keeps every window."
    ),
    "plot": "Generate diagnostic PNG plots (requires matplotlib).",
    "temperature_feature": (
        "Add the window GPU temperature above --temperature-reference-c as "
        "a non-negative design column (W per degree C), so temperature-"
        "dependent leakage power is not attributed to kernels."
    ),
    "temperature_reference_c": (
        "Temperature (C) at which the temperature column is zero."
    ),
    "skip_event_output": (
        "Skip the potentially large per-event CSV while retaining summaries."
    ),
}


def add_config_arguments(
    parser: argparse.ArgumentParser, config_type: type
) -> None:
    """Add one option per dataclass field, using the field's default.

    Args:
      parser: Parser that receives the options.
      config_type: Configuration dataclass; boolean fields become flags and
        fields without a default become required options.
    """
    for field in fields(config_type):
        option = "--" + field.name.replace("_", "-")
        options = {"help": HELP.get(field.name)}
        if field.type is bool:
            parser.add_argument(option, action="store_true", **options)
            continue
        options["type"] = field.type
        if field.default is MISSING:
            options["required"] = True
        else:
            options["default"] = field.default
        if field.name == "attribution":
            options["choices"] = ATTRIBUTION_MODES
        parser.add_argument(option, **options)


def config_from_namespace(
    config_type: type[ConfigT], namespace: argparse.Namespace
) -> ConfigT:
    """Construct a config from an argparse namespace, field by field.

    Args:
      config_type: Configuration dataclass to build.
      namespace: Parsed options, one per field.

    Returns:
      The configuration.

    Raises:
      ValueError: If the namespace lacks a field or carries an option the
        config does not define, instead of silently using a default.
    """
    names = {field.name for field in fields(config_type)}
    provided = set(vars(namespace))
    if names != provided:
        raise ValueError(
            f"{config_type.__name__} fields and CLI options differ: "
            f"missing {sorted(names - provided)}, unexpected "
            f"{sorted(provided - names)}"
        )
    return config_type(**{name: getattr(namespace, name) for name in names})


class _ConsoleFormatter(logging.Formatter):
    """Plain progress messages; warnings keep their historical prefix."""

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if record.levelno >= logging.WARNING:
            return f"{record.levelname}: {message}"
        return message


def configure_logging(level: int = logging.INFO) -> None:
    """Send package log records to stdout as the commands always printed."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_ConsoleFormatter())
    logger = logging.getLogger("energy_attribution")
    logger.handlers[:] = [handler]
    logger.setLevel(level)
    logger.propagate = False


def run_command(
    config_type: type,
    description: str,
    analysis: Callable[[Any], Any],
    report: Callable[[Any], None],
    argv: Optional[Sequence[str]] = None,
) -> Any:
    """Parse ``argv``, run ``analysis`` and print ``report``.

    Configuration and input errors exit with status 1 and a one-line message.

    Args:
      config_type: Configuration dataclass that defines the options.
      description: Help text of the command.
      analysis: Function that runs the analysis on the configuration.
      report: Function that prints the result.
      argv: Arguments to parse; ``None`` reads ``sys.argv``.

    Returns:
      The analysis result.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_config_arguments(parser, config_type)
    namespace = parser.parse_args(argv)
    configure_logging()
    try:
        config = config_from_namespace(config_type, namespace)
        result = analysis(config)
    except ValueError as error:
        parser.exit(1, f"error: {error}\n")
    report(result)
    return result


__all__ = [
    "add_config_arguments",
    "config_from_namespace",
    "configure_logging",
    "run_command",
]
