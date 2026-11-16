"""Write analysis results to CSV, JSON and PNG files.

File names and CSV columns are a stable contract checked by the golden
regression tests.
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from energy_attribution.io import write_csv
from energy_attribution.reporting.plots import create_plots
from energy_attribution.windows import EVENT_CHUNK

if TYPE_CHECKING:
    from energy_attribution.allocation import EventAttribution
    from energy_attribution.workflows.multisession import MultiSessionResult
    from energy_attribution.workflows.single_session import SingleSessionResult

logger = logging.getLogger(__name__)


def write_rows(path: Path, rows: list[dict]) -> None:
    """Write dictionaries as CSV, using the first row's keys as columns."""
    write_csv(path, rows, list(rows[0]))


def write_summary(path: Path, summary: dict) -> None:
    """Write ``summary`` as indented JSON."""
    with path.open("w", encoding="utf-8") as destination:
        json.dump(summary, destination, indent=2)
        destination.write("\n")


def write_event_csv(path: Path, events: "EventAttribution") -> None:
    """Per-event energy in chunks; join to the CUPTI trace via event_id.

    Args:
      path: Destination CSV.
      events: Posterior energy of every kernel event.
    """
    family_names = np.asarray(events.families, dtype=object)
    for offset in range(0, len(events.start_ns), EVENT_CHUNK):
        chunk = slice(offset, offset + EVENT_CHUNK)
        frame = pd.DataFrame(
            {
                "start_ns": events.start_ns[chunk],
                "end_ns": events.end_ns[chunk],
                "kernel_family": family_names[events.family_column[chunk]],
                "bayesian_energy_mean_mj": events.mean[chunk],
                "bayesian_energy_lower_95_mj": events.lower[chunk],
                "bayesian_energy_upper_95_mj": events.upper[chunk],
            }
        )
        if events.event_ids is not None:
            frame.insert(0, "event_id", events.event_ids[chunk])
        frame.to_csv(
            path,
            mode="w" if offset == 0 else "a",
            header=offset == 0,
            index=False,
        )


def _write_kernel_model(
    result: "SingleSessionResult | MultiSessionResult", output_dir: Path
) -> None:
    """Outputs shared by the single- and multi-session kernel models.

    Args:
      result: A kernel-model result.
      output_dir: Destination directory, created when missing.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_dir / "bayesian_energy_windows.csv", result.dataset.rows)
    write_rows(
        output_dir / "bayesian_kernel_coefficients.csv", result.coefficient_rows
    )
    write_rows(
        output_dir / "bayesian_kernel_family_energy.csv", result.family_rows
    )
    write_rows(output_dir / "bayesian_session_energy.csv", result.session_rows)
    if result.config.plot:
        create_plots(
            output_dir,
            result.dataset.rows,
            result.family_rows,
            result.session_rows,
            result.session_metrics,
            result.test_metrics,
        )


def write_single_session(result: "SingleSessionResult") -> None:
    """Write the CSV, JSON and optional PNG outputs of one session."""
    output_dir = result.config.output_dir
    _write_kernel_model(result, output_dir)
    if result.events is not None:
        logger.info("Writing per-event Bayesian attribution...")
        write_event_csv(
            output_dir / "bayesian_kernel_event_energy.csv", result.events
        )
    write_summary(
        output_dir / "bayesian_kernel_energy_summary.json", result.summary
    )


def write_multisession(result: "MultiSessionResult") -> None:
    """Write the CSV, JSON and optional PNG outputs of a pooled model."""
    output_dir = result.config.output_dir
    _write_kernel_model(result, output_dir)
    write_summary(
        output_dir / "bayesian_kernel_energy_summary.json", result.summary
    )
