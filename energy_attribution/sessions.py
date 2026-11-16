"""Discover profiling sessions and aggregate each one into windows."""

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from energy_attribution.families import (
    TEMPERATURE_FAMILY,
    KernelScan,
    family_columns,
    scan_kernel_families,
    select_families,
)
from energy_attribution.io import PowerTrace, load_kernels, load_power_trace
from energy_attribution.windows import aggregate_family_activity, build_windows

logger = logging.getLogger(__name__)

TELEMETRY_FILE = "gpu_telemetry.csv"
KERNELS_FILE = "kernel_activity.csv"


@dataclass
class SessionInput:
    """One session directory and, once scanned, its kernel statistics.

    ``scan.table`` is released after scanning so that many sessions can be
    pooled in bounded memory; ``build_session_windows`` reloads it.
    """

    name: str
    telemetry: Path
    kernels: Path
    scan: Optional[KernelScan] = None


@dataclass
class SessionScan:
    """All usable sessions below a root and their shared family basis."""

    sessions: list
    excluded_sessions: list
    families: list
    selected: set
    duration_by_name: Counter = field(default_factory=Counter)
    event_count_by_name: Counter = field(default_factory=Counter)


@dataclass
class SessionWindows:
    """One session aggregated into windows (rows carry session labels)."""

    name: str
    rows: list
    activity_ns: np.ndarray
    target_power_w: np.ndarray
    train_count: int = 0


def kernel_file(directory: Path) -> Optional[Path]:
    """Return the session's kernel CSV, preferring it over a ``.gz`` copy."""
    for name in (KERNELS_FILE, KERNELS_FILE + ".gz"):
        path = directory / name
        if path.is_file():
            return path
    return None


def session_files(directory: Path) -> tuple[Path, Path]:
    """Return the telemetry and kernel CSV paths of one session directory.

    Args:
      directory: The session directory.

    Returns:
      The telemetry CSV and the (possibly gzip-compressed) kernel CSV.

    Raises:
      ValueError: If either file is missing.
    """
    telemetry = directory / TELEMETRY_FILE
    kernels = kernel_file(directory)
    missing = [str(telemetry)] if not telemetry.is_file() else []
    if kernels is None:
        missing.append(f"{directory / KERNELS_FILE}[.gz]")
    if missing:
        raise ValueError("Session is missing: " + ", ".join(missing))
    return telemetry, kernels


def discover_session_paths(root: Path) -> list[SessionInput]:
    """Sessions are the subdirectories of ``root`` holding both CSV files.

    The kernel CSV may be gzip-compressed (``kernel_activity.csv.gz``).

    Args:
      root: Directory whose subdirectories (or symlinks) are sessions.

    Returns:
      The sessions in name order.

    Raises:
      ValueError: If no subdirectory holds both files.
    """
    sessions = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        telemetry = directory / TELEMETRY_FILE
        kernels = kernel_file(directory)
        if telemetry.is_file() and kernels is not None:
            sessions.append(SessionInput(directory.name, telemetry, kernels))
    if not sessions:
        raise ValueError("No complete profiling sessions were found")
    return sessions


def scan_sessions(
    session_inputs: list[SessionInput], max_families: int
) -> SessionScan:
    """Load every session's kernels and pick one family basis for all of them.

    Sessions without valid kernel events are excluded with their reason.
    Each session's event table is released once its statistics are known,
    so peak memory is one session's trace rather than all of them.

    Args:
      session_inputs: Sessions from :func:`discover_session_paths`.
      max_families: Maximum family columns, including ``__other__``.

    Returns:
      The usable sessions, the excluded ones and the shared family basis.

    Raises:
      ValueError: If no session has valid kernel events.
    """
    active_sessions = []
    excluded_sessions = []
    duration_by_name = Counter()
    event_count_by_name = Counter()
    for session in session_inputs:
        try:
            logger.info("Loading kernel events for %s...", session.name)
            session.scan = scan_kernel_families(
                load_kernels(session.kernels), max_families=2
            )
        except ValueError as error:
            excluded_sessions.append(
                {"session": session.name, "reason": str(error)}
            )
            logger.info("Skipping %s: %s", session.name, error)
            continue
        session.scan.table = None
        duration_by_name.update(session.scan.duration_by_name)
        event_count_by_name.update(session.scan.event_count_by_name)
        active_sessions.append(session)

    if not active_sessions:
        raise ValueError("No session contains valid CUDA kernel events")
    families, selected = select_families(duration_by_name, max_families)
    return SessionScan(
        sessions=active_sessions,
        excluded_sessions=excluded_sessions,
        families=families,
        selected=selected,
        duration_by_name=duration_by_name,
        event_count_by_name=event_count_by_name,
    )


def window_temperature_c(trace: PowerTrace, rows: list[dict]) -> np.ndarray:
    """Telemetry temperature interpolated at every window's midpoint.

    Args:
      trace: Power samples with temperature.
      rows: Window records with ``start_ns`` and ``end_ns``.

    Returns:
      The temperature of each window in degrees C.

    Raises:
      ValueError: If the telemetry has no valid temperature.
    """
    if trace.temperature_c is None:
        raise ValueError(
            "The temperature feature needs a temperature_c telemetry column"
        )
    valid = np.isfinite(trace.temperature_c)
    if not valid.any():
        raise ValueError("Telemetry has no valid temperature_c samples")
    midpoints = np.asarray(
        [(row["start_ns"] + row["end_ns"]) / 2 for row in rows],
        dtype=np.float64,
    )
    return np.interp(
        midpoints,
        trace.timestamps_ns[valid].astype(np.float64),
        trace.temperature_c[valid],
    )


def add_temperature_column(
    trace: PowerTrace,
    rows: list[dict],
    activity_ns: np.ndarray,
    column: int,
    temperature_reference_c: float,
) -> None:
    """Fill ``column`` so its busy fraction is degrees above the reference.

    The design divides each column by the window length, so storing
    ``excess_c * duration_ns`` yields ``max(0, T - reference)`` in degrees C.

    Args:
      trace: Power samples with temperature.
      rows: Window records; each gains ``temperature_c``.
      activity_ns: (windows, families) matrix updated in place.
      column: Index of the temperature column.
      temperature_reference_c: Temperature at which the column is zero.
    """
    temperature = window_temperature_c(trace, rows)
    excess = np.maximum(0.0, temperature - temperature_reference_c)
    durations = np.asarray([row["duration_ns"] for row in rows], dtype=float)
    activity_ns[:, column] = excess * durations
    for row, value in zip(rows, temperature.tolist()):
        row["temperature_c"] = value


def build_session_windows(
    session: SessionInput,
    selected: set,
    family_index: dict,
    window_ns: int,
    idle_power_mw: float,
    temperature_reference_c: Optional[float] = None,
) -> SessionWindows:
    """Aggregate one scanned session; releases its KernelTable afterwards.

    The table is reloaded from disk when ``scan_sessions`` already released it.
    When ``family_index`` holds the temperature column, it is filled with the
    window temperature above ``temperature_reference_c``.

    Args:
      session: A scanned session.
      selected: Kernel names with their own family.
      family_index: Column of each family.
      window_ns: Nominal window length.
      idle_power_mw: Power subtracted for the dynamic energy.
      temperature_reference_c: Reference of the temperature column.

    Returns:
      The session's window rows, family activity and target power.

    Raises:
      ValueError: If telemetry and kernels do not overlap.
    """
    name, scan = session.name, session.scan
    logger.info("\n=== Aggregating %s ===", name)
    trace = load_power_trace(session.telemetry)
    analysis_start = max(trace.first_ns, scan.first_start)
    analysis_end = min(trace.last_ns, scan.last_end)
    if analysis_end <= analysis_start:
        raise ValueError(f"Telemetry and kernels do not overlap in {name}")
    table, scan.table = scan.table, None
    if table is None:
        table = load_kernels(session.kernels)
    activity_ns, union_active_ns = aggregate_family_activity(
        table,
        family_columns(table, selected, family_index),
        len(family_index),
        analysis_start,
        analysis_end,
        window_ns,
    )
    del table
    rows, target_power_w = build_windows(
        trace,
        union_active_ns,
        analysis_start,
        analysis_end,
        window_ns,
        idle_power_mw,
    )
    if TEMPERATURE_FAMILY in family_index:
        add_temperature_column(
            trace,
            rows,
            activity_ns,
            family_index[TEMPERATURE_FAMILY],
            temperature_reference_c,
        )
    for index, row in enumerate(rows):
        row["session"] = name
        row["session_window_index"] = index
    return SessionWindows(name, rows, activity_ns, target_power_w)
