"""Check the constant-load temperature slope estimator."""

import unittest
from pathlib import Path

import numpy as np


from energy_attribution.temperature_slope import (
    COOLDOWN_START_MARKER,
    MEASUREMENT_START_MARKER,
    estimate_temperature_slope,
    measurement_phase_ns,
)
from energy_attribution.io import (
    KernelTable,
)

WINDOW_NS = 100


def table(names, codes, starts):
    """A kernel table with one 10 ns event per start time."""
    starts = np.asarray(starts, dtype=np.int64)
    return KernelTable(
        path=Path("kernels.csv"),
        start_ns=starts,
        end_ns=starts + 10,
        name_codes=np.asarray(codes),
        names=names,
    )


def session(name, load_w, temperatures, slope, rng):
    """Windows of one session: a warmup at low load, then a constant phase.

    The warmup is cold and idle while the phase is hot and loaded, so a
    regression across all windows would mix load with temperature.
    """
    rows, power = [], []
    warmup = [30.0] * 5
    for index, temperature in enumerate(warmup + list(temperatures)):
        loaded = index >= len(warmup)
        rows.append(
            {
                "session": name,
                "start_ns": index * WINDOW_NS,
                "end_ns": (index + 1) * WINDOW_NS,
                "temperature_c": temperature,
            }
        )
        power.append(
            (load_w if loaded else 2.0)
            + slope * temperature
            + rng.normal(0.0, 0.01)
        )
    phase = (len(warmup) * WINDOW_NS, len(rows) * WINDOW_NS)
    return rows, power, phase


class MeasurementPhaseTests(unittest.TestCase):
    """The phase runs between the measurement and cooldown markers."""

    def test_phase_is_found_by_marker_substring(self):
        names = [
            "_ZN4main" + MEASUREMENT_START_MARKER + "Ev",
            "gemm",
            "_ZN4main" + COOLDOWN_START_MARKER + "Ev",
        ]
        kernels = table(names, [1, 0, 1, 2, 1], [5, 10, 20, 900, 950])
        self.assertEqual(measurement_phase_ns(kernels), (10, 900))

    def test_missing_marker_gives_no_phase(self):
        names = ["gemm", "_ZN4main" + MEASUREMENT_START_MARKER + "Ev"]
        self.assertIsNone(measurement_phase_ns(table(names, [0, 1], [5, 10])))


class EstimateTests(unittest.TestCase):
    """Pooled within-phase slope across sessions."""

    def test_recovers_slope_despite_load_confounding(self):
        rng = np.random.default_rng(3)
        rows, power, phases = [], [], {}
        for index, load in enumerate((20.0, 40.0, 60.0)):
            name = f"s{index}"
            heating = np.linspace(50.0, 50.0 + 10.0 * (index + 1), 40)
            session_rows, session_power, phases[name] = session(
                name, load, heating, 0.12, rng
            )
            rows += session_rows
            power += session_power
        estimate = estimate_temperature_slope(rows, power, phases, seed=1)
        self.assertAlmostEqual(estimate.coefficient_w_per_c, 0.12, places=3)
        self.assertLessEqual(estimate.lower_95_w_per_c, 0.12)
        self.assertGreaterEqual(estimate.upper_95_w_per_c, 0.12)
        self.assertEqual(estimate.sessions, ["s0", "s1", "s2"])
        self.assertEqual(estimate.windows, 120)
        self.assertEqual(
            estimate.summary()["coefficient_w_per_c"],
            estimate.coefficient_w_per_c,
        )

    def test_flat_or_unmarked_phases_give_no_estimate(self):
        rng = np.random.default_rng(4)
        rows, power, phase = session("flat", 30.0, [60.0] * 40, 0.12, rng)
        self.assertIsNone(
            estimate_temperature_slope(rows, power, {"flat": phase}, seed=1)
        )
        rows, power, _ = session(
            "unmarked", 30.0, np.linspace(50, 70, 40), 0.12, rng
        )
        self.assertIsNone(
            estimate_temperature_slope(rows, power, {"unmarked": None}, seed=1)
        )

    def test_negative_slope_gives_no_estimate(self):
        rng = np.random.default_rng(5)
        rows, power, phase = session(
            "cooling", 30.0, np.linspace(50, 70, 40), -0.2, rng
        )
        self.assertIsNone(
            estimate_temperature_slope(rows, power, {"cooling": phase}, seed=1)
        )


if __name__ == "__main__":
    unittest.main()
