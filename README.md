# efimon-analyser

Bayesian analyser for GPU sessions captured with the EfiMon NVIDIA profiler.
It aligns NVML power telemetry with the CUPTI kernel timeline in fixed time
windows, learns how much power each kernel family draws, and splits the
measured energy among kernel families and individual kernel events.

| Command | Use it to |
|---|---|
| `nvidia-kernel-energy` | Attribute the energy of **one session** to its kernels. |
| `nvidia-kernel-energy-multisession` | Train **one model on many sessions** and attribute each one. |

## 1. Installation

Requirements: Python 3.10 or newer. A GPU is not needed; the analyser only
reads CSV files.

```bash
git clone https://github.com/ECASLab/efimon-analyser.git
cd efimon-analyser
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[plots]'
```

- `plots` installs matplotlib, needed only for `--plot`.
- Use `'.[plots,test]'` to also install pytest and pylint.

Check the installation:

```bash
nvidia-kernel-energy --help
```

Every command also runs as a module, for example
`python3 -m energy_attribution.commands.kernel_energy --help`.

## 2. Input data

A **session** is one directory written by the profiler:

| File | Required columns | Notes |
|---|---|---|
| `gpu_telemetry.csv` | `timestamp_ns`, `power_mw` | Add `temperature_c` to use `--temperature-feature`. |
| `kernel_activity.csv` | `start_ns`, `end_ns`, `kernel_name` | Rows sorted by `start_ns`; may be gzip-compressed as `kernel_activity.csv.gz`. `event_id` links the per-event output back to the trace. |

Both files must use the same nanosecond clock, as the profiler writes them,
and the kernels must come from a single GPU.

For several sessions, place each session directory (or a symlink to it)
inside one root directory:

```text
training-sessions/
├── qdnn-fp16-rep-01/   gpu_telemetry.csv  kernel_activity.csv.gz
├── md-fp32-rep-01/     gpu_telemetry.csv  kernel_activity.csv.gz
└── llm-qwen14b-rep-01 -> ../campaign/llm/qwen14b-rep-01
```

## 3. Use cases

### Attribute the energy of one session

```bash
nvidia-kernel-energy \
  --telemetry session/gpu_telemetry.csv \
  --kernels session/kernel_activity.csv \
  --window-ms 300 \
  --output-dir session/kernel-energy \
  --plot
```

The first 80% of the windows fit the model and the last 20% check it
(temporal holdout). The output gives the energy of every kernel family and of
every kernel event.

For very long traces, add `--skip-event-output` to keep the summaries and
skip the per-event CSV.

### Train one model on many sessions

```bash
nvidia-kernel-energy-multisession \
  --sessions-root training-sessions \
  --output-dir training-model \
  --window-ms 300 \
  --max-families 64 \
  --temperature-feature \
  --plot
```

- Each session keeps its own last 20% as holdout.
- Sessions are loaded one at a time, so memory stays at one session's trace.
- Use sessions from **one GPU model**: each kernel family gets one
  coefficient, so mixing GPUs averages their power.
- Sessions without valid kernels (for example an idle capture) are skipped
  and listed in the summary.

### Recommended settings

| Situation | Suggested options |
|---|---|
| Typical LLM or benchmark capture | `--window-ms 300 --max-families 64` |
| The GPU warms up during the capture | add `--temperature-feature` |
| The tool stops with "Only N training windows for M parameters" | lower `--max-families`, shorten `--window-ms` or add sessions |
| Exploratory run on a short trace | `--allow-underdetermined` (results are prior-dominated) |
| Reproduce a run exactly | keep the same `--seed` and options |

## 4. Options

Run any command with `--help` to see the same list with defaults.

**Input and output**

| Option | Default | Description |
|---|---|---|
| `--telemetry` | required | `gpu_telemetry.csv` (one-session command). |
| `--kernels` | required | `kernel_activity.csv[.gz]` (one-session command). |
| `--sessions-root` | required | Directory of sessions (multi-session command). |
| `--output-dir` | `bayesian-kernel-energy-analysis` (one session), required (many) | Where CSV, JSON and PNG files are written. |
| `--plot` | off | Write diagnostic PNG files. |
| `--skip-event-output` | off | Skip the per-event CSV (one-session command). |

**Windows and data split**

| Option | Default | Description |
|---|---|---|
| `--window-ms` | 200 | Window length. Shorter windows give more data points but average fewer power samples. |
| `--train-fraction` | 0.8 | Leading share of each session used for fitting; the rest is the holdout. |
| `--min-window-fraction` | 0.9 | Windows shorter than this share of `--window-ms` (the last, truncated one) are left out of fitting and metrics; their energy is still attributed. |
| `--idle-power-mw` | 0 | Power subtracted from every sample before attribution. 0 attributes the whole device power. |

**Model**

| Option | Default | Description |
|---|---|---|
| `--max-families` | 64 | Number of kernel-family columns, including `__other__`. The longest-running kernels get their own column; the rest share `__other__`. |
| `--attribution` | `model` | `model`: the intercept's share of each window is reported as baseline energy. `active-fraction`: legacy rule that splits all busy-time energy among families. |
| `--temperature-feature` | off | Adds the window temperature above `--temperature-reference-c` as one more term (W per °C), so warm-up power is not charged to kernels (multi-session command). |
| `--temperature-reference-c` | 25 | Temperature at which that term is zero. |
| `--min-windows-per-parameter` | 3 | Required training windows per model parameter. |
| `--allow-underdetermined` | off | Fit anyway when there are too few windows (only warns). |

**Sampler** (rarely changed)

| Option | Default | Description |
|---|---|---|
| `--prior-scale-w` | 250 | Prior standard deviation of each scaled coefficient, in W. |
| `--noise-prior-scale-w` | 20 | Scale of the prior on the observation noise, in W. |
| `--burn-in` | 1000 | Gibbs iterations discarded at the start. |
| `--posterior-draws` | 2000 | Draws kept. |
| `--thin` | 2 | Iterations between kept draws. |
| `--seed` | 2026 | Random seed. |

## 5. Outputs

| File | Content |
|---|---|
| `bayesian_kernel_energy_summary.json` | Settings, holdout metrics, identifiability report and energy totals. Start here. |
| `bayesian_kernel_family_energy.csv` | Energy of each kernel family (mean, median and 95% interval, mJ), event count and busy time. |
| `bayesian_kernel_coefficients.csv` | Power of each family per unit of activity, in W, with 95% interval. |
| `bayesian_energy_windows.csv` | One row per window: split, measured energy, predicted power and attributed energy. |
| `bayesian_session_energy.csv` | Measured and predicted holdout energy of each session. |
| `bayesian_kernel_event_energy.csv` | Energy of every kernel event (one-session command). Join to `kernel_activity.csv` on `event_id`. |

With `--plot`: `bayesian_kernel_family_energy.png`,
`bayesian_kernel_power_fit.png`, `bayesian_kernel_energy_test_scatter.png`
and `bayesian_session_energy.png`.

### What to check in the summary

| Key | Meaning |
|---|---|
| `test_power_metrics.rmse_w`, `.r_squared` | Power error on the holdout. R² is not informative when the measured power is almost constant; use RMSE then. |
| `test_power_metrics.predictive_interval_95_coverage` | Share of holdout windows inside the 95% predictive interval. |
| `test_session_energy_metrics.total_energy_error_pct` | Signed error of the holdout energy. |
| `identifiability.aliased_terms` | Families that always run together; their split between them is set by the prior. |
| `identifiability.test_families_unseen_in_training` | Families active only in the holdout; their power comes from the prior. |
| `conservation_error_mj` | Should be close to 0: attributed + baseline energy equals the measured energy. |

Energy conservation shows that the split adds up, not that each kernel's
share is physically exact: there is no per-kernel ground truth.

## 6. Library use

```python
from pathlib import Path
from energy_attribution import SingleSessionConfig, analyze_single_session

result = analyze_single_session(
    SingleSessionConfig(
        telemetry=Path("session/gpu_telemetry.csv"),
        kernels=Path("session/kernel_activity.csv"),
        window_ms=300,
    )
)
print(result.summary["test_power_metrics"]["rmse_w"])
for row in result.family_rows[:5]:
    print(row["kernel_family"], row["energy_posterior_mean_mj"])
```

`analyze_*` functions only read their inputs; `run_*` functions also write
the files listed above, exactly as the commands do.

## 7. Development

```bash
python3 -m pip install -e '.[plots,test]'
python3 -m pytest
pre-commit run --all-files
```

The regression tests run each command on synthetic sessions and compare
every CSV and JSON output with `energy_attribution/tests/golden/`. Code
follows the Google Python style: pyink with 80 columns and pylint with
`pylintrc`.
