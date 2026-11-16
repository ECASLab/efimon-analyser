# efimon-analyser

Bayesian analyser for the GPU sessions captured with the EfiMon NVIDIA
profiler. It aligns NVML telemetry with the CUPTI kernel timeline in fixed
windows, fits a non-negative Bayesian regression of window power on the time
each kernel family is active, and attributes the measured energy to kernel
families and individual kernel events.

## Install

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[plots,test]'
```

The `plots` extra installs matplotlib, which is only needed with `--plot`.

## Input sessions

A session is a directory with the profiler outputs:

- `gpu_telemetry.csv`: NVML power, temperature and clocks.
- `kernel_activity.csv` or a gzip-compressed `kernel_activity.csv.gz`: the
  CUPTI kernel timeline.

## Attribute energy in one session

`nvidia-kernel-energy` fits the kernel-family model on the first part of one
session, evaluates it on the final part, and distributes the measured energy
to kernel families and CUPTI events.

```bash
nvidia-kernel-energy \
  --telemetry "$SESSION/gpu_telemetry.csv" \
  --kernels "$SESSION/kernel_activity.csv" \
  --window-ms 200 \
  --output-dir "$SESSION/bayesian-kernel-energy-analysis" \
  --plot
```

The output holds posterior energy by kernel family, model coefficients,
window diagnostics, a JSON summary and `bayesian_kernel_event_energy.csv`.
The event file holds `event_id`, `start_ns`, `end_ns`, `kernel_family` and the
posterior mean and 95% bounds of each event's energy; join it to
`kernel_activity.csv` on `event_id` for the remaining CUPTI columns. The
method conserves measured energy, but it does not create per-kernel ground
truth: kernels that always run together can remain non-identifiable and get
wide credible intervals.

By default (`--attribution model`) each window's measured dynamic energy is
split in proportion to the fitted terms `b0 + sum(beta_f * activity_f)`; the
intercept's share is reported as baseline energy instead of being charged to
kernels. `--attribution active-fraction` reproduces the earlier rule, which
scaled energy by the kernel-busy fraction and ignored `b0`.

The analysis requires at least `--min-windows-per-parameter` (default 3)
training windows per regression parameter and otherwise stops with a suggested
`--max-families`; `--allow-underdetermined` overrides this. The summary's
`identifiability` block lists terms that are linear combinations of others in
training and families that are active in the holdout but never in training.

The truncated final window of a session (shorter than
`--min-window-fraction`, default 0.9, of `--window-ms`) is marked `excluded`
and left out of fitting, metrics and plots; its energy is still attributed.

Window and session predictions carry two 95% intervals. The credible interval
reflects only coefficient uncertainty; the predictive interval adds the
posterior observation noise and is the one expected to contain measured
values.

## Pool many sessions into one model

`nvidia-kernel-energy-multisession` fits one kernel-family model across every
session below `--sessions-root`. Symlinks to session directories work, so one
root can combine several campaigns. The final 20% of every session is the
temporal holdout. Sessions are loaded one at a time, so peak memory is one
session's trace.

```bash
nvidia-kernel-energy-multisession \
  --sessions-root training-sessions \
  --output-dir training-sessions-model \
  --window-ms 300 \
  --max-families 64 \
  --temperature-feature \
  --plot
```

`--max-families` is the number of kernel columns, including `__other__`;
kernels are ranked by their total duration over all training sessions. Use
sessions from a single GPU model: every kernel has one coefficient, so mixing
platforms averages their power.

`--temperature-feature` adds the window temperature above
`--temperature-reference-c` (default 25 C) as one more non-negative term, in
W per degree C. It is reported as the `__temperature__` family.

## Library use

```python
from pathlib import Path
from energy_attribution import SingleSessionConfig, analyze_single_session

result = analyze_single_session(
    SingleSessionConfig(
        telemetry=Path("session/gpu_telemetry.csv"),
        kernels=Path("session/kernel_activity.csv"),
    )
)
print(result.summary["kernel_families"])
```

`analyze_*` functions only read their inputs; `run_*` functions also write the
outputs as the commands do.

## Tests and style

```bash
python3 -m pytest
```

The regression tests run every command on deterministic synthetic sessions
and compare each CSV and JSON output with the files in
`energy_attribution/tests/golden/`. Code follows the Google Python style:
pyink with 80 columns and pylint with `pylintrc`, also run by
`pre-commit`.
