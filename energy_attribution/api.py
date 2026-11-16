"""Public library API for Bayesian GPU energy attribution.

``analyze_*`` functions only read their inputs and return a typed result;
``run_*`` functions additionally write the CSV/JSON/PNG outputs to
``config.output_dir`` exactly as the command-line tools do.
"""

from energy_attribution.config import (
    MultiSessionConfig,
    SingleSessionConfig,
)
from energy_attribution.reporting.outputs import (
    write_multisession,
    write_single_session,
)
from energy_attribution.workflows import (
    multisession,
    single_session,
)


def analyze_single_session(
    config: SingleSessionConfig,
) -> single_session.SingleSessionResult:
    """Fit one session and attribute its energy; writes nothing."""
    config.validate()
    return single_session.analyze(config)


def analyze_multisession(
    config: MultiSessionConfig,
) -> multisession.MultiSessionResult:
    """Fit one pooled model across sessions; writes nothing."""
    config.validate()
    return multisession.analyze(config)


def run_single_session(
    config: SingleSessionConfig,
) -> single_session.SingleSessionResult:
    """Analyze one session and write its outputs."""
    result = analyze_single_session(config)
    write_single_session(result)
    return result


def run_multisession(
    config: MultiSessionConfig,
) -> multisession.MultiSessionResult:
    """Analyze several sessions with one pooled model and write its outputs."""
    result = analyze_multisession(config)
    write_multisession(result)
    return result
