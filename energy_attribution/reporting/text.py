"""Short labels and console summaries shared by the analyses."""

# Subscripts inside f-strings need single quotes before Python 3.12.
# pylint: disable=inconsistent-quotes


import re

from energy_attribution.diagnostics import log_identifiability_warnings
from energy_attribution.families import OTHER_FAMILY, TEMPERATURE_FAMILY


def short_kernel_name(name, limit=52):
    if name == OTHER_FAMILY:
        return "Other kernels"
    if name == TEMPERATURE_FAMILY:
        return "GPU temperature"
    matrix_vector = re.search(
        r"mul_mat_vec_qIL9ggml_type(\d+)ELi(\d+)ELb([01])ELb([01])ELb([01])",
        name,
    )
    if matrix_vector:
        metric_type, integer, first, second, third = matrix_vector.groups()
        return (
            f"mul_mat_vec_q<t{metric_type},i{integer},"
            f"b{first}{second}{third}>"
        )
    if "rope_norm" in name:
        return "rope_norm<half>" if "6__half" in name else "rope_norm<float>"
    attention = re.search(
        r"flash_attn_tileILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)", name
    )
    if attention:
        return "flash_attn_tile<" + ",".join(attention.groups()) + ">"
    if name.startswith("_Z"):
        position = 2
        while position < len(name) and name[position].isdigit():
            position += 1
        if position > 2:
            length = int(name[2:position])
            candidate = name[position : position + length]
            if candidate:
                return candidate
    return name if len(name) <= limit else name[: limit - 1] + "…"


def short_session_name(name):
    return re.sub(r"-[0-9a-f]{10,}$", "", name)


def format_optional(value, pattern):
    return "n/a" if value is None else pattern.format(value)


def print_coverage(test_metrics, session_metrics):
    credible = format_optional(
        test_metrics["credible_interval_95_coverage"], "{:.1%}"
    )
    predictive = format_optional(
        test_metrics["predictive_interval_95_coverage"], "{:.1%}"
    )
    print(
        "Test 95% coverage per window: "
        f"credible {credible}, predictive {predictive}"
    )
    print(
        "Test 95% coverage of session energy: "
        f"credible {session_metrics['credible_interval_95_coverage']:.1%}, "
        f"predictive {session_metrics['predictive_interval_95_coverage']:.1%}"
    )


def print_energy_balance(accounting, *, detailed):
    if detailed:
        print(
            "Measured dynamic energy: "
            f"{accounting['measured_dynamic_energy_mj']:.3f} mJ"
        )
        print(
            "Bayesian attributed energy: "
            f"{accounting['attributed_posterior_mean_mj']:.3f} mJ"
        )
    print(
        "Baseline (b0) energy: "
        f"{accounting['baseline_posterior_mean_mj']:.3f} mJ"
    )
    if detailed:
        print(
            "Unassigned energy: "
            f"{accounting['unassigned_dynamic_energy_mj']:.3f} mJ"
        )
    print(f"Conservation error: {accounting['conservation_error_mj']:.6g} mJ")


def print_test_fit(test_metrics, session_metrics):
    print(f"Test power RMSE: {test_metrics['rmse_w']:.3f} W")
    if test_metrics["r_squared"] is not None:
        print(f"Test power R-squared: {test_metrics['r_squared']:.4f}")
    print_coverage(test_metrics, session_metrics)


def print_single_session_report(result):
    """Console summary of :class:`SingleSessionResult`."""
    summary = result.summary
    print("=== Bayesian kernel-energy attribution ===")
    print(f"Attribution mode: {result.config.attribution}")
    print(f"Kernel families: {len(result.families)}")
    print(f"Kernel events: {summary['kernel_events']}")
    print_test_fit(result.test_metrics, result.session_metrics)
    print_energy_balance(result.accounting, detailed=True)
    log_identifiability_warnings(result.identifiability, short_kernel_name)
    print(f"Results: {result.config.output_dir}")


def print_multisession_report(result):
    """Console summary of :class:`MultiSessionResult`."""
    print("\n=== Multi-session Bayesian kernel-energy attribution ===")
    print(f"Included sessions: {len(result.included_sessions)}")
    print(f"Excluded sessions: {len(result.excluded_sessions)}")
    print(f"Kernel families: {len(result.families)}")
    print_test_fit(result.test_metrics, result.session_metrics)
    print_energy_balance(result.accounting, detailed=False)
    log_identifiability_warnings(result.identifiability, short_kernel_name)
    print(f"Results: {result.config.output_dir}")
