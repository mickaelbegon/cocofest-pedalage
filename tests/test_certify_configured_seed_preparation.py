import copy

import pytest

from scripts.certify_configured_seed_preparation import _validate_result


def _document():
    return {
        "configurations": {"ipopt": {
            "formulation": "dynamic", "mechanical_formulation": "reduced",
            "ipopt_linear_solver": "ma57", "ipopt_c_compile": False,
            "constant_crank_torque": .1, "crank_torque_role": "resistive",
            "cycles_per_window": 1,
        }},
        "results": [{
            "requested_cycles": 1, "success": True, "solver_success": True,
            "physical_success": True, "solver": "ipopt", "error": None,
            "validated_cycles": 1, "physically_validated_cycles": 1,
            "nlp_solver_stats": [{"return_status": "Solve_Succeeded", "success": True, "iter_count": 12}],
            "windows": [{"feasibility": {"passes_tolerance": True, "trajectories_finite": True,
                                          "maximum_bound_violation": 1e-9}}],
            "nlp_crank_diagnostics": {"is_physical": True, "issues": []},
            "physical_crank_diagnostics": {"is_physical": True, "issues": []},
            "mechanical_equivalence_audit": {
                "available": True, "passes_tolerance": True,
                "passes_configuration_tolerance": True, "passes_velocity_tolerance": True,
                "passes_physical_crank_velocity_bounds": True,
                "maximum_configuration_projection_error_rad": 1e-12,
                "maximum_physical_crank_velocity_bound_violation_rad_s": 0.0,
            },
            "terminal_capacity_reserve": {"physiological_domain_valid": True},
            "min_A_capacity_ratio": .9,
            "control_saturation": [{"lower": .1, "maximum": .2, "upper": .3}],
        }],
    }


def test_validates_complete_dynamic_reduced_ma57_seed():
    metrics = _validate_result(_document(), .1)
    assert metrics["solver_return_status"] == "Solve_Succeeded"
    assert metrics["validated_cycles"] == 1


@pytest.mark.parametrize("path", [
    ("results", 0, "physical_success"),
    ("results", 0, "terminal_capacity_reserve", "physiological_domain_valid"),
])
def test_refuses_missing_physical_or_domain_validation(path):
    document = _document()
    target = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = False
    with pytest.raises(ValueError):
        _validate_result(document, .1)


def test_refuses_excessive_bound_violation():
    document = copy.deepcopy(_document())
    document["results"][0]["windows"][0]["feasibility"]["maximum_bound_violation"] = 2e-7
    with pytest.raises(ValueError, match="bound violation"):
        _validate_result(document, .1)
