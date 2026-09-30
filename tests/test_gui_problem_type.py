import pytest

from cocofest.simulation.config import SimulationConfig
from cocofest.simulation.gui_problem_type import (
    BILATERAL_COMBINED,
    INDEPENDENT_ARMS,
    UNILATERAL,
    main_problem_type,
    set_main_problem_type,
)


def test_main_problem_types_only_change_the_bilateral_mechanical_choice():
    source = SimulationConfig(cycles=37, formulation="dynamic", signed_crank_torque=.15)
    combined = set_main_problem_type(source, BILATERAL_COMBINED)
    assert main_problem_type(combined) == BILATERAL_COMBINED
    assert combined.bilateral_reduced is True
    assert combined.cycles == source.cycles
    assert combined.signed_crank_torque == source.signed_crank_torque
    assert set_main_problem_type(combined, UNILATERAL).bilateral_reduced is False


def test_independent_arms_cannot_be_silently_converted_to_main_schema():
    with pytest.raises(ValueError, match="propre requête"):
        set_main_problem_type(SimulationConfig(), INDEPENDENT_ARMS)
