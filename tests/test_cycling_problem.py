from cocofest.optimization.cycling_problem import CyclingProblemFactory


def test_factory_passes_a_private_copy_of_conditions_and_preserves_provenance():
    calls = []

    def builder(model, mhe_info, cycling_info, conditions):
        calls.append((model, mhe_info, cycling_info, conditions))
        conditions["builder_only"] = True
        return object()

    original = {"solver": "ipopt"}
    context = CyclingProblemFactory(builder).build(
        model="model",
        mhe_info="mhe",
        cycling_info="cycle",
        simulation_conditions=original,
        metadata={"profile": "K5"},
    )

    assert calls[0][:3] == ("model", "mhe", "cycle")
    assert original == {"solver": "ipopt"}
    assert context.simulation_conditions == {"solver": "ipopt", "builder_only": True}
    assert context.metadata == {"profile": "K5"}


def test_factory_rejects_non_callable_builder():
    try:
        CyclingProblemFactory(None)
    except TypeError as error:
        assert "nmpc_builder" in str(error)
    else:
        raise AssertionError("a non-callable builder must fail")
