from hashlib import sha256

import pytest

from cocofest.optimization.capacity_frontier_interpolation import (
    TricepsExchangeSlice, audit_linear_checkpoint_interpolation,
    build_exact_conditioned_exchange_model, diagnostic_frontier_weight_hint,
    lookup_exact_exchange,
)


def slice(cycle, seed):
    digest = lambda value: sha256((seed + value).encode()).hexdigest()
    return TricepsExchangeSlice(sha256(seed.encode()).hexdigest(), cycle,
        ("Delt_ant", "Delt_post", "Biceps", "Triceps"), (.1, 1.), (.99, .99, .96, .32),
        {"Delt_ant": (.6, .99, .96, .35), "Delt_post": (.99, .6, .96, .345),
         "Biceps": (.99, .99, .6, .37)},
        {"Delt_ant": .6, "Delt_post": .6, "Biceps": .6},
        {"minimum_triceps": digest("minimum"), "Delt_ant": digest("ant"),
         "Delt_post": digest("post"), "Biceps": digest("biceps")})


def test_exact_conditioned_model_returns_slopes_only_for_measured_fast_state():
    first, second = slice(120, "first"), slice(160, "second")
    model = build_exact_conditioned_exchange_model((first, second))
    assert lookup_exact_exchange(model, first.condition_sha256)["Biceps"] > lookup_exact_exchange(model, first.condition_sha256)["Delt_ant"]
    assert lookup_exact_exchange(model, sha256(b"unknown").hexdigest()) is None
    assert len(model["model_sha256"]) == 64


def test_exchange_slice_rejects_an_epsilon_witness_without_a_real_tradeoff():
    invalid = slice(120, "invalid")
    object.__setattr__(invalid, "epsilon_capacity", {"Delt_ant": (.6, .99, .96, .31),
        "Delt_post": (.99, .6, .96, .345), "Biceps": (.99, .99, .6, .37)})
    with pytest.raises(ValueError, match="trade"):
        invalid.validate()


def test_model_rejects_mixing_atlas_domains_even_when_other_shapes_match():
    first, second = slice(120, "first"), slice(160, "second")
    object.__setattr__(second, "capacity_domain", (.2, 1.))
    with pytest.raises(ValueError, match="atlas domain"):
        build_exact_conditioned_exchange_model((first, second))


def test_checkpoint_interpolation_is_a_real_holdout_and_returns_a_conservative_band():
    lower, holdout, upper = slice(120, "lower"), slice(140, "holdout"), slice(160, "upper")
    # Construct a deliberately linear synthetic atlas.  The c140 state remains
    # held out: only c120 and c160 form the predicted slope.
    object.__setattr__(lower, "epsilon_capacity", {
        "Delt_ant": (.6, .99, .96, .34), "Delt_post": (.99, .6, .96, .33),
        "Biceps": (.99, .99, .6, .35),
    })
    object.__setattr__(holdout, "epsilon_capacity", {
        "Delt_ant": (.6, .99, .96, .35), "Delt_post": (.99, .6, .96, .345),
        "Biceps": (.99, .99, .6, .37),
    })
    object.__setattr__(upper, "epsilon_capacity", {
        "Delt_ant": (.6, .99, .96, .36), "Delt_post": (.99, .6, .96, .36),
        "Biceps": (.99, .99, .6, .39),
    })

    audit = audit_linear_checkpoint_interpolation(lower, holdout, upper)

    assert audit["passed"]
    assert audit["fraction"] == .5
    assert audit["relative_error"] == pytest.approx({"Delt_ant": 0., "Delt_post": 0., "Biceps": 0.})
    assert audit["conservative_slope_interval"]["Biceps"] == pytest.approx(
        [audit["observed_exchange_slopes"]["Biceps"]] * 2
    )
    assert audit["online_use"] == "forbidden_pending_causal_RHO_holdout"


def test_checkpoint_interpolation_rejects_a_nonheldout_cycle_order():
    lower, holdout, upper = slice(120, "lower"), slice(120, "holdout"), slice(160, "upper")
    with pytest.raises(ValueError, match="strictly between"):
        audit_linear_checkpoint_interpolation(lower, holdout, upper)


def test_diagnostic_weight_hint_rewards_a_useful_exchange_but_prioritizes_triceps():
    weights = diagnostic_frontier_weight_hint({"Delt_ant": .2, "Delt_post": .4, "Biceps": .8})
    assert weights["Biceps"] < weights["Delt_post"] < weights["Delt_ant"] < weights["Triceps"]
    assert sum(weights.values()) / len(weights) == pytest.approx(1.)
