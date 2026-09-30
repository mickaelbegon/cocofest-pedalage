from scripts.run_bilateral_fixed_weight_bo import candidate_from_coordinates


def test_side_specific_coordinates_create_distinct_normalized_arm_weights():
    candidate, geometry = candidate_from_coordinates(
        {"right__Delt_ant": .2, "right__Delt_post": 0., "right__Triceps": -.2,
         "left__Delt_ant": -.2, "left__Delt_post": 0., "left__Triceps": .2},
        total_torque_nm=1.6, initial_right_fraction=.5, min_weight=.25, max_weight=4.,
        separate_arm_weights=True,
    )
    assert geometry["mode"] == "side_specific"
    assert candidate.right_weights != candidate.left_weights
    assert candidate.right_weights["Delt_ant"] > candidate.left_weights["Delt_ant"]
