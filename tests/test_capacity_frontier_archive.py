import json
from hashlib import sha256

import pytest

from cocofest.optimization.capacity_frontier_archive import (
    deserialize_slice, serialize_slice, slice_from_raw_reports, slice_from_summary,
)


ORDER = ["Delt_ant", "Delt_post", "Biceps", "Triceps"]
CONDITION = sha256(b"condition").hexdigest()


def raw_receipt(*, support, capacity, epsilon_upper=None):
    return {
        "condition_sha256": CONDITION, "muscle_order": ORDER, "capacity_domain": [.1, 1.],
        "support": support, "epsilon_upper": epsilon_upper or [None] * 4,
        "support_smoke": {"initial_capacity_ratio": capacity},
    }


def write(path, content):
    path.write_text(json.dumps(content))
    return path


def test_raw_receipts_roundtrip_to_an_exact_checked_slice(tmp_path):
    paths = [
        write(tmp_path / "minimum.json", raw_receipt(support=[0, 0, 0, 1], capacity=[.99, .99, .96, .32])),
        write(tmp_path / "ant.json", raw_receipt(support=[0, 0, 0, 1], capacity=[.6, .99, .96, .35], epsilon_upper=[.6, None, None, None])),
        write(tmp_path / "post.json", raw_receipt(support=[0, 0, 0, 1], capacity=[.99, .6, .96, .345], epsilon_upper=[None, .6, None, None])),
        write(tmp_path / "biceps.json", raw_receipt(support=[0, 0, 0, 1], capacity=[.99, .99, .6, .37], epsilon_upper=[None, None, .6, None])),
    ]
    built = slice_from_raw_reports(paths, checkpoint_cycle=120)
    export = tmp_path / "slice.json"
    export.write_text(json.dumps(serialize_slice(built)))
    restored = deserialize_slice(export)

    assert restored.checkpoint_cycle == 120
    assert restored.exchange_slopes()["Biceps"] > restored.exchange_slopes()["Delt_ant"]


def test_summary_recovery_requires_numerically_accepted_evidence(tmp_path):
    reports = [
        {"numeric_gate_passed": True, "support": [0, 0, 0, 1], "capacity_ratio": [.99, .99, .96, .32],
         "sha256": sha256(b"min").hexdigest(), "epsilon_upper": None},
    ]
    for index, name in enumerate(ORDER[:-1]):
        epsilon = [None] * 4
        epsilon[index] = .6
        capacity = [.99, .99, .96, .32]
        capacity[index], capacity[-1] = .6, (.35, .345, .37)[index]
        reports.append({"numeric_gate_passed": True, "support": [0, 0, 0, 1], "capacity_ratio": capacity,
                        "sha256": sha256(name.encode()).hexdigest(), "epsilon_upper": epsilon})
    summary = write(tmp_path / "summary.json", {"condition_sha256": CONDITION, "muscle_order": ORDER, "reports": reports})
    recovered = slice_from_summary(summary, checkpoint_cycle=140, capacity_domain=(.1, 1.), epsilon_upper=.6)
    assert recovered.checkpoint_cycle == 140

    reports[-1]["numeric_gate_passed"] = False
    summary.write_text(json.dumps({"condition_sha256": CONDITION, "muscle_order": ORDER, "reports": reports}))
    with pytest.raises(ValueError, match="does not contain"):
        slice_from_summary(summary, checkpoint_cycle=140, capacity_domain=(.1, 1.), epsilon_upper=.6)
