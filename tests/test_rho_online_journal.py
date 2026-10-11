import json

import pytest

from scripts.materialize_rho_online_journal import load_journal, materialize


def test_online_journal_materializes_only_a_contiguous_finite_prefix(tmp_path):
    journal = tmp_path / "nlp0_metrics.jsonl"
    journal.write_text(
        json.dumps({"call": 0, "timings": {"t_wall_total": 0.1}}) + "\n"
        + json.dumps({"call": 1, "timings": {"t_wall_total": 0.2}}) + "\n",
        encoding="utf-8",
    )
    output = materialize(journal)
    assert json.loads(output.read_text()) == load_journal(journal)


@pytest.mark.parametrize(
    "text, message",
    [
        ('{"call": 1}\n', "not contiguous"),
        ('{"call": 0, "value": NaN}\n', "non-finite"),
        ('{"call": 0}', "incomplete final line"),
    ],
)
def test_online_journal_refuses_corruption(tmp_path, text, message):
    journal = tmp_path / "nlp0_metrics.jsonl"
    journal.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_journal(journal)
