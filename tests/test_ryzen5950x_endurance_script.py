from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github/scripts/run_ryzen5950x_endurance_sweep.sh"
PROTOCOL = ROOT / "docs/cycling_solver_benchmark/ryzen5950x_endurance.md"


def test_ryzen_endurance_script_keeps_the_scientific_contract():
    script = SCRIPT.read_text()

    assert 'resistances_nm="${RESISTANCES_NM:-0.10 0.15 0.20}"' in script
    assert 'max_rhos="${MAX_RHOS:-2000}"' in script
    assert 'rho_threads="${RHO_THREADS:-16}"' in script
    assert "--max-consecutive-failing 2" in script
    assert "--retry-failed-rho-without-advance" in script
    assert "--mechanical-formulation reduced" in script
    assert "5 scientific-radau5 auto true" in script
    assert 'case_slug="${solver}-r5-fatigue-endurance"' in script
    assert "--madnlp-c-compile" not in script  # delegated to the common runner
    assert "--acados-ipopt-recovery" in script
    assert "--acados-ipopt-fallback-advance" in script
    assert "--acados-ipopt-recovery-collocation-degree 5" in script
    assert "--terminal-wheel-qdot-bound-margin 0.3" in script
    assert 'export OMP_NUM_THREADS="${NUMERIC_THREADS:-1}"' in script
    assert "endurance-summary.csv" in script


def test_ryzen_protocol_does_not_claim_that_smt_is_free_speedup():
    protocol = PROTOCOL.read_text()

    assert "16` cœurs physiques" in protocol
    assert "comparer `16` et `30`" in protocol
    assert 'signed:+0.15' in protocol
    assert 'STRATEGIES="ipopt acados-ipopt"' in protocol
    assert 'STRATEGIES="madnlp"' in protocol
    assert "unconfirmed_endurance_stop" in protocol
