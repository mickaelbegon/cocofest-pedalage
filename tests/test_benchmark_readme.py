import re
import textwrap
from pathlib import Path


BENCHMARK_DOCS = Path(__file__).resolve().parents[1] / "docs" / "cycling_solver_benchmark"
MARKDOWN_DOCUMENTS = (
    BENCHMARK_DOCS / "README.md",
    BENCHMARK_DOCS / "development_history.md",
    BENCHMARK_DOCS / "resume_and_todo.md",
)


def test_display_equations_use_balanced_github_math_fences():
    for document in MARKDOWN_DOCUMENTS:
        lines = document.read_text(encoding="utf-8").splitlines()
        in_math = False
        math_blocks = 0

        for line_number, line in enumerate(lines, start=1):
            stripped = line.strip()
            assert stripped != "$$", (
                f"Display-math delimiter $$ at {document.name}:{line_number} can be "
                "split by Markdown constructs inside a multiline equation."
            )
            if stripped == "```math":
                assert not in_math, f"Nested math fence at {document.name}:{line_number}."
                in_math = True
                math_blocks += 1
            elif stripped == "```" and in_math:
                in_math = False

        assert math_blocks > 0, f"No display math found in {document.name}."
        assert not in_math, f"Unclosed GitHub math fence in {document.name}."


def test_benchmark_readme_separates_current_method_from_history():
    readme = (BENCHMARK_DOCS / "README.md").read_text(encoding="utf-8")
    history = (BENCHMARK_DOCS / "development_history.md").read_text(encoding="utf-8")

    assert "development_history.md" in readme
    assert "README.md" in history
    assert "resume_and_todo.md" in readme
    assert "resume_and_todo.md" in history
    assert "continuation_prompt.md" in readme
    assert "linux_32core_setup.md" in readme
    assert (BENCHMARK_DOCS / "continuation_prompt.md").is_file()
    assert (BENCHMARK_DOCS / "linux_32core_setup.md").is_file()
    assert "corriger plutôt que reproduire" in readme
    assert "ne doit jamais être utilisée comme oracle" in readme


def test_linux_32core_setup_tracks_workflow_solver_pins():
    setup = (BENCHMARK_DOCS / "linux_32core_setup.md").read_text(encoding="utf-8")
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "cycling_solver_benchmark_linux.yml"
    ).read_text(encoding="utf-8")

    pinned_variables = (
        "CASADI_VERSION",
        "CASADI_MADNLP_COMMIT",
        "BIOPTIM_PRODUCTION_COMMIT",
        "ACADOS_COMMIT",
        "LIBMAD_COMMIT",
        "JULIAC_COMMIT",
        "JULIA_VERSION",
    )
    for variable in pinned_variables:
        match = re.search(rf"^  {variable}:\s*[\"']?([^\"'\s]+)", workflow, re.MULTILINE)
        assert match, f"Missing {variable} in the benchmark workflow."
        assert match.group(1) in setup, f"The Linux setup does not document {variable}."

    assert "cocofest-rho32" in setup
    assert "cocofest-madnlp32" in setup
    assert "OMP_NUM_THREADS=1" in setup
    assert "CMAKE_BUILD_PARALLEL_LEVEL=32" in setup


def test_acados_ipopt_hybrid_reuses_the_shared_seed_physical_parameters():
    """The hybrid consumer must represent the exact OCP certified by its seed."""

    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "cycling_solver_benchmark_linux.yml"
    ).read_text(encoding="utf-8")
    hybrid_job = workflow.split("\n  acados-ipopt-hybrid:", maxsplit=1)[1].split(
        "\n  prepare-acados-stack:", maxsplit=1
    )[0]

    assert '--common-initial-solution benchmark-seed/common-reduced.npz' in hybrid_job
    assert '--objective fatigue' in hybrid_job
    assert '--cycles-per-window "${{ inputs.cycles_per_window }}"' in hybrid_job
    assert '--stimulations-per-cycle 30' in hybrid_job
    assert '--crank-assistance "${{ inputs.crank_assistance_nm }}"' in hybrid_job
    assert '--first-node-wheel-q-slack 0' in hybrid_job
    assert '--terminal-wheel-q-slack "${{ inputs.terminal_wheel_q_slack }}"' in hybrid_job
    assert "--acados-ipopt-recovery" in hybrid_job
    assert "--madnlp-linear-solver" not in hybrid_job
    assert "prepare-acados-stack" in hybrid_job
    assert "ACADOS_SOURCE_DIR:" not in hybrid_job
    assert "CONDA_PREFIX" in hybrid_job


def test_acados_hybrid_workflow_stays_below_github_expression_limit():
    """Long run scalars containing expressions cannot exceed GitHub's 21 kB limit."""

    repository_root = Path(__file__).resolve().parents[1]
    workflow = (
        repository_root / ".github" / "workflows" / "cycling_solver_benchmark_linux.yml"
    ).read_text(encoding="utf-8")
    hybrid_job = workflow.split("\n  acados-ipopt-hybrid:", maxsplit=1)[1].split(
        "\n  prepare-acados-stack:", maxsplit=1
    )[0]
    run_block = hybrid_job.split(
        "      - name: Run the full/reduced hybrid recovery gates", maxsplit=1
    )[1].split("\n      - name:", maxsplit=1)[0]

    run_scalar = run_block.split("        run: |\n", maxsplit=1)[1]
    assert len(textwrap.dedent(run_scalar).encode("utf-8")) < 21_000
    assert "bash .github/scripts/run_acados_dropout.sh" in run_block

    dropout_script = (
        repository_root / ".github" / "scripts" / "run_acados_dropout.sh"
    ).read_text(encoding="utf-8")
    assert "--acados-forced-iteration-cap-rhos" in dropout_script
    assert "--acados-forced-iteration-cap" in dropout_script
    assert "dropout-summary.json" in dropout_script


def test_rho7_recovery_uses_a_pre_failure_checkpoint_and_acados_recertification():
    repository_root = Path(__file__).resolve().parents[1]
    workflow = (
        repository_root / ".github" / "workflows" / "cycling_solver_benchmark_linux.yml"
    ).read_text(encoding="utf-8")
    script = (
        repository_root / ".github" / "scripts" / "run_acados_rho7_recovery.sh"
    ).read_text(encoding="utf-8")

    assert "inputs.cycles == 'acados_rho7_recovery'" in workflow
    assert "run_acados_rho7_recovery.sh" in workflow
    assert 'ACADOS_RHO7_REFERENCE_RUN_ID: "32377237731"' in workflow
    assert "reference-reduced-feasible-seed.npz" in workflow
    assert "--rho-prepared-checkpoint-windows 6" in script
    assert "prepared-after-6-for-7.npz" in script
    assert "--acados-ipopt-recovery-collocation-degree 5" in script
    assert '[[ "$qdot_margin" != "2.6" ]]' in script
    assert "--acados-max-iter 100" in script
    assert "--acados-maxiter-retries" not in script
    assert '"$BENCHMARK_MODE" == "acados_rho7_recovery"' in workflow
    assert "--wheel-qdot-bound-margin 2.6" in workflow
    assert "--acados-transfer-irk-rollout" in script
    assert "--acados-transfer-bound-homotopy" in script
    assert "--disable-periodic-fes-warmup-projection" in script
    assert "--acados-initial-irk-rollout" not in script
    assert "--acados-ipopt-fallback-advance" not in script
    assert ".results[0].validated_cycles >= 1" in script
    assert ".target_rho == 1 and .advanced == true" in script


def test_workflow_dispatch_respects_github_input_limit():
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "cycling_solver_benchmark_linux.yml"
    ).read_text(encoding="utf-8")
    dispatch_inputs = workflow.split("    inputs:\n", maxsplit=1)[1].split(
        "\npermissions:", maxsplit=1
    )[0]
    input_names = re.findall(r"^      ([a-z][a-z0-9_]*):$", dispatch_inputs, re.MULTILINE)

    assert len(input_names) <= 25
    assert "acados_dropout_rhos" not in input_names
    assert "acados_dropout_followup_rhos" not in input_names
    assert 'ACADOS_DROPOUT_RHOS: "100,150,430"' in workflow
    assert 'ACADOS_DROPOUT_FOLLOWUP_RHOS: "30"' in workflow
