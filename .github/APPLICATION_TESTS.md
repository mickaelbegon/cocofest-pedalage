# Application and scientific test coverage

`application_tests_linux.yml` adds Linux PR coverage alongside the unchanged
Windows `run_tests_win.yml` shard jobs. It installs only the pip dependencies in
`application-test-requirements.txt`: no Bioptim/Biorbd build, ACADOS, HSL licence,
Julia or graphical display is required. The GUI is exercised headlessly and
the DOP853 cross-rollout tests use small synthetic trajectories.

`.github/test-suites.json` is the explicit module inventory. Every `test*.py`
under `tests/` must appear exactly once. An unclassified new module, duplicate,
deleted path or missing mandatory PR test fails the inventory/PR check. Add a
new module to `pr_unit` (configuration and fixture contracts), `pr_integration`
(small numerical/replay/subprocess tests), `scientific_root` (production stack,
NLP solves/code generation or scientific numerical validation), or the existing
Windows shard group. Each group identifies its CI job and rationale.

For a module whose scientific imports are deferred until a particular test,
`scientific_tests` can move explicit node IDs to the scientific job while its
pure tests remain in PR. The runner validates those overrides against the
collected names and reports deselection; it does not skip the scientific test
as though it had been executed in the light job.

The runner applies the registered pytest markers `unit`, `integration` and
`solver` from this inventory. Scientific modules are selected by file before
collection; `pytest -m 'not solver' tests` alone would still import their heavy
dependencies before deselection. These markers are available when using the
runner, which keeps the existing direct pytest/Windows commands unchanged.

```bash
python -m pip install -r .github/application-test-requirements.txt
python .github/scripts/run_application_tests.py --inventory-only
python .github/scripts/run_application_tests.py --suite pr -q
python .github/scripts/run_application_tests.py --suite pr --collect-only -q
```

Configuration round-trip, GUI round-trip and the continuous DOP853 fatigue-cost
regression must actually **pass** in PR. A skip, deselection or missing test for
these contracts is a failure. Other existing optional tests retain their skip
policy. `summary.json` separates collected/executed/passed/failed counts from
dependency skips and other skips; each skip includes its node id and reason.
JUnit and the summary are uploaded even after a test failure. A pytest failure
always fails the job; skips cannot stand in for the required contracts.

Select **Run workflow** for `Application tests (Linux)` to run the additional
`scientific-application-tests` job. It uses `environment.yml` and the same pinned
production Bioptim integration as the benchmark workflow. It runs all
`scientific_root` modules, including Bioptim factory/CLI integration, compilation,
small NLP solves and the Radau numerical regression. Optional HSL/ACADOS/MadNLP
capabilities must be installed for their corresponding assertions to execute;
reported skips are not evidence of solver validation. To run this suite locally
in an existing scientific environment:

```bash
python .github/scripts/run_application_tests.py --suite scientific -q
```

The long physical endurance and multi-solver campaigns remain in the existing
manual `cycling_solver_benchmark_linux.yml` workflow. Unit/scientific test
success is not a substitute for its convergence, mechanical and DOP853 audits.

The initial light-environment preflight exposed the pre-existing
`test_radau_coupled_mechanics_validation` sensitivity: SciPy can report stagnation
with a residual around `2.8e-16`, which the current scientific helper rejects.
This regression remains visible in the scientific suite; this CI change does
not modify the numerical implementation or relax its assertions.
