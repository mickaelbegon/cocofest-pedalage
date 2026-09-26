# Runtime evidence and compatibility preflight

The first implementation for issue #10 is a read-only, standard-library report:

```sh
python -m cocofest.runtime_preflight --solver madnlp --linear-solver ma57 --output runtime-preflight.json
```

It records the current Python executable, platform and pointer width, installed
distribution metadata, already loaded Python module versions and paths, relevant
Linux native library mappings and selected runtime environment variables. It
observes the Bioptim solver factory and the ACADOS local Ding patch when their
modules are already loaded. A configuration requesting MadNLP `ma57` records
the interface spelling `Ma57Solver`; `mumps` records `MumpsSolver` using the
existing backend mapping. These are **requested** settings: neither is evidence
of the linear solver actually selected by MadNLP.

The report never imports scientific modules, calls `has_nlpsol`, installs a
patch, loads a library or starts an optimization. This makes it safe to collect
even when native plugin loading might crash the process. The CLI creates a new
file exclusively; it refuses to replace existing evidence.

## Integration seam

The report must be collected **inside the scientific worker**. The benchmark
comparison driver now automatically writes one sidecar per requested backend
when `output_json` is set. GUI and campaign runs using this driver inherit it.
Files are named `<result-stem>.runtime-preflight.<solver>.<attempt-id>.json`,
beside the result JSON, and are created exclusively. Every invocation gets a
new attempt identifier, preserving earlier runtime evidence.

The observation runs after the worker imports and before `solve_case`, including
its native warmup and code generation. The report identifies this stage as
`before_solve_case`. The optional ACADOS Ding patch can be installed later by
`solve_case`, after a warmup; its absence at this early stage is recorded without
requiring it or declaring an incompatibility. A caller needing post-patch
evidence can additionally collect a report at that later point:

```python
from cocofest.runtime_preflight import collect_runtime_preflight

report = collect_runtime_preflight(
    solver="acados",
    require_acados_ding_patch=True,
    expected_versions={"casadi": "3.7.2"},  # use the campaign's own exact pin
)
```

Each serialized solver result contains `runtime_preflight`, giving the stage,
sidecar path, collection/persistence status, report status and observation time.
Setup failures handled by the benchmark retain this metadata. If a native
process crashes before result serialization, the already-written sidecar remains.
Collection or write failures are reported as `unavailable`, with their error,
and do not block the solve. Incompatibility is likewise observational here;
this integration does not introduce a new solver rejection policy. Preflight
time is included in case end-to-end time and reported separately; it is outside
the solver's timings and hot-cycle populations.

Calls without `output_json` and legacy direct invocations of `solve_case` are
not automatically instrumented. The launcher remains free of scientific imports.
A launcher-side report would describe only the launcher's interpreter, not a
worker running under another prefix. Reports only describe native libraries
already mapped at the observation point; a second report after solver
construction is needed for newly loaded-library evidence.

## Interpretation and limits

`incompatible` means an actual contradiction was observed: an exact expected
loaded version differs, a loaded Bioptim namespace lacks the requested factory,
a required loaded ACADOS interface lacks its patch, or an inspected mapped ELF
has the wrong pointer width. Unloaded modules and missing distribution metadata
are `unknown`; source checkouts need not have distribution metadata. Loaded
versions take precedence as evidence, while installed metadata is retained
separately for diagnosing environment shadowing.

Without such contradictions the report is `incomplete`, never a successful
native compatibility certificate. CLI exit 0 means evidence collection succeeded
without a detected contradiction; exit 1 means incompatible, exit 2 invalid input
or an output error. Consumers must inspect the status field.

Mapped libraries do not prove a solver's selected linear backend, its version,
Fortran integer width, BLAS ABI or numerical convergence. Actual native versions
and linear backend remain unknown until an isolated smoke solve and solver
banner are captured. Existing `solve_ipopt_ma57_probe` remains the separate
opt-in native diagnostic. No native probe runs as part of this report. The
version check is caller-supplied exact pinning, not a curated supported-version
matrix. Patch evidence currently covers the opt-in ACADOS local Ding adapter,
not every historical Bioptim patch. Additional worker entry points, backend
smoke probes and the supported-version matrix remain further work for issue #10.
