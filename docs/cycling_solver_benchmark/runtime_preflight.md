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

The useful production call is **inside the scientific worker**, after its
imports and required patch installation, immediately before solver construction:

```python
from cocofest.runtime_preflight import collect_runtime_preflight

report = collect_runtime_preflight(
    solver="acados",
    require_acados_ding_patch=True,
    expected_versions={"casadi": "3.7.2"},  # use the campaign's own exact pin
)
```

Attach that JSON object to the result's runtime provenance or save it beside the
resolved configuration. Collect a second report after solver construction if
loaded-library evidence is needed. A launcher-side report describes only the
launcher process: its packages, environment and loaded patches do not establish
anything about a worker running under another interpreter or prefix. This first
tranche exposes the API/CLI; legacy drivers are not automatically instrumented.

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
not every historical Bioptim patch. Full worker wiring, backend smoke probes and
the supported-version matrix remain further work for issue #10.
