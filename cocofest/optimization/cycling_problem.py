"""Small, solver-agnostic construction seam for periodic cycling problems.

The historical cycling driver owns its CLI and all model-specific policy. This
module owns only the hand-off from validated construction inputs to an NMPC
instance, so other consumers need not import the large command-line driver.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


@dataclass
class CyclingRunContext:
    """Live periodic problem plus a snapshot of construction provenance."""

    nmpc: Any
    model: Any
    mhe_info: Any
    cycling_info: Any
    simulation_conditions: dict[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)


class CyclingProblemFactory:
    """Create a context through an injected ``prepare_nmpc``-style builder."""

    def __init__(self, nmpc_builder: Callable[[Any, Any, Any, dict[str, Any]], Any]):
        if not callable(nmpc_builder):
            raise TypeError("nmpc_builder must be callable")
        self._nmpc_builder = nmpc_builder

    def build(
        self,
        *,
        model: Any,
        mhe_info: Any,
        cycling_info: Any,
        simulation_conditions: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> CyclingRunContext:
        """Build once, copying mutable conditions before passing them onward."""

        conditions = dict(simulation_conditions)
        nmpc = self._nmpc_builder(model, mhe_info, cycling_info, conditions)
        return CyclingRunContext(
            nmpc=nmpc,
            model=model,
            mhe_info=mhe_info,
            cycling_info=cycling_info,
            simulation_conditions=conditions,
            metadata=dict(metadata or {}),
        )
