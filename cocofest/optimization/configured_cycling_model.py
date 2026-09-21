"""Explicit Ding parameter variants and strict seed compatibility.

Factories are patched only inside the caller's context, before an OCP graph
exists. All supported fields are resolved from a Ding model plus the declared
overrides; the fingerprint covers the complete effective muscle parameters.
"""

from contextlib import contextmanager
from hashlib import sha256
import json
import math
import os
from pathlib import Path

import numpy as np


FIELD_ATTRIBUTES = {
    "Fmax": "fmax", "a_scale": "a_scale", "alpha_a": "alpha_a", "tau_fat": "tau_fat",
    "alpha_tau1": "alpha_tau1", "alpha_km": "alpha_km", "tau1_rest": "tau1_rest",
    "km_rest": "km_rest", "tauc": "tauc", "tau2": "tau2", "pd0": "pd0", "pdt": "pdt",
}
REQUIRED_FIELDS = frozenset(("Fmax", "a_scale", "alpha_a", "tau_fat"))
FINGERPRINT_KEY = "muscle_parameter_fingerprint"
WARMUP_CACHE_FINGERPRINT_ENV = "COCOFEST_CONFIGURED_MODEL_FINGERPRINT"


def resolve_model_config(declared):
    """Resolve a version-1 config, refusing unknown fields and invalid domains."""
    from cocofest.models.ding2007.ding2007_with_fatigue import DingModelPulseWidthFrequencyWithFatigue

    if not isinstance(declared, dict) or set(declared) != {"schema_version", "case_id", "provenance", "muscles"}:
        raise ValueError("Model config requires exactly schema_version, case_id, provenance, muscles")
    if type(declared["schema_version"]) is not int or declared["schema_version"] != 1:
        raise ValueError("Only model-config schema_version=1 is supported")
    for field in ("case_id", "provenance"):
        if not isinstance(declared[field], str) or not declared[field].strip():
            raise ValueError(f"Model config requires nonempty {field}")
    muscles = declared["muscles"]
    if not isinstance(muscles, dict) or not muscles:
        raise ValueError("Model config requires an explicit nonempty muscles mapping")
    resolved = {}
    for name, overrides in muscles.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(overrides, dict):
            raise ValueError("Every muscle must have a name and parameter mapping")
        if set(overrides) - set(FIELD_ATTRIBUTES) or not REQUIRED_FIELDS <= set(overrides):
            raise ValueError(f"{name}: unsupported fields or missing {sorted(REQUIRED_FIELDS)}")
        reference = DingModelPulseWidthFrequencyWithFatigue(muscle_name=name, stim_time=[0., .1])
        values = {key: overrides.get(key, getattr(reference, attribute))
                  for key, attribute in FIELD_ATTRIBUTES.items()}
        for key, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name}.{key} must be a finite number")
            valid_sign = (value <= 0 if key == "alpha_a" else value >= 0
                          if key in {"alpha_tau1", "alpha_km", "pd0"} else value > 0)
            if not valid_sign:
                raise ValueError(f"{name}.{key} is outside the supported Ding parameter domain")
        resolved[name] = {key: float(value) for key, value in values.items()}
    encoded = json.dumps({"schema_version": 1, "muscles": resolved}, sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
    return {"schema_version": 1, "case_id": declared["case_id"], "provenance": declared["provenance"],
            "muscles": resolved, FINGERPRINT_KEY: sha256(encoded.encode()).hexdigest()}


def apply_model_config(model, config):
    """Apply parameters before graph construction; synchronize a_rest/a_scale."""
    muscles = model.muscles_dynamics_model
    names = [muscle.muscle_name for muscle in muscles]
    if len(names) != len(set(names)) or set(names) != set(config["muscles"]):
        raise ValueError("Model config muscle names do not match the actual cycling model")
    # Validate all destinations before making the first assignment.
    for muscle in muscles:
        for attribute in FIELD_ATTRIBUTES.values():
            if not hasattr(muscle, attribute):
                raise ValueError(f"Unsupported model attribute {muscle.muscle_name}.{attribute}")
    for muscle in muscles:
        for key, value in config["muscles"][muscle.muscle_name].items():
            setattr(muscle, FIELD_ATTRIBUTES[key], value)
        muscle.a_rest = muscle.a_scale
    return {"muscle_names": names, "actual_parameters": {
        muscle.muscle_name: {key: float(getattr(muscle, attribute))
                             for key, attribute in FIELD_ATTRIBUTES.items()} for muscle in muscles},
        FINGERPRINT_KEY: config[FINGERPRINT_KEY], "applied_before_ocp_construction": True}


@contextmanager
def configured_model_factories(config, records, *, modules=None):
    """Patch both factory aliases, including standard warmup and refinement."""
    if modules is None:
        from examples.fes_multibody.cycling import cycling_pulse_width_mhe as mhe
        from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as periodic
        modules = (mhe, periodic)
    originals = [(module, module.set_fes_model) for module in modules]
    previous_fingerprint = os.environ.get(WARMUP_CACHE_FINGERPRINT_ENV)
    try:
        # The standard-warmup cache is constructed inside this patched-factory
        # context. The .bioMod source alone cannot identify two configured Ding
        # variants, so make the cache key and its metadata variant-specific.
        os.environ[WARMUP_CACHE_FINGERPRINT_ENV] = config[FINGERPRINT_KEY]
        for module, original in originals:
            def factory(*args, _original=original, **kwargs):
                model = _original(*args, **kwargs)
                records.append(apply_model_config(model, config))
                return model
            module.set_fes_model = factory
        yield
    finally:
        if previous_fingerprint is None:
            os.environ.pop(WARMUP_CACHE_FINGERPRINT_ENV, None)
        else:
            os.environ[WARMUP_CACHE_FINGERPRINT_ENV] = previous_fingerprint
        for module, original in originals:
            module.set_fes_model = original


def require_seed_fingerprint(path, fingerprint):
    """An unversioned or different-parameter seed is not certified for a variant."""
    path = Path(path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as seed:
        metadata = json.loads(str(seed["metadata__json"].item())) if "metadata__json" in seed else {}
    actual = metadata.get(FINGERPRINT_KEY)
    if actual != fingerprint:
        raise ValueError(f"Seed {path} has incompatible {FINGERPRINT_KEY}: {actual!r}; expected {fingerprint}. "
                         "Generate a seed through the configured wrapper for this model variant.")
    return {"path": str(path), FINGERPRINT_KEY: actual, "compatible": True}


def annotate_generated_seed(path, config, *, condition):
    """Attach model provenance to a newly generated output (never an input seed)."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as seed:
        payload = {key: seed[key].copy() for key in seed.files}
    metadata = json.loads(str(payload["metadata__json"].item())) if "metadata__json" in payload else {}
    metadata.update({FINGERPRINT_KEY: config[FINGERPRINT_KEY],
                     "configured_muscle_parameters": config["muscles"],
                     "configured_model_case_id": config["case_id"], "configured_condition": condition})
    payload["metadata__json"] = np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False))
    # The wrapper only calls this for outputs proven absent before this run.
    from tempfile import NamedTemporaryFile
    import os
    with NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **payload)
    os.replace(temporary, path)
