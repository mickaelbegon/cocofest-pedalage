"""Write the two bundled Ding variants descriptor without running an OCP.

Seed and certificate paths may refer to pending outputs. This command only
declares their locations; it does not create or approve a seed certificate.
"""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples/fes_multibody/cycling/two_model_ding_campaign"


def build_variants(*, run_directory, reduced_profile, seed_x0p5=None,
                   seed_certificate_x0p5=None, seed_x2=None, seed_certificate_x2=None):
    """Return absolute input paths for the provided alpha_a x0.5 and x2 cases."""
    directory = Path(run_directory).expanduser().resolve()
    profile = str(Path(reduced_profile).expanduser().resolve())
    variants = []
    for suffix, seed, certificate in (
        ("x0p5", seed_x0p5, seed_certificate_x0p5),
        ("x2", seed_x2, seed_certificate_x2),
    ):
        variants.append({
            "model_id": f"ding_triceps_alpha_a_{suffix}",
            "model_config": str((EXAMPLES / f"ding_triceps_alpha_a_{suffix}.json").resolve(strict=True)),
            "weights_config": str((EXAMPLES / f"weights_triceps_alpha_a_{suffix}.json").resolve(strict=True)),
            "seed": str(Path(seed if seed is not None else directory / suffix / "seed.npz")
                        .expanduser().resolve()),
            "seed_certificate": str(Path(certificate if certificate is not None else
                                         directory / suffix / "seed-certificate.json")
                                    .expanduser().resolve()),
            "reduced_profile": profile,
        })
    for key in ("seed", "seed_certificate"):
        if variants[0][key] == variants[1][key]:
            raise ValueError(f"The two Ding variants need distinct {key} paths")
    return variants


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", required=True, type=Path,
                        help="Default seeds/certificates live in x0p5/ and x2/ below this directory")
    parser.add_argument("--reduced-profile", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Default: RUN_DIRECTORY/variants.json; never overwritten")
    for suffix in ("x0p5", "x2"):
        parser.add_argument(f"--seed-{suffix}", type=Path)
        parser.add_argument(f"--seed-certificate-{suffix}", type=Path)
    args = vars(parser.parse_args(argv))
    destination = (args.pop("output") or args["run_directory"] / "variants.json").expanduser().resolve()
    try:
        variants = build_variants(**args)
        if destination in {Path(value) for variant in variants for key, value in variant.items()
                           if key != "model_id"}:
            raise ValueError("Descriptor output must differ from all input artifact paths")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(variants, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Variant descriptor written: {destination}; no solver started or certificate created")
    missing = sorted({value for variant in variants for key, value in variant.items()
                      if key != "model_id" and not Path(value).is_file()})
    if missing:
        print("Campaign preparation is incomplete. Required artifacts still missing:")
        for path in missing:
            print(f"  {path}")
    print("The campaign builder will verify the seeds and their physical-review certificates.")


if __name__ == "__main__":
    main()
