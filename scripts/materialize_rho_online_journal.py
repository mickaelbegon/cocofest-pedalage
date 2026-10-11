#!/usr/bin/env python3
"""Validate and materialize an append-only RHO JSONL journal offline.

The online RHO writer emits one solver-call record per line.  This command is
the explicit boundary where that durable prefix becomes the legacy JSON array
consumed by detailed analysis code.  It never runs an NLP or declares a
mechanical/physiological result valid.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _finite(value) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def load_journal(path: Path) -> list[dict]:
    """Read a contiguous, finite call sequence without accepting a partial line."""
    path = Path(path).resolve(strict=True)
    rows: list[dict] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.endswith("\n"):
                raise ValueError(f"Journal has an incomplete final line at {line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at journal line {line_number}") from error
            if not isinstance(row, dict) or row.get("call") != len(rows):
                raise ValueError(f"Journal call sequence is not contiguous at line {line_number}")
            if not _finite(row):
                raise ValueError(f"Journal contains a non-finite value at line {line_number}")
            rows.append(row)
    if not rows:
        raise ValueError("Journal is empty")
    return rows


def materialize(path: Path, output: Path | None = None) -> Path:
    path = Path(path).resolve(strict=True)
    rows = load_journal(path)
    destination = (
        Path(output).resolve()
        if output is not None
        else path.with_name(path.name.removesuffix(".jsonl") + ".json")
    )
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite an existing materialization: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(materialize(args.journal, args.output))


if __name__ == "__main__":
    main()
