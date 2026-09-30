"""Persistent, dependency-free launch history for the desktop simulation GUI."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4


class CampaignHistory:
    """Store GUI launch receipts; results remain authoritative in each output directory."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def entries(self) -> list[dict[str, object]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except (OSError, json.JSONDecodeError):
            return []
        entries = data.get("entries") if isinstance(data, dict) else None
        return [dict(entry) for entry in entries if isinstance(entry, dict)] if isinstance(entries, list) else []

    def add(self, **details: object) -> str:
        entry_id = uuid4().hex
        entries = self.entries()
        entries.append({
            "id": entry_id,
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "en cours",
            **details,
        })
        self._write(entries)
        return entry_id

    def set_status(self, entry_id: str | None, status: str) -> None:
        if not entry_id:
            return
        entries = self.entries()
        for entry in entries:
            if entry.get("id") == entry_id:
                entry["status"] = status
                entry["ended_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                self._write(entries)
                return

    def _write(self, entries: list[dict[str, object]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps({"schema_version": 1, "entries": entries}, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
        temporary.replace(self.path)
