"""scenes_manifest.json: по каждой найденной дате — сцена, облачность в поле, решение и причина."""
from __future__ import annotations

import json
from pathlib import Path

NAME = "scenes_manifest.json"


def load(out_dir: Path) -> dict:
    p = Path(out_dir) / NAME
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save(out_dir: Path, run: dict, entries: dict[str, dict], previous: dict) -> Path:
    """Записи прошлых запусков по датам вне текущего интервала сохраняются."""
    merged = {e["date"]: e for e in previous.get("scenes", [])}
    merged.update(entries)
    doc = {**run, "scenes": [merged[d] for d in sorted(merged)]}
    p = Path(out_dir) / NAME
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return p
