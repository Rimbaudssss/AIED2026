from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def add_artifact(
    manifest: dict,
    *,
    kind: str,
    model: str,
    dataset: str,
    path: str,
    meta: Optional[dict] = None,
) -> None:
    entry = {
        "kind": str(kind),
        "model": str(model),
        "path": str(path),
        "meta": dict(meta) if meta is not None else {},
    }
    entry["meta"].setdefault("dataset", str(dataset))
    manifest.setdefault("artifacts", []).append(entry)


def write_manifest(manifest: dict, out_json: str) -> None:
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if "created_at" not in manifest:
        manifest["created_at"] = datetime.now().isoformat(timespec="seconds")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)
