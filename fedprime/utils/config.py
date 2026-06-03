from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".json"}:
        return json.loads(text)

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "YAML configs require PyYAML. Install dependencies with "
            "`pip install -r requirements.txt`, or use a .json config."
        ) from exc

    return yaml.safe_load(text)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def get_nested(config: dict[str, Any], key: str, default: Any = None) -> Any:
    cur: Any = config
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

