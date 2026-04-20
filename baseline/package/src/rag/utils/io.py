"""File I/O helpers shared across pipeline stages."""

import json
import pathlib
from typing import Any


def load_json(path: str | pathlib.Path) -> Any:
    """Read and parse a JSON file, trying UTF-8 then UTF-8-BOM encodings.

    Args:
        path: Path to the JSON file (string or ``pathlib.Path``).

    Returns:
        Parsed Python object (dict, list, etc.).

    Raises:
        RuntimeError: If the file cannot be decoded with either encoding.
    """
    p = pathlib.Path(path)
    for enc in ("utf-8", "utf-8-sig"):
        try:
            return json.loads(p.read_text(encoding=enc))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Cannot read {path} as UTF-8")


def save_json(data: Any, path: str | pathlib.Path, *, indent: int = 2) -> None:
    """Serialise ``data`` to a JSON file, creating parent directories as needed.

    Args:
        data: JSON-serialisable Python object.
        path: Destination file path (string or ``pathlib.Path``).
        indent: JSON indentation level (default 2).
    """
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=indent, ensure_ascii=False), encoding="utf-8")
