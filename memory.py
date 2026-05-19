"""
Persistent memory store using a local JSON file.
Survives browser refreshes and server restarts, unlike the
previous in-memory list which reset on every run.
"""

import json
import os
from datetime import datetime

MEMORY_FILE = "research_memory.json"


def _load() -> list:
    """Load memory from disk, return empty list if file doesn't exist."""
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def _save(data: list) -> None:
    """Write memory to disk."""
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_memory(content: str, query: str = "") -> None:
    """
    Save a memory entry with timestamp and optional query label.
    Each entry is a dict so it's queryable later.
    """
    data = _load()
    data.append({
        "timestamp": datetime.now().isoformat(),
        "query":     query,
        "content":   content
    })
    _save(data)


def get_memory() -> list:
    """Return all memory entries from disk."""
    return _load()


def get_last_report() -> str:
    """
    Return the content of the most recent memory entry.
    Used by app.py on startup to restore previous context
    even after a server restart.
    """
    data = _load()
    if not data:
        return ""
    return data[-1].get("content", "")


def clear_memory() -> None:
    """Wipe all memory — used for fresh research sessions."""
    _save([])
