"""Fixture loader utility.

Reads JSON fixture files from data/samples/ so all service adapters have a
single, consistent way to load fallback data.  Never call this outside the
services layer — routes must not reference fixtures directly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Resolve samples directory relative to repo root regardless of CWD.
_REPO_ROOT = Path(__file__).resolve().parents[2]  # backend/services/fixtures.py -> repo root
_SAMPLES_DIR = Path(os.getenv("DATA_ROOT", str(_REPO_ROOT / "data"))) / "samples"


def _load(filename: str) -> dict[str, Any]:
    """Load a JSON fixture file from data/samples/. Raises RuntimeError on failure."""
    path = _SAMPLES_DIR / filename
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Fixture file '{path}' not found. "
            "Ensure data/samples/ contains the demo fixture files."
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Fixture file '{path}' is not valid JSON: {exc}") from exc


def load_threats() -> list[dict[str, Any]]:
    """Return the list of threat fixture objects."""
    data = _load("threats.json")
    return data["threats"]


def load_trajectories() -> dict[str, Any]:
    """Return trajectory fixtures keyed by threat_id."""
    data = _load("trajectories.json")
    return data["trajectories"]


def load_precursors() -> dict[str, Any]:
    """Return precursor series fixtures keyed by threat_id."""
    data = _load("precursors.json")
    return data["precursors"]


def load_transitions() -> dict[str, Any]:
    """Return transition fixtures keyed by threat_id."""
    data = _load("transitions.json")
    return data["transitions"]


def load_footprints() -> dict[str, Any]:
    """Return footprint fixtures keyed by threat_id."""
    data = _load("footprints.json")
    return data["footprints"]


def load_alerts() -> list[dict[str, Any]]:
    """Return alert fixture list."""
    data = _load("alerts.json")
    return data["alerts"]
