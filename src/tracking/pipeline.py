"""Tracking pipeline orchestration from detector frames to canonical trajectories."""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.tracking.tracker import ThreatTracker, TrackingConfig
from src.tracking.trajectory import tracks_to_trajectories

logger = logging.getLogger(__name__)
DEFAULT_OUTPUT_DIR = Path(os.getenv("DATA_ROOT", "./data")) / "processed"
DEFAULT_OUTPUT_FILE = DEFAULT_OUTPUT_DIR / "trajectories.json"


def run_tracking_pipeline(
    detector_results: Sequence[dict[str, Any]] | dict[str, Any] | str | Path,
    output_path: str | Path | None = None,
    default_year: int | None = 2026,
    merge_existing: bool = True,
    config: TrackingConfig | None = None,
) -> dict[str, Any]:
    """Track detector frames and serialize canonical trajectory contracts."""
    frames = _load_frames(detector_results)
    tracker = ThreatTracker(config=config, default_year=default_year)
    tracks = tracker.process_frames(frames)
    trajectories = tracks_to_trajectories(tracks)
    serialized = {threat_id: trajectory.to_dict() for threat_id, trajectory in trajectories.items()}

    target_path = Path(output_path) if output_path is not None else DEFAULT_OUTPUT_FILE
    target_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if merge_existing and target_path.exists():
        try:
            with target_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                existing = loaded
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read existing trajectories file %s: %s", target_path, exc)

    existing_trajectories = existing.get("trajectories", {})
    if not isinstance(existing_trajectories, dict):
        existing_trajectories = {}
    merged = {**existing_trajectories, **serialized}
    payload: dict[str, Any] = {
        "_meta": {
            "source": "src.tracking.pipeline",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "count": len(merged),
            "tracks_count": len(merged),
        },
        "trajectories": merged,
    }
    with target_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("Wrote %d trajectories to %s", len(merged), target_path)
    return payload


def _load_frames(data: Sequence[dict[str, Any]] | dict[str, Any] | str | Path) -> list[dict[str, Any]]:
    """Load detector frames from memory or a JSON file."""
    if isinstance(data, (str, Path)):
        path = Path(data)
        if not path.exists():
            raise FileNotFoundError(f"Detector output file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            content = json.load(handle)
    else:
        content = data
    return _extract_frames_from_dict_or_list(content)


def _extract_frames_from_dict_or_list(content: Any) -> list[dict[str, Any]]:
    """Extract frame dictionaries from the detector's list or wrapper object."""
    if isinstance(content, list):
        return content
    if isinstance(content, dict):
        frames = content.get("frames")
        return frames if isinstance(frames, list) else [content]
    if isinstance(content, Sequence):
        return list(content)
    return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Threat-X tracking pipeline runner")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to detector output JSON file (e.g. output/anomalies_amphan.json)")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_FILE),
                        help=f"Path to write trajectories JSON (default: {DEFAULT_OUTPUT_FILE})")
    parser.add_argument("--year", type=int, default=2026,
                        help="Default year for threat ID minting")
    parser.add_argument("--merge", action="store_true", default=True,
                        help="Merge with existing output file if present")
    args = parser.parse_args()
    run_tracking_pipeline(detector_results=args.input, output_path=args.output,
                          default_year=args.year, merge_existing=args.merge)
