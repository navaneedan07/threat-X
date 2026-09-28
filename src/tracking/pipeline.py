"""Tracking orchestration pipeline for executing multi-frame threat tracking.

Connects anomaly detection frame inputs to the ThreatTracker, converts
tracked trajectories into standard Trajectory contract representations
via tracks_to_trajectories, and serializes the results to JSON output files
matching backend/services/trajectory_service.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

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
    """Execute the end-to-end tracking pipeline from detection frames to serialized trajectories.

    Parameters
    ----------
    detector_results : Sequence | dict | str | Path
        Anomaly detector results. Can be a list of frame dicts, a dict containing
        a 'frames' key, or a filesystem path to an anomaly JSON output file.
    output_path : str | Path | None
        Path where trajectories.json should be written. If None, defaults to
        data/processed/trajectories.json.
    default_year : int | None
        Fallback year for ID minting (default: 2026). If None, uses frame timestamps.
    merge_existing : bool
        If True and output_path exists, load and merge with existing output.
    config : TrackingConfig | None
        Optional custom TrackingConfig instance.

    Returns
    -------
    dict[str, Any]
        The complete top-level payload written to disk, containing '_meta'
        and 'trajectories' keyed by threat_id.
    """
    frames = _load_frames(detector_results)

    tracker = ThreatTracker(config=config, default_year=default_year)
    tracks = tracker.process_frames(frames)
    trajectories = tracks_to_trajectories(tracks)

    serialized_trajectories: dict[str, dict[str, Any]] = {
        threat_id: traj.to_dict() for threat_id, traj in trajectories.items()
    }

    target_path = Path(output_path) if output_path is not None else DEFAULT_OUTPUT_FILE
    target_path.parent.mkdir(parents=True, exist_ok=True)

    existing_data: dict[str, Any] = {}
    if merge_existing and target_path.exists():
        try:
            with target_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    existing_data = loaded
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read existing trajectories file %s: %s", target_path, exc)

    existing_trajectories = existing_data.get("trajectories", {})
    if not isinstance(existing_trajectories, dict):
        existing_trajectories = {}

    merged_trajectories = {**existing_trajectories, **serialized_trajectories}

    payload: dict[str, Any] = {
        "_meta": {
            "source": "src.tracking.pipeline",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "count": len(merged_trajectories),
            "tracks_count": len(merged_trajectories),
        },
        "trajectories": merged_trajectories,
    }

    with target_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info("Wrote %d trajectories to %s", len(merged_trajectories), target_path)
    return payload


def _load_frames(
    data: Sequence[dict[str, Any]] | dict[str, Any] | str | Path,
) -> list[dict[str, Any]]:
    """Normalize input detector results into a list of frame dicts."""
    if isinstance(data, (str, Path)):
        path = Path(data)
        if not path.exists():
            raise FileNotFoundError(f"Detector output file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            content = json.load(f)
        return _extract_frames_from_dict_or_list(content)
    return _extract_frames_from_dict_or_list(data)


def _extract_frames_from_dict_or_list(content: Any) -> list[dict[str, Any]]:
    """Extract list of frame dicts from loaded content."""
    if isinstance(content, list):
        return content
    if isinstance(content, dict):
        if "frames" in content and isinstance(content["frames"], list):
            return content["frames"]
        return [content]
    if isinstance(content, Sequence):
        return list(content)
    return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Threat-X tracking pipeline runner")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to detector output JSON file (e.g. output/anomalies_amphan.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_FILE),
        help=f"Path to write trajectories JSON (default: {DEFAULT_OUTPUT_FILE})",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2026,
        help="Default year for threat ID minting",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        default=True,
        help="Merge with existing output file if present",
    )
    args = parser.parse_args()

    run_tracking_pipeline(
        detector_results=args.input,
        output_path=args.output,
        default_year=args.year,
        merge_existing=args.merge,
    )
