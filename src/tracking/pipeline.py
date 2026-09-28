"""Tracking orchestration pipeline for executing multi-frame threat tracking.

Connects anomaly detection frame inputs to the ThreatTracker and converts
tracked trajectories into standard Trajectory contract representations suitable
for downstream services and JSON output files.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.shared.contracts import Trajectory, TrajectoryPoint
from src.tracking.tracker import ThreatTracker, TrackingConfig


def run_tracking_pipeline(
    detector_results: Sequence[dict[str, Any]] | str | Path,
    output_path: str | Path | None = None,
    default_year: int = 2026,
    merge_existing: bool = False,
    config: TrackingConfig | None = None,
) -> dict[str, Any]:
    """Run tracking pipeline over frame sequence or detector result JSON.

    Args:
        detector_results: List of frame dicts or path to detector output JSON file.
        output_path: Optional path to save resulting trajectories JSON.
        default_year: Fallback year for ID minting if timestamp lacks year.
        merge_existing: If True and output_path exists, load and merge with existing output.
        config: Custom TrackingConfig instance.

    Returns:
        Dict containing serialized trajectories and metadata.
    """
    existing_trajectories: dict[str, Any] = {}

    if output_path is not None:
        output_path = Path(output_path)
        if merge_existing and output_path.exists():
            try:
                with output_path.open("r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                    if isinstance(existing_data, dict) and "trajectories" in existing_data:
                        existing_trajectories = existing_data["trajectories"]
            except Exception:
                existing_trajectories = {}

    if isinstance(detector_results, (str, Path)):
        file_path = Path(detector_results)
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "frames" in data:
            frames = data["frames"]
        elif isinstance(data, list):
            frames = data
        else:
            frames = []
    else:
        frames = list(detector_results)

    tracker = ThreatTracker(config=config, default_year=default_year)
    tracker.process_frames(frames)

    trajectories_dict: dict[str, Any] = dict(existing_trajectories)

    for track in tracker.tracks:
        if not track.detections:
            continue

        steps: list[TrajectoryPoint] = []
        for det in track.detections:
            step = TrajectoryPoint(
                timestamp=det.timestamp,
                centroid_lat=det.centroid_lat,
                centroid_lon=det.centroid_lon,
            )
            steps.append(step)

        traj = Trajectory(
            threat_id=track.threat_id,
            historical_steps=steps,
        )
        trajectories_dict[track.threat_id] = traj.to_dict()

    payload = {
        "_meta": {
            "source": "src.tracking.pipeline",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "count": len(trajectories_dict),
        },
        "trajectories": trajectories_dict,
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    return payload
