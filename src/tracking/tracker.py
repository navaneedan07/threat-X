"""Multi-detection association, track state management, and persistent threat IDs.

STATUS: Minimal core tracking engine implementing detection representation from
detector boxes, persistent THR-YYYY-NNNN threat identifiers, gap handling
according to configs/tracking.yaml semantics, and optimal bipartite matching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.shared.config import get_value, stage_config
from src.shared.contracts import THREAT_ID_PATTERN, parse_timestamp
from src.shared.geo import haversine_km

CONFIG_FILE = "tracking.yaml"


def _positive_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        num = float(value)
        return num if num > 0 else None
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        num = int(value)
        return num if num >= 0 else None
    except (TypeError, ValueError):
        return None


def _compute_iou(
    bbox1: tuple[float, float, float, float],
    bbox2: tuple[float, float, float, float],
) -> float:
    """Intersection-over-Union between two (lat_min, lat_max, lon_min, lon_max) boxes."""
    lat_min1, lat_max1, lon_min1, lon_max1 = bbox1
    lat_min2, lat_max2, lon_min2, lon_max2 = bbox2

    inter_lat = max(0.0, min(lat_max1, lat_max2) - max(lat_min1, lat_min2))
    inter_lon = max(0.0, min(lon_max1, lon_max2) - max(lon_min1, lon_min2))
    inter_area = inter_lat * inter_lon

    area1 = max(0.0, lat_max1 - lat_min1) * max(0.0, lon_max1 - lon_min1)
    area2 = max(0.0, lat_max2 - lat_min2) * max(0.0, lon_max2 - lon_min2)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0.0 else 0.0


@dataclass
class Detection:
    """An observed anomaly region at a single time step."""

    centroid_lat: float
    centroid_lon: float
    timestamp: datetime
    peak_zscore: float | None = None
    lat_min: float | None = None
    lat_max: float | None = None
    lon_min: float | None = None
    lon_max: float | None = None
    cell_count: int | None = None
    label_id: int | None = None

    def __post_init__(self) -> None:
        self.timestamp = parse_timestamp(self.timestamp)
        self.centroid_lat = float(self.centroid_lat)
        self.centroid_lon = float(self.centroid_lon)
        if self.peak_zscore is not None:
            self.peak_zscore = float(self.peak_zscore)
        if self.cell_count is not None:
            self.cell_count = int(self.cell_count)
        if self.label_id is not None:
            self.label_id = int(self.label_id)

    @classmethod
    def from_box(cls, box: dict[str, Any], timestamp: datetime | str) -> Detection:
        """Construct a Detection from an anomaly detector box dictionary."""
        return cls(
            centroid_lat=float(box["centroid_lat"]),
            centroid_lon=float(box["centroid_lon"]),
            timestamp=timestamp,
            peak_zscore=float(box["peak_zscore"]) if box.get("peak_zscore") is not None else None,
            lat_min=float(box["lat_min"]) if box.get("lat_min") is not None else None,
            lat_max=float(box["lat_max"]) if box.get("lat_max") is not None else None,
            lon_min=float(box["lon_min"]) if box.get("lon_min") is not None else None,
            lon_max=float(box["lon_max"]) if box.get("lon_max") is not None else None,
            cell_count=int(box["cell_count"]) if box.get("cell_count") is not None else None,
            label_id=int(box["label_id"]) if box.get("label_id") is not None else None,
        )

    @property
    def bbox(self) -> tuple[float, float, float, float] | None:
        """Bounding box tuple (lat_min, lat_max, lon_min, lon_max) if all present."""
        if None not in (self.lat_min, self.lat_max, self.lon_min, self.lon_max):
            return (self.lat_min, self.lat_max, self.lon_min, self.lon_max)
        return None


@dataclass(frozen=True)
class TrackingConfig:
    """Thresholds and settings loaded from configs/tracking.yaml.

    Null values remain None to preserve null semantics without guessing.
    """

    max_centroid_distance_km: float | None = None
    min_iou: float | None = None
    intensity_tolerance: float | None = None
    require_motion_consistency: bool = True
    max_gap_steps: int | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> TrackingConfig:
        loaded = config if config is not None else stage_config(CONFIG_FILE)
        assoc = loaded.get("association") or {}
        return cls(
            max_centroid_distance_km=_positive_or_none(
                get_value(assoc, "max_centroid_distance_km")
            ),
            min_iou=_positive_or_none(get_value(assoc, "min_iou")),
            intensity_tolerance=_positive_or_none(get_value(assoc, "intensity_tolerance")),
            require_motion_consistency=bool(
                get_value(assoc, "require_motion_consistency", default=True)
            ),
            max_gap_steps=_int_or_none(get_value(assoc, "max_gap_steps")),
        )

    def unresolved(self) -> list[str]:
        """Return list of configuration keys that remain null/unresolved."""
        unresolved_keys = []
        if self.max_centroid_distance_km is None:
            unresolved_keys.append("association.max_centroid_distance_km")
        if self.min_iou is None:
            unresolved_keys.append("association.min_iou")
        if self.intensity_tolerance is None:
            unresolved_keys.append("association.intensity_tolerance")
        if self.max_gap_steps is None:
            unresolved_keys.append("association.max_gap_steps")
        return unresolved_keys


@dataclass
class ActiveTrack:
    """In-memory state of an active or coasting threat track."""

    threat_id: str
    detections: list[Detection] = field(default_factory=list)
    consecutive_misses: int = 0
    status: str = "active"

    def __post_init__(self) -> None:
        if not THREAT_ID_PATTERN.match(self.threat_id):
            raise ValueError(f"Invalid threat_id: {self.threat_id}")

    @property
    def timestamps(self) -> list[datetime]:
        return [det.timestamp for det in self.detections]

    @property
    def centroids(self) -> list[tuple[float, float]]:
        return [(det.centroid_lat, det.centroid_lon) for det in self.detections]

    @property
    def latest_centroid(self) -> tuple[float, float] | None:
        if self.detections:
            last = self.detections[-1]
            return (last.centroid_lat, last.centroid_lon)
        return None

    @property
    def latest_detection(self) -> Detection | None:
        return self.detections[-1] if self.detections else None


class ThreatTracker:
    """Associates detections across time steps into persistent threat tracks."""

    def __init__(
        self,
        config: TrackingConfig | None = None,
        default_year: int | None = None,
        id_start: int = 1,
    ) -> None:
        self.config = config if config is not None else TrackingConfig.from_config()
        self.default_year = default_year
        self._next_id_seq = id_start
        self.tracks: list[ActiveTrack] = []

    def _mint_id(self, timestamp: datetime) -> str:
        year = timestamp.year if self.default_year is None else self.default_year
        threat_id = f"THR-{year:04d}-{self._next_id_seq:04d}"
        self._next_id_seq += 1
        return threat_id

    def _cost_matrix(
        self,
        active_tracks: list[ActiveTrack],
        detections: list[Detection],
    ) -> np.ndarray:
        matrix = np.full((len(active_tracks), len(detections)), float("inf"), dtype=float)

        for i, track in enumerate(active_tracks):
            last_det = track.latest_detection
            if last_det is None:
                continue

            for j, det in enumerate(detections):
                dist_km = float(
                    haversine_km(
                        last_det.centroid_lat,
                        last_det.centroid_lon,
                        det.centroid_lat,
                        det.centroid_lon,
                    )
                )

                if (
                    self.config.max_centroid_distance_km is not None
                    and dist_km > self.config.max_centroid_distance_km
                ):
                    continue

                if self.config.min_iou is not None:
                    if last_det.bbox is None or det.bbox is None:
                        continue
                    iou_val = _compute_iou(last_det.bbox, det.bbox)
                    if iou_val < self.config.min_iou:
                        continue

                if self.config.intensity_tolerance is not None:
                    if last_det.peak_zscore is None or det.peak_zscore is None:
                        continue
                    z_diff = abs(last_det.peak_zscore - det.peak_zscore)
                    if z_diff > self.config.intensity_tolerance:
                        continue

                matrix[i, j] = dist_km

        return matrix

    def update(
        self,
        frame: dict[str, Any] | list[Detection],
        timestamp: datetime | str | None = None,
    ) -> list[ActiveTrack]:
        """Update tracks with detections from one time step.

        Accepts either a detector frame dict (with 'timestamp' and 'boxes')
        or a list of Detection objects with an explicit timestamp.
        """
        if isinstance(frame, dict):
            frame_time = parse_timestamp(frame["timestamp"])
            boxes = frame.get("boxes", [])
            detections = [Detection.from_box(b, frame_time) for b in boxes]
        else:
            if timestamp is None:
                raise ValueError("timestamp must be provided when passing a list of detections")
            frame_time = parse_timestamp(timestamp)
            detections = list(frame)

        eligible_tracks = [t for t in self.tracks if t.status in ("active", "coasting")]

        if not eligible_tracks:
            for det in detections:
                threat_id = self._mint_id(frame_time)
                new_track = ActiveTrack(
                    threat_id=threat_id,
                    detections=[det],
                    consecutive_misses=0,
                    status="active",
                )
                self.tracks.append(new_track)
            return [t for t in self.tracks if t.status in ("active", "coasting")]

        if not detections:
            for track in eligible_tracks:
                track.consecutive_misses += 1
                if (
                    self.config.max_gap_steps is None
                    or track.consecutive_misses > self.config.max_gap_steps
                ):
                    track.status = "terminated"
                else:
                    track.status = "coasting"
            return [t for t in self.tracks if t.status in ("active", "coasting")]

        cost_matrix = self._cost_matrix(eligible_tracks, detections)

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()

        valid_row_mask = np.any(np.isfinite(cost_matrix), axis=1)
        valid_col_mask = np.any(np.isfinite(cost_matrix), axis=0)

        if np.any(valid_row_mask) and np.any(valid_col_mask):
            valid_rows = np.where(valid_row_mask)[0]
            valid_cols = np.where(valid_col_mask)[0]
            sub_matrix = cost_matrix[np.ix_(valid_rows, valid_cols)]

            try:
                sub_r, sub_c = linear_sum_assignment(sub_matrix)
                for sr, sc in zip(sub_r, sub_c, strict=True):
                    if np.isfinite(sub_matrix[sr, sc]):
                        r = int(valid_rows[sr])
                        c = int(valid_cols[sc])
                        track = eligible_tracks[r]
                        det = detections[c]
                        track.detections.append(det)
                        track.consecutive_misses = 0
                        track.status = "active"
                        matched_tracks.add(r)
                        matched_detections.add(c)
            except ValueError:
                finite_edges = []
                for sr_idx, r_orig in enumerate(valid_rows):
                    for sc_idx, c_orig in enumerate(valid_cols):
                        val = sub_matrix[sr_idx, sc_idx]
                        if np.isfinite(val):
                            finite_edges.append((float(val), int(r_orig), int(c_orig)))
                finite_edges.sort(key=lambda x: x[0])
                for _cost, r, c in finite_edges:
                    if r not in matched_tracks and c not in matched_detections:
                        track = eligible_tracks[r]
                        det = detections[c]
                        track.detections.append(det)
                        track.consecutive_misses = 0
                        track.status = "active"
                        matched_tracks.add(r)
                        matched_detections.add(c)

        for i, track in enumerate(eligible_tracks):
            if i not in matched_tracks:
                track.consecutive_misses += 1
                if (
                    self.config.max_gap_steps is None
                    or track.consecutive_misses > self.config.max_gap_steps
                ):
                    track.status = "terminated"
                else:
                    track.status = "coasting"

        for j, det in enumerate(detections):
            if j not in matched_detections:
                threat_id = self._mint_id(frame_time)
                new_track = ActiveTrack(
                    threat_id=threat_id,
                    detections=[det],
                    consecutive_misses=0,
                    status="active",
                )
                self.tracks.append(new_track)

        return [t for t in self.tracks if t.status in ("active", "coasting")]

    def process_frames(self, frames: list[dict[str, Any]]) -> list[ActiveTrack]:
        """Process a sequence of frames as emitted by anomaly detection."""
        for frame in frames:
            self.update(frame)
        return self.tracks
