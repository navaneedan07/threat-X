"""Deterministic threat lifecycle state machine.

Assigns a lifecycle state to every time step of a threat's history:

    FORMATION → INTENSIFICATION → EXPANSION → PEAK → DECAY

and reports the discrete events ``MERGER``, ``SPLIT``, ``WEAKENING`` and
``RELOCATION``.

No training, no model, no probability. This is a rule-based state machine, which
is why it can ship now while the learned transition model
(``src/models/transition/``) still waits on paired data.

Two deliberate properties
-------------------------
**The phase rules are threshold-free.** They use the sign of step-to-step change
and the relative size of an intensity change against a footprint change. There is
no tuned constant anywhere, so no phase can be an artefact of a number somebody
picked.

**Events that need a threshold report themselves as not evaluated.** Relocation
needs a distance cut-off and weakening needs an intensity tolerance. Both live in
``configs/tracking.yaml`` as ``null`` because they must be derived from observed
displacement in real data. Rather than defaulting them, this module returns an
explicit "not evaluated" with the config key that would enable it. A lifecycle
that quietly invented a 50 km relocation radius would be worse than one that
admits it cannot tell yet.

MERGER and SPLIT are never derivable from a single threat's history — they are
properties of two tracks. They are always reported as not evaluated here, with a
pointer to the tracker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.shared.config import get_value, stage_config
from src.shared.contracts import (
    LifecycleEvent,
    LifecycleEventRecord,
    LifecyclePhase,
    ThreatHistory,
    ThreatObject,
    format_timestamp,
)
from src.shared.geo import step_distances_km

CONFIG_FILE = "tracking.yaml"

_PHASE_ORDER: tuple[LifecyclePhase, ...] = (
    LifecyclePhase.FORMATION,
    LifecyclePhase.INTENSIFICATION,
    LifecyclePhase.EXPANSION,
    LifecyclePhase.PEAK,
    LifecyclePhase.DECAY,
)
"""Canonical forward order. DECAY may be followed by another intensification in a
real event, which is reported as a phase change rather than suppressed."""


@dataclass(frozen=True)
class LifecycleThresholds:
    """Cut-offs for event detection, read from ``configs/tracking.yaml``.

    Every field is optional. ``None`` means the config value is still ``null``, and
    the corresponding event is reported as not evaluated instead of being decided
    against a guess. A non-positive value is normalised to ``None`` too: these are
    cut-offs, and a zero cut-off would fire on every step.
    """

    relocation_km: float | None = None
    intensity_tolerance: float | None = None

    def __post_init__(self) -> None:
        # Normalised here as well as in from_config, so a non-positive value is
        # treated as unresolved however it was constructed. A zero distance cut-off
        # is not a usable threshold and would flag every step as a relocation.
        object.__setattr__(self, "relocation_km", _positive_or_none(self.relocation_km))
        object.__setattr__(
            self, "intensity_tolerance", _positive_or_none(self.intensity_tolerance)
        )

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> LifecycleThresholds:
        loaded = config if config is not None else stage_config(CONFIG_FILE)
        return cls(
            relocation_km=_positive_or_none(
                get_value(loaded, "association.max_centroid_distance_km")
            ),
            intensity_tolerance=_positive_or_none(
                get_value(loaded, "association.intensity_tolerance")
            ),
        )

    def unresolved(self) -> list[str]:
        """Config keys that are still ``null``, so the caller can report why."""
        names: list[str] = []
        if self.relocation_km is None:
            names.append("association.max_centroid_distance_km")
        if self.intensity_tolerance is None:
            names.append("association.intensity_tolerance")
        return names


def _positive_or_none(value: Any) -> float | None:
    """Accept a positive number, otherwise treat the parameter as unresolved."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


@dataclass
class PhaseAssignment:
    """The lifecycle state at one time step, with the reason it was chosen."""

    timestamp: datetime
    phase: LifecyclePhase | None
    reason: str
    intensity: float | None = None
    footprint_radius_km: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": format_timestamp(self.timestamp),
            "phase": self.phase.value if self.phase else None,
            "reason": self.reason,
            "intensity": self.intensity,
            "footprint_radius_km": self.footprint_radius_km,
        }


@dataclass
class EventOutcome:
    """Whether a lifecycle event could be decided, and why not when it could not."""

    event: LifecycleEvent
    evaluated: bool
    detail: str
    timestamps: list[datetime] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event.value,
            "evaluated": self.evaluated,
            "detail": self.detail,
            "timestamps": [format_timestamp(t) for t in self.timestamps],
        }


@dataclass
class LifecycleSequence:
    """The full lifecycle verdict for one threat history."""

    threat_id: str
    phases: list[PhaseAssignment] = field(default_factory=list)
    events: list[EventOutcome] = field(default_factory=list)
    thresholds: LifecycleThresholds = field(default_factory=LifecycleThresholds)
    notes: list[str] = field(default_factory=list)

    @property
    def current_phase(self) -> LifecyclePhase | None:
        """Phase at the final time step, or ``None`` if it could not be decided."""
        return self.phases[-1].phase if self.phases else None

    @property
    def peak_phase_time(self) -> datetime | None:
        """Timestamp of the step assigned PEAK, or ``None`` if no peak was found."""
        for assignment in self.phases:
            if assignment.phase is LifecyclePhase.PEAK:
                return assignment.timestamp
        return None

    def phase_at(self, timestamp: datetime) -> LifecyclePhase | None:
        for assignment in self.phases:
            if assignment.timestamp == timestamp:
                return assignment.phase
        return None

    def change_points(self) -> list[tuple[datetime, LifecyclePhase | None, LifecyclePhase | None]]:
        """``(timestamp, previous, new)`` at every phase change."""
        changes: list[tuple[datetime, LifecyclePhase | None, LifecyclePhase | None]] = []
        previous: LifecyclePhase | None = None
        for index, assignment in enumerate(self.phases):
            if index == 0 or assignment.phase is not previous:
                changes.append((assignment.timestamp, previous, assignment.phase))
            previous = assignment.phase
        return changes

    def unevaluated_events(self) -> list[EventOutcome]:
        return [outcome for outcome in self.events if not outcome.evaluated]

    def to_dict(self) -> dict[str, Any]:
        return {
            "threat_id": self.threat_id,
            "current_phase": self.current_phase.value if self.current_phase else None,
            "peak_phase_time": format_timestamp(self.peak_phase_time)
            if self.peak_phase_time
            else None,
            "phases": [assignment.to_dict() for assignment in self.phases],
            "events": [outcome.to_dict() for outcome in self.events],
            "unevaluated_events": [outcome.event.value for outcome in self.unevaluated_events()],
            "unresolved_thresholds": self.thresholds.unresolved(),
            "notes": list(self.notes),
            "provenance": {
                "source": "src.transition.lifecycle.assign_lifecycle",
                "trained": False,
                "state_machine": [phase.value for phase in _PHASE_ORDER],
            },
        }


# ---------------------------------------------------------------------------
# Series extraction
# ---------------------------------------------------------------------------


def _first_finite(values: list[float | None]) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def _relative_change(series: list[float | None], index: int) -> float | None:
    """Change at ``index`` relative to the series' initial scale.

    Normalising by the initial magnitude is what lets an intensity change be
    compared with a footprint change without picking a tolerance: both become
    fractions of their own starting size, so the comparison needs only the sign.
    """
    if index == 0:
        return None
    previous, current = series[index - 1], series[index]
    if previous is None or current is None:
        return None
    scale = _first_finite(series)
    if scale is None:
        return None
    denominator = abs(scale)
    if denominator < 1e-12:
        denominator = max(abs(previous), abs(current), 1e-12)
    return (current - previous) / denominator


def _assign_phases(
    ordered: list[ThreatObject],
    intensity: list[float | None],
    radius: list[float | None],
) -> tuple[list[PhaseAssignment], list[str]]:
    """Walk the history once, assigning a phase and a reason to each step."""
    notes: list[str] = []
    finite_intensity = [value for value in intensity if value is not None]
    finite_radius = [value for value in radius if value is not None]

    if not finite_intensity and not finite_radius:
        reason = (
            "no intensity or footprint was recorded for any step, so no lifecycle "
            "state can be decided"
        )
        return (
            [
                PhaseAssignment(timestamp=snapshot.timestamp, phase=None, reason=reason)
                for snapshot in ordered
            ],
            [reason],
        )

    peak_index: int | None = None
    if finite_intensity:
        peak_index = max(
            range(len(intensity)),
            key=lambda index: intensity[index]
            if intensity[index] is not None
            else float("-inf"),
        )

    global_peak = max(finite_intensity) if finite_intensity else None

    assignments: list[PhaseAssignment] = []
    running_max: float | None = None
    for index, snapshot in enumerate(ordered):
        current_intensity = intensity[index]
        current_radius = radius[index]
        radius_change = _relative_change(radius, index)

        if index == 0:
            # The first observed step is where the track begins, by convention.
            phase, reason = LifecyclePhase.FORMATION, "first step of the track"
        elif current_intensity is None:
            # Intensity missing at this step; footprint growth is all we can claim.
            if radius_change is not None and radius_change > 0:
                phase, reason = (
                    LifecyclePhase.EXPANSION,
                    "footprint growing; intensity not recorded so intensification "
                    "cannot be distinguished",
                )
            else:
                phase, reason = (
                    LifecyclePhase.FORMATION,
                    "footprint not growing; intensity not recorded",
                )
        elif running_max is None:
            phase, reason = LifecyclePhase.FORMATION, "no earlier intensity to compare against"
        elif current_intensity < running_max:
            # Falling back from a level already reached. Using the running maximum
            # rather than the global one is what makes a dip before a later, higher
            # peak read as DECAY instead of as if the threat were still forming.
            phase = LifecyclePhase.DECAY
            reason = (
                f"intensity {current_intensity:g} is below the {running_max:g} already reached"
            )
        elif global_peak is not None and current_intensity == global_peak:
            phase = LifecyclePhase.PEAK
            reason = "at the maximum intensity observed in this history" + (
                "" if index == peak_index else " (tied plateau)"
            )
        else:
            intensity_change = _relative_change(intensity, index)
            intensity_growing = intensity_change is not None and intensity_change > 0
            radius_growing = radius_change is not None and radius_change > 0
            if radius_growing and (
                not intensity_growing or radius_change > (intensity_change or 0.0)
            ):
                phase = LifecyclePhase.EXPANSION
                reason = f"footprint grew {radius_change:.3f} of its initial scale" + (
                    f", more than intensity ({intensity_change:.3f})"
                    if intensity_change is not None
                    else ""
                )
            elif intensity_growing:
                phase = LifecyclePhase.INTENSIFICATION
                reason = f"intensity rising {intensity_change:.3f} of its initial scale"
            else:
                phase = LifecyclePhase.FORMATION
                reason = "neither intensity nor footprint growing yet"

        if current_intensity is not None:
            running_max = (
                current_intensity if running_max is None else max(running_max, current_intensity)
            )

        assignments.append(
            PhaseAssignment(
                timestamp=snapshot.timestamp,
                phase=phase,
                reason=reason,
                intensity=current_intensity,
                footprint_radius_km=current_radius,
            )
        )

    if peak_index == 0 and len(ordered) > 1:
        notes.append(
            "maximum intensity occurs at the first step, so no intensification or "
            "expansion phase precedes the peak"
        )
    return assignments, notes


# ---------------------------------------------------------------------------
# Event detection
# ---------------------------------------------------------------------------


def _detect_relocation(
    ordered: list[ThreatObject],
    thresholds: LifecycleThresholds,
) -> EventOutcome:
    if thresholds.relocation_km is None:
        return EventOutcome(
            event=LifecycleEvent.RELOCATION,
            evaluated=False,
            detail=(
                "requires association.max_centroid_distance_km in configs/tracking.yaml, "
                "which is null until observed centroid displacement is measured"
            ),
        )

    centroids = [
        (snapshot.location.centroid_lat, snapshot.location.centroid_lon)
        for snapshot in ordered
        if snapshot.location is not None
    ]
    if len(centroids) < 2:
        return EventOutcome(
            event=LifecycleEvent.RELOCATION,
            evaluated=False,
            detail="fewer than two steps carry a centroid",
        )

    distances = step_distances_km(centroids)
    flagged = [
        ordered[index + 1].timestamp
        for index, distance in enumerate(distances)
        if distance > thresholds.relocation_km
    ]
    return EventOutcome(
        event=LifecycleEvent.RELOCATION,
        evaluated=True,
        detail=(
            f"{len(flagged)} step(s) exceed {thresholds.relocation_km:g} km "
            f"(max step {float(distances.max()):.1f} km)"
        ),
        timestamps=flagged,
    )


def _detect_weakening(
    ordered: list[ThreatObject],
    intensity: list[float | None],
    thresholds: LifecycleThresholds,
) -> EventOutcome:
    if thresholds.intensity_tolerance is None:
        return EventOutcome(
            event=LifecycleEvent.WEAKENING,
            evaluated=False,
            detail=(
                "requires association.intensity_tolerance in configs/tracking.yaml, "
                "which is null until the intensity distribution is measured"
            ),
        )

    flagged: list[datetime] = []
    for index in range(1, len(ordered)):
        previous, current = intensity[index - 1], intensity[index]
        if previous is None or current is None:
            continue
        if (previous - current) > thresholds.intensity_tolerance:
            flagged.append(ordered[index].timestamp)
    return EventOutcome(
        event=LifecycleEvent.WEAKENING,
        evaluated=True,
        detail=(
            f"{len(flagged)} step(s) drop by more than "
            f"{thresholds.intensity_tolerance:g}"
        ),
        timestamps=flagged,
    )


def _cross_threat_events() -> list[EventOutcome]:
    """MERGER and SPLIT are never decidable from one threat's history."""
    return [
        EventOutcome(
            event=event,
            evaluated=False,
            detail=(
                "requires more than one simultaneously tracked threat; decided by the "
                "tracker (src/tracking/), not by a single threat history"
            ),
        )
        for event in (LifecycleEvent.MERGER, LifecycleEvent.SPLIT)
    ]


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def assign_lifecycle(
    history: ThreatHistory,
    *,
    thresholds: LifecycleThresholds | None = None,
    config: dict[str, Any] | None = None,
) -> LifecycleSequence:
    """Assign a lifecycle state per time step and report the lifecycle events.

    ``history`` is ordered internally and rejects duplicate timestamps, so an
    input with two states at one time step fails loudly rather than producing a
    phase sequence that cannot be trusted.
    """
    selected = thresholds if thresholds is not None else LifecycleThresholds.from_config(config)
    ordered = history.ordered()

    if not ordered:
        return LifecycleSequence(
            threat_id=history.threat_id,
            thresholds=selected,
            notes=["history is empty; nothing to classify"],
        )

    intensity = [
        snapshot.evolution.intensity_anomaly_sigma if snapshot.evolution else None
        for snapshot in ordered
    ]
    radius = [
        snapshot.location.footprint_radius_km if snapshot.location else None
        for snapshot in ordered
    ]

    phases, notes = _assign_phases(ordered, intensity, radius)
    events = [
        _detect_relocation(ordered, selected),
        _detect_weakening(ordered, intensity, selected),
        *_cross_threat_events(),
    ]

    unresolved = selected.unresolved()
    if unresolved:
        notes.append(
            "event thresholds still null in configs/tracking.yaml: "
            + ", ".join(unresolved)
        )

    sequence = LifecycleSequence(
        threat_id=history.threat_id,
        phases=phases,
        events=events,
        thresholds=selected,
        notes=notes,
    )
    events_from_record = _events_from_history(history)
    sequence.events.extend(events_from_record)
    return sequence


def _events_from_history(history: ThreatHistory) -> list[EventOutcome]:
    """Fold in any events the tracker already recorded on the history."""
    records: dict[LifecycleEvent, list[datetime]] = {}
    for record in history.events:
        records.setdefault(record.event, []).append(record.timestamp)
    return [
        EventOutcome(
            event=event,
            evaluated=True,
            detail=f"recorded on the threat history ({len(timestamps)} occurrence(s))",
            timestamps=sorted(timestamps),
        )
        for event, timestamps in records.items()
    ]


def lifecycle_state_per_step(history: ThreatHistory, **kwargs: Any) -> list[dict[str, Any]]:
    """The narrow answer the pipeline wants: one state label per time step."""
    return [assignment.to_dict() for assignment in assign_lifecycle(history, **kwargs).phases]


def make_event(
    timestamp: datetime,
    event: LifecycleEvent,
) -> LifecycleEventRecord:
    """Build a lifecycle event record for the tracker to attach to a history."""
    return LifecycleEventRecord(timestamp=timestamp, event=event)
