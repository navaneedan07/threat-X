"""Tests for the deterministic lifecycle state machine.

Run:  pytest tests/test_lifecycle.py -v

The synthetic translating event is the oracle: it is built with a known peak at a
known step, so the phase sequence has a checkable answer rather than just "looks
plausible". The event half of the file asserts the opposite property — that
anything needing a threshold reports itself as **not evaluated** while
``configs/tracking.yaml`` still has those thresholds null.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.shared.contracts import (
    EventType,
    Evolution,
    LifecycleEvent,
    LifecyclePhase,
    Location,
    Severity,
    ThreatHistory,
    ThreatObject,
)
from src.shared.synthetic import translating_event
from src.transition.lifecycle import (
    LifecycleThresholds,
    assign_lifecycle,
    lifecycle_state_per_step,
    make_event,
)

THREAT_ID = "THR-2026-0001"


def _snapshot(
    hour: int,
    *,
    intensity: float | None,
    radius_km: float | None = 20.0,
    centroid: tuple[float, float] = (13.0, 80.0),
    severity: Severity = Severity.MODERATE,
) -> ThreatObject:
    return ThreatObject(
        threat_id=THREAT_ID,
        event_type=EventType.EXTREME_RAINFALL,
        timestamp=datetime(2026, 9, 24, hour, tzinfo=timezone.utc),
        severity=severity,
        evolution=Evolution(intensity_anomaly_sigma=intensity) if intensity is not None else None,
        location=Location(
            centroid_lat=centroid[0], centroid_lon=centroid[1], footprint_radius_km=radius_km
        )
        if radius_km is not None
        else None,
    )


@pytest.fixture
def event_history() -> ThreatHistory:
    """The synthetic event: amplitude peaks at step 3 of 6."""
    return ThreatHistory(
        threat_id=THREAT_ID,
        snapshots=[
            _snapshot(
                event.step,
                intensity=event.amplitude,
                radius_km=20.0 + event.step * 6.0,
                centroid=event.centre,
            )
            for event in translating_event()
        ],
    )


class TestPhaseSequence:
    def test_phases_follow_the_known_peak(self, event_history: ThreatHistory):
        sequence = assign_lifecycle(event_history)
        phases = [assignment.phase for assignment in sequence.phases]
        assert phases == [
            LifecyclePhase.FORMATION,
            LifecyclePhase.INTENSIFICATION,
            LifecyclePhase.INTENSIFICATION,
            LifecyclePhase.PEAK,
            LifecyclePhase.DECAY,
            LifecyclePhase.DECAY,
        ]

    def test_peak_lands_on_the_highest_intensity_step(self, event_history: ThreatHistory):
        sequence = assign_lifecycle(event_history)
        peak_time = sequence.peak_phase_time
        assert peak_time is not None
        assert peak_time.hour == 3

    def test_current_phase_is_the_last_step(self, event_history: ThreatHistory):
        assert assign_lifecycle(event_history).current_phase is LifecyclePhase.DECAY

    def test_every_step_carries_a_reason(self, event_history: ThreatHistory):
        """Each phase must be auditable, not just asserted."""
        for assignment in assign_lifecycle(event_history).phases:
            assert assignment.phase is not None
            assert assignment.reason

    def test_change_points_name_each_transition(self, event_history: ThreatHistory):
        changes = assign_lifecycle(event_history).change_points()
        observed = [(previous, new) for _, previous, new in changes]
        assert (None, LifecyclePhase.FORMATION) in observed
        assert (LifecyclePhase.INTENSIFICATION, LifecyclePhase.PEAK) in observed
        assert (LifecyclePhase.PEAK, LifecyclePhase.DECAY) in observed

    def test_state_per_step_is_one_label_per_timestep(self, event_history: ThreatHistory):
        steps = lifecycle_state_per_step(event_history)
        assert len(steps) == 6
        assert all("timestamp" in step and "phase" in step for step in steps)

    def test_serialised_sequence_is_json_safe(self, event_history: ThreatHistory):
        import json

        payload = assign_lifecycle(event_history).to_dict()
        assert json.loads(json.dumps(payload))["current_phase"] == "DECAY"
        assert payload["provenance"]["trained"] is False


class TestPhaseRules:
    def test_expansion_wins_when_the_footprint_outgrows_intensity(self):
        """EXPANSION must be reachable, or the branch is dead code."""
        history = ThreatHistory(
            threat_id=THREAT_ID,
            snapshots=[
                _snapshot(0, intensity=1.0, radius_km=10.0),
                _snapshot(1, intensity=1.05, radius_km=60.0),
                _snapshot(2, intensity=1.6, radius_km=60.0),
            ],
        )
        phases = [assignment.phase for assignment in assign_lifecycle(history).phases]
        assert phases[1] is LifecyclePhase.EXPANSION
        assert phases == [
            LifecyclePhase.FORMATION,
            LifecyclePhase.EXPANSION,
            LifecyclePhase.PEAK,
        ]

    def test_intensification_wins_when_intensity_outgrows_the_footprint(self):
        history = ThreatHistory(
            threat_id=THREAT_ID,
            snapshots=[
                _snapshot(0, intensity=1.0, radius_km=10.0),
                _snapshot(1, intensity=9.0, radius_km=11.0),
                _snapshot(2, intensity=12.0, radius_km=11.0),
            ],
        )
        phases = [assignment.phase for assignment in assign_lifecycle(history).phases]
        assert phases[1] is LifecyclePhase.INTENSIFICATION

    def test_a_tied_plateau_stays_at_peak(self):
        """Equal maxima are one peak, not a decay."""
        history = ThreatHistory(
            threat_id=THREAT_ID,
            snapshots=[
                _snapshot(0, intensity=1.0),
                _snapshot(1, intensity=5.0),
                _snapshot(2, intensity=5.0),
            ],
        )
        phases = [assignment.phase for assignment in assign_lifecycle(history).phases]
        assert phases[1] is LifecyclePhase.PEAK
        assert phases[2] is LifecyclePhase.PEAK

    def test_flat_history_stays_in_formation(self):
        history = ThreatHistory(
            threat_id=THREAT_ID,
            snapshots=[
                _snapshot(0, intensity=2.0, radius_km=10.0),
                _snapshot(1, intensity=2.0, radius_km=10.0),
            ],
        )
        phases = [assignment.phase for assignment in assign_lifecycle(history).phases]
        assert set(phases) == {LifecyclePhase.FORMATION, LifecyclePhase.PEAK}

    def test_single_snapshot_is_formation(self):
        history = ThreatHistory(threat_id=THREAT_ID, snapshots=[_snapshot(0, intensity=3.0)])
        sequence = assign_lifecycle(history)
        assert sequence.current_phase is LifecyclePhase.FORMATION

    def test_early_peak_is_noted(self):
        history = ThreatHistory(
            threat_id=THREAT_ID,
            snapshots=[_snapshot(0, intensity=9.0), _snapshot(1, intensity=1.0)],
        )
        sequence = assign_lifecycle(history)
        assert sequence.phases[0].phase is LifecyclePhase.FORMATION
        assert any("first step" in note for note in sequence.notes)

    def test_reintensification_after_decay_is_reported(self):
        """A second peak is a real pattern; the machine reports it rather than smoothing it."""
        history = ThreatHistory(
            threat_id=THREAT_ID,
            snapshots=[
                _snapshot(0, intensity=1.0),
                _snapshot(1, intensity=5.0),
                _snapshot(2, intensity=2.0),
                _snapshot(3, intensity=7.0),
            ],
        )
        phases = [assignment.phase for assignment in assign_lifecycle(history).phases]
        assert LifecyclePhase.DECAY in phases
        assert phases[-1] is LifecyclePhase.PEAK


class TestMissingInputs:
    def test_empty_history_is_reported_not_raised(self):
        sequence = assign_lifecycle(ThreatHistory(threat_id=THREAT_ID))
        assert sequence.phases == []
        assert sequence.current_phase is None
        assert "empty" in sequence.notes[0]

    def test_no_intensity_and_no_footprint_gives_no_phase(self):
        history = ThreatHistory(
            threat_id=THREAT_ID,
            snapshots=[
                ThreatObject(
                    threat_id=THREAT_ID,
                    event_type=EventType.HEATWAVE,
                    timestamp="2026-09-24T00:00:00Z",
                    severity=Severity.MODERATE,
                )
            ],
        )
        sequence = assign_lifecycle(history)
        assert sequence.phases[0].phase is None
        assert "no intensity or footprint" in sequence.phases[0].reason
        assert sequence.current_phase is None

    def test_footprint_only_never_claims_intensification(self):
        """Without intensity, intensification cannot be distinguished — say so."""
        history = ThreatHistory(
            threat_id=THREAT_ID,
            snapshots=[
                _snapshot(0, intensity=None, radius_km=10.0),
                _snapshot(1, intensity=None, radius_km=25.0),
            ],
        )
        phases = [assignment.phase for assignment in assign_lifecycle(history).phases]
        assert phases[0] is LifecyclePhase.FORMATION
        assert phases[1] is LifecyclePhase.EXPANSION
        assert LifecyclePhase.INTENSIFICATION not in phases
        assert "intensity not recorded" in assign_lifecycle(history).phases[1].reason


class TestThresholds:
    def test_thresholds_are_null_in_the_shipped_config(self):
        thresholds = LifecycleThresholds.from_config()
        assert thresholds.relocation_km is None
        assert thresholds.intensity_tolerance is None
        assert set(thresholds.unresolved()) == {
            "association.max_centroid_distance_km",
            "association.intensity_tolerance",
        }

    def test_events_are_not_evaluated_without_thresholds(self, event_history: ThreatHistory):
        sequence = assign_lifecycle(event_history)
        unevaluated = {outcome.event for outcome in sequence.unevaluated_events()}
        assert LifecycleEvent.RELOCATION in unevaluated
        assert LifecycleEvent.WEAKENING in unevaluated
        assert LifecycleEvent.MERGER in unevaluated
        assert LifecycleEvent.SPLIT in unevaluated

    def test_the_reason_names_the_config_key_to_fill_in(self, event_history: ThreatHistory):
        sequence = assign_lifecycle(event_history)
        relocation = next(
            outcome for outcome in sequence.events if outcome.event is LifecycleEvent.RELOCATION
        )
        assert "max_centroid_distance_km" in relocation.detail
        assert "configs/tracking.yaml" in relocation.detail

    def test_merger_and_split_point_at_the_tracker(self, event_history: ThreatHistory):
        sequence = assign_lifecycle(event_history)
        for event in (LifecycleEvent.MERGER, LifecycleEvent.SPLIT):
            outcome = next(item for item in sequence.events if item.event is event)
            assert outcome.evaluated is False
            assert "tracker" in outcome.detail

    def test_unresolved_thresholds_are_listed_in_the_output(self, event_history: ThreatHistory):
        payload = assign_lifecycle(event_history).to_dict()
        assert "association.max_centroid_distance_km" in payload["unresolved_thresholds"]
        assert LifecycleEvent.RELOCATION.value in payload["unevaluated_events"]

    def test_relocation_is_detected_once_a_threshold_is_supplied(self):
        history = ThreatHistory(
            threat_id=THREAT_ID,
            snapshots=[
                _snapshot(0, intensity=1.0, centroid=(13.0, 80.0)),
                _snapshot(1, intensity=1.5, centroid=(13.5, 80.0)),
                # ~500 km jump
                _snapshot(2, intensity=1.6, centroid=(18.0, 80.0)),
            ],
        )
        thresholds = LifecycleThresholds(relocation_km=100.0)
        sequence = assign_lifecycle(history, thresholds=thresholds)
        relocation = next(
            outcome for outcome in sequence.events if outcome.event is LifecycleEvent.RELOCATION
        )
        assert relocation.evaluated is True
        assert len(relocation.timestamps) == 1

    def test_relocation_is_not_flagged_for_normal_movement(self, event_history: ThreatHistory):
        sequence = assign_lifecycle(
            event_history, thresholds=LifecycleThresholds(relocation_km=500.0)
        )
        relocation = next(
            outcome for outcome in sequence.events if outcome.event is LifecycleEvent.RELOCATION
        )
        assert relocation.evaluated is True
        assert relocation.timestamps == []

    def test_weakening_is_detected_once_a_tolerance_is_supplied(self, event_history: ThreatHistory):
        sequence = assign_lifecycle(
            event_history, thresholds=LifecycleThresholds(intensity_tolerance=20.0)
        )
        weakening = next(
            outcome for outcome in sequence.events if outcome.event is LifecycleEvent.WEAKENING
        )
        assert weakening.evaluated is True
        assert len(weakening.timestamps) == 2

    def test_zero_threshold_is_treated_as_unresolved(self):
        """A zero distance cut-off is not a usable threshold, so it stays undecided."""
        assert LifecycleThresholds(relocation_km=0.0).unresolved() == [
            "association.max_centroid_distance_km",
            "association.intensity_tolerance",
        ]


class TestRecordedEvents:
    def test_events_already_on_the_history_are_folded_in(self, event_history: ThreatHistory):
        event_history.events.append(
            make_event(datetime(2026, 9, 24, 2, tzinfo=timezone.utc), LifecycleEvent.MERGER)
        )
        sequence = assign_lifecycle(event_history)
        folded = [
            outcome
            for outcome in sequence.events
            if outcome.event is LifecycleEvent.MERGER and outcome.evaluated
        ]
        assert len(folded) == 1
        assert len(folded[0].timestamps) == 1

    def test_make_event_builds_a_valid_record(self):
        record = make_event(datetime(2026, 9, 24, 2, tzinfo=timezone.utc), "SPLIT")
        assert record.event is LifecycleEvent.SPLIT
        assert record.to_dict()["timestamp"] == "2026-09-24T02:00:00Z"


class TestHistoryContract:
    def test_duplicate_timestamps_are_refused(self):
        history = ThreatHistory(
            threat_id=THREAT_ID,
            snapshots=[_snapshot(5, intensity=1.0), _snapshot(5, intensity=2.0)],
        )
        with pytest.raises(Exception, match="duplicate snapshot"):
            assign_lifecycle(history)

    def test_history_is_ordered_internally(self):
        history = ThreatHistory(
            threat_id=THREAT_ID,
            snapshots=[
                _snapshot(2, intensity=9.0),
                _snapshot(0, intensity=1.0),
                _snapshot(1, intensity=5.0),
            ],
        )
        sequence = assign_lifecycle(history)
        hours = [assignment.timestamp.hour for assignment in sequence.phases]
        assert hours == [0, 1, 2]
        assert sequence.current_phase is LifecyclePhase.PEAK
