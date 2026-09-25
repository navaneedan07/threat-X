"""Contract tests for ``src/shared/contracts.py``.

Run:  pytest tests/test_contracts.py -v

These tests need no weather data, no xarray and no network — the point of the
contract module is that tracking, GNN, TTIE and validation can all depend on it
on a bare Python install. Two things are being protected here:

1. The contract matches the objects that already exist on disk
   (``data/samples/*.json``), so it is a description of reality, not an
   aspirational schema.
2. The contract and ``backend/schemas/threat.py`` carry the *same* field names,
   so nobody can quietly create a second definition.
"""

from __future__ import annotations

import json
from datetime import timezone
from pathlib import Path

import pytest

from src.shared.contracts import (
    LIFECYCLE_EVENTS,
    LIFECYCLE_STATES,
    SEVERITY_TIERS,
    THREAT_OBJECT_FIELDS,
    ContractError,
    EventType,
    Footprint,
    GateVerdict,
    LifecycleEvent,
    LifecyclePhase,
    Location,
    Movement,
    PrecursorSample,
    PrecursorSeries,
    Severity,
    ThreatHistory,
    ThreatObject,
    Trajectory,
    TransitionEstimate,
    TransitionWindow,
    ValidationResult,
    contract_drift,
    format_timestamp,
    load_threat_history,
    load_threat_objects,
    parse_timestamp,
    write_threat_objects,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES = REPO_ROOT / "data" / "samples"

KNOWN_ID = "THR-2026-0001"


def _load_sample(filename: str) -> dict:
    return json.loads((SAMPLES / filename).read_text(encoding="utf-8"))


def _minimal_threat(**overrides) -> ThreatObject:
    """A Threat Object with only the fields every producer can supply."""
    kwargs = {
        "threat_id": KNOWN_ID,
        "event_type": EventType.EXTREME_RAINFALL,
        "timestamp": "2026-09-24T16:00:00Z",
        "severity": Severity.SEVERE,
    }
    kwargs.update(overrides)
    return ThreatObject(**kwargs)


def _assert_fixture_keys_survive(original: dict, restored: dict, path: str = "") -> None:
    """Assert every fixture key is present in the parsed object with the same value.

    Subset rather than equality: the contract legitimately carries extra keys
    (the null pressure-level precursors, ``unavailable_variables``) that these
    hand-written fixtures predate. Those additions are asserted separately.
    """
    for key, value in original.items():
        assert key in restored, f"{path}{key}: dropped by the contract"
        if isinstance(value, dict):
            _assert_fixture_keys_survive(value, restored[key], path=f"{path}{key}.")
        else:
            assert restored[key] == value, f"{path}{key}: {restored[key]!r} != {value!r}"


# ---------------------------------------------------------------------------
# Round-trip against the committed fixtures
# ---------------------------------------------------------------------------


class TestFixtureRoundTrip:
    """The fixtures on disk are the de-facto contract; parsing must not lose them."""

    def test_threat_fixture_keys_survive(self):
        entry = _load_sample("threats.json")["threats"][0]
        _assert_fixture_keys_survive(entry, ThreatObject.from_dict(entry).to_dict())

    def test_threat_fixture_adds_only_documented_keys(self):
        entry = _load_sample("threats.json")["threats"][0]
        restored = ThreatObject.from_dict(entry).to_dict()
        # Keys the contract carries that this fixture omits, all null here.
        assert set(restored) - set(entry) == {"ensemble_agreement", "footprint", "lifecycle_phase"}
        assert restored["ensemble_agreement"] is None

    def test_trajectory_fixture_round_trips_exactly(self):
        entry = _load_sample("trajectories.json")["trajectories"][KNOWN_ID]
        restored = Trajectory.from_dict(entry)
        assert restored.threat_id == entry["threat_id"]
        assert len(restored.historical_steps) == len(entry["historical_steps"])
        assert len(restored.forecast_steps) == len(entry["forecast_steps"])
        assert restored.to_dict() == entry

    def test_transition_fixture_parses_uncalibrated_state(self):
        entry = _load_sample("transitions.json")["transitions"][KNOWN_ID]
        estimate = TransitionEstimate.from_dict(entry["transition_evaluation"])
        # ``target_state`` is the fixture/endpoint key name; accepted on read.
        assert estimate.target_severity is Severity.EXTREME
        assert estimate.probability is None
        assert estimate.calibrated is False
        assert estimate.model_id is None
        assert estimate.is_publishable is False
        assert LifecyclePhase(entry["lifecycle_phase"].upper()) is LifecyclePhase.INTENSIFICATION

    def test_precursor_fixture_keys_survive(self):
        entry = _load_sample("precursors.json")["precursors"][KNOWN_ID]
        restored = PrecursorSeries.from_dict(entry).to_dict()
        assert set(entry) <= set(restored)
        assert restored["variables_analyzed"] == entry["variables_analyzed"]
        assert restored["explanatory_summary"] == entry["explanatory_summary"]
        assert len(restored["series"]) == len(entry["series"])
        for original, parsed in zip(entry["series"], restored["series"], strict=True):
            _assert_fixture_keys_survive(original, parsed)

    def test_precursor_series_declares_unavailable_fields(self):
        series = PrecursorSeries(threat_id=KNOWN_ID)
        assert series.unavailable_variables == []
        assert "unavailable_variables" in series.to_dict()

    def test_precursor_series_carries_timestamps(self):
        entry = _load_sample("precursors.json")["precursors"][KNOWN_ID]
        restored = PrecursorSeries.from_dict(entry)
        assert all(point.timestamp is not None for point in restored.series)

    def test_pressure_level_precursors_stay_null(self):
        """Fields the dataset cannot support must be null, not fabricated."""
        entry = _load_sample("precursors.json")["precursors"][KNOWN_ID]
        restored = PrecursorSeries.from_dict(entry).to_dict()
        for point in restored["series"]:
            assert point["theta_e_gradient_k_100km"] is None

    def test_footprint_fixture_round_trips_exactly(self):
        entry = _load_sample("footprints.json")["footprints"][KNOWN_ID]
        coarse = Footprint.from_dict(entry["coarse"])
        assert coarse.to_dict() == entry["coarse"]
        downscaled = Footprint.from_dict(entry["downscaled"])
        assert downscaled.resolution_km == 5.0
        assert downscaled.extreme_preserved is True


# ---------------------------------------------------------------------------
# No second definition: parity with the HTTP schema
# ---------------------------------------------------------------------------


class TestApiSchemaParity:
    def test_threat_object_field_names_match_backend(self):
        schemas = pytest.importorskip("backend.schemas.threat")
        api_fields = set(schemas.ThreatObject.model_fields)
        contract_fields = set(_minimal_threat().to_dict())
        # The contract may carry fields the HTTP summary omits, but it must not
        # be missing anything the API promises to serve.
        missing = api_fields - contract_fields
        assert not missing, f"API fields absent from the contract: {sorted(missing)}"

    def test_severity_labels_match_backend_description(self):
        schemas = pytest.importorskip("backend.schemas.threat")
        description = schemas.ThreatObject.model_fields["severity"].description or ""
        for tier in SEVERITY_TIERS:
            assert tier in description, f"severity tier {tier!r} missing from API description"


# ---------------------------------------------------------------------------
# Null semantics
# ---------------------------------------------------------------------------


class TestNullSemantics:
    def test_minimal_object_leaves_unrun_stages_null(self):
        threat = _minimal_threat()
        payload = threat.to_dict()
        for key in (
            "confidence",
            "location",
            "movement",
            "evolution",
            "precursors",
            "transition",
            "validation",
            "provenance",
            "ensemble_agreement",
            "footprint",
            "lifecycle_phase",
        ):
            assert payload[key] is None, f"{key} should be null until its stage runs"

    def test_null_is_not_zero(self):
        """A missing precursor must serialise as null, never as 0.0."""
        sample = PrecursorSample()
        payload = sample.to_dict()
        assert payload["wind_speed_ms"] is None
        assert payload["vorticity_850_s1"] is None

    def test_json_null_survives_serialisation(self):
        payload = json.loads(json.dumps(_minimal_threat().to_dict()))
        assert payload["transition"] is None

    def test_unknown_keys_ignored_for_forward_compatibility(self):
        entry = _load_sample("threats.json")["threats"][0]
        entry["future_field_from_a_later_release"] = 123
        assert ThreatObject.from_dict(entry).threat_id == KNOWN_ID


# ---------------------------------------------------------------------------
# Integrity rules
# ---------------------------------------------------------------------------


class TestIntegrityRules:
    def test_probability_without_model_is_rejected(self):
        with pytest.raises(ContractError, match="model_id"):
            TransitionEstimate(target_severity=Severity.EXTREME, probability=0.76)

    def test_probability_with_named_model_is_accepted(self):
        estimate = TransitionEstimate(
            target_severity=Severity.EXTREME,
            probability=0.76,
            model_id="logreg_precursor_v1",
            window=TransitionWindow(earliest=8, most_likely=11, latest=14),
        )
        assert estimate.is_publishable
        payload = estimate.to_dict()
        assert payload["expected_window_hours"] == {
            "earliest": 8,
            "most_likely": 11,
            "latest": 14,
        }
        # The threat-level summary uses the flat key; the endpoint uses the
        # structured one. One contract, both consumers.
        assert payload["window_hours"] == 11.0

    def test_flat_window_hours_is_readable(self):
        estimate = TransitionEstimate.from_dict({"window_hours": 12})
        assert estimate.window.most_likely == 12

    def test_probability_outside_unit_interval_is_rejected(self):
        with pytest.raises(ContractError):
            TransitionEstimate(probability=1.4, model_id="m")

    def test_transition_window_must_be_ordered(self):
        with pytest.raises(ContractError, match="earliest"):
            TransitionWindow(earliest=14, most_likely=11, latest=8)

    def test_gate_verdict_is_required(self):
        with pytest.raises(ContractError):
            ValidationResult(gate_verdict=None)

    @pytest.mark.parametrize("verdict", ["PASS", "DEGRADE", "SUPPRESS", "pass"])
    def test_gate_verdict_vocabulary(self, verdict):
        assert isinstance(ValidationResult(gate_verdict=verdict).gate_verdict, GateVerdict)

    def test_gate_verdict_rejects_fail(self):
        """The gate vocabulary is PASS/DEGRADE/SUPPRESS; FAIL is not a verdict."""
        with pytest.raises(ContractError):
            ValidationResult(gate_verdict="FAIL")

    def test_lifecycle_event_vocabulary(self):
        for name in ("MERGER", "SPLIT", "WEAKENING", "RELOCATION"):
            assert LifecycleEvent(name).value == name


# ---------------------------------------------------------------------------
# Identifier, timestamp and geometry validation
# ---------------------------------------------------------------------------


class TestIdentifierAndTime:
    @pytest.mark.parametrize("bad_id", ["THR-26-1", "thr-2026-0001", "THR-2026-00001", ""])
    def test_bad_threat_ids_rejected(self, bad_id):
        with pytest.raises(ContractError):
            _minimal_threat(threat_id=bad_id)

    @pytest.mark.parametrize("good_id", ["THR-2020-0001", "THR-2022-0001", "THR-2026-0001"])
    def test_real_pipeline_ids_accepted(self, good_id):
        assert _minimal_threat(threat_id=good_id).threat_id == good_id

    def test_naive_timestamp_rejected(self):
        with pytest.raises(ContractError, match="timezone-aware"):
            _minimal_threat(timestamp="2026-09-24T16:00:00")

    def test_non_utc_offset_is_normalised(self):
        parsed = parse_timestamp("2026-09-24T21:30:00+05:30")
        assert parsed.tzinfo == timezone.utc
        assert format_timestamp(parsed) == "2026-09-24T16:00:00Z"

    def test_timestamp_round_trips_as_iso_utc(self):
        assert _minimal_threat().to_dict()["timestamp"] == "2026-09-24T16:00:00Z"

    def test_bad_timestamp_string_rejected(self):
        with pytest.raises(ContractError):
            parse_timestamp("24-09-2026")


class TestGeometry:
    def test_latitude_out_of_range_rejected(self):
        with pytest.raises(ContractError):
            Location(centroid_lat=113.08, centroid_lon=80.27, footprint_radius_km=42.5)

    def test_negative_speed_rejected(self):
        with pytest.raises(ContractError):
            Movement(direction_deg=72, speed_kmh=-1.0)

    def test_direction_above_360_rejected(self):
        with pytest.raises(ContractError):
            Movement(direction_deg=400.0, speed_kmh=10.0)

    def test_footprint_ring_must_close(self):
        with pytest.raises(ContractError, match="not closed"):
            Footprint(
                resolution_km=5.0,
                coordinates=[[[80.0, 13.0], [80.5, 13.0], [80.5, 13.5], [80.1, 13.1]]],
            )

    def test_footprint_accepts_closed_ring(self):
        footprint = Footprint(
            resolution_km=5.0,
            coordinates=[[[80.0, 13.0], [80.5, 13.0], [80.5, 13.5], [80.0, 13.0]]],
        )
        assert footprint.to_dict()["geometry"]["coordinates"][0][0] == [80.0, 13.0]


# ---------------------------------------------------------------------------
# History — the input to the lifecycle state machine
# ---------------------------------------------------------------------------


class TestThreatHistory:
    def _history(self) -> ThreatHistory:
        return ThreatHistory(
            threat_id=KNOWN_ID,
            snapshots=[
                _minimal_threat(severity=Severity.MODERATE, timestamp="2026-09-24T10:00:00Z"),
                _minimal_threat(severity=Severity.SEVERE, timestamp="2026-09-24T16:00:00Z"),
                _minimal_threat(severity=Severity.MODERATE, timestamp="2026-09-24T13:00:00Z"),
            ],
        )

    def test_snapshots_are_ordered_by_time(self):
        history = self._history()
        assert [s.severity for s in history.ordered()] == [
            Severity.MODERATE,
            Severity.MODERATE,
            Severity.SEVERE,
        ]

    def test_duration_is_measured_not_assumed(self):
        assert self._history().duration_hours == 6.0

    def test_single_snapshot_has_no_measurable_duration(self):
        history = ThreatHistory(threat_id=KNOWN_ID, snapshots=[_minimal_threat()])
        assert history.duration_hours is None

    def test_duplicate_timestamp_rejected(self):
        history = ThreatHistory(
            threat_id=KNOWN_ID,
            snapshots=[_minimal_threat(), _minimal_threat()],
        )
        with pytest.raises(ContractError, match="duplicate snapshot"):
            history.ordered()

    def test_mismatched_threat_id_rejected(self):
        with pytest.raises(ContractError, match="does not"):
            ThreatHistory(
                threat_id="THR-2026-0002",
                snapshots=[_minimal_threat(threat_id="THR-2026-0001")],
            )

    def test_lifecycle_phase_parses_case_insensitively(self):
        threat = _minimal_threat(lifecycle_phase="intensification")
        assert threat.lifecycle_phase is LifecyclePhase.INTENSIFICATION
        assert threat.to_dict()["lifecycle_phase"] == "INTENSIFICATION"


# ---------------------------------------------------------------------------
# File I/O and config drift
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_write_then_load_round_trip(self, tmp_path):
        original = ThreatObject.from_dict(_load_sample("threats.json")["threats"][0])
        target = write_threat_objects([original], tmp_path / "processed" / "threats.json")
        assert target.exists()

        payload = json.loads(target.read_text(encoding="utf-8"))
        assert payload["_meta"]["source"].endswith("write_threat_objects")
        reloaded = load_threat_objects(target)
        assert reloaded == [original]

    def test_load_groups_snapshots_into_histories(self, tmp_path):
        entry = _load_sample("threats.json")["threats"][0]
        second = dict(entry, timestamp="2026-09-24T10:00:00Z")
        path = write_threat_objects(
            [ThreatObject.from_dict(entry), ThreatObject.from_dict(second)],
            tmp_path / "threats.json",
        )
        histories = load_threat_history(path)
        assert len(histories) == 1
        assert len(histories[0].snapshots) == 2

    def test_load_rejects_non_list_payload(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"threats": {"not": "a list"}}), encoding="utf-8")
        with pytest.raises(ContractError):
            load_threat_objects(path)



class TestContractDrift:
    def test_runs_against_the_repo_config(self):
        """The check must execute against configs/tracking.yaml without crashing."""
        drift = contract_drift()
        assert isinstance(drift, list)
        assert all(isinstance(entry, str) for entry in drift)

    def test_repo_config_is_in_sync(self):
        """configs/tracking.yaml and the contract must declare the same fields.

        Locks in the schema freeze: editing the field list on either side without
        the other now fails here instead of surfacing as an unexplained mismatch
        months later.
        """
        pytest.importorskip("yaml")
        assert contract_drift() == []

    def test_lifecycle_mismatch_is_reported(self, tmp_path):
        pytest.importorskip("yaml")
        config = tmp_path / "tracking.yaml"
        config.write_text(
            "lifecycle:\n"
            "  states: [BIRTH, DEATH]\n"
            "  events: [MERGER]\n"
            "threat_object_fields: [threat_id, unknown_future_field]\n",
            encoding="utf-8",
        )
        drift = contract_drift(config)
        joined = " ".join(drift)
        assert "lifecycle.states" in joined
        assert "lifecycle.events" in joined
        assert "unknown_future_field" in joined

    def test_in_sync_config_reports_no_drift(self, tmp_path):
        pytest.importorskip("yaml")
        config = tmp_path / "tracking.yaml"
        config.write_text(
            "lifecycle:\n"
            f"  states: [{', '.join(LIFECYCLE_STATES)}]\n"
            f"  events: [{', '.join(LIFECYCLE_EVENTS)}]\n"
            "severity_bands:\n  moderate: null\n  severe: null\n"
            f"threat_object_fields: [{', '.join(THREAT_OBJECT_FIELDS)}]\n",
            encoding="utf-8",
        )
        assert contract_drift(config) == []

    def test_missing_config_is_reported_not_raised(self, tmp_path):
        assert contract_drift(tmp_path / "absent.yaml")
