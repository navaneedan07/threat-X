"""API integration tests for the Threat-X REST API.

Run:  pytest tests/test_api.py -v

These tests use FastAPI's TestClient (synchronous HTTPX wrapper) so they
require no live server.  They exercise:
  - Response status codes
  - Schema shape (required fields present)
  - Null-semantics contract (null fields where modules are not wired)
  - Query parameter validation
  - 404 behaviour for unknown threat IDs
  - GeoJSON coordinate ordering
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)

_KNOWN_ID = "THR-2026-0001"
_UNKNOWN_ID = "THR-9999-9999"


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_status_200(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_response_keys(self):
        r = client.get("/health")
        body = r.json()
        for key in ("status", "timestamp", "version", "pipeline_stages"):
            assert key in body, f"Missing key: {key}"

    def test_pipeline_stages_present(self):
        body = client.get("/health").json()
        stages = body["pipeline_stages"]
        expected = {
            "data_ingestion",
            "anomaly_detection",
            "threat_tracking",
            "precursor_analysis",
            "transition_intelligence",
            "downscaling",
            "validation_gate",
        }
        assert set(stages.keys()) == expected

    def test_status_is_string(self):
        body = client.get("/health").json()
        assert isinstance(body["status"], str)
        assert body["status"] in ("healthy", "degraded", "unhealthy")


# ---------------------------------------------------------------------------
# GET /api/v1/threats
# ---------------------------------------------------------------------------


class TestThreatList:
    def test_status_200(self):
        r = client.get("/api/v1/threats")
        assert r.status_code == 200

    def test_schema_keys(self):
        body = client.get("/api/v1/threats").json()
        for key in ("count", "timestamp", "threats"):
            assert key in body

    def test_count_matches_threats_length(self):
        body = client.get("/api/v1/threats").json()
        assert body["count"] == len(body["threats"])

    def test_severity_filter_valid(self):
        r = client.get("/api/v1/threats?severity=severe")
        assert r.status_code == 200
        body = r.json()
        for t in body["threats"]:
            assert t["severity"] == "severe"

    def test_severity_filter_invalid(self):
        r = client.get("/api/v1/threats?severity=catastrophic")
        assert r.status_code == 422

    def test_limit_respected(self):
        r = client.get("/api/v1/threats?limit=1")
        assert r.status_code == 200
        body = r.json()
        assert len(body["threats"]) <= 1

    def test_limit_zero_invalid(self):
        r = client.get("/api/v1/threats?limit=0")
        assert r.status_code == 422

    def test_threat_item_required_fields(self):
        body = client.get("/api/v1/threats").json()
        if not body["threats"]:
            pytest.skip("No threats in fixture")
        item = body["threats"][0]
        for field in ("threat_id", "event_type", "severity", "confidence", "location", "movement"):
            assert field in item, f"Missing field: {field}"


# ---------------------------------------------------------------------------
# GET /api/v1/threats/{threat_id}
# ---------------------------------------------------------------------------


class TestThreatDetail:
    def test_known_id_200(self):
        r = client.get(f"/api/v1/threats/{_KNOWN_ID}")
        assert r.status_code == 200

    def test_unknown_id_404(self):
        r = client.get(f"/api/v1/threats/{_UNKNOWN_ID}")
        assert r.status_code == 404

    def test_full_schema_keys(self):
        body = client.get(f"/api/v1/threats/{_KNOWN_ID}").json()
        for key in (
            "threat_id", "event_type", "timestamp", "severity", "confidence",
            "location", "movement", "evolution", "precursors", "transition",
            "validation", "provenance",
        ):
            assert key in body, f"Missing key: {key}"

    def test_null_semantics_transition_probability(self):
        """Transition probability must be null until the transition model is wired."""
        body = client.get(f"/api/v1/threats/{_KNOWN_ID}").json()
        # Spec: null if transition models have not processed the object.
        # Fixtures have null; this should stay null until Navaneedan integrates.
        assert body["transition"]["probability"] is None

    def test_confidence_in_range(self):
        body = client.get(f"/api/v1/threats/{_KNOWN_ID}").json()
        assert 0.0 <= body["confidence"] <= 1.0

    def test_centroid_is_lat_lon_pair(self):
        body = client.get(f"/api/v1/threats/{_KNOWN_ID}").json()
        centroid = body["location"]["centroid"]
        assert len(centroid) == 2
        lat, lon = centroid
        assert -90.0 <= lat <= 90.0
        assert -180.0 <= lon <= 180.0


# ---------------------------------------------------------------------------
# GET /api/v1/threats/{threat_id}/trajectory
# ---------------------------------------------------------------------------


class TestTrajectory:
    def test_known_id_200(self):
        r = client.get(f"/api/v1/threats/{_KNOWN_ID}/trajectory")
        assert r.status_code == 200

    def test_unknown_id_404(self):
        r = client.get(f"/api/v1/threats/{_UNKNOWN_ID}/trajectory")
        assert r.status_code == 404

    def test_schema_keys(self):
        body = client.get(f"/api/v1/threats/{_KNOWN_ID}/trajectory").json()
        for key in ("threat_id", "historical_steps", "forecast_steps"):
            assert key in body

    def test_historical_steps_have_required_fields(self):
        body = client.get(f"/api/v1/threats/{_KNOWN_ID}/trajectory").json()
        for step in body["historical_steps"]:
            for field in ("timestamp", "centroid", "severity"):
                assert field in step

    def test_forecast_steps_have_uncertainty(self):
        body = client.get(f"/api/v1/threats/{_KNOWN_ID}/trajectory").json()
        for step in body["forecast_steps"]:
            for field in ("lead_time_hours", "timestamp", "centroid", "uncertainty_radius_km"):
                assert field in step
            assert step["uncertainty_radius_km"] >= 0.0


# ---------------------------------------------------------------------------
# GET /api/v1/threats/{threat_id}/precursors
# ---------------------------------------------------------------------------


class TestPrecursors:
    def test_known_id_200(self):
        r = client.get(f"/api/v1/threats/{_KNOWN_ID}/precursors")
        assert r.status_code == 200

    def test_unknown_id_404(self):
        r = client.get(f"/api/v1/threats/{_UNKNOWN_ID}/precursors")
        assert r.status_code == 404

    def test_schema_keys(self):
        body = client.get(f"/api/v1/threats/{_KNOWN_ID}/precursors").json()
        for key in ("threat_id", "variables_analyzed", "series"):
            assert key in body

    def test_window_hours_parameter(self):
        r = client.get(f"/api/v1/threats/{_KNOWN_ID}/precursors?window_hours=6")
        assert r.status_code == 200

    def test_window_hours_zero_invalid(self):
        r = client.get(f"/api/v1/threats/{_KNOWN_ID}/precursors?window_hours=0")
        assert r.status_code == 422

    def test_series_timestamps_present(self):
        body = client.get(f"/api/v1/threats/{_KNOWN_ID}/precursors").json()
        for point in body["series"]:
            assert "timestamp" in point


# ---------------------------------------------------------------------------
# GET /api/v1/threats/{threat_id}/transition
# ---------------------------------------------------------------------------


class TestTransition:
    def test_known_id_200(self):
        r = client.get(f"/api/v1/threats/{_KNOWN_ID}/transition")
        assert r.status_code == 200

    def test_unknown_id_404(self):
        r = client.get(f"/api/v1/threats/{_UNKNOWN_ID}/transition")
        assert r.status_code == 404

    def test_schema_keys(self):
        body = client.get(f"/api/v1/threats/{_KNOWN_ID}/transition").json()
        for key in ("threat_id", "current_state", "transition_evaluation", "lifecycle_phase"):
            assert key in body

    def test_probability_null_when_uncalibrated(self):
        """Per spec: probability must be null when model is not calibrated."""
        body = client.get(f"/api/v1/threats/{_KNOWN_ID}/transition").json()
        eval_ = body["transition_evaluation"]
        if not eval_.get("calibrated", False):
            assert eval_["probability"] is None

    def test_probability_in_range_if_present(self):
        body = client.get(f"/api/v1/threats/{_KNOWN_ID}/transition").json()
        prob = body["transition_evaluation"].get("probability")
        if prob is not None:
            assert 0.0 <= prob <= 1.0


# ---------------------------------------------------------------------------
# GET /api/v1/threats/{threat_id}/footprint
# ---------------------------------------------------------------------------


class TestFootprint:
    def test_known_id_200(self):
        r = client.get(f"/api/v1/threats/{_KNOWN_ID}/footprint")
        assert r.status_code == 200

    def test_unknown_id_404(self):
        r = client.get(f"/api/v1/threats/{_UNKNOWN_ID}/footprint")
        assert r.status_code == 404

    def test_geojson_feature_type(self):
        body = client.get(f"/api/v1/threats/{_KNOWN_ID}/footprint").json()
        assert body["type"] == "Feature"

    def test_geometry_is_polygon(self):
        body = client.get(f"/api/v1/threats/{_KNOWN_ID}/footprint").json()
        assert body["geometry"]["type"] == "Polygon"

    def test_geojson_coordinates_lon_lat_order(self):
        """RFC 7946: coordinates are [longitude, latitude]."""
        body = client.get(f"/api/v1/threats/{_KNOWN_ID}/footprint").json()
        coords = body["geometry"]["coordinates"][0]
        for lon, lat in coords:
            assert -180.0 <= lon <= 180.0, f"Longitude out of range: {lon}"
            assert -90.0 <= lat <= 90.0, f"Latitude out of range: {lat}"

    def test_downscaled_flag(self):
        r = client.get(f"/api/v1/threats/{_KNOWN_ID}/footprint?downscaled=true")
        assert r.status_code == 200
        body = r.json()
        # Downscaled resolution should be finer than coarse (5 km vs 12 km).
        assert body["properties"]["resolution_km"] <= 12.0

    def test_properties_present(self):
        body = client.get(f"/api/v1/threats/{_KNOWN_ID}/footprint").json()
        props = body["properties"]
        assert "threat_id" in props
        assert "resolution_km" in props


# ---------------------------------------------------------------------------
# GET /api/v1/alerts
# ---------------------------------------------------------------------------


class TestAlerts:
    def test_status_200(self):
        r = client.get("/api/v1/alerts")
        assert r.status_code == 200

    def test_schema_keys(self):
        body = client.get("/api/v1/alerts").json()
        for key in ("generated_at", "active_alerts_count", "alerts"):
            assert key in body

    def test_count_matches_list(self):
        body = client.get("/api/v1/alerts").json()
        assert body["active_alerts_count"] == len(body["alerts"])

    def test_min_severity_filter(self):
        r = client.get("/api/v1/alerts?min_severity=severe")
        assert r.status_code == 200

    def test_min_severity_invalid(self):
        r = client.get("/api/v1/alerts?min_severity=critical")
        assert r.status_code == 422

    def test_alert_required_fields(self):
        body = client.get("/api/v1/alerts").json()
        if not body["alerts"]:
            pytest.skip("No alerts in fixture")
        alert = body["alerts"][0]
        for field in (
            "alert_id", "threat_id", "headline", "severity",
            "urgency", "target_area_description", "centroid",
            "primary_driver", "validation_status",
        ):
            assert field in alert, f"Missing field: {field}"
