"""Tests for serving transition output through the API service.

Run:  pytest tests/test_transition_serving.py -v

The service must serve **exactly what the publishing gate allows**:
  * the shortest horizon whose ``serving.served`` is true is published;
  * a horizon the gate withheld keeps ``probability = null`` even though its
    metrics are in the report;
  * a threat with no report still falls back to the fixtures.

The fixtures are read from the real ``data/samples/`` regardless of ``DATA_ROOT``
(their directory is resolved at import), so pointing ``DATA_ROOT`` at a temp
directory only changes where reports are read from.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.services import transition_service as S

_THREAT = "THR-2099-0001"


def _horizon(hours: float, *, served: bool, probability: float | None, model_id: str | None):
    return {
        "horizon_hours": hours,
        "publishable": served,
        "dataset": {"escalation_quantile": 0.75, "base_rate": 0.30},
        "model": {"model_id": model_id},
        "serving": {"served": served, "probability": probability},
    }


def _write_report(root: Path, threat_id: str, horizons: list[dict]) -> None:
    directory = root / "processed" / "validation" / threat_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transition_report.json").write_text(
        json.dumps(
            {
                "_meta": {"generated_at": "2026-09-28T00:00:00Z"},
                "can_publish": any(h["serving"]["served"] for h in horizons),
                "horizons": horizons,
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    return tmp_path


class TestServing:
    def test_publishable_horizon_is_served(self, data_root: Path):
        _write_report(
            data_root,
            _THREAT,
            [_horizon(6.0, served=True, probability=0.42, model_id="logreg_h6")],
        )
        response = S.get_transition(_THREAT)
        assert response is not None
        evaluation = response["transition_evaluation"]
        assert evaluation["probability"] == pytest.approx(0.42)
        assert evaluation["calibrated"] is False
        assert evaluation["target_state"] == "above_p0.75"
        assert evaluation["expected_window_hours"]["latest"] == 6
        assert response["provenance"]["model_id"] == "logreg_h6"

    def test_shortest_publishable_horizon_wins(self, data_root: Path):
        _write_report(
            data_root,
            _THREAT,
            [
                _horizon(24.0, served=True, probability=0.9, model_id="logreg_h24"),
                _horizon(6.0, served=True, probability=0.2, model_id="logreg_h6"),
            ],
        )
        response = S.get_transition(_THREAT)
        assert response["provenance"]["model_id"] == "logreg_h6"
        assert response["transition_evaluation"]["probability"] == pytest.approx(0.2)

    def test_withheld_horizon_keeps_probability_null(self, data_root: Path):
        _write_report(
            data_root,
            _THREAT,
            [_horizon(12.0, served=False, probability=0.8, model_id="logreg_h12")],
        )
        response = S.get_transition(_THREAT)
        evaluation = response["transition_evaluation"]
        assert evaluation["probability"] is None
        # No model may be named for a probability that is not being served.
        assert response["provenance"]["model_id"] is None

    def test_null_probability_is_still_a_200_shaped_response(self, data_root: Path):
        _write_report(
            data_root,
            _THREAT,
            [_horizon(12.0, served=False, probability=None, model_id=None)],
        )
        response = S.get_transition(_THREAT)
        for key in ("threat_id", "current_state", "transition_evaluation", "lifecycle_phase"):
            assert key in response
        assert response["current_state"] == "below_p0.75"

    def test_brier_baseline_is_the_no_skill_score(self, data_root: Path):
        _write_report(
            data_root,
            _THREAT,
            [_horizon(6.0, served=True, probability=0.42, model_id="logreg_h6")],
        )
        evaluation = S.get_transition(_THREAT)["transition_evaluation"]
        # Constant base-rate forecast at 0.30 scores 0.30 * 0.70.
        assert evaluation["brier_score_baseline"] == pytest.approx(0.21)


class TestAvailability:
    def test_available_when_a_horizon_is_served(self, data_root: Path):
        _write_report(
            data_root,
            _THREAT,
            [_horizon(6.0, served=True, probability=0.42, model_id="logreg_h6")],
        )
        assert S.pipeline_available() is True

    def test_not_available_when_every_horizon_is_withheld(self, data_root: Path):
        _write_report(
            data_root,
            _THREAT,
            [_horizon(6.0, served=False, probability=None, model_id=None)],
        )
        assert S.pipeline_available() is False

    def test_not_available_with_no_reports(self, data_root: Path):
        assert S.pipeline_available() is False
        assert S.load_reports() == {}

    def test_malformed_report_is_skipped(self, data_root: Path):
        directory = data_root / "processed" / "validation" / _THREAT
        directory.mkdir(parents=True)
        (directory / "transition_report.json").write_text("{not json", encoding="utf-8")
        assert S.load_reports() == {}
        assert S.pipeline_available() is False


class TestFixtureFallback:
    def test_unknown_threat_falls_back_to_fixtures(self, data_root: Path):
        _write_report(
            data_root,
            _THREAT,
            [_horizon(6.0, served=True, probability=0.42, model_id="logreg_h6")],
        )
        fixture = S.get_transition("THR-2026-0001")
        assert fixture is not None
        assert fixture["transition_evaluation"]["probability"] is None

    def test_unknown_id_returns_none(self, data_root: Path):
        assert S.get_transition("THR-9999-9999") is None
