"""Transition service adapter.

Responsibilities:
  - Load transition intelligence output written by
    ``src/models/transition/transition_model.py``.
  - Fall back to ``data/samples/transitions.json`` (which has null probabilities)
    when no report exists for a threat.

How the pipeline output is read
-------------------------------
``train_all_horizons`` writes one report per event to
``<DATA_ROOT>/processed/validation/<threat_id>/transition_report.json``. Each
horizon carries a ``serving`` block holding the probability for the threat's most
recent fully-observed state, but only when that horizon cleared the publishing
gate (``docs/transition.md`` §7). This service **serves exactly what the gate
allows**:

  * the shortest horizon with ``serving.served`` is published (earliest actionable
    warning);
  * a horizon that failed the gate keeps ``probability = null`` here even though
    its metrics are in the report — withholding is a result, not an error;
  * if no horizon is servable the response is still HTTP 200 with null fields.

Degraded-mode contract (from the API spec):
  If the transition model has not produced validated output, this service returns
  an object where probability = null and calibrated = false. The API returns HTTP
  200 with null fields — it does NOT return an error.

Honesty note: ``calibrated`` stays ``false`` and ``provenance.training_run`` names
the proxy. These are prototype, severity-proxy probabilities — not tracked-threat
escalation (``docs/transition.md`` §2).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from . import fixtures

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_ROOT = _REPO_ROOT / "data"


def _data_root() -> Path:
    """Resolve ``DATA_ROOT`` the same way ``fixtures.py`` does, but anchored.

    A relative ``DATA_ROOT`` (the ``./data`` shipped in ``.env.example``) is
    resolved against the repo root so the service cannot become CWD-dependent.
    """
    root = Path(os.getenv("DATA_ROOT", str(_DEFAULT_DATA_ROOT)))
    if not root.is_absolute():
        root = _REPO_ROOT / root
    return root


def _report_root() -> Path:
    return _data_root() / "processed" / "validation"


def load_reports() -> dict[str, dict[str, Any]]:
    """Read every ``<threat_id>/transition_report.json`` under the report dir.

    A malformed or unreadable report is skipped rather than crashing the endpoint:
    a bad artefact must degrade to fixtures, not 500.
    """
    root = _report_root()
    if not root.is_dir():
        return {}
    reports: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("*/transition_report.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            reports[path.parent.name] = payload
    return reports


def _shortest_served(horizons: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The shortest horizon the publishing gate allows to be served."""
    served = [
        horizon
        for horizon in horizons
        if (horizon.get("serving") or {}).get("served")
        and (horizon.get("serving") or {}).get("probability") is not None
    ]
    if not served:
        return None
    return min(served, key=lambda horizon: horizon.get("horizon_hours") or 0.0)


def _quantile(horizon: dict[str, Any] | None) -> float | None:
    if not horizon:
        return None
    value = (horizon.get("dataset") or {}).get("escalation_quantile")
    return float(value) if isinstance(value, (int, float)) else None


def _response_from_report(threat_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Map one stored report onto the ``/transition`` response shape."""
    horizons = [h for h in (payload.get("horizons") or []) if isinstance(h, dict)]
    served = _shortest_served(horizons)
    context_horizon = served or (horizons[0] if horizons else None)

    quantile = _quantile(context_horizon)
    band = f"p{quantile:g}" if quantile is not None else None

    base_rate = None
    if context_horizon:
        rate = (context_horizon.get("dataset") or {}).get("base_rate")
        base_rate = float(rate) if isinstance(rate, (int, float)) else None
    # Brier score of the constant base-rate forecast: p(1 - p). The reference the
    # model's own score has to beat, so it is reported next to it.
    baseline = base_rate * (1.0 - base_rate) if base_rate is not None else None

    horizon_hours = served.get("horizon_hours") if served else None
    probability = (served.get("serving") or {}).get("probability") if served else None
    model_id = ((served.get("model") or {}).get("model_id")) if served else None

    generated = (payload.get("_meta") or {}).get("generated_at")
    training_run = (
        f"{generated} (prototype: severity proxy, not tracked-threat escalation)"
        if generated
        else "prototype: severity proxy"
    )

    return {
        "threat_id": threat_id,
        "current_state": f"below_{band}" if band else "unknown",
        "transition_evaluation": {
            "target_state": f"above_{band}" if band else None,
            "probability": probability,
            "calibrated": False,
            "brier_score_baseline": baseline,
            "expected_window_hours": (
                {
                    "earliest": None,
                    "most_likely": None,
                    "latest": int(horizon_hours),
                }
                if horizon_hours is not None
                else None
            ),
        },
        # The tracked-threat lifecycle comes from src/transition/lifecycle.py, which
        # needs a threat history; a transition report does not carry one.
        "lifecycle_phase": "unknown",
        "provenance": {"model_id": model_id, "training_run": training_run},
    }


def _load_from_pipeline() -> dict[str, dict[str, Any]] | None:
    """Transition responses keyed by ``threat_id``, or ``None`` when none exist."""
    reports = load_reports()
    if not reports:
        return None
    return {
        threat_id: _response_from_report(threat_id, payload)
        for threat_id, payload in reports.items()
    }


def pipeline_available() -> bool:
    """True when at least one stored report has a horizon the gate allows serving.

    This is the condition behind ``/health`` → ``transition_intelligence``. It is
    derived from the artefacts rather than hard-coded so the flag cannot claim a
    capability the reports do not back up.
    """
    for payload in load_reports().values():
        horizons = [h for h in (payload.get("horizons") or []) if isinstance(h, dict)]
        if _shortest_served(horizons) is not None:
            return True
    return False


def get_transition(threat_id: str) -> dict | None:
    """Return transition intelligence for a threat.

    Returns None if no transition data exists for the given threat_id.
    Returns an object with null probability fields if the model is withheld or
    uncalibrated (degraded mode — HTTP 200 with null values, as per spec).
    """
    pipeline = _load_from_pipeline()
    if pipeline and threat_id in pipeline:
        return pipeline[threat_id]
    return fixtures.load_transitions().get(threat_id)
