"""Learned threat transition model (Threat Transition Intelligence Engine).

Predicts, for a time step and a horizon ``H``::

    P( severity band reaches TARGET within (t, t + H] | precursor state at t )

The target definition is **not** in this file — it is written down in
``docs/transition.md`` and this module implements exactly that. Read that first;
the decision that matters (what counts as an escalation) is there, deliberately,
because a model whose target lives only in its own source cannot be audited.

Three properties are enforced in code rather than left to discipline:

1. **No future leakage.** Features come from ``t`` only; the label is taken
   strictly after ``t``; rows whose ``t + H`` runs past the end of the series are
   dropped rather than labelled optimistically.
2. **No invented thresholds.** The escalation boundary is a quantile of the
   event's own severity distribution. ``configs/tracking.yaml`` keeps
   ``severity_bands`` null precisely so a guessed cut-off cannot creep in here.
3. **No fitting on data that cannot support it.** Below ``min_rows`` rows or
   ``min_positives`` positives the horizon is reported as **untrained** and its
   probability stays ``null``. Fitting a 6-row logistic regression would produce
   a number, and the number would be meaningless.

Reported metrics are **out-of-fold** where the sample allows it, so they are not
merely the model grading its own homework.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.shared.contracts import EventType, PrecursorSeries, TransitionEstimate, format_timestamp
from src.shared.metrics import MetricTable
from src.validation.transition_metrics import evaluate_transition

__all__ = [
    "FEATURE_FIELDS",
    "HORIZONS_HOURS",
    "MIN_PUBLISHABLE_AUC",
    "MIN_PUBLISHABLE_SKILL",
    "SEVERITY_FIELD_BY_EVENT",
    "FittedTransitionModel",
    "TransitionDataset",
    "TransitionReport",
    "TransitionTarget",
    "build_dataset",
    "derive_threshold",
    "load_precursor_entries",
    "train_all_horizons",
]

FEATURE_FIELDS: tuple[str, ...] = (
    "pressure_tendency_3h_hpa",
    "wind_divergence_s1",
    "wind_speed_ms",
    "precipitation_rate_mm3h",
    "t2m_anomaly_k",
)
"""Precursor features actually computed from the available surface data.

Deliberately excludes ``moisture_flux_convergence_g_kg_s``, ``vorticity_850_s1``
and ``theta_e_gradient_k_100km``: the dataset has no pressure-level variables, so
those are ``null`` for every sample. They are never used and never zero-filled —
imputing them would invent the physics the project claims to explain.
"""

HORIZONS_HOURS: tuple[float, ...] = (6.0, 12.0, 18.0, 24.0)

SEVERITY_FIELD_BY_EVENT: dict[str, str] = {
    "cyclone": "precipitation_rate_mm3h",
    "extreme_rainfall": "precipitation_rate_mm3h",
    "gale_wind": "wind_speed_ms",
    "heatwave": "t2m_anomaly_k",
}
DEFAULT_SEVERITY_FIELD = "precipitation_rate_mm3h"

DEFAULT_MODEL_PREFIX = "logreg_precursor"
PROCESSED_PRECURSORS = Path("data/processed/precursors.json")

# ---------------------------------------------------------------------------
# Publishing gate.
#
# A fitted model is not the same as a publishable one. A horizon may leave this
# module as an API probability only if it beats **both** canonical no-skill
# references: the constant base-rate forecast (Brier skill > 0) and a coin flip
# (ROC-AUC > 0.5). Neither bound is a tuned threshold -- they are the exact
# values that zero information produces, which is why they are the only two
# numbers this gate is allowed to contain. Anything stricter would be a guess
# dressed up as a criterion.
#
# The two conditions catch different failures, and this repo has both in it:
# a model can rank cases well (AUC 0.80) while still losing to the base rate on
# Brier score, and it can beat the base rate while only barely ranking above
# chance. Either way the probability is withheld rather than shipped.
# ---------------------------------------------------------------------------
MIN_PUBLISHABLE_SKILL = 0.0
MIN_PUBLISHABLE_AUC = 0.5


class TransitionDataError(RuntimeError):
    """The precursor input could not be turned into a usable dataset."""


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransitionTarget:
    """The explicit definition of what is being predicted.

    ``threshold`` is derived from the data (see :func:`derive_threshold`), never
    configured by hand, and is carried on the target so a prediction can always
    be traced back to the boundary that produced it.
    """

    horizon_hours: float
    severity_field: str
    escalation_quantile: float
    threshold: float
    step_hours: float

    @property
    def steps_ahead(self) -> int:
        """How many time steps the horizon spans, at this series' cadence."""
        return max(1, int(round(self.horizon_hours / self.step_hours)))

    @property
    def band_label(self) -> str:
        return f"above_p{self.escalation_quantile:g}"

    def describe(self) -> str:
        return (
            f"P({self.severity_field} >= {self.threshold:.4g} within "
            f"{self.horizon_hours:g}h | state at t), boundary = "
            f"p{self.escalation_quantile:g} of the event's own distribution"
        )


def derive_threshold(severity: Sequence[float | None], quantile: float) -> float | None:
    """Escalation boundary as a quantile of the severity series.

    Returns ``None`` when there is nothing finite to take a quantile of, so the
    caller reports "no boundary" instead of receiving a default number.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError("escalation_quantile must lie strictly between 0 and 1")
    finite = np.asarray([value for value in severity if value is not None], dtype=float)
    if finite.size == 0:
        return None
    return float(np.quantile(finite, quantile))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class TransitionDataset:
    """A built dataset for one horizon, with everything needed to audit it."""

    features: np.ndarray
    labels: np.ndarray
    feature_names: list[str]
    timestamps: list[datetime]
    threat_ids: list[str]
    target: TransitionTarget
    severity: list[float | None] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.labels.size)

    @property
    def positives(self) -> int:
        return int(self.labels.sum()) if self.labels.size else 0

    @property
    def negatives(self) -> int:
        return int(self.labels.size - self.positives)

    @property
    def base_rate(self) -> float | None:
        """Fraction of rows that escalated. Always report this next to a score."""
        return float(self.labels.mean()) if self.labels.size else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": len(self),
            "positives": self.positives,
            "negatives": self.negatives,
            "base_rate": self.base_rate,
            "feature_names": list(self.feature_names),
            "threats": sorted(set(self.threat_ids)),
            "threshold": self.target.threshold,
            "escalation_quantile": self.target.escalation_quantile,
            "horizon_hours": self.target.horizon_hours,
            "steps_ahead": self.target.steps_ahead,
            "step_hours": self.target.step_hours,
            "target": self.target.describe(),
            "notes": list(self.notes),
        }


def _series_step_hours(timestamps: list[datetime]) -> float | None:
    """Median cadence of a series, or ``None`` when it has fewer than two steps."""
    if len(timestamps) < 2:
        return None
    deltas = np.diff(np.array([stamp.timestamp() for stamp in timestamps]))
    if deltas.size == 0 or not np.all(deltas > 0):
        return None
    return float(np.median(deltas) / 3600.0)


def build_dataset(
    entries: dict[str, dict[str, Any]],
    *,
    horizon_hours: float,
    escalation_quantile: float = 0.75,
    feature_fields: Sequence[str] = FEATURE_FIELDS,
    severity_field_override: str | None = None,
) -> TransitionDataset:
    """Build one horizon's dataset from precursor pipeline output.

    ``entries`` is the ``precursors`` mapping from ``data/processed/precursors.json``
    (keyed by ``threat_id``). A row is emitted per time step where:

    * the current severity is **below** the escalation boundary — a step already
      at or above it has nothing left to transition to, and keeping those rows
      would inflate the base rate;
    * every feature is present at ``t`` — a partially-missing feature row is
      dropped rather than imputed;
    * ``t + horizon`` still exists in the series.
    """
    features: list[list[float]] = []
    labels: list[int] = []
    stamps: list[datetime] = []
    threat_ids: list[str] = []
    all_severity: list[float | None] = []
    notes: list[str] = []
    target: TransitionTarget | None = None

    for threat_id in sorted(entries):
        entry = entries[threat_id]
        try:
            series = PrecursorSeries.from_dict(entry)
        except Exception as exc:  # a malformed entry must name itself
            raise TransitionDataError(
                f"{threat_id}: could not parse precursor series ({exc})"
            ) from exc

        if not series.series:
            notes.append(f"{threat_id}: empty series, skipped")
            continue

        event_type = entry.get("event_type")
        field = (
            severity_field_override
            or SEVERITY_FIELD_BY_EVENT.get(str(event_type), DEFAULT_SEVERITY_FIELD)
        )
        if field not in FEATURE_FIELDS:
            raise TransitionDataError(
                f"{threat_id}: severity field {field!r} is not an available feature"
            )

        stamps_here = [sample.timestamp for sample in series.series]
        if any(stamp is None for stamp in stamps_here):
            notes.append(f"{threat_id}: series has timestamps missing, skipped")
            continue
        step_hours = _series_step_hours(stamps_here)  # type: ignore[arg-type]
        if step_hours is None:
            notes.append(f"{threat_id}: series has no usable cadence, skipped")
            continue

        severity = [getattr(sample, field) for sample in series.series]
        all_severity.extend(severity)
        threshold = derive_threshold(severity, escalation_quantile)
        if threshold is None:
            notes.append(f"{threat_id}: no finite {field}, skipped")
            continue

        if target is None:
            target = TransitionTarget(
                horizon_hours=horizon_hours,
                severity_field=field,
                escalation_quantile=escalation_quantile,
                threshold=threshold,
                step_hours=step_hours,
            )
        elif target.severity_field != field:
            # Mixing events whose severity is measured differently would make the
            # threshold meaningless, so it is refused rather than silently pooled.
            raise TransitionDataError(
                f"{threat_id}: severity field {field!r} differs from "
                f"{target.severity_field!r} already in this dataset; build one dataset "
                "per event type instead of pooling them"
            )
        elif not math.isclose(threshold, target.threshold, rel_tol=1e-9, abs_tol=1e-12):
            # The boundary is a quantile of *each event's own* severity, so pooling
            # two events would quietly label the second one against the first one's
            # boundary. Two events of the same type still have different severity
            # distributions, so this is a relabelling bug rather than a rounding
            # detail: refuse it and let the caller build one dataset per event.
            raise TransitionDataError(
                f"{threat_id}: escalation threshold {threshold:.6g} differs from "
                f"{target.threshold:.6g} already in this dataset; the boundary is a "
                "quantile of one event's own severity, so build one dataset per event"
            )

        steps = max(1, int(round(horizon_hours / step_hours)))
        for index in range(len(series.series) - steps):
            current = severity[index]
            if current is None or current >= threshold:
                continue
            row = [getattr(series.series[index], name) for name in feature_fields]
            if any(value is None for value in row):
                continue
            future = severity[index + 1 : index + 1 + steps]
            if any(value is None for value in future):
                continue
            features.append([float(value) for value in row])
            labels.append(int(any(value >= threshold for value in future)))
            stamps.append(stamps_here[index])  # type: ignore[arg-type]
            threat_ids.append(threat_id)

    if target is None:
        target = TransitionTarget(
            horizon_hours=horizon_hours,
            severity_field=severity_field_override or DEFAULT_SEVERITY_FIELD,
            escalation_quantile=escalation_quantile,
            threshold=float("nan"),
            step_hours=float("nan"),
        )
        notes.append("no usable series, so no threshold was derived")

    return TransitionDataset(
        features=np.asarray(features, dtype=float).reshape(len(features), len(feature_fields)),
        labels=np.asarray(labels, dtype=int),
        feature_names=list(feature_fields),
        timestamps=stamps,
        threat_ids=threat_ids,
        target=target,
        severity=all_severity,
        notes=notes,
    )


def load_precursor_entries(
    path: str | Path = PROCESSED_PRECURSORS,
) -> dict[str, dict[str, Any]]:
    """Read the ``precursors`` mapping from the precursor pipeline's output.

    Raises rather than returning empty when the file is missing, because the
    caller needs to know it should run
    ``python -m src.precursors.pipeline`` first.
    """
    target = Path(path)
    if not target.exists():
        raise TransitionDataError(
            f"{target} not found. Run: python -m src.precursors.pipeline"
        )
    payload = json.loads(target.read_text(encoding="utf-8"))
    entries = payload.get("precursors") or {}
    if not entries:
        raise TransitionDataError(f"{target} contains no precursor entries")
    return entries


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _estimator() -> Pipeline:
    """Standardised logistic regression.

    Scaling is not optional here: ``wind_divergence_s1`` is ~1e-5 while
    ``precipitation_rate_mm3h`` reaches ~1e2, and an unscaled logistic regression
    on those would effectively ignore everything but the largest-scale feature.
    """
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=2000)),
        ]
    )


@dataclass
class FittedTransitionModel:
    """One fitted model for one horizon, plus where it came from."""

    target: TransitionTarget
    model_id: str
    estimator: Pipeline
    feature_names: list[str]
    training_rows: int
    base_rate: float | None
    notes: list[str] = field(default_factory=list)

    def predict_probability(self, features: Sequence[float] | dict[str, float]) -> float | None:
        """Probability this state escalates within the horizon.

        Returns ``None`` — never a fabricated number — when a required feature is
        missing, because an incomplete state has no probability.
        """
        if isinstance(features, dict):
            if any(features.get(name) is None for name in self.feature_names):
                return None
            vector = [float(features[name]) for name in self.feature_names]
        else:
            vector = [float(value) for value in features]
        if len(vector) != len(self.feature_names):
            raise ValueError(
                f"expected {len(self.feature_names)} features, got {len(vector)}"
            )
        probability = self.estimator.predict_proba(np.asarray([vector], dtype=float))[0, 1]
        return float(probability)

    def to_estimate(self, features: Sequence[float] | dict[str, float]) -> TransitionEstimate:
        """Build the contract object the API publishes.

        ``probability`` is only set when a prediction is actually possible, and
        ``model_id`` is always set — which is what the contract requires before a
        probability may be published at all.
        """
        probability = self.predict_probability(features)
        return TransitionEstimate(
            probability=probability,
            horizon_hours=int(self.target.horizon_hours),
            confidence=None,
            calibrated=False,
            model_id=self.model_id if probability is not None else None,
            drivers=None
            if probability is None
            else [f"{self.target.severity_field} escalation within {self.target.horizon_hours:g}h"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "horizon_hours": self.target.horizon_hours,
            "target": self.target.describe(),
            "threshold": self.target.threshold,
            "feature_names": list(self.feature_names),
            "training_rows": self.training_rows,
            "base_rate": self.base_rate,
            "trained": True,
            "notes": list(self.notes),
        }


@dataclass
class HorizonResult:
    """One horizon's outcome — a model, or an explicit reason there is none."""

    horizon_hours: float
    dataset: TransitionDataset
    model: FittedTransitionModel | None = None
    metrics: MetricTable | None = None
    reason: str | None = None

    @property
    def trained(self) -> bool:
        return self.model is not None

    @property
    def publish_blockers(self) -> list[str]:
        """Reasons this horizon's probability must not be served. Empty == publishable.

        Every blocker names the measurement that failed, so a withheld horizon is
        still reportable evidence rather than a silent gap.
        """
        if self.model is None:
            return [f"no model was fitted ({self.reason})"]
        if self.metrics is None:
            return ["the model was never scored, so its skill is unknown"]

        blockers: list[str] = []
        skill = self.metrics.get("skill_score")
        auc = self.metrics.get("roc_auc")
        if skill is None:
            blockers.append("Brier skill score unavailable")
        elif skill <= MIN_PUBLISHABLE_SKILL:
            blockers.append(
                f"Brier skill {skill:.4g} does not beat the constant base-rate forecast"
            )
        if auc is None:
            blockers.append("ROC-AUC unavailable")
        elif auc <= MIN_PUBLISHABLE_AUC:
            blockers.append(f"ROC-AUC {auc:.4g} is no better than chance")
        return blockers

    @property
    def publishable(self) -> bool:
        return not self.publish_blockers

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon_hours": self.horizon_hours,
            "trained": self.trained,
            "publishable": self.publishable,
            "publish_blockers": self.publish_blockers,
            "reason": self.reason,
            "dataset": self.dataset.to_dict(),
            "model": self.model.to_dict() if self.model else None,
            "metrics": self.metrics.to_dict() if self.metrics else None,
        }


@dataclass
class TransitionReport:
    """Every horizon's result, with the caveats attached."""

    results: list[HorizonResult] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now())
    notes: list[str] = field(default_factory=list)

    def trained_horizons(self) -> list[float]:
        return [result.horizon_hours for result in self.results if result.trained]

    def untrained_horizons(self) -> list[float]:
        return [result.horizon_hours for result in self.results if not result.trained]

    def publishable_horizons(self) -> list[float]:
        """Horizons whose probability may be served by the API."""
        return [result.horizon_hours for result in self.results if result.publishable]

    def withheld_horizons(self) -> list[float]:
        """Horizons that were fitted but may not be published, with their reasons."""
        return [
            result.horizon_hours
            for result in self.results
            if result.trained and not result.publishable
        ]

    def can_publish(self) -> bool:
        """Whether *any* horizon may be published.

        This is the check behind ``backend/main.py`` -> ``transition_intelligence``.
        That flag may only be flipped to True when this returns True: a stage flag
        that claims a capability the model cannot back up is exactly the kind of
        invented number the project forbids.
        """
        return bool(self.publishable_horizons())

    def to_dict(self) -> dict[str, Any]:
        return {
            "_meta": {
                "generated_at": self.generated_at.isoformat(),
                "source": "src.models.transition.transition_model.train_all_horizons",
                "note": (
                    "Prototype. Target is a SEVERITY PROXY derived from the precursor "
                    "series, not tracked-threat escalation — see docs/transition.md. "
                    "Metrics are out-of-fold where the sample allowed it."
                ),
            },
            "trained_horizons": self.trained_horizons(),
            "untrained_horizons": self.untrained_horizons(),
            "publishable_horizons": self.publishable_horizons(),
            "withheld_horizons": self.withheld_horizons(),
            "can_publish": self.can_publish(),
            "horizons": [result.to_dict() for result in self.results],
            "notes": list(self.notes),
        }

    def to_markdown(self) -> str:
        lines = [
            "## Transition model report",
            "",
            f"- Generated: {format_timestamp(self.generated_at)}",
            f"- Trained horizons: {self.trained_horizons() or '—'}",
            f"- Untrained horizons: {self.untrained_horizons() or '—'}",
            "",
            "| H | Rows | Positives | Base rate | Brier | Skill | AUC | Calibration |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for result in self.results:
            dataset = result.dataset
            if not result.trained or result.metrics is None:
                lines.append(
                    f"| {result.horizon_hours:g}h | {len(dataset)} | {dataset.positives} | "
                    f"{self._rate(dataset.base_rate)} | — | — | — | — |"
                )
                continue
            metrics = result.metrics
            lines.append(
                f"| {result.horizon_hours:g}h | {len(dataset)} | {dataset.positives} | "
                f"{self._rate(dataset.base_rate)} | {self._fmt(metrics.get('brier_score'))} | "
                f"{self._fmt(metrics.get('skill_score'))} | {self._fmt(metrics.get('roc_auc'))} | "
                f"{self._fmt(metrics.get('calibration'))} |"
            )
        lines.extend(
            [
                "",
                "### Publishing gate",
                "",
                "A trained model is not a publishable one. A horizon is served only if it",
                f"beats both the base-rate forecast (Brier skill > {MIN_PUBLISHABLE_SKILL:g})",
                f"and chance (ROC-AUC > {MIN_PUBLISHABLE_AUC:g}).",
            ]
        )
        publishable = [result for result in self.results if result.publishable]
        if publishable:
            lines.append(
                "- **Publishable:** "
                + ", ".join(f"{result.horizon_hours:g}h" for result in publishable)
            )
        else:
            lines.append("- **Publishable:** none")
        for result in self.results:
            if result.publishable:
                continue
            lines.append(
                f"- **{result.horizon_hours:g}h withheld** — "
                + "; ".join(result.publish_blockers)
            )
        lines.append(
            "- Serve a probability only while `can_publish()` is True; "
            "`backend/main.py` -> `transition_intelligence` follows this report."
        )

        untrained = [result for result in self.results if not result.trained]
        if untrained:
            lines.extend(["", "### Untrained horizons", ""])
            lines.extend(
                f"- **{result.horizon_hours:g}h** — {result.reason}" for result in untrained
            )
        if self.notes:
            lines.extend(["", "### Notes", ""])
            lines.extend(f"- {note}" for note in self.notes)
        return "\n".join(lines)

    def save(self, directory: str | Path) -> dict[str, Path]:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        json_path = target / "transition_report.json"
        markdown_path = target / "transition_report.md"
        json_path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        markdown_path.write_text(self.to_markdown() + "\n", encoding="utf-8")
        return {"json": json_path, "markdown": markdown_path}

    @staticmethod
    def _fmt(value: float | None) -> str:
        return "—" if value is None else f"{value:.4g}"

    @staticmethod
    def _rate(value: float | None) -> str:
        return "—" if value is None else f"{value:.1%}"


def train(
    dataset: TransitionDataset,
    *,
    model_prefix: str = DEFAULT_MODEL_PREFIX,
    min_rows: int = 20,
    min_positives: int = 5,
) -> tuple[FittedTransitionModel | None, str | None]:
    """Fit one horizon, or explain why it was refused.

    Returns ``(model, None)`` on success and ``(None, reason)`` when the data
    cannot support a fit. The refusal is the point: a logistic regression will
    happily fit six rows, and the resulting probability would be indistinguishable
    from a guess while looking like a model output.
    """
    if len(dataset) < min_rows:
        return None, (
            f"only {len(dataset)} rows (need {min_rows}); fitting would produce a "
            "number with no meaning"
        )
    if dataset.positives < min_positives:
        return None, (
            f"only {dataset.positives} positive rows (need {min_positives}); the "
            "escalation class is too rare to learn from"
        )
    if dataset.negatives < 2:
        return None, f"only {dataset.negatives} negative rows; nothing to discriminate against"

    estimator = _estimator()
    estimator.fit(dataset.features, dataset.labels)
    model_id = f"{model_prefix}_{dataset.target.severity_field}_h{dataset.target.horizon_hours:g}"
    model = FittedTransitionModel(
        target=dataset.target,
        model_id=model_id,
        estimator=estimator,
        feature_names=list(dataset.feature_names),
        training_rows=len(dataset),
        base_rate=dataset.base_rate,
        notes=[
            "target is a severity proxy, not tracked-threat escalation "
            "(docs/transition.md §2)",
            f"boundary = p{dataset.target.escalation_quantile:g} of the event's own "
            f"{dataset.target.severity_field}",
        ],
    )
    return model, None


def evaluate_fitted(
    dataset: TransitionDataset,
    model: FittedTransitionModel,
    *,
    folds: int = 5,
) -> MetricTable:
    """Score the model, out-of-fold where the sample allows it.

    In-sample probabilities would be the model grading its own homework on ~100
    rows. ``cross_val_predict`` gives out-of-fold probabilities instead, so the
    Brier score reflects generalisation. When there are too few rows per class for
    even two folds, it falls back to in-sample and says so in the table.
    """
    rows = len(dataset)
    usable_folds = min(folds, dataset.positives, dataset.negatives)
    if usable_folds >= 2:
        probabilities = cross_val_predict(
            _estimator(), dataset.features, dataset.labels, cv=usable_folds, method="predict_proba"
        )[:, 1]
        scheme = f"{usable_folds}-fold out-of-fold"
    else:
        probabilities = model.estimator.predict_proba(dataset.features)[:, 1]
        scheme = "in-sample (too few rows per class for cross-validation)"

    table = evaluate_transition(
        probabilities,
        dataset.labels,
        target_state=model.target.band_label,
        model_id=model.model_id,
        context={
            "horizon_hours": dataset.target.horizon_hours,
            "rows": rows,
            "evaluation": scheme,
            "proxy_target": True,
        },
    )
    table.notes["evaluation"] = (
        f"{scheme}; features and the proxy label come from the same series, so some "
        "of this reflects autocorrelation rather than a physical precursor relationship"
    )
    return table


def train_all_horizons(
    entries: dict[str, dict[str, Any]],
    *,
    horizons: Sequence[float] = HORIZONS_HOURS,
    escalation_quantile: float = 0.75,
    severity_field_override: str | None = None,
    min_rows: int = 20,
    min_positives: int = 5,
) -> TransitionReport:
    """Build, fit and score one dataset per horizon.

    Each horizon has its own sample count because rows whose ``t + H`` exceeds the
    series are dropped, so the horizons are **not comparable to one another** and
    the report says so.
    """
    report = TransitionReport()
    for horizon in horizons:
        dataset = build_dataset(
            entries,
            horizon_hours=float(horizon),
            escalation_quantile=escalation_quantile,
            severity_field_override=severity_field_override,
        )
        model, reason = train(
            dataset, min_rows=min_rows, min_positives=min_positives
        )
        if model is None:
            report.results.append(
                HorizonResult(
                    horizon_hours=float(horizon), dataset=dataset, reason=reason
                )
            )
            continue
        report.results.append(
            HorizonResult(
                horizon_hours=float(horizon),
                dataset=dataset,
                model=model,
                metrics=evaluate_fitted(dataset, model),
            )
        )

    report.notes.append(
        "Prototype target: severity is proxied from the precursor series because no "
        "threat-severity series exists in the repo yet — see docs/transition.md §2 and §6."
    )
    report.notes.append(
        "Horizons have different row counts by construction (see docs/transition.md §3), "
        "so scores across horizons are not directly comparable."
    )
    report.notes.append(
        "Publishing gate (docs/transition.md §7): a horizon is publishable only if it beats "
        f"both the base-rate forecast (Brier skill > {MIN_PUBLISHABLE_SKILL:g}) and chance "
        f"(ROC-AUC > {MIN_PUBLISHABLE_AUC:g}). Check TransitionReport.can_publish()."
    )
    if not report.trained_horizons():
        report.notes.append(
            "No horizon had enough data to fit. This is the honest outcome, not a "
            "failure to run: the available two events yield far too few rows."
        )
    return report


def severity_field_for(event_type: str | EventType | None) -> str:
    """Which field defines severity for an event type."""
    key = event_type.value if isinstance(event_type, EventType) else str(event_type)
    return SEVERITY_FIELD_BY_EVENT.get(key, DEFAULT_SEVERITY_FIELD)


if __name__ == "__main__":  # pragma: no cover - manual end-to-end run
    # One dataset per event, deliberately. The escalation boundary is a quantile of
    # each event's own severity field, so pooling events would label one against
    # another's boundary -- ``build_dataset`` refuses to do it, and this CLI does not
    # try. The per-event split is also how the numbers in docs/experiments.md were
    # produced, and only per-event reports can be read as an honest skill claim.
    entries = load_precursor_entries()
    output = Path("data/processed/validation")
    for threat_id in sorted(entries):
        report = train_all_horizons({threat_id: entries[threat_id]})
        paths = report.save(output / threat_id)
        print(f"\n\n===================== {threat_id} =====================")
        print(report.to_markdown())
        verdict = "PUBLISHABLE" if report.can_publish() else "NOT PUBLISHABLE"
        print(f"\nPublishing gate: {verdict}")
        print(f"Written to {paths['json']}")
