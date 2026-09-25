"""Metrics for the transition intelligence layer.

Brier score, ROC-AUC, calibration, lead-time error — and, always, the **base
rate**.

Why the base rate is not optional: transitions to a higher severity are rare. A
model that predicts "no transition" for every sample scores a very good Brier
score and an undefined-or-terrible AUC while having learnt nothing. Reporting a
score without the base rate is the single easiest way to present a useless model
as a working one, so ``evaluate_transition`` always emits it and flags when a set
of predictions is no better than the constant base-rate forecast.

No thresholds here. Everything is a measured number; deciding what counts as good
is the gate's job (``src/validation/evaluation.py``), and the gate's thresholds
stay ``null`` until real distributions are measured.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.stats import rankdata

from src.shared.metrics import MetricTable

__all__ = [
    "base_rate",
    "brier_score",
    "brier_skill_score",
    "calibration_table",
    "expected_calibration_error",
    "lead_time_error",
    "roc_auc",
    "evaluate_transition",
]


def _validate(
    probabilities: Sequence[float],
    outcomes: Sequence[bool | int],
) -> tuple[np.ndarray, np.ndarray]:
    """Coerce inputs and check they are paired, in range, and non-empty."""
    predicted = np.asarray(probabilities, dtype=float)
    observed = np.asarray(outcomes, dtype=float)
    if predicted.shape != observed.shape:
        raise ValueError(
            f"probabilities {predicted.shape} and outcomes {observed.shape} must pair up"
        )
    if predicted.size == 0:
        raise ValueError("no samples supplied")
    if not np.isfinite(predicted).all():
        raise ValueError(
            "probabilities contain non-finite values; an unknown probability is null, "
            "not NaN"
        )
    if (predicted < 0).any() or (predicted > 1).any():
        raise ValueError("probabilities must lie in [0, 1]")
    if not np.isin(observed, (0.0, 1.0)).all():
        raise ValueError("outcomes must be binary (0/1 or False/True)")
    return predicted, observed


def base_rate(outcomes: Sequence[bool | int]) -> float | None:
    """Share of samples in which the transition actually happened.

    The reference every other number here has to be read against.
    """
    observed = np.asarray(outcomes, dtype=float)
    if observed.size == 0:
        return None
    return float(np.mean(observed))


def brier_score(
    probabilities: Sequence[float],
    outcomes: Sequence[bool | int],
) -> float | None:
    """Mean squared error of the probability. Lower is better; 0.25 is the
    no-skill level for a 50/50 forecast."""
    predicted, observed = _validate(probabilities, outcomes)
    return float(np.mean((predicted - observed) ** 2))


def brier_skill_score(
    probabilities: Sequence[float],
    outcomes: Sequence[bool | int],
) -> float | None:
    """``1 - brier / brier_reference`` against the constant base-rate forecast.

    ``0`` means no better than always predicting the base rate, which for a rare
    event is a genuinely hard baseline to beat. Negative means worse than it.
    Returns ``None`` when the reference score is zero (a degenerate all-or-nothing
    sample) because the ratio is then undefined.
    """
    predicted, observed = _validate(probabilities, outcomes)
    reference_rate = float(np.mean(observed))
    reference = float(np.mean((reference_rate - observed) ** 2))
    if reference <= 0:
        return None
    return float(1.0 - float(np.mean((predicted - observed) ** 2)) / reference)


def roc_auc(
    probabilities: Sequence[float],
    outcomes: Sequence[bool | int],
) -> float | None:
    """Rank-based ROC-AUC, with ties averaged.

    Returns ``None`` when only one class is present: AUC is undefined there, and
    reporting ``0.5`` would imply a measured discrimination that was never
    measured.
    """
    predicted, observed = _validate(probabilities, outcomes)
    positives = observed == 1
    n_positive = int(positives.sum())
    n_negative = int(observed.size - n_positive)
    if n_positive == 0 or n_negative == 0:
        return None
    ranks = rankdata(predicted)
    auc = (ranks[positives].sum() - n_positive * (n_positive + 1) / 2.0) / (
        n_positive * n_negative
    )
    return float(auc)


def calibration_table(
    probabilities: Sequence[float],
    outcomes: Sequence[bool | int],
    *,
    bins: int = 10,
) -> list[dict[str, Any]]:
    """Reliability table: predicted vs observed frequency per probability bin.

    Bins with no samples carry ``observed_frequency=None`` rather than ``0.0``, so
    an empty bin can never be mistaken for a bin where nothing happened.
    """
    if bins < 1:
        raise ValueError("bins must be at least 1")
    predicted, observed = _validate(probabilities, outcomes)
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.clip(np.digitize(predicted, edges[1:-1], right=False), 0, bins - 1)

    table: list[dict[str, Any]] = []
    for index in range(bins):
        mask = indices == index
        count = int(mask.sum())
        table.append(
            {
                "bin_lower": float(edges[index]),
                "bin_upper": float(edges[index + 1]),
                "count": count,
                "mean_predicted": float(predicted[mask].mean()) if count else None,
                "observed_frequency": float(observed[mask].mean()) if count else None,
            }
        )
    return table


def expected_calibration_error(
    probabilities: Sequence[float],
    outcomes: Sequence[bool | int],
    *,
    bins: int = 10,
) -> float | None:
    """Sample-weighted mean gap between predicted and observed frequency.

    ``0`` is perfect calibration. A model can be well-calibrated and still
    useless, which is why this is reported next to the base rate and not instead
    of it.
    """
    table = calibration_table(probabilities, outcomes, bins=bins)
    total = sum(row["count"] for row in table)
    if total == 0:
        return None
    gaps = [
        row["count"] * abs(row["observed_frequency"] - row["mean_predicted"])
        for row in table
        if row["count"] and row["observed_frequency"] is not None
    ]
    if not gaps:
        return None
    return float(sum(gaps) / total)


def lead_time_error(
    predicted_window_hours: Sequence[float | None],
    actual_transition_hours: Sequence[float | None],
) -> float | None:
    """Mean absolute error in hours between the predicted window centre and reality.

    Calculated only over samples where both sides are known; a prediction with no
    observed transition to compare against is dropped rather than counted as
    correct. Returns ``None`` when nothing was comparable — reporting ``0.0`` there
    would read as a perfect result.
    """
    if len(predicted_window_hours) != len(actual_transition_hours):
        raise ValueError("predicted and actual series must be the same length")
    errors = [
        abs(float(predicted) - float(actual))
        for predicted, actual in zip(
            predicted_window_hours, actual_transition_hours, strict=True
        )
        if predicted is not None and actual is not None
    ]
    if not errors:
        return None
    return float(np.mean(errors))


def evaluate_transition(
    probabilities: Sequence[float],
    outcomes: Sequence[bool | int],
    *,
    predicted_window_hours: Sequence[float | None] | None = None,
    actual_transition_hours: Sequence[float | None] | None = None,
    target_state: str | None = None,
    model_id: str | None = None,
    calibration_bins: int = 10,
    context: dict[str, Any] | None = None,
) -> MetricTable:
    """Full transition metric table.

    Metric names match ``configs/validation.yaml`` → ``transition.metrics``
    (``brier_score``, ``roc_auc``, ``calibration``, ``lead_time_error_hours``),
    with the base rate and Brier skill score added because a score without them is
    not interpretable for a rare event.

    ``calibration`` is reported as the upper-tail gap (expected calibration error)
    because that is the number a reliability diagram is summarised by.
    """
    rate = base_rate(outcomes)
    table = MetricTable(
        units="hours",
        context={
            "samples": len(outcomes),
            "positives": int(np.sum(np.asarray(outcomes, dtype=float))),
            "target_state": target_state,
            "model_id": model_id,
            "trained": model_id is not None,
            **(context or {}),
        },
    )

    table.metrics["base_rate"] = rate
    table.notes["base_rate"] = (
        f"{rate:.1%} of samples transitioned — a model predicting 'no transition' "
        "everywhere scores the reference Brier below"
        if rate is not None
        else "no samples"
    )

    table.metrics["brier_score"] = brier_score(probabilities, outcomes)
    table.metrics["skill_score"] = brier_skill_score(probabilities, outcomes)
    table.metrics["roc_auc"] = roc_auc(probabilities, outcomes)
    table.metrics["calibration"] = expected_calibration_error(
        probabilities, outcomes, bins=calibration_bins
    )

    if table.metrics["skill_score"] is not None:
        if table.metrics["skill_score"] <= 0:
            table.notes["skill_score"] = (
                "no better than always predicting the base rate — this is not a "
                "working model"
            )
        else:
            table.notes["skill_score"] = "improvement over the constant base-rate forecast"

    if predicted_window_hours is not None and actual_transition_hours is not None:
        table.metrics["lead_time_error_hours"] = lead_time_error(
            predicted_window_hours, actual_transition_hours
        )
    else:
        table.metrics["lead_time_error_hours"] = None
        table.notes["lead_time_error_hours"] = (
            "no observed transition times supplied to compare against"
        )

    if table.metrics["roc_auc"] is None and rate not in (None, 0.0, 1.0):
        table.notes["roc_auc"] = "undefined: only one class present in the sample"
    return table
