"""Extreme-preservation metrics for downscaling.

This is the module that makes any downscaling claim defensible. Generic image
similarity is **not** enough: a field can score a fine RMSE while having flattened
the one value the alert depends on. So the primary output here is not RMSE but
whether the extreme survived.

The bar this module is held to (from ``team/Navaneedan.md``): *a test case where
interpolation obviously fails must actually be flagged as a failure. If your metric
passes a smoothed field, the metric is wrong.* ``tests/test_downscaling.py``
enforces exactly that against a deliberately smoothed field.

Conventions
-----------
* An "extreme" is the **maximum** of the field. Cold extremes (minimum) need the
  field sign-flipped by the caller — stated here rather than guessed at.
* Every metric compares only cells that are finite in **both** fields, and records
  what fraction that was. A metric computed over a handful of overlapping cells is
  not evidence, so below ``MIN_VALID_FRACTION`` the metric is reported as ``None``.
* A metric that cannot be computed returns ``None``. It never returns ``nan``, a
  ``0.0``, or an estimate — the same null rule the API follows.

No threshold is baked in anywhere. Gate thresholds stay ``null`` in
``configs/validation.yaml`` until real distributions are measured (see
``docs/experiments.md``), and the functions that need one require the caller to
pass it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.shared.fields import GriddedField
from src.shared.metrics import DIMENSIONLESS_METRICS, MetricTable

__all__ = [
    "DIMENSIONLESS_METRICS",
    "MIN_VALID_FRACTION",
    "MetricTable",
    "dice",
    "evaluate_downscaling",
    "extreme_bias",
    "iou",
    "mae",
    "peak_preservation",
    "percentile_error",
    "resolve_extreme_threshold",
    "rmse",
    "threshold_exceedance_change",
    "threshold_exceedance_fraction",
    "threshold_mask",
]

MIN_VALID_FRACTION = 0.5
"""Below this share of comparable cells, a metric is reported as not computable."""


def _aligned_pair(a: GriddedField, b: GriddedField) -> tuple[GriddedField, GriddedField]:
    """Orient two fields the same way and confirm they share a grid.

    Both operands are put in ascending-latitude order, so a comparison between an
    ERA5 slice (descending) and a target grid (ascending) does not silently
    misalign rows. Raises rather than guessing on a genuine grid mismatch.
    """
    left = a.ascending()
    right = b.ascending()
    if left.shape != right.shape:
        raise ValueError(
            f"fields have different shapes {left.shape} and {right.shape}; "
            "resample onto a common grid before comparing "
            "(see src.downscaling.baseline.downscale_to_reference)"
        )
    if not (
        np.allclose(left.latitude, right.latitude) and np.allclose(left.longitude, right.longitude)
    ):
        raise ValueError(
            "fields have different coordinate axes; resample onto a common grid first"
        )
    return left, right


def _common_finite(a: GriddedField, b: GriddedField) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the values finite in both aligned fields, plus their fraction."""
    left, right = _aligned_pair(a, b)
    mask = np.isfinite(left.values) & np.isfinite(right.values)
    return left.values[mask], right.values[mask], float(np.mean(mask))


def _aligned_masks(
    predicted: GriddedField,
    reference: GriddedField,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Thresholded masks for two fields, aligned onto a shared orientation."""
    left, right = _aligned_pair(predicted, reference)
    return threshold_mask(left, threshold), threshold_mask(right, threshold)


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------


def rmse(predicted: GriddedField, reference: GriddedField) -> float | None:
    """Root mean squared error over comparable cells."""
    left, right, fraction = _common_finite(predicted, reference)
    if fraction < MIN_VALID_FRACTION or left.size == 0:
        return None
    return float(np.sqrt(np.mean((left - right) ** 2)))


def mae(predicted: GriddedField, reference: GriddedField) -> float | None:
    """Mean absolute error over comparable cells."""
    left, right, fraction = _common_finite(predicted, reference)
    if fraction < MIN_VALID_FRACTION or left.size == 0:
        return None
    return float(np.mean(np.abs(left - right)))


def peak_preservation(predicted: GriddedField, reference: GriddedField) -> float | None:
    """``predicted peak / reference peak``.

    The headline metric. ``1.0`` means the extreme survived, ``0.62`` means it was
    flattened by a third. Returns ``None`` when the reference peak is zero or
    non-finite, because a ratio against zero is not a number worth reporting.
    """
    reference_peak = reference.peak
    predicted_peak = predicted.peak
    if reference_peak is None or predicted_peak is None:
        return None
    if reference_peak == 0:
        return None
    return float(predicted_peak / reference_peak)


def extreme_bias(predicted: GriddedField, reference: GriddedField) -> float | None:
    """``predicted extreme - reference extreme``, in the field's own units.

    Signed: negative means the extreme was weakened, positive means it was
    exaggerated. An exaggerated extreme is also a failure, just a louder one.
    """
    reference_peak = reference.peak
    predicted_peak = predicted.peak
    if reference_peak is None or predicted_peak is None:
        return None
    return float(predicted_peak - reference_peak)


def percentile_error(
    predicted: GriddedField,
    reference: GriddedField,
    percentiles: tuple[float, ...] = (95.0, 99.0),
) -> dict[str, float | None]:
    """Absolute error at each upper-tail percentile, ``{percentile: error}``.

    The upper tail is where the alert lives. A field can match the reference
    median and still be wrong at p99, which is the value that matters.
    """
    left, right, fraction = _common_finite(predicted, reference)
    result: dict[str, float | None] = {}
    for percentile in percentiles:
        key = f"p{percentile:g}"
        if fraction < MIN_VALID_FRACTION or left.size == 0:
            result[key] = None
            continue
        result[key] = float(abs(np.percentile(left, percentile) - np.percentile(right, percentile)))
    return result


def threshold_mask(field: GriddedField, threshold: float) -> np.ndarray:
    """Boolean mask of cells at or above ``threshold``.

    NaN cells are ``False``: an unknown value is not an extreme, and treating it as
    one would inflate the footprint.
    """
    return np.isfinite(field.values) & (field.values >= threshold)


def threshold_exceedance_fraction(field: GriddedField, threshold: float) -> float | None:
    """Share of finite cells at or above ``threshold``."""
    finite = np.isfinite(field.values)
    if not finite.any():
        return None
    return float(np.sum(finite & (field.values >= threshold)) / np.sum(finite))


def iou(predicted: GriddedField, reference: GriddedField, threshold: float) -> float | None:
    """Intersection over union of the two thresholded footprints."""
    left, right = _aligned_masks(predicted, reference, threshold)
    union = int(np.sum(left | right))
    if union == 0:
        # Both footprints are empty. That is mutual agreement, but IoU has no
        # value for two empty sets, so it is reported as not computable.
        return None
    return float(np.sum(left & right) / union)


def dice(predicted: GriddedField, reference: GriddedField, threshold: float) -> float | None:
    """Dice coefficient of the two thresholded footprints."""
    left, right = _aligned_masks(predicted, reference, threshold)
    total = int(np.sum(left) + np.sum(right))
    if total == 0:
        return None
    return float(2 * np.sum(left & right) / total)


def threshold_exceedance_change(
    predicted: GriddedField,
    reference: GriddedField,
    threshold: float,
) -> dict[str, float | None]:
    """Exceedance fraction before vs after, and the absolute change.

    A downscaling step that doubles the exceedance fraction has invented area, even
    if it happened to reproduce the peak.
    """
    before = threshold_exceedance_fraction(reference, threshold)
    after = threshold_exceedance_fraction(predicted, threshold)
    change = None if before is None or after is None else float(after - before)
    return {"reference_fraction": before, "predicted_fraction": after, "change": change}


# ---------------------------------------------------------------------------
# The evaluation entry point
# ---------------------------------------------------------------------------


def resolve_extreme_threshold(
    reference: GriddedField,
    extreme_threshold: float | None,
    threshold_percentile: float,
) -> tuple[float | None, str | None]:
    """Decide the threshold used for footprint and exceedance metrics.

    With no threshold supplied, it is taken from the **reference** field at
    ``threshold_percentile``. This is a derivation, not a guess, and the value and
    its source are both written into the table so the choice is visible.
    """
    if extreme_threshold is not None:
        return float(extreme_threshold), "supplied by caller"
    derived = reference.percentile(threshold_percentile)
    if derived is None:
        return None, None
    return float(derived), f"reference field p{threshold_percentile:g}"


def evaluate_downscaling(
    predicted: GriddedField,
    reference: GriddedField,
    *,
    extreme_threshold: float | None = None,
    threshold_percentile: float = 99.0,
    percentiles: tuple[float, ...] = (95.0, 99.0),
    context: dict[str, Any] | None = None,
) -> MetricTable:
    """Full extreme-preservation table for a downscaling output.

    Metric names match ``configs/validation.yaml`` →
    ``downscaling.metrics`` (``rmse``, ``mae``, ``peak_preservation``,
    ``extreme_bias``, ``iou``, ``dice``, ``percentile_error``), plus the
    per-percentile detail as separate entries.
    """
    table = MetricTable(
        units=reference.units or predicted.units,
        context={
            "predicted": predicted.name,
            "reference": reference.name,
            "predicted_shape": predicted.shape,
            "reference_shape": reference.shape,
            "predicted_resolution_km": round(predicted.resolution_km(), 3),
            "trained": predicted.attrs.get("trained", False),
            "synthetic": predicted.attrs.get("synthetic", False),
            **(context or {}),
        },
    )

    table.metrics["rmse"] = rmse(predicted, reference)
    table.metrics["mae"] = mae(predicted, reference)
    table.metrics["peak_preservation"] = peak_preservation(predicted, reference)
    table.metrics["extreme_bias"] = extreme_bias(predicted, reference)

    threshold, source = resolve_extreme_threshold(
        reference, extreme_threshold, threshold_percentile
    )
    table.extreme_threshold = threshold
    table.extreme_threshold_source = source

    if threshold is not None:
        table.metrics["iou"] = iou(predicted, reference, threshold)
        table.metrics["dice"] = dice(predicted, reference, threshold)
        exceedance = threshold_exceedance_change(predicted, reference, threshold)
        table.metrics["exceedance_change"] = exceedance["change"]
        table.notes["exceedance_change"] = (
            f"predicted {exceedance['predicted_fraction']:.4g} vs reference "
            f"{exceedance['reference_fraction']:.4g} of finite cells"
            if exceedance["predicted_fraction"] is not None
            and exceedance["reference_fraction"] is not None
            else "not computable"
        )
    else:
        table.metrics["iou"] = None
        table.metrics["dice"] = None
        table.metrics["exceedance_change"] = None
        table.notes["iou"] = "no extreme threshold available"

    tail = percentile_error(predicted, reference, percentiles)
    table.metrics.update({f"percentile_error_{key}": value for key, value in tail.items()})
    computable = [value for value in tail.values() if value is not None]
    table.metrics["percentile_error"] = (
        float(np.mean(computable)) if computable else None
    )
    table.notes["percentile_error"] = (
        "mean absolute error across " + ", ".join(f"p{q:g}" for q in percentiles)
    )

    _, _, valid_fraction = _common_finite(predicted, reference)
    table.valid_fraction = valid_fraction
    if valid_fraction < MIN_VALID_FRACTION:
        table.notes["valid_fraction"] = (
            f"only {valid_fraction:.1%} of cells are finite in both fields — metrics "
            f"below are not evidence"
        )
    return table
