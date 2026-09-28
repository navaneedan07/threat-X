"""Learned downscaling: one learned convolution layer, trained on the real pair.

What is actually implemented
----------------------------
A **single learned linear filter** over a local neighbourhood, fitted by ridge
regression on the real coarse/fine pair (ERA5 0.25 deg -> ERA5-Land 0.10 deg).
That is the smallest thing that deserves the word *learned*, and it is the honest
description of this file: it is not a CNN with depth, and nothing here is called
one. ``describe()['architecture']`` records the size actually fitted.

Why this is a real super-resolution model and not a smoothed interpolation
-------------------------------------------------------------------------
For each fine cell the design vector is ``[1, centre, neighbour_1 - centre, ...]``.
The residual block makes the fitted filter an *unsharp mask* — it can add back
high-frequency structure the bilinear upsample averaged away — and the centre
coefficient can exceed 1, so the peak can be amplified. A plain linear smoother
would be a convex combination and could never exceed the coarse maximum; that is
precisely why the interpolation baseline flattens extremes. Both capabilities are
needed for the one metric that matters here, ``peak_preservation``.

Fitting details that are deliberate
-----------------------------------
* **Regularisation is selected, not guessed.** ``alpha`` is chosen on a held-out
  slice of the *training* columns, and the whole sweep is written into the
  artefact. The bilinear validation RMSE is recorded next to it, so "did the
  learned filter beat simply copying the input" is answerable from the output.
* **The holdout is spatial.** The evaluation columns are never seen during
  fitting — no cell is used for both training and scoring. The split is recorded.
* **Nothing is randomised.** No shuffling, no seeds, no dropout: the same data
  gives the same weights and the same table every run.
* **The baseline is always scored too,** on exactly the same cells, because the
  rung has to be reported alongside the model (README Phase 1).

Requires the raw archives; it never falls back to synthetic data, and reports the
gate verdict as ``undecided`` while ``configs/validation.yaml -> gates.pass`` is
``null`` rather than inventing a threshold.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.downscaling.baseline import downscale_to_reference
from src.downscaling.metrics import MIN_VALID_FRACTION, evaluate_downscaling
from src.downscaling.real_pair import (
    COARSE_FILES,
    EVENTS,
    FINE_FILES,
    as_field,
    peak_covered,
)
from src.shared.config import get_value, stage_config
from src.shared.fields import GriddedField
from src.shared.metrics import MetricTable

__all__ = [
    "DEFAULT_ALPHAS",
    "FLOOR_BY_VARIABLE",
    "LearnedDownscaler",
    "accumulate",
    "build_features",
    "holdout_split",
    "load_event_sequence",
    "select_regularisation",
    "train_and_evaluate",
]

DEFAULT_ALPHAS: tuple[float, ...] = (1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0)
"""Candidate ridge penalties, log-spaced. The selected one lands in the artefact."""

FLOOR_BY_VARIABLE: dict[str, float] = {"tp": 0.0}
"""Physical lower bounds, applied after prediction.

A negative precipitation rate is a bug rather than a prediction, so it is clipped
and the clipping is recorded. Temperature has no such bound and is left alone —
clipping a temperature field to zero would silently hide a genuine model error.
"""


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def feature_count(radius: int) -> int:
    """Columns in the design matrix: a constant, the centre value, the residuals."""
    if radius < 0:
        raise ValueError("radius must be non-negative")
    return 2 + (2 * radius + 1) ** 2


def _neighbourhood(values: np.ndarray, radius: int) -> np.ndarray:
    """Stack the ``(2r+1) x (2r+1)`` neighbourhood around every cell.

    Edges are padded by replication, so no cell is dropped and no NaN is invented
    at the border.
    """
    padded = np.pad(values, radius, mode="edge")
    height, width = values.shape
    patches = [
        padded[radius + row : radius + row + height, radius + col : radius + col + width]
        for row in range(-radius, radius + 1)
        for col in range(-radius, radius + 1)
    ]
    return np.stack(patches, axis=-1)


def build_features(values: np.ndarray, radius: int) -> np.ndarray:
    """Design matrix of shape ``(height, width, feature_count(radius))``.

    ``[1, centre, neighbourhood - centre]``: an intercept and a centre term carry the
    amplitude, and the residual block carries the sharpening the bilinear baseline
    cannot express.
    """
    patches = _neighbourhood(np.asarray(values, dtype=float), radius)
    centre = patches[..., patches.shape[-1] // 2]
    height, width, _ = patches.shape
    constant = np.ones((height, width, 1))
    return np.concatenate(
        [constant, centre[..., None], patches - centre[..., None]], axis=-1
    )


# ---------------------------------------------------------------------------
# Ridge via accumulated normal equations
# ---------------------------------------------------------------------------


class _NormalEquations:
    """Accumulated ``XᵀX``, ``Xᵀy``, ``yᵀy`` and ``Σy`` over a set of cells.

    Accumulating instead of materialising the design matrix keeps memory flat in
    the number of time steps, and lets the regularisation sweep be scored without
    refitting the features once per candidate.
    """

    __slots__ = ("gram", "rhs", "yty", "sum_y", "count")

    def __init__(self, features: int) -> None:
        self.gram = np.zeros((features, features))
        self.rhs = np.zeros(features)
        self.yty = 0.0
        self.sum_y = 0.0
        self.count = 0

    def add(self, design: np.ndarray, target: np.ndarray) -> None:
        """Fold one block of cells (already flattened) into the accumulation."""
        self.gram += design.T @ design
        self.rhs += design.T @ target
        self.yty += float(target @ target)
        self.sum_y += float(target.sum())
        self.count += int(target.size)

    def residual_sum_of_squares(self, weights: np.ndarray) -> float:
        """``Σ(y - Xw)²`` for weights from *any* fit, evaluated against this block."""
        error = self.yty - 2.0 * float(weights @ self.rhs) + float(
            weights @ self.gram @ weights
        )
        return float(max(error, 0.0))

    def total_sum_of_squares(self) -> float:
        """``Σ(y - ȳ)²``, for an R²-style comparison against predicting the mean."""
        if self.count == 0:
            return 0.0
        mean = self.sum_y / self.count
        return float(max(self.yty - self.count * mean * mean, 0.0))


def solve_ridge(gram: np.ndarray, rhs: np.ndarray, alpha: float) -> np.ndarray:
    """Solve ``(XᵀX + alpha I) w = Xᵀy``, leaving the intercept unpenalised.

    Penalising the intercept would bias the whole field towards zero as ``alpha``
    grows, which is not what a regularisation parameter should buy.
    """
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    penalty = float(alpha) * np.eye(gram.shape[0])
    penalty[0, 0] = 0.0
    return np.linalg.solve(gram + penalty, rhs)


def select_regularisation(
    train: _NormalEquations,
    validation: _NormalEquations,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
) -> tuple[float, dict[str, Any]]:
    """Pick the ridge penalty by validation RMSE on held-out training columns.

    Also scores the *identity* predictor (copying the bilinear input) on the same
    cells, so the output says whether the learned filter added anything over simply
    passing the coarse field through.
    """
    if train.count == 0 or validation.count == 0:
        raise ValueError(
            "cannot select a penalty without both a fit and a validation block "
            f"(fit cells={train.count}, validation cells={validation.count}); "
            "check that the event's reference field has finite cells in both column sets"
        )

    scores: dict[str, float] = {}
    chosen: float | None = None
    chosen_rmse: float | None = None
    for alpha in alphas:
        weights = solve_ridge(train.gram, train.rhs, alpha)
        sse = validation.residual_sum_of_squares(weights)
        rmse = float(np.sqrt(max(sse, 0.0) / validation.count))
        scores[f"{alpha:g}"] = rmse
        if chosen_rmse is None or rmse < chosen_rmse:
            chosen, chosen_rmse = float(alpha), rmse

    # Feature 1 is the centre value, so this is exactly "copy the bilinear input".
    identity = np.zeros(train.gram.shape[0])
    identity[1] = 1.0
    identity_rmse = float(
        np.sqrt(validation.residual_sum_of_squares(identity) / validation.count)
    )

    assert chosen is not None  # alphas is non-empty, so a candidate was scored
    return chosen, {
        "alphas": scores,
        "chosen": chosen,
        "chosen_rmse": chosen_rmse,
        "bilinear_rmse": identity_rmse,
        "improvement_over_bilinear": (
            None if identity_rmse == 0 else float(identity_rmse - (chosen_rmse or 0.0))
        ),
        "validation_cells": validation.count,
        "note": (
            "penalty chosen on held-out training columns only; the evaluation holdout "
            "is never touched during selection"
        ),
    }


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


@dataclass
class LearnedDownscaler:
    """A learned linear filter over a local neighbourhood.

    Fit with :func:`accumulate` + :meth:`fit`; the dataclass itself holds only the
    hyperparameters and the fitted weights, so it serialises into an artefact.
    """

    radius: int = 2
    alpha: float = 1.0
    floor: float | None = None
    weights: np.ndarray | None = None
    rows: int = 0

    @property
    def kernel_size(self) -> int:
        return 2 * self.radius + 1

    def predict(self, values: np.ndarray) -> np.ndarray:
        """Apply the fitted filter to an upsampled coarse field."""
        if self.weights is None:
            raise RuntimeError("LearnedDownscaler.predict called before fit")
        design = build_features(np.asarray(values, dtype=float), self.radius)
        predicted = design @ self.weights
        if self.floor is not None:
            predicted = np.maximum(predicted, self.floor)
        return predicted

    def fit(self, equations: _NormalEquations) -> LearnedDownscaler:
        """Solve for the weights from accumulated normal equations."""
        self.weights = solve_ridge(equations.gram, equations.rhs, self.alpha)
        self.rows = equations.count
        return self

    def kernel(self) -> np.ndarray | None:
        """The learned filter as a ``(kernel_size, kernel_size)`` map, for reading."""
        if self.weights is None:
            return None
        centre, residuals = self.weights[1], self.weights[2:]
        kernel = residuals.reshape(self.kernel_size, self.kernel_size).copy()
        middle = self.kernel_size // 2
        kernel[middle, middle] += centre
        return kernel

    def describe(self) -> dict[str, Any]:
        """Provenance block: what was fitted, on how much data, with which penalty."""
        kernel = self.kernel()
        return {
            "architecture": f"learned_linear_filter_{self.kernel_size}x{self.kernel_size}",
            "architecture_note": (
                "one learned convolution layer (ridge-fitted linear filter over a "
                "neighbourhood); not a deep network, and not described as one"
            ),
            "trained": True,
            "radius": self.radius,
            "kernel_size": self.kernel_size,
            "alpha": self.alpha,
            "floor": self.floor,
            "parameters": None if self.weights is None else int(self.weights.size),
            "training_cells": self.rows,
            "kernel": None if kernel is None else kernel.tolist(),
            "coefficients": None if self.weights is None else self.weights.tolist(),
        }


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_event_sequence(event: str, *, order: int | None = None) -> dict[str, Any]:
    """Load the coarse/fine pair for **every** time step, aligned on the fine grid.

    ``inputs`` are the coarse fields bilinearly upsampled onto the fine grid — the
    exact input the baseline is scored with, so the learned model and the baseline
    see identical data. Both are blanked where the reference is NaN, so a land-only
    ERA5-Land reference defines the region that is compared and trained on.
    """
    from src.data.loader import load_dataset, spatial_resolution_deg

    for path in (COARSE_FILES[event], FINE_FILES[event]):
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{path} missing. Fetch it first: "
                "python -m src.data.cds_fetch --event <event> and "
                "python -m src.data.land_fetch --event <event>"
            )

    variable, units, scale = EVENTS[event]
    coarse_ds = load_dataset(COARSE_FILES[event])
    fine_ds = load_dataset(FINE_FILES[event])
    steps = int(
        min(
            coarse_ds.sizes.get("valid_time", 1),
            fine_ds.sizes.get("valid_time", 1),
        )
    )
    if steps == 0:
        raise ValueError(f"{event}: no time steps in the coarse/fine pair")

    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    valid_times: list[str] = []
    latitude = longitude = None
    upscale_factor = None

    for index in range(steps):
        coarse_time = str(coarse_ds["valid_time"].values[index])[:19]
        fine_time = str(fine_ds["valid_time"].values[index])[:19]
        if coarse_time != fine_time:
            raise ValueError(
                f"{event}: coarse time {coarse_time} does not match fine time "
                f"{fine_time} at step {index}"
            )

        coarse = as_field(coarse_ds, variable, units, scale, index).ascending()
        fine = as_field(fine_ds, variable, units, scale, index).ascending()
        if latitude is None:
            latitude, longitude = fine.latitude, fine.longitude

        upsampled = downscale_to_reference(coarse, fine, order=order)
        if upscale_factor is None:
            upscale_factor = float(upsampled.attrs["upscale_factor"][0])

        inputs.append(np.where(np.isfinite(fine.values), upsampled.values, np.nan))
        targets.append(np.asarray(fine.values, dtype=float))
        valid_times.append(fine_time)

    return {
        "event": event,
        "variable": variable,
        "units": units,
        "latitude": latitude,
        "longitude": longitude,
        "inputs": inputs,
        "targets": targets,
        "valid_times": valid_times,
        "steps": steps,
        "coarse_resolution_deg": spatial_resolution_deg(coarse_ds),
        "fine_resolution_deg": spatial_resolution_deg(fine_ds),
        "upscale_factor": upscale_factor,
    }


def holdout_split(longitude: np.ndarray, test_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Split the domain by longitude into training and evaluation columns.

    A spatial split rather than a random cell split: neighbouring cells of the same
    field are almost identical, so a random split would score the model on cells it
    had effectively already memorised.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    cut = float(np.quantile(longitude, 1.0 - test_fraction))
    train = np.flatnonzero(longitude < cut)
    test = np.flatnonzero(longitude >= cut)
    if train.size == 0 or test.size == 0:
        raise ValueError(
            f"a {test_fraction:.0%} split leaves an empty column set "
            f"(train={train.size}, test={test.size}); the domain is too narrow"
        )
    return train, test


def accumulate(
    sequence: dict[str, Any],
    column_sets: dict[str, np.ndarray],
    radius: int,
) -> dict[str, _NormalEquations]:
    """Build normal equations for several column sets in one pass over the steps."""
    equations = {name: _NormalEquations(feature_count(radius)) for name in column_sets}
    for input_values, target_values in zip(
        sequence["inputs"], sequence["targets"], strict=True
    ):
        design = build_features(input_values, radius)
        for name, columns in column_sets.items():
            block = design[:, columns, :].reshape(-1, design.shape[-1])
            target = target_values[:, columns].reshape(-1)
            keep = np.isfinite(target) & np.isfinite(block).all(axis=1)
            equations[name].add(block[keep], target[keep])
    return equations


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _aggregate(
    tables: list[MetricTable], *, units: str | None, context: dict[str, Any]
) -> MetricTable:
    """Mean of the per-time-step metric tables, ``None`` where it never computed.

    Averaging the per-step *metrics* rather than the fields: a time-mean field
    smooths away exactly the extremes this stage is judged on, so peak preservation
    has to be measured per step and then averaged.
    """
    if not tables:
        raise ValueError("no per-time-step tables to aggregate")

    table = MetricTable(units=units, context=dict(context))
    for name in sorted({name for item in tables for name in item.metrics}):
        values = [item.get(name) for item in tables]
        present = [value for value in values if value is not None]
        table.metrics[name] = float(np.mean(present)) if present else None
        if not present:
            table.notes[name] = (
                f"not computable at any of the {len(tables)} evaluated time steps"
            )

    fractions = [item.valid_fraction for item in tables if item.valid_fraction is not None]
    table.valid_fraction = float(np.mean(fractions)) if fractions else None
    thresholds = [
        item.extreme_threshold for item in tables if item.extreme_threshold is not None
    ]
    table.extreme_threshold = float(np.mean(thresholds)) if thresholds else None
    table.extreme_threshold_source = (
        "mean across time steps of the reference field's p99" if thresholds else None
    )
    table.notes["aggregation"] = (
        f"mean of {len(tables)} per-time-step tables on the spatial holdout; "
        "peak_preservation is the mean of the per-step ratios, not a ratio of "
        "time-averaged fields"
    )
    return table


def train_and_evaluate(
    event: str,
    *,
    test_fraction: float = 0.3,
    radius: int = 2,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    order: int | None = None,
) -> dict[str, Any]:
    """Fit the learned filter, then score it and the baseline on held-out columns.

    Returns the tables, the model provenance, the regularisation sweep, the
    per-metric comparison and the fields for plotting. Raises on any real-data
    problem rather than degrading into a synthetic result.
    """
    sequence = load_event_sequence(event, order=order)
    latitude = sequence["latitude"]
    longitude = sequence["longitude"]
    units = sequence["units"]
    train_columns, test_columns = holdout_split(longitude, test_fraction)

    def _field(values: np.ndarray) -> GriddedField:
        return GriddedField(values, latitude, longitude, units=units)

    # Same two refusals as the baseline experiment, for the same reasons: a pair whose
    # extreme the reference does not cover, or whose holdout is mostly NaN, cannot
    # produce a peak-preservation number -- only a misleading one. Reported, never scored.
    peak_step = int(np.nanargmax([np.nanmax(values) for values in sequence["inputs"]]))
    coarse_peak = _field(sequence["inputs"][peak_step])
    reference_peak = _field(sequence["targets"][peak_step])
    if not peak_covered(coarse_peak, reference_peak):
        return {
            "event": event,
            "variable": sequence["variable"],
            "status": "not_scoreable",
            "reason": (
                "the coarse field peaks where the reference is NaN (ERA5-Land is "
                "land-only), so the extreme to be preserved is outside the reference's "
                "footprint and no peak-preservation number can be computed"
            ),
            "reference_nan_fraction": round(
                float(np.mean([np.mean(~np.isfinite(t)) for t in sequence["targets"]])), 4
            ),
        }

    holdout_coverage = float(
        np.mean([np.mean(np.isfinite(t[:, test_columns])) for t in sequence["targets"]])
    )
    if holdout_coverage < MIN_VALID_FRACTION:
        return {
            "event": event,
            "variable": sequence["variable"],
            "status": "not_scoreable",
            "reason": (
                f"only {holdout_coverage:.1%} of the holdout cells have a reference value "
                f"(below the {MIN_VALID_FRACTION:.0%} floor), so a metric computed there "
                "would not be evidence"
            ),
            "holdout_coverage": round(holdout_coverage, 4),
        }

    # Inner split of the *training* columns, used only to choose the penalty.
    inner_cut = max(1, int(round(0.85 * train_columns.size)))
    inner_fit = train_columns[:inner_cut]
    inner_validation = train_columns[inner_cut:]
    if inner_validation.size == 0:
        inner_fit = inner_validation = train_columns

    equations = accumulate(
        sequence,
        {"fit": inner_fit, "validate": inner_validation, "train": train_columns},
        radius,
    )
    alpha, selection = select_regularisation(
        equations["fit"], equations["validate"], alphas
    )
    model = LearnedDownscaler(
        radius=radius, alpha=alpha, floor=FLOOR_BY_VARIABLE.get(sequence["variable"])
    ).fit(equations["train"])

    learned_tables: list[MetricTable] = []
    baseline_tables: list[MetricTable] = []
    frames: list[dict[str, Any]] = []
    peak_step = 0
    best_peak = -np.inf

    for index, (input_values, target_values) in enumerate(
        zip(sequence["inputs"], sequence["targets"], strict=True)
    ):
        reference = GriddedField(
            values=target_values[:, test_columns],
            latitude=latitude,
            longitude=longitude[test_columns],
            name=f"{sequence['variable']}_reference",
            units=units,
            attrs={"source": "era5-land", "reference": True, "synthetic": False},
        )
        learned = GriddedField(
            values=model.predict(input_values)[:, test_columns],
            latitude=latitude,
            longitude=longitude[test_columns],
            name=f"{sequence['variable']}_learned",
            units=units,
            attrs={
                "source": "learned-downscaler",
                "trained": True,
                "synthetic": False,
                "architecture": model.describe()["architecture"],
            },
        )
        baseline = GriddedField(
            values=input_values[:, test_columns],
            latitude=latitude,
            longitude=longitude[test_columns],
            name=f"{sequence['variable']}_baseline",
            units=units,
            attrs={
                "source": "interpolation-baseline",
                "trained": False,
                "synthetic": False,
            },
        )

        learned_tables.append(evaluate_downscaling(learned, reference))
        baseline_tables.append(evaluate_downscaling(baseline, reference))

        peak = reference.peak
        if peak is not None and peak > best_peak:
            best_peak, peak_step = peak, index
        frames.append(
            {"learned": learned, "baseline": baseline, "reference": reference}
        )

    context = {
        "event": event,
        "variable": sequence["variable"],
        "steps_evaluated": len(learned_tables),
        "holdout": (
            f"east {test_fraction:.0%} of the domain "
            f"({test_columns.size} of {longitude.size} columns), never seen in training"
        ),
        "coarse_resolution_deg": sequence["coarse_resolution_deg"],
        "fine_resolution_deg": sequence["fine_resolution_deg"],
        "upscale_factor": round(float(sequence["upscale_factor"]), 4),
        "coarse_source": "ERA5",
        "reference_source": "ERA5-Land (different run, not truth; land-only)",
        "holdout_reference_coverage": round(holdout_coverage, 4),
    }
    learned_table = _aggregate(learned_tables, units=units, context={**context, "model": "learned"})
    baseline_table = _aggregate(
        baseline_tables, units=units, context={**context, "model": "interpolation"}
    )

    comparison: dict[str, dict[str, float | None]] = {}
    for name in sorted(set(learned_table.metrics) | set(baseline_table.metrics)):
        learned_value = learned_table.get(name)
        baseline_value = baseline_table.get(name)
        comparison[name] = {
            "baseline": baseline_value,
            "learned": learned_value,
            "delta": (
                None
                if learned_value is None or baseline_value is None
                else float(learned_value - baseline_value)
            ),
        }

    gate = get_value(stage_config("validation.yaml"), "gates.pass", default=None)
    verdict = {
        "primary_metric": "peak_preservation",
        "pass_threshold": gate,
        "status": "undecided" if gate is None else "decided",
        "reason": (
            "configs/validation.yaml -> gates.pass is null: no threshold has been "
            "justified on measured distributions yet, so neither model passes or fails "
            "here. The comparison is the result; the verdict waits for the gate."
            if gate is None
            else "compared against the configured pass threshold"
        ),
    }

    peak_frame = frames[peak_step]
    return {
        "event": event,
        "variable": sequence["variable"],
        "units": units,
        "status": "scored",
        "model": model.describe(),
        "selection": selection,
        "holdout": {
            "test_fraction": test_fraction,
            "train_columns": int(train_columns.size),
            "test_columns": int(test_columns.size),
            "split": "spatial (by longitude)",
        },
        "learned": learned_table,
        "baseline": baseline_table,
        "comparison": comparison,
        "verdict": verdict,
        "frame": {
            "index": peak_step,
            "valid_time": sequence["valid_times"][peak_step],
            "reason": "time step with the largest reference peak on the holdout",
            "learned": peak_frame["learned"],
            "baseline": peak_frame["baseline"],
            "reference": peak_frame["reference"],
            "coarse": GriddedField(
                values=sequence["inputs"][peak_step][:, test_columns],
                latitude=latitude,
                longitude=longitude[test_columns],
                name=f"{sequence['variable']}_coarse",
                units=units,
                attrs={"source": "ERA5 upsampled", "trained": False},
            ),
        },
    }


def _print(learned: MetricTable, baseline: MetricTable) -> None:
    names = sorted(set(learned.metrics) | set(baseline.metrics))
    width = max(len(name) for name in names)
    header = f"| {'metric'.ljust(width)} | {'baseline':>12} | {'learned':>12} | {'delta':>12} |"
    print(header)
    print("|" + "-" * (width + 2) + "|" + "-" * 14 + "|" + "-" * 14 + "|" + "-" * 14 + "|")
    for name in names:
        left, right = baseline.get(name), learned.get(name)
        delta = None if left is None or right is None else right - left
        rows = [baseline.render(name, left), learned.render(name, right)]
        delta_text = "—" if delta is None else f"{delta:+.4g}"
        print(
            f"| {name.ljust(width)} | {rows[0]:>12} | {rows[1]:>12} | {delta_text:>12} |"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the learned downscaler on the real ERA5 / ERA5-Land pair and score it "
            "against the interpolation baseline."
        )
    )
    parser.add_argument("--event", choices=list(EVENTS), help="One event.")
    parser.add_argument("--all-events", action="store_true", help="Every event.")
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.3,
        help="Share of the domain (by longitude) held out for scoring.",
    )
    parser.add_argument("--radius", type=int, default=2, help="Neighbourhood radius.")
    parser.add_argument(
        "--output-dir", default="data/processed/validation/downscaling"
    )
    parser.add_argument("--plots-dir", default="data/processed/plots/downscaling")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    events = list(EVENTS) if args.all_events else ([args.event] if args.event else [])
    if not events:
        parser.print_help()
        return 2

    out = Path(args.output_dir)
    for event in events:
        print()
        print("=" * 74)
        print(f"learned downscaling — {event}")
        print("=" * 74)
        try:
            result = train_and_evaluate(
                event, test_fraction=args.test_fraction, radius=args.radius
            )
        except (FileNotFoundError, ValueError) as error:
            print(f"NOT SCORED — {error}")
            continue

        if result["status"] != "scored":
            print(f"NOT SCOREABLE — {result['reason']}")
            artifact = out / f"{event}_learned.json"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(f"[written] {artifact}")
            continue

        model = result["model"]
        selection = result["selection"]
        print(
            f"model      {model['architecture']}  "
            f"({model['parameters']} parameters, {model['training_cells']} training cells)"
        )
        print(
            f"input      ERA5 "
            f"{result['learned'].context['coarse_resolution_deg']:g}° -> "
            f"ERA5-Land {result['learned'].context['fine_resolution_deg']:g}°  "
            f"(factor {result['learned'].context['upscale_factor']:g}, real pair)"
        )
        print(
            f"holdout    {result['holdout']['split']}, "
            f"{result['holdout']['test_columns']} columns held out of "
            f"{result['holdout']['test_columns'] + result['holdout']['train_columns']}"
        )
        print(
            f"alpha      {selection['chosen']:g} "
            f"(validation RMSE {selection['chosen_rmse']:.4g} {result['units']} vs "
            f"bilinear {selection['bilinear_rmse']:.4g} {result['units']})"
        )
        print(f"frame      {result['frame']['valid_time']} ({result['frame']['reason']})")
        print()
        _print(result["learned"], result["baseline"])
        print()
        print(f"gate       {result['verdict']['status']} — {result['verdict']['reason']}")

        artifact = out / f"{event}_{result['variable']}_learned.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(
            json.dumps(
                {
                    "event": event,
                    "variable": result["variable"],
                    "units": result["units"],
                    "status": "scored",
                    "model": model,
                    "selection": selection,
                    "holdout": result["holdout"],
                    "verdict": result["verdict"],
                    "comparison": result["comparison"],
                    "learned": result["learned"].to_dict(),
                    "baseline": result["baseline"].to_dict(),
                    "frame": {
                        "index": result["frame"]["index"],
                        "valid_time": result["frame"]["valid_time"],
                        "reason": result["frame"]["reason"],
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[written] {artifact}")

        if not args.no_plots:
            from src.shared.visualization import plot_downscaling_comparison

            frame = result["frame"]
            plot_downscaling_comparison(
                frame["coarse"],
                frame["baseline"],
                frame["learned"],
                frame["reference"],
                title=(
                    f"{event} — learned filter vs interpolation baseline, "
                    f"holdout {frame['valid_time']}"
                ),
                save_to=Path(args.plots_dir) / f"{event}_learned_vs_baseline.png",
            )
            print(f"[figure]  {Path(args.plots_dir) / f'{event}_learned_vs_baseline.png'}")
    return 0


if __name__ == "__main__":  # pragma: no cover - needs the local raw archives
    raise SystemExit(main())
