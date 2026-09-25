"""The metric-table container, shared by every validated stage.

A validation result has to be an **artefact**, not a claim in prose: the final run
must emit a table to ``configs/validation.yaml`` → ``reporting.output_dir`` and
the README or a slide must reference that file. So the table type lives in
``shared/`` where the downscaling, detection, tracking and transition metric
modules can all produce one.

Two rules are built into the type rather than left to a caller's discipline:

* A value that could not be computed is ``None`` and renders as ``—``. It is never
  ``0.0``, never ``nan``, and never an estimate.
* A metric that requires a threshold takes it as an argument. There is no default,
  because a default would launder a guess into an apparent pass.

Units are attached only to dimensioned metrics, so a dimensionless ratio is never
printed with the field's units next to it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DIMENSIONLESS_METRICS = frozenset(
    {
        "peak_preservation",
        "iou",
        "dice",
        "exceedance_change",
        "valid_fraction",
        "precision",
        "recall",
        "f1",
        "false_alarm_rate",
        "miss_rate",
        "track_continuity",
        "id_consistency",
        "roc_auc",
        "base_rate",
        "brier_score",
        "skill_score",
    }
)
"""Ratios and fractions. Kept explicit so the rendered table never labels a
dimensionless value with the field's units — printing a unit beside a number that
does not have one is how a reader learns to distrust the whole table."""

NOT_COMPUTABLE = "—"
"""How a null metric is rendered. Never rendered as 0."""


@dataclass
class MetricTable:
    """A named set of metric values, with the provenance of how they were obtained.

    ``notes`` records *how* each number was produced, so a table quoted in a slide
    carries its own audit trail.
    """

    metrics: dict[str, float | None] = field(default_factory=dict)
    units: str | None = None
    notes: dict[str, str] = field(default_factory=dict)
    extreme_threshold: float | None = None
    extreme_threshold_source: str | None = None
    valid_fraction: float | None = None
    context: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> float | None:
        return self.metrics.get(key)

    def get(self, key: str, default: float | None = None) -> float | None:
        return self.metrics.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self.metrics

    def render(self, name: str, value: float | None) -> str:
        """Format one value, attaching units only where they apply."""
        if value is None:
            return NOT_COMPUTABLE
        if name in DIMENSIONLESS_METRICS or not self.units:
            return f"{value:.4g}"
        return f"{value:.4g} {self.units}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": dict(self.metrics),
            "units": self.units,
            "extreme_threshold": self.extreme_threshold,
            "extreme_threshold_source": self.extreme_threshold_source,
            "valid_fraction": self.valid_fraction,
            "notes": dict(self.notes),
            "context": dict(self.context),
        }

    def to_markdown(self) -> str:
        """Render as a Markdown table for ``docs/experiments.md`` or a slide."""
        lines = [
            "| Metric | Value | Notes |",
            "|---|---|---|",
        ]
        for name, value in self.metrics.items():
            lines.append(
                f"| `{name}` | {self.render(name, value)} | {self.notes.get(name, '')} |"
            )
        if self.extreme_threshold is not None:
            unit = f" {self.units}" if self.units else ""
            lines.append(
                f"| `extreme_threshold` | {self.extreme_threshold:.4g}{unit} | "
                f"{self.extreme_threshold_source or 'supplied by caller'} |"
            )
        return "\n".join(lines)

    def save_json(self, path: str | Path) -> Path:
        """Write the table as an artefact, creating parent directories."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return target

    def unavailable(self) -> list[str]:
        """Names of metrics that could not be computed.

        A caller should surface these rather than let a table of mostly-``None``
        values read as a clean run.
        """
        return [name for name, value in self.metrics.items() if value is None]

    def flags_extreme_loss(self, min_peak_preservation: float) -> bool:
        """True when the extreme was lost, or when it could not be measured.

        ``min_peak_preservation`` is required: there is no defensible default, and
        one here would launder a guess into an apparent pass. An unmeasurable peak
        counts as loss — the failure mode being guarded against is a metric that
        passes because it never ran.
        """
        value = self.metrics.get("peak_preservation")
        return value is None or value < min_peak_preservation
