"""The validation gate and the final metric table.

Turns measured metrics into a verdict a demo can show:

    PASS      the stage met its primary metric threshold
    DEGRADE   it cleared the minimum but not the target — serve it, flagged
    SUPPRESS  it did not clear the minimum, or the stage produced nothing to judge

and writes the whole thing to ``configs/validation.yaml`` →
``reporting.output_dir`` as an artefact, because a validation result that exists
only in prose cannot be checked.

The one thing this module refuses to do
---------------------------------------
**It will not invent a threshold.** ``configs/validation.yaml`` ships
``gates.pass``, ``gates.degrade`` and ``gates.suppress`` as ``null`` on purpose:
they must be derived from measured metric distributions and justified in
``docs/experiments.md``. Until then every verdict comes back as ``None`` with the
reason naming the keys to fill in. A gate with a guessed threshold is worse than
no gate, because it launders a guess into an apparent pass.

Direction of each metric is fixed here (higher-is-better vs lower-is-better)
because that is metric semantics, not a tunable parameter. Thresholds differ per
stage and, for ``brier_score``, run the opposite way to ``f1`` and
``peak_preservation`` — so a single shared triple is only usable once the
per-stage overrides below are filled in. That limitation is reported, not hidden.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.shared.config import CONFIGS_DIR, get_value, stage_config
from src.shared.contracts import GateVerdict, format_timestamp
from src.shared.metrics import MetricTable

GATE_CONFIG_FILE = "validation.yaml"

HIGHER_IS_BETTER = frozenset(
    {
        "f1",
        "precision",
        "recall",
        "peak_preservation",
        "iou",
        "dice",
        "track_continuity",
        "id_consistency",
        "roc_auc",
        "skill_score",
        "confidence",
    }
)

LOWER_IS_BETTER = frozenset(
    {
        "brier_score",
        "false_alarm_rate",
        "miss_rate",
        "rmse",
        "mae",
        "centroid_error_km",
        "trajectory_error_km",
        "duration_error_hours",
        "lead_time_error_hours",
        "calibration",
        "percentile_error",
    }
)

_METRIC_DIRECTION: dict[str, bool] = {
    **{name: True for name in HIGHER_IS_BETTER},
    **{name: False for name in LOWER_IS_BETTER},
}

_RANK: dict[GateVerdict, int] = {
    GateVerdict.PASS: 0,
    GateVerdict.DEGRADE: 1,
    GateVerdict.SUPPRESS: 2,
}


def metric_is_higher_better(metric: str) -> bool | None:
    """Direction of a metric, or ``None`` when the metric is unknown.

    Unknown direction means the gate cannot compare it, which is reported rather
    than assumed — assuming "higher is better" for an error metric would invert
    the verdict.
    """
    return _METRIC_DIRECTION.get(metric)


@dataclass(frozen=True)
class StageGate:
    """Thresholds and direction for one pipeline stage's primary metric."""

    stage: str
    primary_metric: str
    pass_at: float | None = None
    degrade_at: float | None = None
    suppress_at: float | None = None
    justification: str | None = None

    @property
    def higher_is_better(self) -> bool | None:
        return metric_is_higher_better(self.primary_metric)

    @property
    def configured(self) -> bool:
        """True only when both decision boundaries have real values."""
        return self.pass_at is not None and self.degrade_at is not None

    def unresolved_keys(self) -> list[str]:
        keys: list[str] = []
        if self.pass_at is None:
            keys.append(f"gates.per_stage.{self.stage}.pass")
        if self.degrade_at is None:
            keys.append(f"gates.per_stage.{self.stage}.degrade")
        return keys

    def ordering_problem(self) -> str | None:
        """Detect a threshold set that cannot produce the intended ordering."""
        if not self.configured:
            return None
        higher = self.higher_is_better
        if higher is None:
            return None
        if higher and self.pass_at < self.degrade_at:
            return (
                f"{self.primary_metric} is higher-is-better but pass ({self.pass_at}) "
                f"is below degrade ({self.degrade_at})"
            )
        if not higher and self.pass_at > self.degrade_at:
            return (
                f"{self.primary_metric} is lower-is-better but pass ({self.pass_at}) "
                f"is above degrade ({self.degrade_at})"
            )
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "primary_metric": self.primary_metric,
            "pass_at": self.pass_at,
            "degrade_at": self.degrade_at,
            "suppress_at": self.suppress_at,
            "higher_is_better": self.higher_is_better,
            "configured": self.configured,
            "justification": self.justification,
        }


@dataclass(frozen=True)
class GateConfig:
    """Per-stage gates plus where the report is written."""

    stages: dict[str, StageGate] = field(default_factory=dict)
    output_dir: str | None = None
    notes: str | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> GateConfig:
        """Read ``configs/validation.yaml``.

        Supported shapes::

            gates:
              primary_metric:            # required: which metric decides each stage
                detection: f1
              pass: null                 # shared fallback, still null today
              degrade: null
              per_stage:                 # optional per-stage override
                downscaling:
                  pass: 0.90
                  degrade: 0.70
                  justification: "baseline interpolation reaches 0.91"

        The per-stage block exists because the shared triple cannot serve stages
        with opposite metric directions. When only the shared values are set, they
        are applied to every stage and the report says so.
        """
        loaded = config if config is not None else stage_config(GATE_CONFIG_FILE)
        primary = get_value(loaded, "gates.primary_metric", default={}) or {}
        shared_pass = _number(get_value(loaded, "gates.pass"))
        shared_degrade = _number(get_value(loaded, "gates.degrade"))
        shared_suppress = _number(get_value(loaded, "gates.suppress"))

        stages: dict[str, StageGate] = {}
        for stage, metric in primary.items():
            stage_pass = _number(get_value(loaded, f"gates.per_stage.{stage}.pass"))
            stage_degrade = _number(get_value(loaded, f"gates.per_stage.{stage}.degrade"))
            stage_suppress = _number(get_value(loaded, f"gates.per_stage.{stage}.suppress"))
            stages[stage] = StageGate(
                stage=stage,
                primary_metric=str(metric),
                pass_at=stage_pass if stage_pass is not None else shared_pass,
                degrade_at=(
                    stage_degrade if stage_degrade is not None else shared_degrade
                ),
                # Explicit None checks, not `or`: a threshold of 0.0 is a real
                # value and would otherwise be discarded as falsy.
                suppress_at=(
                    stage_suppress if stage_suppress is not None else shared_suppress
                ),
                justification=(
                    get_value(loaded, f"gates.per_stage.{stage}.justification")
                    or get_value(loaded, "gates.notes")
                ),
            )

        return cls(
            stages=stages,
            output_dir=get_value(loaded, "reporting.output_dir"),
            notes=get_value(loaded, "gates.notes"),
        )

    def stage(self, name: str) -> StageGate | None:
        return self.stages.get(name)

    def unresolved(self) -> list[str]:
        keys: list[str] = []
        for gate in self.stages.values():
            keys.extend(gate.unresolved_keys())
        return keys


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class StageVerdict:
    """The gate's decision for one stage, with the reasoning attached."""

    stage: str
    verdict: GateVerdict | None
    primary_metric: str
    value: float | None
    reason: str
    gate: StageGate

    @property
    def decided(self) -> bool:
        return self.verdict is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "verdict": self.verdict.value if self.verdict else None,
            "primary_metric": self.primary_metric,
            "value": self.value,
            "reason": self.reason,
            "gate": self.gate.to_dict(),
        }


def decide_stage(
    gate: StageGate,
    value: float | None,
    *,
    produced_output: bool = True,
) -> StageVerdict:
    """Apply one gate.

    ``produced_output=False`` means the stage did not run at all, which is a
    SUPPRESS regardless of thresholds — an absent stage is not a passing one.
    """
    if not produced_output:
        return StageVerdict(
            stage=gate.stage,
            verdict=GateVerdict.SUPPRESS,
            primary_metric=gate.primary_metric,
            value=None,
            reason=f"{gate.stage} produced no output",
            gate=gate,
        )

    if gate.higher_is_better is None:
        return StageVerdict(
            stage=gate.stage,
            verdict=None,
            primary_metric=gate.primary_metric,
            value=value,
            reason=(
                f"direction of metric '{gate.primary_metric}' is unknown; add it to "
                "HIGHER_IS_BETTER or LOWER_IS_BETTER in src/validation/evaluation.py"
            ),
            gate=gate,
        )

    problem = gate.ordering_problem()
    if problem:
        return StageVerdict(
            stage=gate.stage,
            verdict=None,
            primary_metric=gate.primary_metric,
            value=value,
            reason=f"thresholds are inconsistent: {problem}",
            gate=gate,
        )

    if not gate.configured:
        return StageVerdict(
            stage=gate.stage,
            verdict=None,
            primary_metric=gate.primary_metric,
            value=value,
            reason=(
                "thresholds unresolved: "
                + ", ".join(gate.unresolved_keys())
                + " are null in configs/validation.yaml. Measure the metric "
                "distribution first and justify the value in docs/experiments.md "
                "— the gate will not guess."
            ),
            gate=gate,
        )

    if value is None:
        return StageVerdict(
            stage=gate.stage,
            verdict=None,
            primary_metric=gate.primary_metric,
            value=None,
            reason=(
                f"primary metric '{gate.primary_metric}' could not be computed, so the "
                "gate cannot be decided (this is not a pass)"
            ),
            gate=gate,
        )

    if gate.higher_is_better:
        if value >= gate.pass_at:
            verdict, reason = GateVerdict.PASS, f"{value:.4g} >= pass {gate.pass_at:.4g}"
        elif value >= gate.degrade_at:
            verdict, reason = (
                GateVerdict.DEGRADE,
                f"{value:.4g} below pass {gate.pass_at:.4g} but >= degrade {gate.degrade_at:.4g}",
            )
        else:
            verdict, reason = (
                GateVerdict.SUPPRESS,
                f"{value:.4g} below degrade {gate.degrade_at:.4g}",
            )
    else:
        if value <= gate.pass_at:
            verdict, reason = GateVerdict.PASS, f"{value:.4g} <= pass {gate.pass_at:.4g}"
        elif value <= gate.degrade_at:
            verdict, reason = (
                GateVerdict.DEGRADE,
                f"{value:.4g} above pass {gate.pass_at:.4g} but <= degrade {gate.degrade_at:.4g}",
            )
        else:
            verdict, reason = (
                GateVerdict.SUPPRESS,
                f"{value:.4g} above degrade {gate.degrade_at:.4g}",
            )

    return StageVerdict(
        stage=gate.stage,
        verdict=verdict,
        primary_metric=gate.primary_metric,
        value=value,
        reason=reason,
        gate=gate,
    )


@dataclass
class ValidationReport:
    """Per-stage verdicts, the tables behind them, and an overall decision."""

    stages: list[StageVerdict] = field(default_factory=list)
    tables: dict[str, MetricTable] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    subject: str | None = None

    @property
    def overall_verdict(self) -> GateVerdict | None:
        """Worst stage verdict, or ``None`` while any stage is undecided.

        ``None`` and not PASS: a pipeline with an undecidable stage has not been
        validated, and reporting the best decided stage as the overall answer
        would hide exactly the gaps the gate exists to expose.
        """
        decided = [stage.verdict for stage in self.stages if stage.decided]
        if not decided or len(decided) != len(self.stages):
            return None
        return max(decided, key=lambda verdict: _RANK[verdict])

    def stage(self, name: str) -> StageVerdict | None:
        for stage in self.stages:
            if stage.stage == name:
                return stage
        return None

    def undecided(self) -> list[str]:
        return [stage.stage for stage in self.stages if not stage.decided]

    def to_dict(self) -> dict[str, Any]:
        overall = self.overall_verdict
        return {
            "_meta": {
                "generated_at": format_timestamp(self.generated_at),
                "source": "src.validation.evaluation.evaluate_validation",
                "note": (
                    "Null means not computable or not decided. A null verdict is not "
                    "a pass."
                ),
            },
            "subject": self.subject,
            "overall_verdict": overall.value if overall else None,
            "undecided_stages": self.undecided(),
            "stages": [stage.to_dict() for stage in self.stages],
            "tables": {name: table.to_dict() for name, table in self.tables.items()},
            "notes": list(self.notes),
        }

    def to_markdown(self) -> str:
        overall = self.overall_verdict
        lines = [
            "## Validation report",
            "",
            f"- Generated: {format_timestamp(self.generated_at)}",
            f"- Subject: {self.subject or '—'}",
            f"- **Overall verdict: {overall.value if overall else 'NOT DECIDED'}**",
            "",
            "| Stage | Metric | Value | Verdict | Reason |",
            "|---|---|---|---|---|",
        ]
        for stage in self.stages:
            value = "—" if stage.value is None else f"{stage.value:.4g}"
            verdict = stage.verdict.value if stage.verdict else "—"
            lines.append(
                f"| {stage.stage} | `{stage.primary_metric}` | {value} | {verdict} | "
                f"{stage.reason} |"
            )
        for name, table in self.tables.items():
            lines.extend(["", f"### {name}", "", table.to_markdown()])
        if self.notes:
            lines.extend(["", "### Notes", ""])
            lines.extend(f"- {note}" for note in self.notes)
        return "\n".join(lines)

    def save(self, directory: str | Path) -> dict[str, Path]:
        """Write ``validation_report.json`` and ``.md`` to ``directory``."""
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        json_path = target / "validation_report.json"
        markdown_path = target / "validation_report.md"
        json_path.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        markdown_path.write_text(self.to_markdown() + "\n", encoding="utf-8")
        return {"json": json_path, "markdown": markdown_path}


def evaluate_validation(
    tables: dict[str, MetricTable],
    *,
    config: dict[str, Any] | None = None,
    subject: str | None = None,
    base_rate: float | None = None,
) -> ValidationReport:
    """Gate each supplied metric table and assemble the report.

    Only stages present in ``tables`` and in ``gates.primary_metric`` are gated. A
    stage with a table but no configured primary metric is reported as undecided
    with that reason, rather than being skipped silently.
    """
    gate_config = GateConfig.from_config(config)
    report = ValidationReport(subject=subject)

    for stage, gate in gate_config.stages.items():
        table = tables.get(stage)
        if table is None:
            report.stages.append(decide_stage(gate, None, produced_output=False))
            report.notes.append(
                f"{stage}: no metric table supplied, so it was not validated"
            )
            continue
        report.tables[stage] = table
        report.stages.append(decide_stage(gate, table.get(gate.primary_metric)))

    for stage in tables:
        if stage not in gate_config.stages:
            report.notes.append(
                f"{stage}: a metric table was supplied but configs/validation.yaml has "
                "no gates.primary_metric entry for it, so no verdict was produced"
            )

    # The rare-event rule: a transition score without its base rate is not
    # interpretable, so it is called out here rather than left to the reader.
    transition = tables.get("transition")
    if transition is not None and (
        transition.get("brier_score") is not None or transition.get("roc_auc") is not None
    ):
        rate = base_rate if base_rate is not None else transition.get("base_rate")
        if rate is None:
            report.notes.append(
                "transition: a Brier/AUC score is present but the base rate is "
                "unknown, so the score cannot be interpreted — report the base rate "
                "alongside it"
            )
        else:
            report.notes.append(
                f"transition: base rate {rate:.1%} — compare every transition score "
                "against a model that always predicts this rate"
            )

    unresolved = gate_config.unresolved()
    if unresolved:
        report.notes.append(
            "gate thresholds still null in configs/validation.yaml: "
            + ", ".join(unresolved)
            + ". Verdicts stay undecided until these are measured and justified in "
            "docs/experiments.md."
        )
    for gate in gate_config.stages.values():
        problem = gate.ordering_problem()
        if problem:
            report.notes.append(f"{gate.stage}: {problem}")

    return report


def default_output_dir(config: dict[str, Any] | None = None) -> Path:
    """Where the report is written, from ``reporting.output_dir``.

    Falls back to ``data/processed/validation`` (the value shipped in the config)
    only when the key is absent, so the artefact always lands somewhere findable.
    """
    loaded = config if config is not None else stage_config(GATE_CONFIG_FILE)
    configured = get_value(loaded, "reporting.output_dir", default="./data/processed/validation")
    path = Path(configured)
    if not path.is_absolute():
        path = CONFIGS_DIR.parent / path
    return path
