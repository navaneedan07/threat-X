"""Evidence plots shared across stages.

Two figures the work cards require and the repo did not have:

* **coarse vs refined** (``plot_coarse_vs_refined``) — the side-by-side that makes a
  downscaling claim checkable rather than asserted. All panels share one colour
  scale, because per-panel autoscaling is exactly how a smoothed field can be made
  to look as sharp as its reference.
* **threat lifecycle** (``plot_lifecycle``) — the deterministic state machine's
  output as a timeline, with the intensity series behind it.

Both run headless (``Agg``) and save nothing unless asked, so they are safe in CI.
Everything plotted comes from the fields and the lifecycle sequence handed in —
nothing is fabricated for a nicer figure, and a synthetic input carries its
``synthetic`` attribute into the title.

Usage::

    from src.shared.visualization import plot_coarse_vs_refined
    fig = plot_coarse_vs_refined(coarse, refined, reference, save_to="out.png")
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: no display required
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from src.shared.fields import GriddedField  # noqa: E402
from src.transition.lifecycle import LifecyclePhase, LifecycleSequence  # noqa: E402

logger = logging.getLogger(__name__)

_PHASE_ORDER = [
    LifecyclePhase.FORMATION,
    LifecyclePhase.INTENSIFICATION,
    LifecyclePhase.EXPANSION,
    LifecyclePhase.PEAK,
    LifecyclePhase.DECAY,
]
_PHASE_INDEX = {phase: index for index, phase in enumerate(_PHASE_ORDER)}


def _save(fig: Figure, save_to: str | Path | None) -> None:
    if not save_to:
        return
    path = Path(save_to)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info("figure saved: %s", path)


def plot_coarse_vs_refined(
    coarse: GriddedField,
    refined: GriddedField,
    reference: GriddedField | None = None,
    *,
    title: str | None = None,
    save_to: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Side-by-side of the coarse input, the refined field and the reference.

    ``reference`` is optional: the synthetic pair has one, but a live run may not.
    All panels share the colour bounds derived from the **largest peak present**, so
    a panel with a weakened extreme visibly looks weaker.
    """
    panels: list[tuple[str, GriddedField]] = [("Coarse input", coarse), ("Refined", refined)]
    if reference is not None:
        panels.append(("Reference", reference))

    peaks = [field.peak for _, field in panels if field.peak is not None]
    vmax = max(peaks) if peaks else None
    synthetic = any(field.attrs.get("synthetic") for _, field in panels)

    fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 4.6), squeeze=False)
    mesh = None
    for axis, (label, field) in zip(axes[0], panels, strict=True):
        mesh = axis.pcolormesh(
            field.longitude,
            field.latitude,
            field.values,
            shading="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
        )
        dlat, dlon = field.resolution_deg()
        peak = field.peak
        axis.set_title(
            f"{label}\n{field.shape[0]}×{field.shape[1]} @ {dlat:g}°"
            f"  peak {peak:.1f}" if peak is not None else f"{label}\n{field.shape}",
            fontsize=10,
        )
        axis.set_xlabel("longitude")
        axis.set_ylabel("latitude")

    units = refined.units or coarse.units or ""
    if mesh is not None:
        bar = fig.colorbar(mesh, ax=list(axes[0]), fraction=0.046, pad=0.02)
        bar.set_label(units or "value")

    header = title or "Coarse vs refined"
    if synthetic:
        header += "  [SYNTHETIC — interface check, not a result]"
    fig.suptitle(header, fontsize=12, fontweight="bold")

    if show:  # pragma: no cover - interactive only
        plt.show()
    _save(fig, save_to)
    return fig


def plot_lifecycle(
    sequence: LifecycleSequence,
    *,
    title: str | None = None,
    save_to: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Timeline of lifecycle phases, with the intensity series behind it.

    The phase track shows the discrete state per step; the intensity panel shows the
    value the state machine read, so a phase that looks wrong can be traced to the
    numbers rather than argued about.
    """
    phases = sequence.phases
    fig, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(11, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    timestamps = [assignment.timestamp for assignment in phases]
    if timestamps:
        indices = [
            _PHASE_INDEX[assignment.phase] if assignment.phase in _PHASE_INDEX else -1
            for assignment in phases
        ]
        top.step(timestamps, indices, where="post", color="black", linewidth=1.4)
        top.scatter(timestamps, indices, c=indices, cmap="plasma", vmin=0, vmax=4, zorder=3)
        top.set_yticks(range(len(_PHASE_ORDER)))
        top.set_yticklabels([phase.value for phase in _PHASE_ORDER], fontsize=9)
        top.set_ylim(-1, len(_PHASE_ORDER) - 1 + 0.5)

        intensities = [assignment.intensity for assignment in phases]
        finite = [
            (t, v) for t, v in zip(timestamps, intensities, strict=True) if v is not None
        ]
        if finite:
            bottom.plot(
                [item[0] for item in finite],
                [item[1] for item in finite],
                color="crimson",
                marker="o",
                markersize=3,
            )
            bottom.set_ylabel("intensity", fontsize=9)
        else:
            bottom.text(0.5, 0.5, "no intensity recorded", ha="center", va="center")
        bottom.grid(True, alpha=0.3)
        bottom.set_xlabel("time (UTC)")
    else:
        top.text(0.5, 0.5, "empty history", ha="center", va="center")

    top.set_title(
        (title or f"Threat lifecycle — {sequence.threat_id}")
        + f"\ncurrent phase: {sequence.current_phase.value if sequence.current_phase else '—'}",
        fontsize=11,
        fontweight="bold",
    )
    top.grid(True, axis="x", alpha=0.3)

    unevaluated = sequence.unevaluated_events()
    if unevaluated:
        note = "Not evaluated (needs a threshold): " + ", ".join(
            outcome.event.value for outcome in unevaluated
        )
        fig.text(0.5, 0.005, note, ha="center", fontsize=7, color="grey", style="italic")

    fig.autofmt_xdate(rotation=30)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    if show:  # pragma: no cover - interactive only
        plt.show()
    _save(fig, save_to)
    return fig


def _demo(argv: list[str] | None = None) -> int:  # pragma: no cover - manual run
    """Generate both figures from synthetic inputs and save them under data/processed."""
    import argparse
    from datetime import datetime, timedelta, timezone

    from src.downscaling.baseline import downscale_to_reference, resample
    from src.shared.contracts import (
        EventType,
        Evolution,
        Location,
        Severity,
        ThreatHistory,
        ThreatObject,
    )
    from src.shared.synthetic import sharp_peak_field, translating_event
    from src.transition.lifecycle import assign_lifecycle

    parser = argparse.ArgumentParser(description="Generate evidence plots (synthetic inputs)")
    parser.add_argument("--output-dir", default="data/processed/plots")
    args = parser.parse_args(argv)
    out = Path(args.output_dir)

    reference = sharp_peak_field()
    coarse = resample(reference, reference.latitude[::10], reference.longitude[::10], order=1)
    refined = downscale_to_reference(coarse, reference)
    plot_coarse_vs_refined(
        coarse,
        refined,
        reference,
        title="Interpolation baseline — 10× coarsening",
        save_to=out / "downscaling" / "coarse_vs_refined.png",
    )

    start = datetime(2026, 9, 23, tzinfo=timezone.utc)
    snapshots = [
        ThreatObject(
            threat_id="THR-2026-0001",
            event_type=EventType.EXTREME_RAINFALL,
            timestamp=start + timedelta(hours=event.hours_after_start),
            severity=Severity.MODERATE,
            location=Location(
                centroid_lat=event.centre[0],
                centroid_lon=event.centre[1],
                footprint_radius_km=40.0 + 6.0 * event.step,
            ),
            evolution=Evolution(intensity_anomaly_sigma=event.amplitude),
        )
        for event in translating_event()
    ]
    sequence = assign_lifecycle(ThreatHistory(threat_id="THR-2026-0001", snapshots=snapshots))
    plot_lifecycle(
        sequence,
        title="Threat lifecycle — synthetic translating event",
        save_to=out / "lifecycle" / "threat_lifecycle.png",
    )
    print(f"[done] wrote figures under {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual run
    raise SystemExit(_demo())
