"""Tests for the shared evidence plots.

Run:  pytest tests/test_visualization.py -v

These plot functions must run headless and produce a saved artefact — the work
cards require a coarse-vs-refined comparison and a lifecycle visual, and a figure
that only exists in a notebook is not an artefact.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

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
from src.shared.visualization import (
    _shared_bounds,
    plot_coarse_vs_refined,
    plot_downscaling_comparison,
    plot_field_panels,
    plot_lifecycle,
)
from src.transition.lifecycle import assign_lifecycle


@pytest.fixture(scope="module")
def fields():
    reference = sharp_peak_field(lat_range=(8.0, 18.0), lon_range=(75.0, 85.0), resolution_deg=0.1)
    coarse = resample(reference, reference.latitude[::5], reference.longitude[::5], order=1)
    refined = downscale_to_reference(coarse, reference)
    return coarse, refined, reference


@pytest.fixture(scope="module")
def sequence():
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
    return assign_lifecycle(ThreatHistory(threat_id="THR-2026-0001", snapshots=snapshots))


class TestCoarseVsRefined:
    def test_saves_a_figure(self, fields, tmp_path):
        coarse, refined, reference = fields
        target = tmp_path / "nested" / "coarse_vs_refined.png"
        fig = plot_coarse_vs_refined(coarse, refined, reference, save_to=target)
        assert target.exists()
        assert fig is not None

    def test_reference_is_optional(self, fields, tmp_path):
        coarse, refined, _ = fields
        target = tmp_path / "pair.png"
        plot_coarse_vs_refined(coarse, refined, save_to=target)
        assert target.exists()

    def test_synthetic_input_is_labelled(self, fields):
        coarse, refined, reference = fields
        fig = plot_coarse_vs_refined(coarse, refined, reference)
        assert "SYNTHETIC" in fig._suptitle.get_text()


class TestSharedColourScale:
    """The pooled scale is a correctness property: it is what stops a smoothed field
    from being autoscaled into looking as sharp as the reference it flattened."""

    def test_bounds_are_pooled_across_every_panel(self, fields):
        coarse, _, reference = fields
        vmin, vmax = _shared_bounds([("coarse", coarse), ("reference", reference)])
        assert vmin == pytest.approx(min(np.nanmin(coarse.values), np.nanmin(reference.values)))
        assert vmax == pytest.approx(max(coarse.peak, reference.peak))

    def test_a_weakened_panel_does_not_get_its_own_scale(self, fields):
        _, refined, reference = fields
        _, vmax = _shared_bounds([("refined", refined), ("reference", reference)])
        assert vmax == pytest.approx(reference.peak)
        assert vmax > (refined.peak or 0.0)

    def test_an_empty_panel_list_is_an_error(self):
        with pytest.raises(ValueError):
            plot_field_panels([])


class TestDownscalingComparison:
    def test_saves_a_figure(self, fields, tmp_path):
        coarse, refined, reference = fields
        target = tmp_path / "nested" / "learned_vs_baseline.png"
        fig = plot_downscaling_comparison(
            coarse, refined, refined, reference, save_to=target
        )
        assert target.exists()
        assert fig is not None


class TestLifecycle:
    def test_saves_a_figure(self, sequence, tmp_path):
        target = tmp_path / "lifecycle.png"
        fig = plot_lifecycle(sequence, save_to=target)
        assert target.exists()
        assert fig is not None

    def test_empty_history_does_not_crash(self, tmp_path):
        empty = assign_lifecycle(ThreatHistory(threat_id="THR-2026-0001"))
        target = tmp_path / "empty.png"
        plot_lifecycle(empty, save_to=target)
        assert target.exists()

    def test_current_phase_appears_in_the_title(self, sequence):
        fig = plot_lifecycle(sequence)
        assert sequence.current_phase.value in fig.axes[0].get_title()
