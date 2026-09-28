"""Tests for the real-data downscaling helpers.

Run:  pytest tests/test_real_pair.py -v

These cover the two decisions that make the real-data number trustworthy, and both
run on synthetic fields so they need no downloaded archives:

* a pair whose coarse peak sits where the reference is NaN must be refused;
* both fields must be compared over the same footprint.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.downscaling.real_pair import _mask_to_reference, peak_covered
from src.shared.fields import GriddedField


def _field(values: np.ndarray, name: str = "f") -> GriddedField:
    axis = np.arange(values.shape[0], dtype=float)
    return GriddedField(values=values, latitude=axis, longitude=axis, name=name, units="K")


def _coarse_peaking_at(row: int, column: int, size: int = 4) -> GriddedField:
    """A field whose maximum sits at ``(row, column)`` on a 0..size-1 axis."""
    values = np.zeros((size, size))
    values[row, column] = 10.0
    return _field(values)


class TestPeakCovered:
    def test_fully_finite_reference_covers_the_peak(self):
        reference = _field(np.full((4, 4), 1.0))
        assert peak_covered(_coarse_peaking_at(3, 3), reference) is True

    def test_a_hole_at_the_peak_is_refused(self):
        """Coarse peaks bottom-right; reference is NaN exactly there."""
        values = np.full((4, 4), 1.0)
        values[3, 3] = np.nan
        assert peak_covered(_coarse_peaking_at(3, 3), _field(values)) is False

    def test_a_hole_away_from_the_peak_is_fine(self):
        values = np.full((4, 4), 1.0)
        values[3, 3] = np.nan
        assert peak_covered(_coarse_peaking_at(0, 3), _field(values)) is True


class TestMaskToReference:
    def test_cells_the_reference_does_not_cover_become_nan(self):
        predicted = _field(np.full((3, 3), 2.0), name="predicted")
        values = np.ones((3, 3))
        values[0, :] = np.nan
        masked = _mask_to_reference(predicted, _field(values))
        assert masked.shape == (3, 3)
        assert np.isnan(masked.values[0, :]).all()
        assert (masked.values[1:, :] == 2.0).all()

    def test_masking_preserves_the_units_and_name(self):
        predicted = _field(np.full((3, 3), 2.0), name="predicted")
        masked = _mask_to_reference(predicted, _field(np.ones((3, 3))))
        assert masked.name == "predicted"
        assert masked.units == "K"

    def test_descending_and_ascending_orientations_agree(self):
        predicted = _field(np.full((3, 3), 2.0))
        reference = _field(np.ones((3, 3)))
        flipped = GriddedField(
            values=reference.values[::-1].copy(),
            latitude=reference.latitude[::-1].copy(),
            longitude=reference.longitude.copy(),
        )
        assert _mask_to_reference(predicted, flipped).equals(
            _mask_to_reference(predicted, reference)
        )


class TestEventConfig:
    def test_every_event_has_a_coarse_and_fine_file(self):
        from src.downscaling.real_pair import COARSE_FILES, EVENTS, FINE_FILES

        assert set(EVENTS) == set(COARSE_FILES) == set(FINE_FILES)

    def test_missing_archive_names_the_command_to_run(self, monkeypatch):
        from src.downscaling import real_pair

        monkeypatch.setitem(real_pair.COARSE_FILES, "heatwave", "data/raw/nope_a.nc")
        monkeypatch.setitem(real_pair.FINE_FILES, "heatwave", "data/raw/nope_b.nc")
        with pytest.raises(FileNotFoundError, match="land_fetch"):
            real_pair.build_pair("heatwave")
