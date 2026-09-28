"""Tests for historical ERA5 acquisition planning, splitting, and normalization."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.cds_fetch import GNN_AREA, GNN_TIMES, GNN_VARIABLES, build_gnn_request

HISTORICAL_PATH = (
    Path(__file__).resolve().parents[1]
    / "weights"
    / "gnn"
    / "03_prepare_historical_era5.py"
)
HISTORICAL_SPEC = importlib.util.spec_from_file_location(
    "historical_era5_preparation", HISTORICAL_PATH
)
assert HISTORICAL_SPEC is not None and HISTORICAL_SPEC.loader is not None
historical = importlib.util.module_from_spec(HISTORICAL_SPEC)
HISTORICAL_SPEC.loader.exec_module(historical)


def test_annual_and_daily_cds_requests_have_expected_scope():
    monthly = build_gnn_request("2010-01-01", "2010-01-31")
    probe = build_gnn_request("2022-01-01", "2022-01-01")

    assert monthly["date"] == "2010-01-01/2010-01-31"
    assert probe["date"] == "2022-01-01"
    assert monthly["variable"] == GNN_VARIABLES
    assert monthly["area"] == GNN_AREA
    assert monthly["time"] == GNN_TIMES
    assert len(monthly["time"]) == 24
    assert monthly["data_format"] == "netcdf"
    assert monthly["download_format"] == "zip"


def test_invalid_cds_date_range_is_rejected():
    with pytest.raises(ValueError, match="on or before"):
        build_gnn_request("2020-01-02", "2020-01-01")


def test_split_boundaries_are_chronological_and_non_overlapping():
    timestamps = pd.DatetimeIndex(
        [
            "2019-12-31 23:00",
            "2020-01-01 00:00",
            "2021-12-31 23:00",
            "2022-01-01 00:00",
        ]
    )
    splits = historical.split_timestamps(timestamps)

    assert list(splits) == ["train", "validation", "test"]
    assert splits["train"].tolist() == [timestamps[0]]
    assert splits["validation"].tolist() == [timestamps[1], timestamps[2]]
    assert splits["test"].tolist() == [timestamps[3]]
    assert sum(map(len, splits.values())) == len(timestamps)
    assert not (set(splits["train"]) & set(splits["validation"]))
    assert not (set(splits["validation"]) & set(splits["test"]))


def test_expected_hourly_counts_include_leap_years():
    assert historical.EXPECTED_SPLIT_COUNTS == {
        "train": 87_648,
        "validation": 17_544,
        "test": 26_304,
    }
    assert sum(historical.EXPECTED_SPLIT_COUNTS.values()) == 131_496
    assert len(historical.expected_month_timestamps(2020, 2)) == 696
    assert len(historical.expected_month_timestamps(2019, 2)) == 672


def test_missing_duplicate_and_out_of_order_timestamps_are_rejected():
    expected = pd.date_range("2020-01-01 00:00", periods=4, freq="h")

    with pytest.raises(ValueError, match="missing"):
        historical.validate_timestamp_axis(expected.delete(2), expected, "fixture")
    with pytest.raises(ValueError, match="duplicate"):
        historical.validate_timestamp_axis(
            expected.insert(2, expected[1]), expected, "fixture"
        )
    with pytest.raises(ValueError, match="chronologically"):
        historical.validate_timestamp_axis(expected[[0, 2, 1, 3]], expected, "fixture")


def test_training_statistics_ignore_validation_and_test_values():
    train = np.arange(4 * 66 * 5, dtype=np.float32).reshape(4, 66, 5)
    validation = np.full((2, 66, 5), 1e6, dtype=np.float32)
    test = np.full((2, 66, 5), -1e6, dtype=np.float32)
    _, _, _, mean, standard_deviation = historical.normalize_splits(
        train, validation, test
    )

    validation[:] = -8e8
    test[:] = 9e8
    _, _, _, changed_mean, changed_std = historical.normalize_splits(train, validation, test)

    np.testing.assert_array_equal(mean, changed_mean)
    np.testing.assert_array_equal(standard_deviation, changed_std)
    np.testing.assert_allclose(mean, train.mean(axis=(0, 1), dtype=np.float64))
    np.testing.assert_allclose(
        standard_deviation, train.std(axis=(0, 1), dtype=np.float64)
    )


def test_same_training_statistics_normalize_all_splits():
    train = np.arange(4 * 66 * 5, dtype=np.float32).reshape(4, 66, 5)
    validation = train[:2] + 20.0
    test = train[:2] - 30.0
    normalized_train, normalized_validation, normalized_test, mean, std = (
        historical.normalize_splits(train, validation, test)
    )

    np.testing.assert_allclose(normalized_train, (train - mean) / std, rtol=1e-6)
    np.testing.assert_allclose(normalized_validation, (validation - mean) / std, rtol=1e-6)
    np.testing.assert_allclose(normalized_test, (test - mean) / std, rtol=1e-6)


def test_expected_tensor_and_storage_sizes():
    assert historical.tensor_shape(168) == (168, 66, 5)
    storage = historical.expected_storage_bytes()
    assert storage["combined_output_tensors"] == 173_574_720
    assert storage["uncompressed_source_fields_float32"] == 1_669_473_216