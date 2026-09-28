"""Build normalized, chronological GNN tensors from annual ERA5 ZIP blocks."""

from __future__ import annotations

import argparse
import calendar
import importlib.util
import json
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_dataset  # noqa: E402

_PREPARATION_PATH = BASE_DIR / "02_prepare_era5_tensor.py"
_PREPARATION_SPEC = importlib.util.spec_from_file_location(
    "era5_pilot_preparation", _PREPARATION_PATH
)
if _PREPARATION_SPEC is None or _PREPARATION_SPEC.loader is None:
    raise ImportError(f"Cannot load ERA5 feature preparation module: {_PREPARATION_PATH}")
_PREPARATION = importlib.util.module_from_spec(_PREPARATION_SPEC)
_PREPARATION_SPEC.loader.exec_module(_PREPARATION)

FIRST_YEAR = 2010
LAST_YEAR = 2024
FEATURE_NAMES = tuple(_PREPARATION.FEATURE_NAMES)
FEATURE_UNITS = tuple(_PREPARATION.FEATURE_UNITS)
NODE_COUNT = 66
FEATURE_COUNT = 5
SOURCE_LATITUDE_COUNT = 23
SOURCE_LONGITUDE_COUNT = 23
SOURCE_VARIABLE_COUNT = 6
FLOAT32_BYTES = np.dtype(np.float32).itemsize
SPLIT_PERIODS = {
    "train": ("2010-01-01 00:00:00", "2019-12-31 23:00:00"),
    "validation": ("2020-01-01 00:00:00", "2021-12-31 23:00:00"),
    "test": ("2022-01-01 00:00:00", "2024-12-31 23:00:00"),
}
EXPECTED_SPLIT_COUNTS = {
    name: len(pd.date_range(start, end, freq="h"))
    for name, (start, end) in SPLIT_PERIODS.items()
}


def expected_timestamps(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="h")


def expected_month_timestamps(year: int, month: int) -> pd.DatetimeIndex:
    last_day = calendar.monthrange(year, month)[1]
    return expected_timestamps(
        f"{year}-{month:02d}-01 00:00:00",
        f"{year}-{month:02d}-{last_day:02d} 23:00:00",
    )


def validate_timestamp_axis(
    actual: pd.DatetimeIndex,
    expected: pd.DatetimeIndex,
    label: str,
) -> None:
    if actual.hasnans:
        raise ValueError(f"{label} contains missing timestamp values")
    if not actual.is_unique:
        duplicates = actual[actual.duplicated()].unique()
        raise ValueError(f"{label} contains duplicate timestamps: {list(duplicates[:5])}")
    if not actual.is_monotonic_increasing:
        raise ValueError(f"{label} timestamps are not chronologically ordered")
    if actual.equals(expected):
        return

    missing = expected.difference(actual)
    unexpected = actual.difference(expected)
    details = []
    if len(missing):
        details.append(f"{len(missing)} missing, first={list(missing[:5])}")
    if len(unexpected):
        details.append(f"{len(unexpected)} unexpected, first={list(unexpected[:5])}")
    raise ValueError(f"{label} timestamp coverage mismatch: {'; '.join(details)}")


def split_timestamps(timestamps: pd.DatetimeIndex) -> dict[str, pd.DatetimeIndex]:
    if timestamps.hasnans or not timestamps.is_unique or not timestamps.is_monotonic_increasing:
        raise ValueError("Historical timestamps must be non-missing, unique, and chronological")

    result: dict[str, pd.DatetimeIndex] = {}
    assigned = np.zeros(len(timestamps), dtype=np.uint8)
    for name, (start, end) in SPLIT_PERIODS.items():
        mask = (timestamps >= pd.Timestamp(start)) & (timestamps <= pd.Timestamp(end))
        result[name] = timestamps[mask]
        assigned += mask.astype(np.uint8)
    if np.any(assigned > 1):
        raise ValueError("A timestamp appears in more than one chronological split")
    if np.any(assigned == 0):
        raise ValueError("One or more timestamps fall outside the configured split periods")
    return result


def tensor_shape(timestamp_count: int) -> tuple[int, int, int]:
    return (timestamp_count, NODE_COUNT, FEATURE_COUNT)


def expected_storage_bytes() -> dict[str, int]:
    split_hours = EXPECTED_SPLIT_COUNTS
    total_hours = sum(split_hours.values())
    return {
        "train_tensor": split_hours["train"] * NODE_COUNT * FEATURE_COUNT * FLOAT32_BYTES,
        "validation_tensor": split_hours["validation"] * NODE_COUNT * FEATURE_COUNT * FLOAT32_BYTES,
        "test_tensor": split_hours["test"] * NODE_COUNT * FEATURE_COUNT * FLOAT32_BYTES,
        "combined_output_tensors": total_hours * NODE_COUNT * FEATURE_COUNT * FLOAT32_BYTES,
        "uncompressed_source_fields_float32": (
            total_hours
            * SOURCE_LATITUDE_COUNT
            * SOURCE_LONGITUDE_COUNT
            * SOURCE_VARIABLE_COUNT
            * FLOAT32_BYTES
        ),
    }


class FeatureMoments:
    """Numerically stable population moments accumulated over training chunks."""

    def __init__(self) -> None:
        self.count = 0
        self.mean: np.ndarray | None = None
        self.m2: np.ndarray | None = None

    def update(self, tensor: np.ndarray) -> None:
        if tensor.ndim != 3 or tensor.shape[1:] != (NODE_COUNT, FEATURE_COUNT):
            raise ValueError(f"Training chunk must have shape (T, 66, 5); got {tensor.shape}")
        if not np.isfinite(tensor).all():
            raise ValueError("Training chunk contains non-finite values")
        sample = tensor.reshape(-1, FEATURE_COUNT).astype(np.float64)
        batch_count = sample.shape[0]
        if batch_count == 0:
            return
        batch_mean = sample.mean(axis=0)
        batch_m2 = np.square(sample - batch_mean).sum(axis=0)

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        assert self.mean is not None and self.m2 is not None
        combined_count = self.count + batch_count
        delta = batch_mean - self.mean
        self.m2 += batch_m2 + np.square(delta) * self.count * batch_count / combined_count
        self.mean += delta * batch_count / combined_count
        self.count = combined_count

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count == 0 or self.mean is None or self.m2 is None:
            raise ValueError("No training samples were provided for normalization")
        standard_deviation = np.sqrt(self.m2 / self.count)
        if not np.isfinite(self.mean).all() or not np.isfinite(standard_deviation).all():
            raise ValueError("Training normalization statistics are non-finite")
        if np.any(standard_deviation == 0):
            raise ValueError("Training feature has zero standard deviation")
        return self.mean.copy(), standard_deviation


def fit_training_normalization(
    training_chunks: Iterable[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    moments = FeatureMoments()
    for chunk in training_chunks:
        moments.update(chunk)
    return moments.result()


def normalize_tensor(
    tensor: np.ndarray,
    mean: np.ndarray,
    standard_deviation: np.ndarray,
) -> np.ndarray:
    if mean.shape != (FEATURE_COUNT,) or standard_deviation.shape != (FEATURE_COUNT,):
        raise ValueError("Normalization mean/std must have shape (5,)")
    if np.any(standard_deviation <= 0) or not np.isfinite(standard_deviation).all():
        raise ValueError("Normalization standard deviations must be finite and positive")
    if tensor.ndim != 3 or tensor.shape[1:] != (NODE_COUNT, FEATURE_COUNT):
        raise ValueError(f"Tensor must have shape (T, 66, 5); got {tensor.shape}")
    return ((tensor.astype(np.float64) - mean) / standard_deviation).astype(np.float32)


def normalize_splits(
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Small-array helper; statistics are deliberately fitted from train only."""
    mean, standard_deviation = fit_training_normalization([train])
    return (
        normalize_tensor(train, mean, standard_deviation),
        normalize_tensor(validation, mean, standard_deviation),
        normalize_tensor(test, mean, standard_deviation),
        mean,
        standard_deviation,
    )


def load_prepared_month(
    archive_path: Path,
    year: int,
    month: int,
    node_table: pd.DataFrame,
) -> tuple[np.ndarray, pd.DatetimeIndex, dict]:
    dataset = load_dataset(archive_path)
    try:
        dataset.load()
        tensor, timestamps, metadata = _PREPARATION.prepare_dataset(
            dataset, node_table, str(archive_path.resolve())
        )
    finally:
        dataset.close()

    expected = expected_month_timestamps(year, month)
    validate_timestamp_axis(timestamps, expected, f"ERA5 {year}-{month:02d}")
    return tensor, timestamps, metadata


def _split_for_year(year: int) -> str:
    if year <= 2019:
        return "train"
    if year <= 2021:
        return "validation"
    return "test"


def _load_all_blocks(raw_dir: Path) -> list[tuple[int, int, Path]]:
    blocks = [
        (year, month, raw_dir / f"era5_gnn_{year}_{month:02d}.zip")
        for year in range(FIRST_YEAR, LAST_YEAR + 1)
        for month in range(1, 13)
    ]
    archives = [path for _, _, path in blocks]
    missing = [path.name for path in archives if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Historical preparation requires one complete monthly ZIP per month; "
            f"missing {len(missing)} blocks: {', '.join(missing)}"
        )
    return blocks


def _split_time_axes(timestamps: pd.DatetimeIndex) -> dict[str, pd.DatetimeIndex]:
    axes = split_timestamps(timestamps)
    for name, expected_count in EXPECTED_SPLIT_COUNTS.items():
        if len(axes[name]) != expected_count:
            raise ValueError(
                f"{name} has {len(axes[name])} timestamps; expected {expected_count}"
            )
    return axes


def prepare_historical_dataset(
    raw_dir: Path,
    output_dir: Path,
    node_table_path: Path,
) -> dict:
    archives = _load_all_blocks(raw_dir)
    node_table = pd.read_csv(node_table_path)
    _PREPARATION._validate_node_table(node_table)

    all_timestamp_parts: list[pd.DatetimeIndex] = []
    source_records: list[dict] = []
    moments = FeatureMoments()
    print("Pass 1/2: validate monthly blocks and fit training-only normalization")
    for year, month, archive in archives:
        tensor, timestamps, metadata = load_prepared_month(archive, year, month, node_table)
        all_timestamp_parts.append(timestamps)
        source_records.append(
            {
                "year": year,
                "month": month,
                "archive": archive.name,
                "timestamp_count": len(timestamps),
                "first_valid_time": metadata["time"]["first_valid_time"],
                "last_valid_time": metadata["time"]["last_valid_time"],
            }
        )
        if _split_for_year(year) == "train":
            moments.update(tensor)
        print(f"  validated {year}-{month:02d}: {len(timestamps)} hourly timestamps")

    actual_timestamps = all_timestamp_parts[0].append(all_timestamp_parts[1:])
    expected_all = expected_timestamps("2010-01-01 00:00:00", "2024-12-31 23:00:00")
    validate_timestamp_axis(actual_timestamps, expected_all, "ERA5 historical dataset")
    split_axes = _split_time_axes(actual_timestamps)
    train_mean, train_std = moments.result()
    if moments.count != EXPECTED_SPLIT_COUNTS["train"] * NODE_COUNT:
        raise ValueError("Training normalization sample count does not match 2010-2019")

    estimated_storage = expected_storage_bytes()
    free_bytes = shutil.disk_usage(output_dir.parent).free
    if free_bytes < estimated_storage["combined_output_tensors"]:
        raise OSError(
            "Insufficient free disk space for split tensor outputs: "
            f"need {estimated_storage['combined_output_tensors']} bytes, have {free_bytes}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    split_arrays = {
        name: np.lib.format.open_memmap(
            output_dir / f"{filename}_tensor.npy",
            mode="w+",
            dtype=np.float32,
            shape=tensor_shape(EXPECTED_SPLIT_COUNTS[name]),
        )
        for name, filename in (("train", "train"), ("validation", "val"), ("test", "test"))
    }
    split_offsets = {name: 0 for name in split_arrays}

    print("Pass 2/2: apply shared training statistics and write normalized tensors")
    for year, month, archive in archives:
        tensor, _, _ = load_prepared_month(archive, year, month, node_table)
        split_name = _split_for_year(year)
        normalized = normalize_tensor(tensor, train_mean, train_std)
        offset = split_offsets[split_name]
        split_arrays[split_name][offset : offset + len(normalized)] = normalized
        split_offsets[split_name] += len(normalized)
        print(f"  wrote {year}-{month:02d} -> {split_name}")

    for name, array in split_arrays.items():
        array.flush()
        if split_offsets[name] != EXPECTED_SPLIT_COUNTS[name]:
            raise ValueError(
                f"Wrote {split_offsets[name]} {name} timestamps; "
                f"expected {EXPECTED_SPLIT_COUNTS[name]}"
            )
        split_axes[name].to_series().dt.strftime("%Y-%m-%dT%H:%M:%SZ").to_frame(
            name="valid_time"
        ).to_csv(
            output_dir / f"{'val' if name == 'validation' else name}_timestamps.csv",
            index=False,
        )
        del array

    normalization = {
        "feature_order": list(FEATURE_NAMES),
        "feature_units": list(FEATURE_UNITS),
        "fit_period": "2010-01-01 00:00 through 2019-12-31 23:00 UTC",
        "fit_timestamp_count": EXPECTED_SPLIT_COUNTS["train"],
        "fit_node_sample_count": moments.count,
        "standard_deviation_convention": "population (ddof=0)",
        "mean": train_mean.tolist(),
        "standard_deviation": train_std.tolist(),
    }
    with (output_dir / "normalization.json").open("w", encoding="utf-8") as handle:
        json.dump(normalization, handle, indent=2)
        handle.write("\n")

    split_metadata = {}
    for name, timestamps in split_axes.items():
        start, end = SPLIT_PERIODS[name]
        split_metadata[name] = {
            "configured_start": start,
            "configured_end": end,
            "actual_start": timestamps[0].strftime("%Y-%m-%dT%H:%M:%SZ"),
            "actual_end": timestamps[-1].strftime("%Y-%m-%dT%H:%M:%SZ"),
            "timestamp_count": len(timestamps),
            "tensor_shape": list(tensor_shape(len(timestamps))),
        }
    dataset_metadata = {
        "source": {
            "name": "ERA5 reanalysis",
            "dataset": "reanalysis-era5-single-levels",
            "source_variables": list(_PREPARATION.SOURCE_VARIABLES),
            "annual_archives": source_records,
        },
        "features": [
            {"name": name, "units": units}
            for name, units in zip(FEATURE_NAMES, FEATURE_UNITS, strict=True)
        ],
        "spatial": {
            "source_grid_resolution_degrees": 0.25,
            "target_node_count": NODE_COUNT,
            "node_table": str(node_table_path),
            "coordinate_selection": "exact",
            "interpolation": False,
        },
        "time": {
            "coordinate": "valid_time",
            "frequency": "hourly",
            "forecast_reference_time": None,
            "lead_time_hours": None,
        },
        "splits": split_metadata,
        "split_overlap": False,
        "normalization": "Training-only statistics; shared unchanged across all splits.",
        "storage_bytes": estimated_storage,
        "tensor_values_are_normalized": True,
    }
    with (output_dir / "dataset_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset_metadata, handle, indent=2)
        handle.write("\n")

    return {
        "timestamps": actual_timestamps,
        "split_axes": split_axes,
        "normalization": normalization,
        "metadata": dataset_metadata,
        "storage_bytes": estimated_storage,
    }


def _print_plan() -> None:
    timestamps = expected_timestamps("2010-01-01 00:00:00", "2024-12-31 23:00:00")
    axes = _split_time_axes(timestamps)
    storage = expected_storage_bytes()
    print("ACQUISITION STRATEGY: one monthly ZIP request per month, 2010-2024 (180 blocks)")
    print("Each block: six single-level variables, hourly, 23 x 23 padded area, NetCDF in ZIP")
    print("Expected split timestamps:")
    for name, axis in axes.items():
        print(f"  {name}: {axis[0]} through {axis[-1]} ({len(axis)} hours)")
    print("Storage estimates:")
    print(f"  split float32 tensors combined: {storage['combined_output_tensors']:,} bytes")
    print(
        "  six source fields uncompressed float32 on 23 x 23: "
        f"{storage['uncompressed_source_fields_float32']:,} bytes"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument(
        "--node-table", type=Path, default=BASE_DIR / "outputs" / "node_table.csv"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=BASE_DIR / "outputs" / "era5_historical"
    )
    parser.add_argument(
        "--plan-only", action="store_true",
        help="Report expected time counts and storage without requiring/downloading data.",
    )
    args = parser.parse_args()

    if args.plan_only:
        _print_plan()
        return
    result = prepare_historical_dataset(args.raw_dir, args.output_dir, args.node_table)
    print("HISTORICAL ERA5 PREPARATION: PASS")
    for name, timestamps in result["split_axes"].items():
        print(f"{name}: {timestamps[0]} through {timestamps[-1]} ({len(timestamps)} timestamps)")
    print(f"Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()