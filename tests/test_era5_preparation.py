"""Focused tests for timestamped ERA5-to-GNN preparation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

PREPARATION_PATH = (
    Path(__file__).resolve().parents[1] / "weights" / "gnn" / "02_prepare_era5_tensor.py"
)
PREPARATION_SPEC = importlib.util.spec_from_file_location("era5_preparation", PREPARATION_PATH)
assert PREPARATION_SPEC is not None and PREPARATION_SPEC.loader is not None
prep = importlib.util.module_from_spec(PREPARATION_SPEC)
PREPARATION_SPEC.loader.exec_module(prep)


def _node_table() -> pd.DataFrame:
    coordinates = [
        (latitude, longitude)
        for latitude in prep.EXPECTED_LATITUDES
        for longitude in prep.EXPECTED_LONGITUDES
    ]
    return pd.DataFrame(
        {
            "node_id": np.arange(66),
            "latitude": [coordinate[0] for coordinate in coordinates],
            "longitude": [coordinate[1] for coordinate in coordinates],
        }
    )


def _dataset() -> xr.Dataset:
    times = pd.date_range("2020-05-16 00:00", periods=4, freq="h")
    latitudes = np.arange(15.25, 9.75 - 0.001, -0.25)
    longitudes = np.arange(75.75, 81.25 + 0.001, 0.25)
    shape = (len(times), len(latitudes), len(longitudes))
    time_axis = np.arange(shape[0], dtype=np.float64)[:, None, None]
    lat_axis = latitudes[None, :, None]
    lon_axis = longitudes[None, None, :]

    variables = {
        "t2m": ("K", 300.0 + time_axis + lat_axis * 0.01 + lon_axis * 0.001),
        "d2m": ("K", 295.0 + time_axis + lat_axis * 0.01 + lon_axis * 0.001),
        "msl": ("Pa", np.full(shape, 100000.0)),
        "u10": ("m s**-1", np.full(shape, 3.0)),
        "v10": ("m s**-1", np.full(shape, 4.0)),
        "tp": ("m", np.full(shape, 0.001)),
    }
    data_vars = {
        name: (("valid_time", "latitude", "longitude"), values, {"units": units})
        for name, (units, values) in variables.items()
    }
    data_vars["tp"][2]["GRIB_stepType"] = "accum"
    return xr.Dataset(
        data_vars,
        coords={
            "valid_time": times,
            "latitude": latitudes,
            "longitude": longitudes,
        },
    )


def test_tensor_shape_ordering_timestamps_and_node_mapping():
    dataset = _dataset()
    nodes = _node_table()
    tensor, timestamps, metadata = prep.prepare_dataset(dataset, nodes, "pilot.zip")

    assert tensor.shape == (4, 66, 5)
    assert tensor.dtype == np.float32
    assert list(metadata["tensor"]["feature_order"]) == [
        "temperature",
        "pressure",
        "humidity",
        "wind_speed",
        "precipitation",
    ]
    assert timestamps.equals(pd.DatetimeIndex(dataset.valid_time.values))
    assert metadata["nodes"][0] == {"node_id": 0, "latitude": 10.0, "longitude": 76.0}
    assert metadata["nodes"][-1] == {"node_id": 65, "latitude": 15.0, "longitude": 81.0}
    assert metadata["time"]["forecast_reference_time"] is None
    assert metadata["time"]["lead_time_hours"] is None
    assert metadata["source"]["name"] == "ERA5 reanalysis"


def test_feature_conversions_match_raw_fields():
    dataset = _dataset()
    tensor, _, _ = prep.prepare_dataset(dataset, _node_table(), "pilot.zip")
    node = 0
    temperature_c = float(dataset.t2m.values[0, 21, 1] - 273.15)
    dewpoint_c = float(dataset.d2m.values[0, 21, 1] - 273.15)
    expected_rh = 100.0 * prep.saturation_vapour_pressure_hpa(
        np.array(dewpoint_c)
    ) / prep.saturation_vapour_pressure_hpa(np.array(temperature_c))

    assert tensor[0, node, 0] == pytest.approx(temperature_c)
    assert tensor[0, node, 1] == pytest.approx(1000.0)
    assert tensor[0, node, 2] == pytest.approx(float(expected_rh))
    assert tensor[0, node, 3] == pytest.approx(5.0)
    assert tensor[0, node, 4] == pytest.approx(1.0)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_source_values_are_rejected(bad_value: float):
    dataset = _dataset()
    dataset["t2m"].values[0, 0, 0] = bad_value

    with pytest.raises(ValueError, match="NaN|infinite"):
        prep.prepare_dataset(dataset, _node_table(), "pilot.zip")


def test_coordinate_mismatch_is_rejected():
    nodes = _node_table()
    nodes.loc[0, "latitude"] = 10.25

    with pytest.raises(ValueError, match="coordinates/order"):
        prep.prepare_dataset(_dataset(), nodes, "pilot.zip")


def test_duplicate_or_nonhourly_timestamps_are_rejected():
    dataset = _dataset().assign_coords(
        valid_time=pd.to_datetime(
            ["2020-05-16 00:00", "2020-05-16 01:00", "2020-05-16 01:00", "2020-05-16 03:00"]
        )
    )

    with pytest.raises(ValueError, match="duplicate"):
        prep.prepare_dataset(dataset, _node_table(), "pilot.zip")

    dataset = _dataset().assign_coords(
        valid_time=pd.to_datetime(
            ["2020-05-16 00:00", "2020-05-16 01:00", "2020-05-16 02:30", "2020-05-16 03:00"]
        )
    )
    with pytest.raises(ValueError, match="hourly"):
        prep.prepare_dataset(dataset, _node_table(), "pilot.zip")


def test_pilot_archive_reconstructs_expected_shape_without_writing_outputs():
    archive = Path("data/raw/era5_gnn_pilot_20200516_20200522.zip")
    if not archive.is_file():
        pytest.skip("Local pilot archive is not available in this checkout")

    dataset = prep.load_dataset(archive)
    nodes = pd.read_csv(prep.DEFAULT_NODE_TABLE)
    tensor, timestamps, metadata = prep.prepare_dataset(dataset, nodes, str(archive))

    def node_values(name: str) -> np.ndarray:
        return np.stack(
            [
                dataset[name]
                .sel(latitude=float(node.latitude), longitude=float(node.longitude))
                .values.astype(np.float64)
                for node in nodes.itertuples(index=False)
            ],
            axis=1,
        )

    t2m = node_values("t2m")
    d2m = node_values("d2m")
    temperature_c = t2m - 273.15
    dewpoint_c = d2m - 273.15
    expected = np.stack(
        (
            temperature_c,
            node_values("msl") / 100.0,
            100.0
            * prep.saturation_vapour_pressure_hpa(dewpoint_c)
            / prep.saturation_vapour_pressure_hpa(temperature_c),
            np.sqrt(node_values("u10") ** 2 + node_values("v10") ** 2),
            node_values("tp") * 1000.0,
        ),
        axis=-1,
    ).astype(np.float32)

    assert tensor.shape == (168, 66, 5)
    assert tensor.dtype == np.float32
    np.testing.assert_array_equal(tensor, expected)
    assert timestamps[0] == pd.Timestamp("2020-05-16 00:00")
    assert timestamps[-1] == pd.Timestamp("2020-05-22 23:00")
    assert np.isfinite(tensor).all()
    assert metadata["validation"]["node_order_matches_node_table"] is True