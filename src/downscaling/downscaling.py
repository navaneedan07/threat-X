"""
Downscaling module (the "Stage 2" substitute for the diffusion model).
Honest substitute: bicubic interpolation from 12km -> 5km. NOT a trained
model -- say so in the PPT/demo. ESPCN stub at the bottom for a real model later.

USAGE:
  python 03.py --event amphan --variable total_precipitation
"""

import argparse
import json
import os

import numpy as np
import xarray as xr
from scipy.ndimage import zoom

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")

VAR_MAP = {
    "total_precipitation": "tp",
    "2m_temperature": "t2m",
    "mean_sea_level_pressure": "msl",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
}

SOURCE_RES_KM = 12
TARGET_RES_KM = 5
ZOOM_FACTOR = SOURCE_RES_KM / TARGET_RES_KM


def downscale_box(event_key: str, variable: str, time_index: int, box: dict, pad_deg: float = 0.5):
    varname = VAR_MAP[variable]
    event_path = os.path.join(DATA_DIR, f"era5_{event_key}.nc")
    ds = xr.open_dataset(event_path)
    da = ds[varname].isel(time=time_index)

    lat_slice = slice(box["lat_max"] + pad_deg, box["lat_min"] - pad_deg)
    lon_slice = slice(box["lon_min"] - pad_deg, box["lon_max"] + pad_deg)
    cropped = da.sel(latitude=lat_slice, longitude=lon_slice)

    if cropped.size == 0:
        raise ValueError("Crop produced an empty array -- check lat/lon ordering for this dataset")

    arr = cropped.values
    upsampled = zoom(arr, ZOOM_FACTOR, order=3)

    noise = np.random.normal(0, upsampled.std() * 0.02, upsampled.shape)
    upsampled = upsampled + noise

    return {
        "original_shape": list(arr.shape),
        "upsampled_shape": list(upsampled.shape),
        "lat_bounds": [float(cropped.latitude.min()), float(cropped.latitude.max())],
        "lon_bounds": [float(cropped.longitude.min()), float(cropped.longitude.max())],
        "resolution_km": TARGET_RES_KM,
        "values": upsampled.tolist(),
    }


# ---------------------------------------------------------------------------
# Real model stub -- fill in if your team has time to train a small
# super-resolution CNN (ESPCN) on paired low/high-res weather patches.
# ---------------------------------------------------------------------------
# import torch
# import torch.nn as nn
#
# class ESPCN(nn.Module):
#     def __init__(self, upscale_factor=2):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 64, 5, padding=2)
#         self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
#         self.conv3 = nn.Conv2d(32, upscale_factor ** 2, 3, padding=1)
#         self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
#
#     def forward(self, x):
#         x = torch.relu(self.conv1(x))
#         x = torch.relu(self.conv2(x))
#         x = self.pixel_shuffle(self.conv3(x))
#         return x
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True, choices=["amphan", "heatwave"])
    ap.add_argument("--variable", default="total_precipitation", choices=list(VAR_MAP))
    args = ap.parse_args()

    anomaly_path = os.path.join(OUT_DIR, f"anomalies_{args.event}.json")
    if not os.path.exists(anomaly_path):
        raise FileNotFoundError(f"Run 02_anomaly_detection.py first (missing {anomaly_path})")

    with open(anomaly_path) as f:
        data = json.load(f)

    downscaled_frames = []
    for frame in data["frames"]:
        if not frame["boxes"]:
            continue
        top_box = frame["boxes"][0]
        result = downscale_box(args.event, args.variable, frame["time_index"], top_box)
        downscaled_frames.append({
            "time_index": frame["time_index"],
            "timestamp": frame["timestamp"],
            "box": top_box,
            "downscaled": result,
        })
        print(f"[downscale] t={frame['time_index']}  {result['original_shape']} -> {result['upsampled_shape']}")

    out_path = os.path.join(OUT_DIR, f"downscaled_{args.event}.json")
    with open(out_path, "w") as f:
        json.dump(downscaled_frames, f)
    print(f"[done] wrote {out_path}")