"""Fetch REAL ERA5-Land reanalysis as the fine reference for downscaling.

Why this module exists
----------------------
Everything currently in ``data/raw/`` is ERA5 at **0.25 deg (~28 km)**. The
downscaling stage needs a *pair*: a coarse field and a finer field to score against.
Without a finer reference there is no real-data artefact, and
``configs/data.yaml -> grid.fine_resolution_deg`` could not be confirmed — which is
exactly why it stayed ``null`` until this pair was downloaded and measured.

ERA5-Land is the same reanalysis family at **0.1 deg (~9 km)** native resolution.
It is a legitimate fine reference for a real 0.25 deg -> 0.1 deg (2.5x) experiment,
and it is honest about what it is: a *different model run* of the same forcing, not
a truth field. It is **not** the problem statement's 5 km target, and it must not be
described as one.

It reuses the event windows, regions and CDS plumbing in
``src/data/cds_fetch.py`` so the two sources stay in step.

USAGE
-----
    python -m src.data.land_fetch --event amphan
    python -m src.data.land_fetch --event heatwave
    # both events, one command
    python -m src.data.land_fetch --all-events

Writes ``data/raw/era5_land_<event>.nc`` (a CDS ZIP-wrapped NetCDF, same as the
ERA5 event files -- read with ``src.data.loader.load_dataset``).
"""

from __future__ import annotations

import argparse
import os

from src.data.cds_fetch import (
    EVENTS,
    RAW_DIR,
    REGIONS,
    TIMES,
    _download,
    create_client,
)

__all__ = ["LAND_DATASET", "LAND_VARIABLES", "fetch_land_event", "main"]

LAND_DATASET = "reanalysis-era5-land"

# ERA5-Land's variable table differs from ERA5 single-levels: it has no
# mean_sea_level_pressure (it carries surface_pressure instead). These four overlap
# with the ERA5 event download so the coarse/fine pair is like-for-like.
LAND_VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation",
]


def fetch_land_event(event_key: str) -> None:
    """Download one event's ERA5-Land window to ``data/raw/era5_land_<event>.nc``."""
    if event_key not in EVENTS:
        raise ValueError(f"Unknown event '{event_key}'. Choices: {list(EVENTS)}")

    cfg = EVENTS[event_key]
    area = REGIONS[cfg["region"]]

    print()
    print("=" * 70)
    print("ERA5-LAND DOWNLOAD (fine reference, ~9 km)")
    print("=" * 70)
    print(f"Event     : {cfg['description']}")
    print(f"Region    : {cfg['region']}  area={area}")
    print(f"Date      : {cfg['date']}")
    print(f"Variables : {len(LAND_VARIABLES)} (native 0.1 deg)")
    print("=" * 70)

    client = create_client()
    target = os.path.join(RAW_DIR, f"era5_land_{event_key}.nc")
    _download(
        client,
        LAND_DATASET,
        {
            "product_type": "reanalysis",
            "variable": LAND_VARIABLES,
            "date": cfg["date"],
            "time": TIMES,  # 3-hourly, to match the ERA5 coarse field
            "area": area,
            "format": "netcdf",
        },
        target,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download REAL ERA5-Land (~9 km) as the fine downscaling reference."
    )
    parser.add_argument("--event", choices=list(EVENTS), help="Fetch one event.")
    parser.add_argument(
        "--all-events", action="store_true", help="Fetch every defined event."
    )
    args = parser.parse_args()

    if args.all_events:
        for event_key in EVENTS:
            fetch_land_event(event_key)
    elif args.event:
        fetch_land_event(args.event)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
