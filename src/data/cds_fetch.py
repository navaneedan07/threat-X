"""
Fetch ERA5 reanalysis data for a target extreme-weather event, plus a
climatology baseline window, from the Copernicus Climate Data Store (CDS).

SETUP:
    1. Register at:
       https://cds.climate.copernicus.eu

    2. Create/configure your CDS API credentials.

    3. Install dependencies:
       pip install cdsapi

USAGE:
    python 01_fetch.py --event amphan
    python 01_fetch.py --event heatwave
    python 01_fetch.py --climatology
"""

import argparse
import os
import sys

# ---------------------------------------------------------------------
# CDS API import
# ---------------------------------------------------------------------
try:
    import cdsapi
except ImportError as exc:
    print("ERROR: cdsapi is not installed.")
    print("Run:")
    print("    python -m pip install cdsapi")
    raise exc


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))

os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Regions
#
# CDS area format:
#   [North, West, South, East]
# ---------------------------------------------------------------------
REGIONS = {
    "bay_of_bengal": [25, 80, 5, 95],
    "north_india": [35, 68, 20, 90],
}


# ---------------------------------------------------------------------
# Extreme-weather events
# ---------------------------------------------------------------------
EVENTS = {
    "amphan": {
        "region": "bay_of_bengal",
        "date": "2020-05-16/2020-05-21",
        "description": "Cyclone Amphan, Bay of Bengal, May 2020",
    },
    "heatwave": {
        "region": "north_india",
        "date": "2022-05-01/2022-05-10",
        "description": "North India heatwave, May 2022",
    },
}


# ---------------------------------------------------------------------
# ERA5 variables
# ---------------------------------------------------------------------
VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
    "total_precipitation",
]


# Every 3 hours: 00:00, 03:00, ..., 21:00
TIMES = [f"{hour:02d}:00" for hour in range(0, 24, 3)]


# ---------------------------------------------------------------------
# CDS client
# ---------------------------------------------------------------------
def create_client():
    """
    Create a CDS API client.

    The client reads credentials from the CDS configuration
    available to the current user.
    """

    try:
        client = cdsapi.Client()
        return client

    except Exception as exc:
        print()
        print("=" * 70)
        print("ERROR: Could not connect to the Copernicus Climate Data Store.")
        print("=" * 70)
        print()
        print("Check that your CDS API credentials are configured correctly.")
        print()
        print("Original error:")
        print(exc)
        print()
        sys.exit(1)


# ---------------------------------------------------------------------
# Fetch an extreme-weather event
# ---------------------------------------------------------------------
def fetch_event(event_key: str):

    if event_key not in EVENTS:
        raise ValueError(
            f"Unknown event '{event_key}'. "
            f"Choices: {list(EVENTS.keys())}"
        )

    cfg = EVENTS[event_key]

    region_key = cfg["region"]
    area = REGIONS[region_key]

    start, end = cfg["date"].split("/")

    print()
    print("=" * 70)
    print("ERA5 EVENT DATA DOWNLOAD")
    print("=" * 70)
    print(f"Event      : {cfg['description']}")
    print(f"Region     : {region_key}")
    print(f"Area       : {area}")
    print(f"Start date : {start}")
    print(f"End date   : {end}")
    print(f"Variables  : {len(VARIABLES)}")
    print(f"Times      : {len(TIMES)} per day")
    print("=" * 70)
    print()

    client = create_client()

    target = os.path.join(
        OUT_DIR,
        f"era5_{event_key}.nc"
    )

    # Avoid accidentally overwriting an existing download.
    if os.path.exists(target):
        print(f"[skip] File already exists:")
        print(f"       {target}")
        print()
        print("Delete the file if you want to download it again.")
        return

    request = {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": VARIABLES,
        "date": cfg["date"],
        "time": TIMES,
        "area": area,
    }

    print("[fetch] Sending request to CDS...")
    print()

    try:
        client.retrieve(
            "reanalysis-era5-single-levels",
            request,
            target,
        )

    except Exception as exc:
        print()
        print("=" * 70)
        print("DOWNLOAD FAILED")
        print("=" * 70)
        print(exc)
        print()
        sys.exit(1)

    print()
    print("=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"File: {target}")
    print("=" * 70)
    print()


# ---------------------------------------------------------------------
# Fetch climatology
# ---------------------------------------------------------------------
def fetch_climatology(
    region_key: str = "bay_of_bengal",
    years=range(1991, 2021),
):

    if region_key not in REGIONS:
        raise ValueError(
            f"Unknown region '{region_key}'. "
            f"Choices: {list(REGIONS.keys())}"
        )

    area = REGIONS[region_key]

    # May 18 is used as the climatology reference date.
    month_day = "05-18"

    output_dir = os.path.join(
        OUT_DIR,
        "climatology"
    )

    os.makedirs(output_dir, exist_ok=True)

    print()
    print("=" * 70)
    print("ERA5 CLIMATOLOGY DOWNLOAD")
    print("=" * 70)
    print(f"Region : {region_key}")
    print(f"Area   : {area}")
    print(f"Years  : {min(years)} - {max(years)}")
    print("=" * 70)
    print()

    client = create_client()

    for year in years:

        date_str = f"{year}-{month_day}"

        target = os.path.join(
            output_dir,
            f"era5_clim_{year}.nc"
        )

        # Resume capability: don't download files that already exist.
        if os.path.exists(target):
            print(f"[skip] {target}")
            continue

        print()
        print(f"[fetch] climatology {date_str}")
        print(f"        area={area}")

        request = {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": VARIABLES,
            "date": date_str,
            "time": TIMES,
            "area": area,
        }

        try:
            client.retrieve(
                "reanalysis-era5-single-levels",
                request,
                target,
            )

        except Exception as exc:
            print()
            print(f"[ERROR] Failed to download {date_str}")
            print(exc)
            print()
            print("Continuing with the next year...")
            continue

        print(f"[done] {target}")

    print()
    print("=" * 70)
    print("CLIMATOLOGY DOWNLOAD COMPLETE")
    print("=" * 70)
    print()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():

    parser = argparse.ArgumentParser(
        description=(
            "Download ERA5 reanalysis data for extreme-weather "
            "events or climatology."
        )
    )

    parser.add_argument(
        "--event",
        choices=list(EVENTS.keys()),
        help="Fetch a specific extreme-weather event.",
    )

    parser.add_argument(
        "--climatology",
        action="store_true",
        help="Fetch the 1991-2020 climatology baseline.",
    )

    parser.add_argument(
        "--region",
        default="bay_of_bengal",
        choices=list(REGIONS.keys()),
        help="Region for climatology download.",
    )

    args = parser.parse_args()

    if args.event:
        fetch_event(args.event)

    elif args.climatology:
        fetch_climatology(args.region)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
