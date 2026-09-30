"""
Fetch REAL ERA5 reanalysis data from the Copernicus Climate Data Store (CDS).

This module downloads:
  1. Event windows  -- the extreme-weather case itself (e.g. Cyclone Amphan).
  2. Climatology    -- a multi-year SEASONAL window used as the anomaly baseline.

It never generates synthetic data. If a download fails, it fails loudly; the
pipeline is expected to run on real archives only.

SETUP
-----
  1. Register at https://cds.climate.copernicus.eu and get API credentials:
     https://cds.climate.copernicus.eu/how-to-api
  2. Put them in ~/.cdsapirc, OR as CDSAPI_URL / CDSAPI_KEY in a gitignored
     .env file (see .env.example).
  3. pip install cdsapi

USAGE
-----
    # One event, surface variables (writes data/raw/era5_<event>.nc)
    python -m src.data.cds_fetch --event amphan
    python -m src.data.cds_fetch --event heatwave

    # Same, but also pull pressure-level fields for the precursor module
    python -m src.data.cds_fetch --event amphan --pressure-levels

    # Climatology baseline: full May, 1991-2020, BOTH regions.
    #   -> data/climatology/era5_clim_<region>_<year>.nc
    python -m src.data.cds_fetch --climatology --all-regions

    # Narrower / sharper baseline window (e.g. +/- 7 days around 18 May):
    python -m src.data.cds_fetch --climatology --all-regions --days 11-25

    # GNN frozen-checkpoint pilot (2020-05-16..22, 168 hourly steps)
    #   -> data/raw/era5_gnn_pilot_20200516_20200522.zip
    python -m src.data.cds_fetch --gnn-pilot

WHY A WINDOW, NOT ONE DAY
-------------------------
An earlier version of this script requested a single day (18 May) per year.
That yields 30 samples of ONE synoptic situation -- a day-to-day weather sample,
not a climatological distribution. A defensible anomaly baseline needs many
days around the event date, across many years, so each grid cell has a real
distribution to score against. See docs/dataset.md and docs/experiments.md.
"""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import os
import sys

# ---------------------------------------------------------------------
# CDS API import
# ---------------------------------------------------------------------
try:
    import cdsapi
except ImportError as exc:  # pragma: no cover - environment guard
    print("ERROR: cdsapi is not installed.")
    print("Run:  pip install cdsapi")
    raise exc


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# src/data/ -> repo root
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
CLIM_DIR = os.path.join(DATA_DIR, "climatology")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLIM_DIR, exist_ok=True)

# Load CDSAPI_URL / CDSAPI_KEY from a gitignored .env if present, so credentials
# never have to be hard-coded or exported by hand. cdsapi also reads ~/.cdsapirc.
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(REPO_ROOT, ".env"), override=False)
except ImportError:  # python-dotenv is optional; ~/.cdsapirc still works
    pass


# ---------------------------------------------------------------------
# Regions -- CDS area format is [North, West, South, East]
# ---------------------------------------------------------------------
REGIONS = {
    "bay_of_bengal": [25, 80, 5, 95],   # Cyclone Amphan domain
    "north_india": [35, 68, 20, 90],    # North India heatwave domain
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
# Variables
# ---------------------------------------------------------------------
# Surface single-level fields (dataset: reanalysis-era5-single-levels).
SURFACE_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
    "total_precipitation",
]

# Pressure-level fields (dataset: reanalysis-era5-pressure-levels). These are
# what the precursor module needs for moisture-flux convergence, 850 hPa
# vorticity, theta-e gradients and bulk shear -- all of which are currently
# null because the surface-only download does not carry them.
PRESSURE_VARIABLES = [
    "geopotential",
    "relative_humidity",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
]
PRESSURE_LEVELS = ["500", "700", "850"]

# Every 3 hours: 00:00, 03:00, ..., 21:00
TIMES = [f"{hour:02d}:00" for hour in range(0, 24, 3)]

# GNN-specific ERA5 single-level request. Kept separate from the existing
# precursor/event request so its cadence, feature definitions, and area cannot
# be silently changed by those workflows.
GNN_DATASET = "reanalysis-era5-single-levels"
GNN_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation",
]
GNN_AREA = [15.25, 75.75, 9.75, 81.25]
GNN_TIMES = [f"{hour:02d}:00" for hour in range(24)]
GNN_FIRST_YEAR = 2010
GNN_LAST_YEAR = 2024

# The frozen-checkpoint pilot. It is the only GNN input with a verified
# wall-clock time axis, so it is the one used for the retrospective ERA5
# evaluation (weights/gnn/08) and the GNN->detection->tracking demo.
# Seven full days at 24 hourly steps = the 168 timesteps that
# weights/gnn/02_prepare_era5_tensor.py validates against.
GNN_PILOT_START = "2020-05-16"
GNN_PILOT_END = "2020-05-22"
GNN_PILOT_FILENAME = "era5_gnn_pilot_20200516_20200522.zip"
GNN_PILOT_TIMESTEPS = 168


# ---------------------------------------------------------------------
# CDS client
# ---------------------------------------------------------------------
def create_client() -> cdsapi.Client:
    """Create a CDS API client, or exit with an actionable message.

    Credentials are read by cdsapi from ~/.cdsapirc or from the CDSAPI_URL /
    CDSAPI_KEY environment variables. They are never hard-coded here and never
    committed.
    """
    try:
        return cdsapi.Client()
    except Exception as exc:  # pragma: no cover - environment guard
        print()
        print("=" * 70)
        print("ERROR: Could not create a CDS API client.")
        print("=" * 70)
        print()
        print("Configure credentials first:")
        print("  1. Register: https://cds.climate.copernicus.eu")
        print("  2. Get URL + key: https://cds.climate.copernicus.eu/how-to-api")
        print("  3. Put them in ~/.cdsapirc, or set CDSAPI_URL / CDSAPI_KEY")
        print("     in a gitignored .env file (see .env.example).")
        print()
        print("Original error:")
        print(exc)
        print()
        sys.exit(1)


def _download(client, dataset: str, request: dict, target: str) -> None:
    """Run one CDS retrieval with resume + loud failure."""
    if os.path.exists(target):
        print(f"[skip] already exists: {target}")
        return

    print(f"[fetch] {dataset} -> {os.path.basename(target)}")
    try:
        client.retrieve(dataset, request, target)
    except Exception as exc:
        # Never fall back to synthetic data -- surface the real error.
        print()
        print("=" * 70)
        print(f"DOWNLOAD FAILED: {target}")
        print("=" * 70)
        print(exc)
        print()
        raise
    print(f"[done] {target}")


def build_gnn_request(start_date: str, end_date: str) -> dict:
    """Build one bounded hourly ERA5 request for a GNN probe or annual block."""
    start = dt.date.fromisoformat(start_date)
    end = dt.date.fromisoformat(end_date)
    if start > end:
        raise ValueError("start_date must be on or before end_date")
    date_range = start.isoformat()
    if end != start:
        date_range = f"{date_range}/{end.isoformat()}"
    return {
        "product_type": ["reanalysis"],
        "variable": list(GNN_VARIABLES),
        "date": date_range,
        "time": list(GNN_TIMES),
        "area": list(GNN_AREA),
        "data_format": "netcdf",
        "download_format": "zip",
    }


def _fetch_gnn_period(start_date: str, end_date: str, target: str) -> str:
    if os.path.exists(target):
        print(f"[skip] already exists: {target}")
        return target

    request = build_gnn_request(start_date, end_date)
    print(
        f"[fetch] {GNN_DATASET} {start_date} through {end_date} "
        f"({len(GNN_TIMES)} hourly steps/day) -> {os.path.basename(target)}"
    )
    result = create_client().retrieve(GNN_DATASET, request)
    result.download(target)
    print(f"[done] {target} ({os.path.getsize(target)} bytes)")
    return target


def fetch_gnn_month(year: int, month: int) -> str:
    """Fetch one calendar month as a resumable, spatially bounded ZIP."""
    if not GNN_FIRST_YEAR <= year <= GNN_LAST_YEAR:
        raise ValueError(f"GNN historical year must be {GNN_FIRST_YEAR}-{GNN_LAST_YEAR}")
    if not 1 <= month <= 12:
        raise ValueError("GNN historical month must be 1-12")
    start_date = f"{year:04d}-{month:02d}-01"
    end_day = calendar.monthrange(year, month)[1]
    end_date = f"{year:04d}-{month:02d}-{end_day:02d}"
    target = os.path.join(RAW_DIR, f"era5_gnn_{year:04d}_{month:02d}.zip")
    return _fetch_gnn_period(start_date, end_date, target)


def fetch_gnn_year(year: int) -> list[str]:
    """Fetch a year as twelve monthly requests to stay below CDS cost limits."""
    if not GNN_FIRST_YEAR <= year <= GNN_LAST_YEAR:
        raise ValueError(f"GNN historical year must be {GNN_FIRST_YEAR}-{GNN_LAST_YEAR}")
    return [fetch_gnn_month(year, month) for month in range(1, 13)]


def fetch_gnn_probe(probe_date: str) -> str:
    """Fetch one representative day without downloading a complete year."""
    parsed_date = dt.date.fromisoformat(probe_date)
    if not GNN_FIRST_YEAR <= parsed_date.year <= GNN_LAST_YEAR:
        raise ValueError(f"Probe year must be {GNN_FIRST_YEAR}-{GNN_LAST_YEAR}")
    target = os.path.join(RAW_DIR, f"era5_gnn_probe_{parsed_date:%Y%m%d}.zip")
    return _fetch_gnn_period(parsed_date.isoformat(), parsed_date.isoformat(), target)


def fetch_gnn_pilot(
    start_date: str = GNN_PILOT_START,
    end_date: str = GNN_PILOT_END,
) -> str:
    """Fetch the frozen-checkpoint GNN pilot archive.

    One bounded request covering the full pilot window at 24 hourly steps, six
    surface variables, over ``GNN_AREA``. The target file name is fixed because
    ``weights/gnn/02_prepare_era5_tensor.py`` validates the timestamp count and
    span that this window produces.
    """
    start = dt.date.fromisoformat(start_date)
    end = dt.date.fromisoformat(end_date)
    expected_days = (end - start).days + 1
    expected_steps = expected_days * len(GNN_TIMES)
    if expected_steps != GNN_PILOT_TIMESTEPS:
        raise ValueError(
            f"Pilot window {start_date}..{end_date} yields {expected_steps} hourly "
            f"steps; the prepared tensor contract expects {GNN_PILOT_TIMESTEPS}"
        )
    target = os.path.join(RAW_DIR, GNN_PILOT_FILENAME)
    return _fetch_gnn_period(start_date, end_date, target)


# ---------------------------------------------------------------------
# Fetch an extreme-weather event
# ---------------------------------------------------------------------
def fetch_event(event_key: str, pressure_levels: bool = False) -> None:
    if event_key not in EVENTS:
        raise ValueError(f"Unknown event '{event_key}'. Choices: {list(EVENTS)}")

    cfg = EVENTS[event_key]
    area = REGIONS[cfg["region"]]

    print()
    print("=" * 70)
    print("ERA5 EVENT DOWNLOAD")
    print("=" * 70)
    print(f"Event     : {cfg['description']}")
    print(f"Region    : {cfg['region']}  area={area}")
    print(f"Date      : {cfg['date']}")
    print(f"Variables : {len(SURFACE_VARIABLES)} surface"
          + (f" + {len(PRESSURE_VARIABLES)} pressure-level" if pressure_levels else ""))
    print("=" * 70)

    client = create_client()

    target = os.path.join(RAW_DIR, f"era5_{event_key}.nc")
    _download(
        client,
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": SURFACE_VARIABLES,
            "date": cfg["date"],
            "time": TIMES,
            "area": area,
            "format": "netcdf",
        },
        target,
    )

    if pressure_levels:
        pl_target = os.path.join(RAW_DIR, f"era5_{event_key}_plev.nc")
        _download(
            client,
            "reanalysis-era5-pressure-levels",
            {
                "product_type": "reanalysis",
                "variable": PRESSURE_VARIABLES,
                "pressure_level": PRESSURE_LEVELS,
                "date": cfg["date"],
                "time": TIMES,
                "area": area,
                "format": "netcdf",
            },
            pl_target,
        )


# ---------------------------------------------------------------------
# Fetch the climatology baseline
# ---------------------------------------------------------------------
def _month_day_codes(month: str, days: str) -> list[str]:
    """Parse a day spec like '1-31' or '11-25' into zero-padded day codes."""
    if "-" in days:
        lo, hi = days.split("-")
        lo_i, hi_i = int(lo), int(hi)
    else:
        lo_i = hi_i = int(days)
    if not (1 <= lo_i <= hi_i <= 31):
        raise ValueError(f"Invalid day range '{days}' (expected e.g. 1-31 or 11-25)")
    if month != "02":
        return [f"{d:02d}" for d in range(lo_i, hi_i + 1)]
    # February: clamp to 28 to avoid a hard CDS rejection.
    return [f"{d:02d}" for d in range(lo_i, min(hi_i, 28) + 1)]


def fetch_climatology(
    region_key: str = "bay_of_bengal",
    years=range(1991, 2021),
    month: str = "05",
    days: str = "1-31",
    pressure_levels: bool = False,
) -> None:
    """Download a multi-year seasonal climatology window for one region.

    Each year is downloaded as its own file so the loop is resumable and CDS
    request sizes stay sane. Files are named with the region so both event
    domains can coexist:

        data/climatology/era5_clim_<region>_<year>.nc
    """
    if region_key not in REGIONS:
        raise ValueError(f"Unknown region '{region_key}'. Choices: {list(REGIONS)}")

    area = REGIONS[region_key]
    day_codes = _month_day_codes(month, days)

    print()
    print("=" * 70)
    print("ERA5 CLIMATOLOGY DOWNLOAD")
    print("=" * 70)
    print(f"Region    : {region_key}  area={area}")
    print(f"Years     : {min(years)}-{max(years)}  ({len(list(years))} files)")
    print(f"Window    : month {month}, days {day_codes[0]}-{day_codes[-1]}"
          f"  ({len(day_codes)} days x {len(TIMES)} times x "
          f"{len(list(years))} years = "
          f"{len(day_codes) * len(TIMES) * len(list(years))} samples/cell)")
    print("Variables : surface"
          + (f" + pressure-level {PRESSURE_LEVELS}" if pressure_levels else ""))
    print("=" * 70)

    client = create_client()

    for year in years:
        target = os.path.join(
            CLIM_DIR, f"era5_clim_{region_key}_{year}.nc"
        )
        _download(
            client,
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": SURFACE_VARIABLES,
                "year": str(year),
                "month": month,
                "day": day_codes,
                "time": TIMES,
                "area": area,
                "format": "netcdf",
            },
            target,
        )

        if pressure_levels:
            pl_target = os.path.join(
                CLIM_DIR, f"era5_clim_{region_key}_{year}_plev.nc"
            )
            _download(
                client,
                "reanalysis-era5-pressure-levels",
                {
                    "product_type": "reanalysis",
                    "variable": PRESSURE_VARIABLES,
                    "pressure_level": PRESSURE_LEVELS,
                    "year": str(year),
                    "month": month,
                    "day": day_codes,
                    "time": TIMES,
                    "area": area,
                    "format": "netcdf",
                },
                pl_target,
            )

    print()
    print("=" * 70)
    print("CLIMATOLOGY DOWNLOAD COMPLETE")
    print("=" * 70)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download REAL ERA5 event and climatology data from CDS."
    )
    parser.add_argument("--event", choices=list(EVENTS), help="Fetch one event.")
    parser.add_argument(
        "--climatology", action="store_true",
        help="Fetch the multi-year seasonal climatology window.",
    )
    parser.add_argument(
        "--region", default="bay_of_bengal", choices=list(REGIONS),
        help="Region for the climatology download.",
    )
    parser.add_argument(
        "--all-regions", action="store_true",
        help="Fetch climatology for every region (needed for both events).",
    )
    parser.add_argument("--month", default="05", help="Month code, e.g. 05.")
    parser.add_argument(
        "--days", default="1-31",
        help="Day window within the month, e.g. '1-31' or '11-25'.",
    )
    parser.add_argument(
        "--years", default="1991-2020",
        help="Year range for the climatology, e.g. 1991-2020.",
    )
    parser.add_argument(
        "--pressure-levels", action="store_true",
        help="Also download pressure-level fields (for precursor analysis).",
    )
    parser.add_argument(
        "--gnn-year", action="append", type=int, default=[],
        help="Fetch a year as twelve monthly ZIP requests (repeat; 2010-2024).",
    )
    parser.add_argument(
        "--gnn-month", action="append", default=[], metavar="YYYY-MM",
        help="Fetch one month (repeatable), e.g. --gnn-month 2010-01.",
    )
    parser.add_argument(
        "--gnn-probe-date", action="append", default=[],
        help="Fetch one GNN validation day (repeatable, YYYY-MM-DD).",
    )
    parser.add_argument(
        "--gnn-pilot", action="store_true",
        help=(
            "Fetch the frozen-checkpoint GNN pilot archive "
            f"({GNN_PILOT_START}..{GNN_PILOT_END}, {GNN_PILOT_TIMESTEPS} hourly steps)."
        ),
    )

    args = parser.parse_args()

    gnn_modes = sum(
        bool(value)
        for value in (args.gnn_year, args.gnn_month, args.gnn_probe_date, args.gnn_pilot)
    )
    if gnn_modes > 1:
        parser.error(
            "choose one of --gnn-year, --gnn-month, --gnn-probe-date, or --gnn-pilot"
        )
    if (args.gnn_year or args.gnn_month or args.gnn_probe_date or args.gnn_pilot) and (
        args.event or args.climatology or args.pressure_levels
    ):
        parser.error("GNN requests cannot be combined with event/climatology options")

    if args.gnn_pilot:
        fetch_gnn_pilot()
    elif args.gnn_year:
        for year in args.gnn_year:
            fetch_gnn_year(year)
    elif args.gnn_month:
        for value in args.gnn_month:
            try:
                year_text, month_text = value.split("-", maxsplit=1)
                fetch_gnn_month(int(year_text), int(month_text))
            except (ValueError, TypeError):
                parser.error(f"invalid --gnn-month {value!r}; expected YYYY-MM")
    elif args.gnn_probe_date:
        for probe_date in args.gnn_probe_date:
            fetch_gnn_probe(probe_date)
    elif args.event:
        fetch_event(args.event, pressure_levels=args.pressure_levels)
    elif args.climatology:
        y0, y1 = args.years.split("-")
        years = range(int(y0), int(y1) + 1)
        regions = list(REGIONS) if args.all_regions else [args.region]
        for region in regions:
            fetch_climatology(
                region, years=years, month=args.month, days=args.days,
                pressure_levels=args.pressure_levels,
            )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
