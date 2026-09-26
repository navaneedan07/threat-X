"""
Fetch REAL medium-range forecast data from ECMWF Open Data.

Why this module exists
----------------------
The SIH problem is about *medium-range forecasts*, but the repo's event data is
ERA5 reanalysis. Reanalysis is not a forecast: it has no lead time, so any
"lead time" or "transition window" computed from it is meaningless. ECMWF Open
Data provides the real IFS forecast (IFS HRES at 0.25 deg), free and without an
account, which is the documented source in `docs/dataset_sources.md` (3a).

Nothing here is synthetic. If a retrieval fails, it fails loudly.

Access notes
------------
* No credentials required. Open data is CC BY 4.0 -- attribute ECMWF.
* Only the most recent runs are mirrored (a rolling window of a few days), so
  this is real-time input, not an archive. For historical forecasts, use a
  MARS/ECDS or NCMRWF/IMD request.
* The `area` keyword is NOT supported by open data; files arrive global. We
  subset to the project region locally in `load_open_data`.

USAGE
-----
    # Surface forecast, default steps 0..240 h every 24 h
    python -m src.data.open_data_fetch

    # Include pressure-level fields for the precursor module
    python -m src.data.open_data_fetch --pressure-levels

    # A specific run, denser steps
    python -m src.data.open_data_fetch --date 2026-09-25 --time 0 \
        --steps 0,6,12,18,24,48,72

Outputs
-------
    data/raw/ecmwf_oper_<YYYYMMDD><HH>_sfc.grib2
    data/raw/ecmwf_oper_<YYYYMMDD><HH>_pl.grib2
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

import xarray as xr

try:
    from ecmwf.opendata import Client
except ImportError as exc:  # pragma: no cover - environment guard
    print("ERROR: ecmwf-opendata is not installed.")
    print("Run:  pip install ecmwf-opendata cfgrib eccodes")
    raise exc


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Request defaults
# ---------------------------------------------------------------------
# Surface parameters (ECMWF short names).
SURFACE_PARAMS = ["2t", "msl", "tp", "10u", "10v"]

# Pressure-level parameters and levels. Needed by the precursor module for
# vorticity / moisture-flux / shear, which is why surface-only data leaves
# those precursors null.
PRESSURE_PARAMS = ["t", "q", "u", "v"]
PRESSURE_LEVELS = [500, 850]

# Lead times in hours. 0-240 h = the medium range (up to 10 days).
DEFAULT_STEPS = [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240]

# Local region subset (India domain spanning both event areas), as
# (north, south, west, east) matching the CDS `area` convention.
INDIA_AREA = (40, 5, 65, 100)


# ---------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------
def _index_url(base: str, date: str, hour: int) -> str:
    stamp = f"{date.replace('-', '')}{hour:02d}"
    return (f"{base.rstrip('/')}/forecasts/{stamp[:8]}/{hour:02d}z/ifs/0p25/oper/"
            f"{stamp}0000-0h-oper-fc.index")


def _latest_run(base: str = "https://data.ecmwf.int", max_lookback: int = 4) -> tuple[str, int]:
    """Return the most recent (date, hour) run that is actually published.

    Uses a cheap HTTP HEAD on the run index rather than a retrieval, so it does
    not consume download quota or trip the open-data rate limiter. Open data
    lags real time by a few hours, so the newest calendar day is often not
    published yet.
    """
    import requests

    today = dt.date.today()
    for back in range(max_lookback + 1):
        d = today - dt.timedelta(days=back)
        for hour in (12, 6, 0):
            url = _index_url(base, d.strftime("%Y-%m-%d"), hour)
            try:
                if requests.head(url, timeout=15).status_code == 200:
                    return d.strftime("%Y-%m-%d"), hour
            except Exception:
                continue
    raise RuntimeError(
        "Could not find an available ECMWF open-data run in the last "
        f"{max_lookback + 1} days. Check network access or the open-data status."
    )


def fetch_open_data(
    date: str | None = None,
    time: int | None = None,
    steps: list[int] | None = None,
    pressure_levels: bool = False,
    source: str = "ecmwf",
) -> dict[str, str]:
    """Download one real IFS run and return the written file paths."""
    steps = list(steps) if steps is not None else list(DEFAULT_STEPS)
    client = Client(source=source)
    # The public mirrors occasionally return 429/503 under load and default to a
    # 120 s server-requested backoff. Cap retries and shorten the wait so a
    # transient throttle does not stall the pipeline for minutes.
    client.maximum_retries = 8
    client.retry_after = 30

    if date is None or time is None:
        date, time = _latest_run()

    stamp = f"{date.replace('-', '')}{time:02d}"
    written: dict[str, str] = {}

    print()
    print("=" * 70)
    print("ECMWF OPEN DATA FORECAST DOWNLOAD")
    print("=" * 70)
    print(f"Source    : {source} (IFS HRES, 0.25 deg)")
    print(f"Run       : {date} {time:02d}Z")
    print(f"Steps     : {steps} h")
    print(f"Surface   : {len(SURFACE_PARAMS)} params")
    print(f"Pressure  : {PRESSURE_LEVELS if pressure_levels else 'not requested'}")
    print("=" * 70)

    sfc_target = os.path.join(RAW_DIR, f"ecmwf_oper_{stamp}_sfc.grib2")
    try:
        client.retrieve(
            type="fc", stream="oper", param=SURFACE_PARAMS,
            date=date, time=time, step=steps, target=sfc_target,
        )
        written["surface"] = sfc_target
        n = _count_messages(sfc_target)
        print(f"[done] {os.path.basename(sfc_target)}  ({n} fields, "
              f"{os.path.getsize(sfc_target) / 1e6:.1f} MB)")
    except Exception as exc:
        print(f"ERROR: surface forecast retrieval failed: {exc}")
        raise

    if pressure_levels:
        pl_target = os.path.join(RAW_DIR, f"ecmwf_oper_{stamp}_pl.grib2")
        try:
            client.retrieve(
                type="fc", stream="oper", levtype="pl",
                levelist=PRESSURE_LEVELS, param=PRESSURE_PARAMS,
                date=date, time=time, step=steps, target=pl_target,
            )
            written["pressure"] = pl_target
            n = _count_messages(pl_target)
            print(f"[done] {os.path.basename(pl_target)}  ({n} fields, "
                  f"{os.path.getsize(pl_target) / 1e6:.1f} MB)")
        except Exception as exc:
            print(f"ERROR: pressure-level retrieval failed: {exc}")
            raise

    return written


def _count_messages(path: str) -> int:
    """Count GRIB messages without decoding the data."""
    import eccodes

    n = 0
    with open(path, "rb") as fh:
        while True:
            gid = eccodes.codes_grib_new_from_file(fh)
            if gid is None:
                break
            n += 1
            eccodes.codes_release(gid)
    return n


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------
def load_open_data(
    path: str,
    area: tuple[float, float, float, float] = INDIA_AREA,
    *,
    level: int | None = None,
) -> xr.Dataset:
    """Open a downloaded GRIB2 file as a clean, optionally subset Dataset.

    Parameters
    ----------
    path : str
        Path to the GRIB2 file written by :func:`fetch_open_data`.
    area : (north, south, west, east)
        Region subset. Pass ``None`` to keep the global field.
    level : int, optional
        For pressure-level files, select a single pressure level in hPa.

    Notes
    -----
    Notes
    -----
    * ``decode_timedelta=False`` is required: current xarray raises
      ``AssertionError`` in ``decode_cf_timedelta`` for cfgrib's non-nanosecond
      step coordinate. With it off, ``step`` stays a plain float array.
    * A single GRIB file can mix level types (2 m temperature and 10 m wind both
      live at ``heightAboveGround``), which makes a plain ``open_dataset`` fail
      with ``DatasetBuildError``. ``cfgrib.open_datasets`` splits the file into
      consistent groups; we drop the scalar level coordinates that only differ
      between groups and merge the rest.
    """
    import cfgrib

    groups = cfgrib.open_datasets(
        path,
        backend_kwargs={"indexpath": "", "decode_timedelta": False},
    )

    cleaned: list[xr.Dataset] = []
    for group in groups:
        # Scalar coordinates that conflict across level types (e.g.
        # heightAboveGround=2 for t2m vs 10 for u10) carry no information the
        # variable name does not already encode, so drop them before merging.
        scalar_coords = [
            name for name, var in group.coords.items()
            if var.ndim == 0 and name != "time"
        ]
        group = group.drop_vars(scalar_coords) if scalar_coords else group

        # cfgrib uses specific level type names depending on the file.
        if level is not None:
            for coord in ("isobaricInhPa", "level", "plev"):
                if coord in group.coords:
                    group = group.sel({coord: level})
                    break
        cleaned.append(group)

    ds = xr.merge(cleaned, compat="override") if len(cleaned) > 1 else cleaned[0]

    if area is not None:
        north, south, west, east = area
        ds = ds.sel(
            latitude=slice(north, south),
            longitude=slice(west, east),
        )
    return ds


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def _parse_steps(spec: str) -> list[int]:
    return [int(s) for s in spec.split(",") if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download REAL ECMWF open-data medium-range forecasts."
    )
    parser.add_argument("--date", default=None, help="Run date, e.g. 2026-09-25.")
    parser.add_argument("--time", type=int, default=None, choices=[0, 6, 12, 18],
                        help="Run hour (UTC).")
    parser.add_argument("--steps", default=None,
                        help="Comma-separated lead times in hours, e.g. 0,24,48.")
    parser.add_argument("--pressure-levels", action="store_true",
                        help="Also fetch pressure-level fields.")
    parser.add_argument("--source", default="ecmwf",
                        help="Open-data mirror: ecmwf | aws | azure | google.")
    args = parser.parse_args()

    steps = _parse_steps(args.steps) if args.steps else None
    written = fetch_open_data(
        date=args.date, time=args.time, steps=steps,
        pressure_levels=args.pressure_levels, source=args.source,
    )

    print()
    print("=" * 70)
    print("FORECAST DOWNLOAD COMPLETE")
    print("=" * 70)
    for kind, path in written.items():
        print(f"  {kind:9s}: {path}")
    print()
    print("Verify with:")
    print("  python -c \"from src.data.open_data_fetch import load_open_data; "
          "print(load_open_data(r'" + written["surface"] + "'))\"")
    print()
    sys.stdout.flush()


if __name__ == "__main__":
    main()
