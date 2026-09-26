"""
Loader for REAL IBTrACS cyclone best tracks.

IBTrACS (International Best Track Archive for Climate Stewardship, NOAA NCEI
v04r01) is the reference ground truth for cyclone position and intensity. It is
the documented source in `docs/dataset_sources.md` (5a) and is what the tracking
stage is scored against -- not a synthetic track.

Download (no account required):

    curl -L -o data/raw/ibtracs.NI.list.v04r01.csv \
      https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NI.list.v04r01.csv

The "NI" file is the North Indian Ocean basin (covers Cyclone Amphan). Other
basins use the same CSV layout.

Format note
-----------
The IBTrACS CSV has TWO header lines: the column names, then a units row. The
units row must be skipped or every column is read as text.

USAGE
-----
    from src.data.ibtracs import load_ibtracs, get_track

    df = load_ibtracs("data/raw/ibtracs.NI.list.v04r01.csv")
    amphan = get_track(df, name="AMPHAN", season=2020)
    print(amphan[["iso_time", "lat", "lon", "usa_wind", "usa_pres"]].head())
"""

from __future__ import annotations

import argparse

import pandas as pd

# Columns we keep, and their tidy names. USA_* holds the JTWC intensity record
# for the North Indian Ocean (WMO_* is frequently blank there).
_COLUMNS = {
    "SID": "sid",
    "SEASON": "season",
    "NAME": "name",
    "ISO_TIME": "iso_time",
    "NATURE": "nature",
    "LAT": "lat",
    "LON": "lon",
    "USA_WIND": "usa_wind",
    "USA_PRES": "usa_pres",
    "USA_STATUS": "usa_status",
    "WMO_WIND": "wmo_wind",
    "WMO_PRES": "wmo_pres",
    "BASIN": "basin",
    "DIST2LAND": "dist2land_km",
    "LANDFALL": "landfall_km",
}


def load_ibtracs(path: str) -> pd.DataFrame:
    """Load an IBTrACS CSV into a tidy DataFrame.

    Skips the units row, keeps the useful columns, coerces numerics and parses
    ISO timestamps. Rows are kept even when intensity is missing, because some
    North Indian Ocean records only have position on early fixes -- dropping
    them would silently shorten the reference track.
    """
    # Row index 1 after the header is the units row -> skiprows=[1].
    raw = pd.read_csv(path, skiprows=[1], low_memory=False)

    present = {k: v for k, v in _COLUMNS.items() if k in raw.columns}
    df = raw[list(present)].rename(columns=present).copy()

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    df["name"] = df["name"].str.upper()
    if "season" in df:
        df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    for col in ("lat", "lon", "usa_wind", "usa_pres", "wmo_wind", "wmo_pres",
                "dist2land_km", "landfall_km"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "iso_time" in df:
        df["iso_time"] = pd.to_datetime(df["iso_time"], errors="coerce")

    return df.reset_index(drop=True)


def get_track(
    df: pd.DataFrame,
    name: str | None = None,
    season: int | None = None,
    sid: str | None = None,
) -> pd.DataFrame:
    """Return one storm's track, ordered by time.

    Identify the storm by ``sid`` (exact) or by ``name`` (and optionally
    ``season``, since storm names repeat across years).
    """
    mask = pd.Series(True, index=df.index)
    if sid is not None:
        mask &= df["sid"] == sid
    if name is not None:
        mask &= df["name"] == name.upper().strip()
    if season is not None and "season" in df:
        mask &= df["season"] == season

    track = df.loc[mask].copy()
    if "iso_time" in track:
        track = track.sort_values("iso_time")
    return track.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect REAL IBTrACS tracks.")
    parser.add_argument("--path", default="data/raw/ibtracs.NI.list.v04r01.csv")
    parser.add_argument("--name", default=None, help="Storm name, e.g. AMPHAN.")
    parser.add_argument("--season", type=int, default=None, help="Season year.")
    args = parser.parse_args()

    df = load_ibtracs(args.path)
    print(f"Loaded {len(df):,} IBTrACS rows from {args.path}")
    print(f"  seasons: {df['season'].min()}..{df['season'].max()}")
    print(f"  distinct storms: {df['sid'].nunique():,}")

    if args.name:
        track = get_track(df, name=args.name, season=args.season)
        if track.empty:
            print(f"No track found for {args.name} {args.season or ''}")
            return
        peak = track["usa_wind"].max()
        print(f"\n{args.name.upper()} {args.season or ''}: {len(track)} fixes, "
              f"peak USA wind = {peak} kt")
        cols = [c for c in ("iso_time", "lat", "lon", "usa_wind", "usa_pres")
                if c in track]
        print(track[cols].head(6).to_string(index=False))
        print("..." if len(track) > 6 else "")


if __name__ == "__main__":
    main()
