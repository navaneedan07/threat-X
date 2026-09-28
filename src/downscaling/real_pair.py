"""Real-data coarse/fine downscaling experiment.

Builds the pair the downscaling stage was blocked on:

* **coarse** — ERA5 reanalysis, 0.25 deg (~28 km), ``data/raw/era5_<event>.nc``
* **fine**   — ERA5-Land reanalysis, 0.1 deg (~11 km), ``data/raw/era5_land_<event>.nc``

Same 3-hourly window, same domain, so the interpolation baseline can be scored
against a finer field that actually exists.

What this is, and is not
------------------------
ERA5-Land is a *different model run* of the same forcing at higher resolution — a
legitimate fine reference, **not** a truth field, and **not** the problem
statement's 5 km target. The honest claim is "the baseline is scored on a real
0.25 deg -> 0.1 deg pair (factor 2.5)", nothing stronger.

ERA5-Land is **land-only**: ocean cells are NaN. That matters. For a cyclone whose
extreme sits over the Bay of Bengal, the fine field does not even cover the peak,
and scoring it would compare an oceanic coarse peak against a land-only reference
peak — a misleading number dressed up as a result. So the experiment refuses an
event whose coarse peak falls outside the reference's finite footprint, and says
why. It is a coverage check, not a performance threshold.

The time step scored is the one where the fine field's spatial peak is largest, so
the comparison is about the event's extreme rather than an arbitrary frame. That
choice is recorded in the output.

Writes ``data/processed/validation/downscaling/<event>_<variable>.json`` and a
figure, and prints the metric table. Requires the local raw files; it does not
download and never falls back to synthetic data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.downscaling.baseline import downscale_to_reference, resample
from src.downscaling.metrics import evaluate_downscaling
from src.shared.fields import GriddedField
from src.shared.metrics import MetricTable

__all__ = ["EVENTS", "build_pair", "evaluate_event", "main", "peak_covered"]

# event -> (variable, units, scale-to-canonical-units)
EVENTS: dict[str, tuple[str, str, float]] = {
    # tp arrives in metres of accumulated precipitation; the project reports mm.
    "amphan": ("tp", "mm/3h", 1000.0),
    "heatwave": ("t2m", "K", 1.0),
}

COARSE_FILES = {
    "amphan": "data/raw/era5_amphan.nc",
    "heatwave": "data/raw/era5_heatwave.nc",
}
FINE_FILES = {
    "amphan": "data/raw/era5_land_amphan.nc",
    "heatwave": "data/raw/era5_land_heatwave.nc",
}


def _as_field(data, variable: str, units: str, scale: float) -> GriddedField:
    """One time step of a variable as a :class:`GriddedField` in canonical units."""
    field = GriddedField.from_dataarray(data[variable], name=variable)
    return GriddedField(
        values=field.values * scale,
        latitude=field.latitude,
        longitude=field.longitude,
        name=variable,
        units=units,
        attrs={"source": "era5-family"},
    )


def _peak_step(dataset, variable: str) -> int:
    """Index of the time step with the largest spatial maximum.

    Deterministic and reported in the output, so the frame being scored is always
    reproducible rather than picked by eye.
    """
    values = dataset[variable].values
    if values.ndim == 2:
        return 0
    per_time = np.nanmax(values.reshape(values.shape[0], -1), axis=1)
    return int(np.nanargmax(per_time))


def peak_covered(predicted: GriddedField, reference: GriddedField) -> bool:
    """Whether the reference has a finite value where the prediction peaks.

    If it does not, the extreme that has to be preserved is not covered by the
    reference at all, so no peak-preservation number can be computed from the pair.
    """
    left = predicted.ascending()
    right = reference.ascending()
    row, column = np.unravel_index(int(np.nanargmax(left.values)), left.values.shape)
    latitude, longitude = float(left.latitude[row]), float(left.longitude[column])
    nearest_row = int(np.argmin(np.abs(right.latitude - latitude)))
    nearest_column = int(np.argmin(np.abs(right.longitude - longitude)))
    return bool(np.isfinite(right.values[nearest_row, nearest_column]))


def _mask_to_reference(predicted: GriddedField, reference: GriddedField) -> GriddedField:
    """Blank cells the reference does not cover, so both peak over the same area.

    Without this the coarse peak (over land *and* ocean) would be compared against a
    land-only reference peak, inflating the apparent loss. Both fields are returned
    ascending so the shapes line up.
    """
    left = predicted.ascending()
    right = reference.ascending()
    values = np.where(np.isfinite(right.values), left.values, np.nan)
    return GriddedField(
        values=values,
        latitude=right.latitude,
        longitude=right.longitude,
        name=left.name,
        units=left.units,
        attrs=dict(left.attrs),
    )


def build_pair(event: str) -> dict[str, Any]:
    """Load the coarse/fine pair for one event and report the grids."""
    from src.data.loader import load_dataset, spatial_resolution_deg

    for path in (COARSE_FILES[event], FINE_FILES[event]):
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{path} missing. Fetch it first: "
                "python -m src.data.cds_fetch --event <event> and "
                "python -m src.data.land_fetch --event <event>"
            )

    variable, units, scale = EVENTS[event]
    coarse_ds = load_dataset(COARSE_FILES[event])
    fine_ds = load_dataset(FINE_FILES[event])

    step = _peak_step(fine_ds, variable)
    coarse = _as_field(coarse_ds, variable, units, scale)
    reference = _as_field(fine_ds, variable, units, scale)

    coarse_time = str(coarse_ds["valid_time"].values[step])[:19]
    fine_time = str(fine_ds["valid_time"].values[step])[:19]
    if coarse_time != fine_time:
        raise ValueError(
            f"{event}: coarse time {coarse_time} does not match fine time {fine_time}"
        )

    return {
        "event": event,
        "variable": variable,
        "units": units,
        "time_index": step,
        "valid_time": fine_time,
        "coarse": coarse,
        "reference": reference,
        "coarse_resolution_deg": spatial_resolution_deg(coarse_ds),
        "fine_resolution_deg": spatial_resolution_deg(fine_ds),
    }


def _decimated(reference: GriddedField, factor: int) -> GriddedField:
    """Sample the fine field every ``factor`` cells, then refine it back up.

    Shows how the metric decays as the peak gets harder to resolve, on the same
    real field rather than a synthetic bump.
    """
    coarse = resample(
        reference,
        reference.latitude[::factor],
        reference.longitude[::factor],
        order=1,
    )
    return downscale_to_reference(coarse, reference, order=1)


def evaluate_event(event: str) -> dict[str, Any]:
    """Score the baseline on the real pair, or explain why the pair is unscorable."""
    pair = build_pair(event)
    coarse, reference = pair["coarse"], pair["reference"]

    if not peak_covered(coarse, reference):
        return {
            "event": event,
            "variable": pair["variable"],
            "status": "not_scoreable",
            "reason": (
                "the coarse field peaks where the reference is NaN (ERA5-Land is "
                "land-only), so the extreme to be preserved is outside the reference's "
                "footprint and no peak-preservation number can be computed"
            ),
            "reference_nan_fraction": round(reference.nan_fraction, 4),
        }

    reference = reference.ascending()
    refined = _mask_to_reference(
        downscale_to_reference(coarse, reference, order=1), reference
    )
    table = evaluate_downscaling(
        refined,
        reference,
        context={
            "event": event,
            "valid_time": pair["valid_time"],
            "coarse_resolution_deg": pair["coarse_resolution_deg"],
            "fine_resolution_deg": pair["fine_resolution_deg"],
            "upscale_factor": round(float(refined.attrs["upscale_factor"][0]), 4),
            "coarse_source": "ERA5",
            "reference_source": "ERA5-Land (different run, not truth; land-only)",
            "reference_nan_fraction": round(reference.nan_fraction, 4),
            "masked_to_reference_footprint": True,
        },
    )

    sweep: dict[str, float | None] = {}
    for factor in (2, 4, 8):
        degraded = _decimated(reference, factor)
        sweep[f"upscale_factor_x{factor}"] = round(
            float(degraded.attrs["upscale_factor"][0]), 3
        )
        sweep[f"peak_preservation_x{factor}"] = evaluate_downscaling(
            degraded, reference
        ).get("peak_preservation")

    return {
        "event": event,
        "variable": pair["variable"],
        "units": pair["units"],
        "status": "scored",
        "table": table,
        "sweep": sweep,
        "coarse": coarse,
        "refined": refined,
        "reference": reference,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score the interpolation baseline on the real ERA5 / ERA5-Land pair."
    )
    parser.add_argument("--event", choices=list(EVENTS), help="One event.")
    parser.add_argument("--all-events", action="store_true", help="Every event.")
    parser.add_argument("--output-dir", default="data/processed/validation/downscaling")
    parser.add_argument("--plots-dir", default="data/processed/plots/downscaling")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    events = list(EVENTS) if args.all_events else ([args.event] if args.event else [])
    if not events:
        parser.print_help()
        return 2

    out = Path(args.output_dir)
    for event in events:
        result = evaluate_event(event)
        print()
        print("=" * 70)
        print(f"{event}  {result['variable']}")
        print("=" * 70)

        if result["status"] != "scored":
            print(f"NOT SCOREABLE — {result['reason']}")
            print(f"reference NaN fraction: {result['reference_nan_fraction']:.3f}")
            artifact = out / f"{event}_{result['variable']}.json"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(f"[written] {artifact}")
            continue

        table: MetricTable = result["table"]
        print(
            f"coarse {table.context['coarse_resolution_deg']:g} deg -> "
            f"fine {table.context['fine_resolution_deg']:g} deg  "
            f"(factor {table.context['upscale_factor']:g})  [{table.units}]"
        )
        print(f"frame: {table.context['valid_time']}")
        print(table.to_markdown())
        print("\ndecay sweep on the fine field:")
        for factor in (2, 4, 8):
            peak = result["sweep"][f"peak_preservation_x{factor}"]
            print(
                f"  x{factor}: peak_preservation = "
                f"{'—' if peak is None else round(peak, 4)}"
            )

        artifact = out / f"{event}_{result['variable']}.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(
            json.dumps(
                {
                    "event": event,
                    "variable": result["variable"],
                    "status": "scored",
                    "sweep": result["sweep"],
                    "metrics": table.to_dict(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[written] {artifact}")

        if not args.no_plots:
            from src.shared.visualization import plot_coarse_vs_refined

            plot_coarse_vs_refined(
                result["coarse"],
                result["refined"],
                result["reference"],
                title=(
                    f"{event} — ERA5 0.25 deg -> ERA5-Land 0.1 deg "
                    f"(real pair, factor 2.5)"
                ),
                save_to=Path(args.plots_dir) / f"real_{event}_coarse_vs_refined.png",
            )
    return 0


if __name__ == "__main__":  # pragma: no cover - needs the local raw archives
    raise SystemExit(main())
