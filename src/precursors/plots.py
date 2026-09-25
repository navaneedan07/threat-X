"""Precursor diagnostic plots.

Generates time-series plots of atmospheric precursor signals for a given
threat event. All plots are derived from actual computed values — no values
are fabricated.

Usage:
    from src.precursors.plots import plot_precursor_series
    fig = plot_precursor_series(series, event_name="Cyclone Amphan", save_to="...")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no display required)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from src.precursors.engine import PrecursorResult

logger = logging.getLogger(__name__)


def _to_datetimes(series: list[PrecursorResult]) -> list:
    """Convert ISO timestamp strings to Python datetime objects."""
    from datetime import datetime, timezone
    dts = []
    for r in series:
        ts = r.timestamp.rstrip("Z")
        dt = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
        dts.append(dt)
    return dts


def plot_precursor_series(
    series: list[PrecursorResult],
    event_name: str = "Event",
    centroid_lat: float = 0.0,
    centroid_lon: float = 0.0,
    save_to: Optional[str | Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Generate a multi-panel time-series figure of all available precursors.

    Parameters
    ----------
    series : list[PrecursorResult]
        Output from compute_precursor_series().
    event_name : str
        Label used in the plot title.
    centroid_lat, centroid_lon : float
        Threat centroid coordinates (used in title).
    save_to : path-like, optional
        If provided, save the figure to this path (PNG).
    show : bool
        If True, display the figure interactively.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not series:
        logger.warning("Empty series passed to plot_precursor_series — nothing to plot.")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    times = _to_datetimes(series)

    # Collect data arrays (None-safe)
    tend = [r.pressure_tendency_3h_hpa for r in series]
    div = [r.wind_divergence_s1 for r in series]
    wspd = [r.wind_speed_ms for r in series]
    precip = [r.precipitation_rate_mm3h for r in series]
    t2m_anom = [r.t2m_anomaly_k for r in series]

    # Determine which panels have data
    panels = []
    if any(v is not None for v in tend):
        panels.append(("MSLP Tendency (hPa / 3h)", tend, "navy", "hPa/3h", True))
    if any(v is not None for v in div):
        # Convert to 10^-5 s^-1 for readability
        div_scaled = [v * 1e5 if v is not None else None for v in div]
        panels.append(("10m Wind Divergence (×10⁻⁵ s⁻¹)", div_scaled, "darkorange", "×10⁻⁵ s⁻¹", True))
    if any(v is not None for v in wspd):
        panels.append(("10m Wind Speed (m/s)", wspd, "green", "m/s", False))
    if any(v is not None for v in precip):
        panels.append(("Precipitation Rate (mm/3h)", precip, "steelblue", "mm/3h", False))
    if any(v is not None for v in t2m_anom):
        panels.append(("2m Temp Anomaly (K)", t2m_anom, "crimson", "K", True))

    n_panels = max(len(panels), 1)
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(
        f"Atmospheric Precursors — {event_name}\n"
        f"Centroid: ({centroid_lat:.2f}°N, {centroid_lon:.2f}°E)  "
        f"[ERA5 surface-level data]",
        fontsize=12,
        fontweight="bold",
    )

    for ax, (title, values, colour, ylabel, zero_line) in zip(axes, panels):
        # Replace None with NaN for plotting
        y = np.array([v if v is not None else np.nan for v in values], dtype=float)
        ax.plot(times, y, color=colour, linewidth=2, marker="o", markersize=4)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(True, alpha=0.3)
        if zero_line:
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%HZ"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        # Shade NaN gaps
        ax.fill_between(times, ax.get_ylim()[0], ax.get_ylim()[1],
                        where=np.isnan(y), alpha=0.1, color="grey",
                        label="No data")

    axes[-1].set_xlabel("Time (UTC)", fontsize=9)
    fig.autofmt_xdate(rotation=30)

    # Annotation: unavailable variables
    unavail = series[0].unavailable_variables if series else []
    if unavail:
        note = "Not computed (data absent): " + ", ".join(
            [v.split("(")[0].strip() for v in unavail]
        )
        fig.text(0.5, 0.01, note, ha="center", fontsize=7, color="grey",
                 style="italic")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    if save_to:
        save_path = Path(save_to)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Precursor plot saved: %s", save_path)

    if show:
        plt.show()

    return fig
