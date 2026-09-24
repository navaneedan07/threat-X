"""Learned downscaling: CNN super-resolution and (if feasible) conditional diffusion.

The deterministic interpolation baseline stays in `src/downscaling/baseline.py`.
Whatever model lives here must be scored against that baseline on
extreme-preservation metrics, not just visual similarity.
"""

from __future__ import annotations

__all__: list[str] = []
