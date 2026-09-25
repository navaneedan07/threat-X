"""Backend services package.

Each sub-module is a thin adapter that:
  1. Tries to load real pipeline output (when the upstream module produces it).
  2. Falls back to data/samples/ fixtures if the pipeline has not run yet.

This allows the API to serve coherent responses in degraded mode during the
demo and while upstream team members are still integrating their modules.

Integration points are clearly marked with # TODO: INTEGRATE comments so
team members know exactly where to connect their outputs.
"""
