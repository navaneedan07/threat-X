"""Learned models -- machine learning and deep learning.

This package holds ONLY models that are fitted or trained. Deterministic
processing (interpolation, thresholding, clustering, geospatial math) belongs in
the sibling stage packages, not here.

Keeping the two apart is deliberate: the README requires the interpolation
baseline to stay available and to be compared against any learned model, so the
baseline must not live inside the package it is being benchmarked against.

Trained artefacts are NOT stored here -- they live in the top-level `weights/`
directory. Code is source, weights are data.
"""
