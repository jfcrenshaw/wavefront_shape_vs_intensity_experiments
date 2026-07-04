"""Shared library for the shape-vs-intensity wavefront experiments.

Holds the pieces every study and notebook reuses:

* :mod:`shape_vs_intensity.config` -- physical constants and experiment knobs.
* :mod:`shape_vs_intensity.sim` -- the danish-based simulate-and-fit engine.
* :mod:`shape_vs_intensity.plotutils` -- shared figure helpers.

Install editable (``pip install -e .``) so the study scripts in ``scripts/``
and the notebooks import these from anywhere, with no ``sys.path`` surgery.

The shared figure style lives in :func:`shape_vs_intensity.plotutils.use_style`.
"""
