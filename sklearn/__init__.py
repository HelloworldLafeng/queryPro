"""Minimal local sklearn stub for Transformers optional imports.

This workspace shadows a broken global scikit-learn installation that pulls in
an incompatible SciPy build. Transformers only probes ``sklearn.metrics`` for
assisted generation utilities, which are not used in this experiment.
"""

from . import metrics

__all__ = ["metrics"]
