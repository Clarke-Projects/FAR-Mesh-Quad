"""
Refined adaptive hole-fill core package.

This package is the clean rewrite namespace for the post-H-ADAPT solver.

Current rule:
- keep existing flat far_mesh.core.hole_* modules stable
- add new refined modules here
- migrate one responsibility at a time
- switch public adaptive_surface only after tests and smoke validation

No geometry code is executed at import time.
"""

from __future__ import annotations

__all__: tuple[str, ...] = ("surface_target",)
