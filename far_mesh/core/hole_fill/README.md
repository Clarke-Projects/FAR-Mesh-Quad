# Refined adaptive hole-fill core

This package is the new home for the cleaned-up adaptive hole-fill solver.

## Why this package exists

The first H-ADAPT implementation proved the C0/G1/G2/fairing/end-layer pipeline,
but the research code grew across the flat `far_mesh/core/hole_*` modules.

The next phase separates responsibilities so the older files can eventually be
retired cleanly.

## Migration rule

Do not delete or rewrite the old flat modules yet.

1. Add refined modules here.
2. Keep old adaptive_surface behavior as fallback.
3. Port one responsibility at a time.
4. Validate with compileall, collect-only, full tests, and smoke tests.
5. Switch the public adaptive_surface route only after validation.

## Planned modules

- `seed_surface.py` — seed/support surface alignment diagnostics and later confidence-weighted seed target.
- `seed_uvdelaunay.py` — UV frame, interior sampling, Delaunay topology.
- `surface_target.py` — MLS, sphere, plane/PCA fallback, confidence-weighted target blend.
- `adaptive_controller.py` — seed, relaxation, fairing, diagnostics, and selection orchestration.
- `adaptive_policy.py` — C0/G1/G2/fairing/end-layer gates.
- `diagnostics.py` — unified seam, quality, G1, G2, dimple/deviation reports.
- `metadata.py` — stable metadata keys and conversion helpers.

## Current priority

H-CORE-R1 should add `seed_surface.py` diagnostics first.

The persistent dimple likely comes from the base seed / support target, not from
post-process smoothing.
