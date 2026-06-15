# Hole-fill core migration plan

## H-CORE-R0A

Create this package scaffold.

Geometry change: no.

## H-CORE-R1

Add seed-surface diagnostics.

Geometry change: no.

Expected outputs:
- seed surface kind
- seed backend
- sphere fit RMS / max error
- MLS projection mean / max distance
- signed seed offset mean / min / max
- effective surface weight
- seed alignment status
- seed alignment reasons

## H-CORE-R2

Move seed/support diagnostic helpers out of `hole_fill_preview.py`.

Geometry change: no.

## H-CORE-R3

Create `adaptive_controller.py` and move orchestration out of `hole_fill_preview.py`.

Geometry change: no intended selected-preview change.

## H-CORE-R4

Create `adaptive_policy.py` and move gates/scoring out of GUI and preview routing code where possible.

Geometry change: no.

## H-CORE-R5

Implement confidence-weighted seed surface target.

Geometry change: yes.

## H-CORE-R6

Compare old seed and new seed across smoke meshes.

Geometry change: route selection only after validation.

## H-CORE-R7

Retire experimental end-layer dimple hacks that become unnecessary.

## H-CORE-R8

Delete or archive old flat `hole_*` files that have been fully migrated.
