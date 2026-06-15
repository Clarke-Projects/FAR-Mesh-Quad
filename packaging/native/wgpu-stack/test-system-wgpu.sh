#!/usr/bin/env bash
set -euo pipefail

/usr/bin/python - <<PY
mods = ["PySide6", "rendercanvas", "wgpu", "pylinalg", "pygfx"]
for m in mods:
    __import__(m)
    print("OK", m)

from PySide6 import QtWidgets
from rendercanvas.qt import QRenderWidget
print("rendercanvas Qt + WGPU stack OK under system Python")
PY
