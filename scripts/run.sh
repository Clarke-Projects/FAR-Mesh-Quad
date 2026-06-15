#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT" || exit 1
source venv_sysqt314/bin/activate

unset VTK_DEFAULT_OPENGL_WINDOW
unset VTK_USE_EGL
unset VTK_RENDER_WINDOW_TYPE

export QT_QPA_PLATFORM=xcb

python -m far_mesh.main
