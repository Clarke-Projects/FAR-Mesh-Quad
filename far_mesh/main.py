from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from far_mesh.gui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("FAR MESH Quad Native")

    # KDE / desktop shell identity. Safe: does not modify process argv
    # and does not affect Python multiprocessing / Phase 6 workers.
    for method_name, value in (
        ("setApplicationDisplayName", "FAR MESH Quad Native"),
        ("setOrganizationName", "FAR MESH"),
        ("setOrganizationDomain", "far-mesh.local"),
        ("setDesktopFileName", "far-mesh-quad-native"),
    ):
        setter = getattr(app, method_name, None)
        if setter is not None:
            try:
                setter(value)
            except Exception:
                pass

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
