from __future__ import annotations

import traceback
from typing import Callable

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QWidget

from far_mesh.core.bore.exceptions import BoreError


class WorkerThread(QThread):
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, task: Callable[[], object], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._task = task

    def run(self) -> None:
        try:
            result = self._task()
        except Exception as exc:
            message = str(exc)
            controlled_preview_rejection = (
                isinstance(exc, BoreError)
                or "surface-context safety gate" in message
                or "Unsafe surface_uvdelaunay_relaxed preview rejected" in message
                or "Bore measured-patch quad rebuild could not find a watertight measured delete patch" in message
                or "Bore wall preview is not rebuild-ready" in message
                or "Bore wall rebuild is disabled for measured fragmented-rim previews" in message
                or "Promoted BOREHOLE quad rebuild blocked" in message
                or "Geometry changed: no" in message
            )

            if controlled_preview_rejection:
                self.failed.emit(message)
            else:
                self.failed.emit(traceback.format_exc())
            return

        self.succeeded.emit(result)
