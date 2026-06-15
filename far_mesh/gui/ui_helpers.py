from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import QToolButton, QWidget

FAR_DEEP_BLACK = "#0A0A0A"
FAR_PANEL_BLACK = "#121720"
FAR_PANEL_BLUE_BLACK = "#151D2A"
FAR_PANEL_SOFT = "#1B2432"
FAR_BORDER = "#26384D"
FAR_ELECTRIC_BLUE = "#0088FF"
FAR_CYAN_GLOW = "#00E6FF"
FAR_SIGNAL_ORANGE = "#FF6A00"
FAR_WARM_AMBER = "#FFB000"
FAR_TEXT = "#E8F7FF"
FAR_TEXT_MUTED = "#93A9BD"


@dataclass(slots=True)
class ToolPage:
    key: str
    title: str
    button: QToolButton
    widget: QWidget


__all__ = [
    "FAR_DEEP_BLACK",
    "FAR_PANEL_BLACK",
    "FAR_PANEL_BLUE_BLACK",
    "FAR_PANEL_SOFT",
    "FAR_BORDER",
    "FAR_ELECTRIC_BLUE",
    "FAR_CYAN_GLOW",
    "FAR_SIGNAL_ORANGE",
    "FAR_WARM_AMBER",
    "FAR_TEXT",
    "FAR_TEXT_MUTED",
    "ToolPage",
]
