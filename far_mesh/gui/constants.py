from __future__ import annotations

from pathlib import Path

CAD_PRESETS: dict[str, dict[str, float | bool] | None] = {
    "Custom": None,
    "CAD High Quality": {
        "quadwild_sharp": 35.0,
        "quadwild_alpha": 0.02,
        "quadwild_scale": 2.0,
        "auto_reduce": True,
        "auto_reduce_target": 80000,
        "auto_reduce_boundary_weight": 8.0,
    },
    "CAD Balanced": {
        "quadwild_sharp": 35.0,
        "quadwild_alpha": 0.02,
        "quadwild_scale": 4.0,
        "auto_reduce": True,
        "auto_reduce_target": 50000,
        "auto_reduce_boundary_weight": 6.0,
    },
    "CAD Lightweight": {
        "quadwild_sharp": 35.0,
        "quadwild_alpha": 0.02,
        "quadwild_scale": 5.5,
        "auto_reduce": True,
        "auto_reduce_target": 25000,
        "auto_reduce_boundary_weight": 5.0,
    },
}

DISPLAY_PRESET_LABELS: dict[str, str] = {
    "inspection_edges": "Inspection Edges",
    "viewer_clean": "Viewer Clean",
    "repair_selection": "Repair Selection",
    "shaded_only": "Shaded Only",
    "shaded + wireframe": "Shaded + Wireframe",
    "wireframe": "Wireframe",
}

COMPARE_MODE_LABELS: dict[str, str] = {
    "current_only": "Current Only",
    "original_only": "Original Only",
    "overlay_ghost": "Ghost Overlay",
}

SELECTION_MODE_LABELS: dict[str, str] = {
    "none": "None",
    "point": "Point",
    "face": "Face",
    "edge": "Edge",
    "mesh": "Mesh",
}

CAMERA_PRESET_LABELS: dict[str, str] = {
    "isometric": "Isometric",
    "front": "Front",
    "back": "Back",
    "left": "Left",
    "right": "Right",
    "top": "Top",
    "bottom": "Bottom",
}

REPAIR_ADVANCED_METHODS: set[str] = {
    "cad_safe",
    "cad_preserve_features",
    "cad_safe_pymeshlab",
    "pymeshlab",
    "topology_cleanup",
    "hybrid",
}

REPAIR_STRICT_PRESERVE_METHODS: set[str] = {
    "cad_preserve_features",
}

GUI_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
FAR_MESH_APP_ICON = GUI_ASSETS_DIR / "icons" / "Icon_FAR_Mesh_Quad.png"
FAR_MESH_LOGO = GUI_ASSETS_DIR / "logos" / "FARMeshQuadLogo_v8.png"
FAR_MESH_DESIGN_REFERENCE = GUI_ASSETS_DIR / "design" / "design.png"

__all__ = [
    "CAD_PRESETS",
    "DISPLAY_PRESET_LABELS",
    "COMPARE_MODE_LABELS",
    "SELECTION_MODE_LABELS",
    "CAMERA_PRESET_LABELS",
    "REPAIR_ADVANCED_METHODS",
    "REPAIR_STRICT_PRESERVE_METHODS",
    "GUI_ASSETS_DIR",
    "FAR_MESH_APP_ICON",
    "FAR_MESH_LOGO",
    "FAR_MESH_DESIGN_REFERENCE",
]
