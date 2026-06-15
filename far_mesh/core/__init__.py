# far_mesh/core/__init__.py
"""
Core processing package for Farmesh 3.

This package exposes the main processing façade and the active worker/runner
classes used by the application.

Public API:
- MeshProcessor: high-level mesh workflow façade
- MeshRepairer: repair backend service
- MeshReducer: reduction backend service
- MeshReductionResult: structured reduction result
- QuadWildBiMDFRunner: QuadWild-BiMDF remesh runner
- QuadWildBiMDFRunResult: structured QuadWild-BiMDF run result
- InstantMeshesRunner: Instant Meshes binary runner

Phase 2A topology API:
- FaceAdjacency: face-neighbour graph derived from shared mesh edges
- BoundaryLoop: extracted closed boundary loop or open boundary chain
- TopologyReport: compact topology summary for a mesh or selected face region
- build_face_adjacency: build a face adjacency graph
- extract_connected_components: extract connected face components
- find_boundary_edges: find whole-mesh or selected-region boundary edges
- extract_boundary_loops: convert boundary edges into loops/chains
- region_grow_faces: simple constrained face-region growth
- analyze_selection_topology: compact topology report helper

Phase 2B boundary-loop classification API:
- BoundaryLoopKind: semantic boundary-loop classification enum
- BoundaryLoopMeasurement: measured geometric hints for a boundary loop
- ClassifiedBoundaryLoop: boundary loop plus classification hints
- measure_boundary_loop: measure perimeter, centroid, and projected area hint
- classify_boundary_loops: classify loops as outer, hole, open chain, or selected-region boundary

Phase 2C hole-candidate API:
- HoleCandidate: read-only candidate hole boundary description
- find_hole_candidates: detect likely hole candidates from classified boundary loops

Design note:
The GUI should normally route execution through MeshProcessor, but the runner
classes are also exported because Farmesh 3 needs backend-specific metadata and
diagnostics for the special QuadWild-BiMDF fork and the bundled Instant Meshes
binary.

Phase 2 topology note:
Topology helpers are exported here as pure core utilities. They must remain
GUI-free, viewport-free, and independent of SelectionController/MainWindow.
"""

from .mesh_processor import MeshProcessor
from .mesh_repairer import MeshRepairer
from .mesh_reducer import MeshReducer, MeshReductionResult
from .quadwild_bimdf_runner import QuadWildBiMDFRunner, QuadWildBiMDFRunResult
from .remesher_wrapper import InstantMeshesRunner
from .selection_topology import (
    BoundaryLoop,
    BoundaryLoopKind,
    BoundaryLoopMeasurement,
    ClassifiedBoundaryLoop,
    FaceAdjacency,
    HoleCandidate,
    HoleCandidateDiagnostics,
    HoleCandidateKind,
    TopologyReport,
    analyze_selection_topology,
    diagnose_hole_candidate,
    diagnose_hole_candidates,
    build_face_adjacency,
    classify_boundary_loops,
    extract_boundary_loops,
    extract_connected_components,
    find_boundary_edges,
    find_hole_candidates,
    measure_boundary_loop,
    region_grow_faces,
)

__all__ = [
    "MeshProcessor",
    "MeshRepairer",
    "MeshReducer",
    "MeshReductionResult",
    "QuadWildBiMDFRunner",
    "QuadWildBiMDFRunResult",
    "InstantMeshesRunner",
    "BoundaryLoop",
    "BoundaryLoopKind",
    "BoundaryLoopMeasurement",
    "ClassifiedBoundaryLoop",
    "FaceAdjacency",
    "HoleCandidate",
    "HoleCandidateDiagnostics",
    "HoleCandidateKind",
    "TopologyReport",
    "analyze_selection_topology",
    "diagnose_hole_candidate",
    "diagnose_hole_candidates",
    "build_face_adjacency",
    "classify_boundary_loops",
    "extract_boundary_loops",
    "extract_connected_components",
    "find_boundary_edges",
    "find_hole_candidates",
    "measure_boundary_loop",
    "region_grow_faces",
]
