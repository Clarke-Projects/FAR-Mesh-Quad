# H-CORE-V2 LEGACY TOOL BOUNDARY
# This file remains available as a topology recovery / comparison tool.
# It is not the default Adaptive Surface Fill v2 seed path.
# New v2 seed construction belongs in far_mesh/core/hole_fill/.
#
"""Topology helpers for seam-sealed hole-fill patch seeds.

The collar consumes every original hole-boundary edge exactly once by adding
an inward ring and a forced quad strip:

    B_i ---- B_j
     |    /   |
     |  /     |
    I_i ---- I_j

The quad is split into two triangles:
    (B_i, B_j, I_j)
    (B_i, I_j, I_i)

This leaves only the inner ring as the boundary for the remaining fill.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class SealedQuadCollarTopology:
    """Connectivity-only description of a sealed boundary collar."""

    boundary_vertex_ids: tuple[int, ...]
    inner_vertex_ids: tuple[int, ...]
    collar_faces: tuple[tuple[int, int, int], ...]
    expected_seam_edges: tuple[tuple[int, int], ...]
    inner_ring_edges: tuple[tuple[int, int], ...]


def _edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def _loop_edges(vertex_ids: Sequence[int]) -> tuple[tuple[int, int], ...]:
    count = len(vertex_ids)
    return tuple(
        _edge_key(int(vertex_ids[i]), int(vertex_ids[(i + 1) % count]))
        for i in range(count)
    )


def count_undirected_face_edges(
    faces: Iterable[Sequence[int]],
) -> Counter[tuple[int, int]]:
    """Count undirected edges used by triangular faces."""

    counts: Counter[tuple[int, int]] = Counter()
    for face in faces:
        if len(face) != 3:
            raise ValueError("Only triangular faces are supported.")
        a, b, c = (int(face[0]), int(face[1]), int(face[2]))
        counts[_edge_key(a, b)] += 1
        counts[_edge_key(b, c)] += 1
        counts[_edge_key(c, a)] += 1
    return counts


def build_sealed_quad_collar_topology(
    boundary_vertex_ids: Sequence[int],
    inner_vertex_ids: Sequence[int],
) -> SealedQuadCollarTopology:
    """Build a deterministic quad-strip collar around a boundary loop.

    Requirements:
    - boundary and inner loops have the same length
    - at least three vertices
    - all ids are unique within each loop
    - no id is shared between the two loops
    """

    boundary = tuple(int(v) for v in boundary_vertex_ids)
    inner = tuple(int(v) for v in inner_vertex_ids)

    if len(boundary) < 3:
        raise ValueError("A sealed collar requires at least three boundary vertices.")
    if len(boundary) != len(inner):
        raise ValueError("Boundary and inner collar loops must have the same length.")
    if len(set(boundary)) != len(boundary):
        raise ValueError("Boundary loop contains duplicate vertex ids.")
    if len(set(inner)) != len(inner):
        raise ValueError("Inner collar loop contains duplicate vertex ids.")
    if set(boundary).intersection(inner):
        raise ValueError("Boundary and inner collar ids must be disjoint.")

    faces: list[tuple[int, int, int]] = []
    count = len(boundary)

    for i in range(count):
        bi = boundary[i]
        bj = boundary[(i + 1) % count]
        ii = inner[i]
        ij = inner[(i + 1) % count]

        # Split the quad (bi, bj, ij, ii). This consumes boundary edge bi-bj
        # exactly once and leaves inner edge ii-ij for the inner fill.
        faces.append((bi, bj, ij))
        faces.append((bi, ij, ii))

    return SealedQuadCollarTopology(
        boundary_vertex_ids=boundary,
        inner_vertex_ids=inner,
        collar_faces=tuple(faces),
        expected_seam_edges=_loop_edges(boundary),
        inner_ring_edges=_loop_edges(inner),
    )


def validate_sealed_quad_collar_topology(
    topology: SealedQuadCollarTopology,
) -> dict[str, int | bool]:
    """Return deterministic edge-use checks for the collar alone."""

    edge_counts = count_undirected_face_edges(topology.collar_faces)

    seam_uses = [
        edge_counts[edge] for edge in topology.expected_seam_edges
    ]
    inner_uses = [
        edge_counts[edge] for edge in topology.inner_ring_edges
    ]

    radial_edges = []
    count = len(topology.boundary_vertex_ids)
    for i in range(count):
        radial_edges.append(
            _edge_key(topology.boundary_vertex_ids[i], topology.inner_vertex_ids[i])
        )
    radial_uses = [edge_counts[edge] for edge in radial_edges]

    return {
        "seam_edge_count": len(topology.expected_seam_edges),
        "inner_ring_edge_count": len(topology.inner_ring_edges),
        "radial_edge_count": len(radial_edges),
        "all_seam_edges_used_once": all(use == 1 for use in seam_uses),
        "all_inner_ring_edges_used_once": all(use == 1 for use in inner_uses),
        "all_radial_edges_used_twice": all(use == 2 for use in radial_uses),
        "face_count": len(topology.collar_faces),
    }
