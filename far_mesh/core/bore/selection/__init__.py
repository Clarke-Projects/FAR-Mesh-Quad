"""Selection/AOI helpers for FAR MESH BoreTool.

This package contains neutral selection, rim-resolution, and mesh-realization
evidence helpers. It must not classify features, emit CandidateData, authorize
delete patches, or mutate topology.
"""

from .region_select import *
from .rim_resolver import *
from .mesh_realization import *
