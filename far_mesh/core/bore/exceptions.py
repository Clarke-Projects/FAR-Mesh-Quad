"""Typed Bore exceptions.

Worker/UI code should catch these as controlled messages.  Full tracebacks are
reserved for programming errors, not expected recognition/rebuild rejections.
"""

from __future__ import annotations


class BoreError(Exception):
    """Base class for Bore-domain errors."""

    user_visible: bool = True


class BoreRecognitionError(BoreError):
    """Recognition could not promote a physical feature object."""


class BoreRebuildRejected(BoreError):
    """A rebuild target was rejected without mutating geometry."""


class BoreTargetInvalid(BoreError):
    """Recognition produced an invalid or incomplete rebuild target."""


class BoreTopologyError(BoreError):
    """Topology evidence was inconsistent or incomplete."""
