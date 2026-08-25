"""Shared physical drop-event metric for simulation extraction."""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence


DROP_EVENT_DISPLACEMENT_THRESHOLD_M = 0.05


def meets_drop_displacement_threshold(value: Any) -> bool:
    """Return whether a finite post-release downward displacement is a drop.

    This helper only evaluates the metric boundary.  Callers must separately
    establish the formal event context: a confirmed grasp followed by
    robot-object contact loss, or an escaped-simulation event.  Ordinary
    vertical transport while grasp contact remains is therefore never enough.
    """
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= DROP_EVENT_DISPLACEMENT_THRESHOLD_M
    )


def escape_drop_displacement_m(
    reference_position: Any,
    escaped_position: Any,
    escape_step: Any,
) -> Optional[float]:
    """Return the downward displacement to a first out-of-bounds sample.

    An object already out of bounds in the first recorded frame has no prior
    in-simulation state and therefore cannot establish a drop event.  The
    returned raw displacement is intentionally not thresholded here.
    """
    if not isinstance(escape_step, int) or isinstance(escape_step, bool) or escape_step <= 0:
        return None
    if not isinstance(reference_position, Sequence) or isinstance(reference_position, (str, bytes)):
        return None
    if not isinstance(escaped_position, Sequence) or isinstance(escaped_position, (str, bytes)):
        return None
    if len(reference_position) < 3 or len(escaped_position) < 3:
        return None
    try:
        reference_z = float(reference_position[2])
        escaped_z = float(escaped_position[2])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(reference_z) and math.isfinite(escaped_z)):
        return None
    return max(0.0, reference_z - escaped_z)
