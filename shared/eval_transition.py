"""Temporal-consensus helper for the public GT-mask controller."""

from __future__ import annotations


def subtask_completion_ready(
    consecutive_complete: int,
    required_confirmations: int,
) -> bool:
    """Require the configured causal temporal consensus for every action.

    Gripper aperture is not evidence that an object was grasped: the same
    aperture can result from an empty closure or from objects with different
    geometry.  Physical grasp completion is checked separately from RGB.
    """

    return int(consecutive_complete) >= int(required_confirmations)
