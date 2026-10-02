"""Process-local bindings and fixture support for wooden-tray choices.

The installed assets are not edited. The shared fixture installer
only corrects the fixed wooden-tray root height from its current collision
geometry; the tray's native ``contain_region`` remains the sole destination.
"""

from __future__ import annotations

from copy import deepcopy


TARGET_REGION = "wooden_tray_1_contain_region"
SUPPORTED_MANIPULANDS = (
    "plate_1",
    "plate_2",
    "plate_3",
    "akita_black_bowl_1",
    "glazed_rim_porcelain_ramekin_1",
    "white_bowl_1",
)


def install_wooden_tray_object_choices_support() -> None:
    """Install the fixed-fixture sampler overlay in this process."""
    from .novel_feedback_fixture_geometry import install_feedback_fixture_geometry

    install_feedback_fixture_geometry()


def register_object_bindings(action_objects=None):
    """Register every exact pickup instance and the one native tray site."""
    from .novel_feedback_object_bindings import (
        register_object_bindings as register_feedback_bindings,
    )

    action_objects = register_feedback_bindings(action_objects)
    additions = {
        object_id: {
            "name": object_id,
            "body_name": object_id + "_main",
            "body_ids": [0],
            "geom_ids": [],
            "segmentable": True,
        }
        for object_id in SUPPORTED_MANIPULANDS
    }
    additions[TARGET_REGION] = {
        "name": TARGET_REGION,
        "body_name": TARGET_REGION,
        "body_ids": [],
        "geom_ids": [],
        "segmentable": False,
    }
    action_objects.update(deepcopy(additions))
    return action_objects
