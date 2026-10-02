"""Exact process-local RAIN bindings for the novel-scene asset instances.

No evaluator, simulator, or global registry is imported until registration is
explicitly requested. Geometry sites use segmentable=False and their full site
name; articulated drawer targets bind to their exact moving body instead.
"""

from __future__ import annotations

from copy import deepcopy


def _body(name: str) -> dict:
    return dict(name=name, body_name=name, body_ids=[0], geom_ids=[], segmentable=True)


def _site(name: str) -> dict:
    return dict(name=name, body_name=name, body_ids=[], geom_ids=[], segmentable=False)


EXTRA_ACTION_OBJECTS = {
    instance: _body(instance + "_main")
    for instance in (
        "popcorn_1", "macaroni_and_cheese_1", "yellow_book_1", "new_salad_dressing_1",
        "white_bowl_1", "wooden_tray_1", "wooden_shelf_1", "wooden_two_layer_shelf_1",
        "white_storage_box_1", "bowl_drainer_1", "short_cabinet_1", "short_fridge_1", "rack_1",
    )
}

for instance, suffixes in {
    "wooden_shelf_1": ("top_region", "middle_region", "bottom_region", "top_side"),
    "wooden_two_layer_shelf_1": ("top_region", "bottom_region", "top_side"),
    "white_storage_box_1": ("top_side", "bottom_side", "left_side", "right_side"),
    "bowl_drainer_1": ("left_region", "right_region"),
    "wooden_tray_1": ("contain_region",),
    "desk_caddy_1": ("back_contain_region", "front_contain_region", "left_contain_region", "right_contain_region"),
    "microwave_1": ("heating_region", "top_side"),
    "white_cabinet_1": ("top_side",),
    "wooden_cabinet_1": ("top_side",),
    "short_fridge_1": ("upper_region", "middle_region", "lower_region"),
}.items():
    for suffix in suffixes:
        full_name = instance + "_" + suffix
        EXTRA_ACTION_OBJECTS[full_name] = _site(full_name)

for slot in ("top", "middle", "bottom"):
    EXTRA_ACTION_OBJECTS[f"short_cabinet_1_{slot}_region"] = _body(f"short_cabinet_1_drawer_{slot}")
    for cabinet in ("white_cabinet_1", "wooden_cabinet_1"):
        EXTRA_ACTION_OBJECTS[f"{cabinet}_{slot}_region"] = _body(f"{cabinet}_cabinet_{slot}")

# The relative-placement candidate declares this finite workspace rectangle;
# its runtime geometry is resolved through the same non-segmentable adapter.
EXTRA_ACTION_OBJECTS["kitchen_table_novel_plate_right_region"] = _site("kitchen_table_novel_plate_right_region")


def register_object_bindings(action_objects: dict | None = None) -> dict:
    """Install independent binding copies into a supplied or RAIN registry."""
    if action_objects is None:
        from final_libero_ex_eval.impl import benchmark_support

        action_objects = benchmark_support.ACTION_OBJECTS
    action_objects.update(deepcopy(EXTRA_ACTION_OBJECTS))
    return action_objects
