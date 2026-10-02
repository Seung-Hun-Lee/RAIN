"""Exact RAIN body/site bindings; register explicitly per process."""

from copy import deepcopy


def _body(name):
    return dict(name=name, body_name=name, body_ids=[0], geom_ids=[], segmentable=True)


def _site(name):
    return dict(name=name, body_name=name, body_ids=[], geom_ids=[], segmentable=False)


EXTRA_ACTION_OBJECTS = {
    f"{kind}_{index}": _body(f"{kind}_{index}_main")
    for kind in (
        "plate", "wine_bottle", "akita_black_bowl", "milk", "popcorn", "butter",
        "cream_cheese", "chocolate_pudding", "porcelain_mug", "white_yellow_mug",
        "red_coffee_mug", "black_book", "yellow_book", "macaroni_and_cheese",
        "moka_pot", "new_salad_dressing", "wooden_tray", "bowl_drainer",
        "dining_set_group", "short_fridge", "white_storage_box",
        "wooden_two_layer_shelf", "short_cabinet", "white_cabinet", "wooden_cabinet",
    )
    for index in (1, 2)
}

for instance, suffixes in {
    "short_fridge_1": ("feedback_top_region", "upper_region", "middle_region", "lower_region"),
    "white_storage_box_1": ("feedback_contain_region", "top_side"),
    "wooden_two_layer_shelf_1": ("feedback_roof_region", "top_side", "top_region", "bottom_region"),
    "dining_set_group_1": ("plate_support_region",),
    "bowl_drainer_1": ("left_region", "right_region"),
    "wooden_tray_1": ("contain_region",),
    "microwave_1": ("heating_region", "top_side"),
}.items():
    for suffix in suffixes:
        name = instance + "_" + suffix
        EXTRA_ACTION_OBJECTS[name] = _site(name)

# Manipulating the microwave means its articulated door, never the whole shell.
EXTRA_ACTION_OBJECTS["microwave_1"] = _body("microwave_1_microdoorroot")
for slot in ("top", "middle", "bottom"):
    EXTRA_ACTION_OBJECTS[f"short_cabinet_1_{slot}_region"] = _body(f"short_cabinet_1_drawer_{slot}")
    for cabinet in ("white_cabinet_1", "wooden_cabinet_1"):
        EXTRA_ACTION_OBJECTS[f"{cabinet}_{slot}_region"] = _body(f"{cabinet}_cabinet_{slot}")


def register_object_bindings(action_objects=None):
    if action_objects is None:
        from final_libero_ex_eval.impl import benchmark_support
        action_objects = benchmark_support.ACTION_OBJECTS
    action_objects.update(deepcopy(EXTRA_ACTION_OBJECTS))
    return action_objects
