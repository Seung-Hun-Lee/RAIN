from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
import re
from typing import Dict, List, NamedTuple, Optional

import yaml


SOURCE_PRIOR_TASKS: Dict[int, str] = {
    1: "Put the bowl on the plate",
    2: "Put the wine bottle on the rack",
    3: "Open the top drawer and put the bowl inside",
    4: "Put the cream cheese in the bowl",
    5: "Put the wine bottle on top of the cabinet",
    6: "Push the plate to the front of the stove",
    7: "Turn on the stove",
    8: "Put the bowl on the stove",
    9: "Put the bowl on top of the cabinet",
    10: "Open the middle drawer of the cabinet",
    301: "Pick the alphabet soup and place it in the basket",
    302: "Pick the cream cheese and place it in the basket",
    303: "Pick the salad dressing and place it in the basket",
    304: "Pick the bbq sauce and place it in the basket",
    305: "Pick the ketchup and place it in the basket",
    306: "Pick the tomato sauce and place it in the basket",
    307: "Pick the butter and place it in the basket",
    308: "Pick the milk and place it in the basket",
    309: "Pick the chocolate pudding and place it in the basket",
    310: "Pick the orange juice and place it in the basket",
}

ACTION_OBJECTS: Dict[str, dict] = {
    "porcelain_mug_1": {
        "name": "porcelain_mug_1_main",
        "body_name": "porcelain_mug_1_main",
        "body_ids": [0],
        "geom_ids": [],
        "segmentable": True,
    },
    "white_yellow_mug_1": {
        "name": "white_yellow_mug_1_main",
        "body_name": "white_yellow_mug_1_main",
        "body_ids": [0],
        "geom_ids": [],
        "segmentable": True,
    },
    "red_coffee_mug_1": {
        "name": "red_coffee_mug_1_main",
        "body_name": "red_coffee_mug_1_main",
        "body_ids": [0],
        "geom_ids": [],
        "segmentable": True,
    },
    "akita_black_bowl_1": {
        "name": "akita_black_bowl_1_main",
        "body_name": "akita_black_bowl_1_main",
        "body_ids": [0],
        "geom_ids": [],
        "segmentable": True,
    },
    "cream_cheese_1": {
        "name": "cream_cheese_1_main",
        "body_name": "cream_cheese_1_main",
        "body_ids": [0],
        "geom_ids": [],
        "segmentable": True,
    },
    "alphabet_soup_1": {
        "name": "alphabet_soup_1_main",
        "body_name": "alphabet_soup_1_main",
        "body_ids": [0],
        "geom_ids": [],
        "segmentable": True,
    },
    "salad_dressing_1": {
        "name": "salad_dressing_1_main",
        "body_name": "salad_dressing_1_main",
        "body_ids": [0],
        "geom_ids": [],
        "segmentable": True,
    },
    "bbq_sauce_1": {
        "name": "bbq_sauce_1_main",
        "body_name": "bbq_sauce_1_main",
        "body_ids": [0],
        "geom_ids": [],
        "segmentable": True,
    },
    "ketchup_1": {
        "name": "ketchup_1_main",
        "body_name": "ketchup_1_main",
        "body_ids": [0],
        "geom_ids": [],
        "segmentable": True,
    },
    "tomato_sauce_1": {
        "name": "tomato_sauce_1_main",
        "body_name": "tomato_sauce_1_main",
        "body_ids": [0],
        "geom_ids": [],
        "segmentable": True,
    },
    "butter_1": {
        "name": "butter_1_main",
        "body_name": "butter_1_main",
        "body_ids": [0],
        "geom_ids": [],
        "segmentable": True,
    },
    "milk_1": {
        "name": "milk_1_main",
        "body_name": "milk_1_main",
        "body_ids": [0],
        "geom_ids": [],
        "segmentable": True,
    },
    "chocolate_pudding_1": {
        "name": "chocolate_pudding_1_main",
        "body_name": "chocolate_pudding_1_main",
        "body_ids": [0],
        "geom_ids": [],
        "segmentable": True,
    },
    "orange_juice_1": {
        "name": "orange_juice_1_main",
        "body_name": "orange_juice_1_main",
        "body_ids": [0],
        "geom_ids": [],
        "segmentable": True,
    },
    "wine_bottle_1": {
        "name": "wine_bottle_1_main",
        "body_name": "wine_bottle_1_main",
        "body_ids": [0],
        "geom_ids": [],
        "segmentable": True,
    },
    "plate_1": {
        "name": "plate_1_main",
        "body_name": "plate_1_main",
        "body_ids": [0],
        "geom_ids": [],
        "segmentable": True,
    },
    "plate_2": {
        "name": "plate_2_main",
        "body_name": "plate_2_main",
        "body_ids": [0],
        "geom_ids": [],
        "segmentable": True,
    },
    "akita_black_bowl_2": {
        "name": "akita_black_bowl_2_main",
        "body_name": "akita_black_bowl_2_main",
        "body_ids": [0],
        "geom_ids": [],
        "segmentable": True,
    },
    "glazed_rim_porcelain_ramekin_1": {
        "name": "glazed_rim_porcelain_ramekin_1_main",
        "body_name": "glazed_rim_porcelain_ramekin_1_main",
        "body_ids": [0],
        "geom_ids": [],
        "segmentable": True,
    },
    "cookies_1": {
        "name": "cookies_1_main",
        "body_name": "cookies_1_main",
        "body_ids": [0],
        "geom_ids": [],
        "segmentable": True,
    },
    "wine_rack_1_top_region": {
        "name": "wine_rack_1_main",
        "body_name": "wine_rack_1_main",
        "body_ids": [0],
        "geom_ids": [],
        "segmentable": True,
    },
    "wooden_cabinet_1_top_region": {
        "name": "wooden_cabinet_1_cabinet_top",
        "body_name": "wooden_cabinet_1_cabinet_top",
        "body_ids": [0],
        "geom_ids": [],
        "segmentable": True,
    },
    "wooden_cabinet_1_middle_region": {
        "name": "wooden_cabinet_1_cabinet_middle",
        "body_name": "wooden_cabinet_1_cabinet_middle",
        "body_ids": [0],
        "geom_ids": [],
        "segmentable": True,
    },
    "wooden_cabinet_1_top_side": {
        "name": "wooden_cabinet_1_main",
        "body_name": "wooden_cabinet_1_main",
        "body_ids": [0],
        "geom_ids": [],
        "segmentable": True,
    },
    "flat_stove_1": {
        "name": "flat_stove_1_button",
        "body_name": "flat_stove_1_button",
        "body_ids": [0],
        "geom_ids": [],
        "segmentable": True,
    },
    "flat_stove_1_cook_region": {
        "name": "flat_stove_1_burner",
        "body_name": "flat_stove_1_burner",
        "body_ids": [0],
        "geom_ids": [],
        "segmentable": True,
    },
    "main_table_stove_front_region": {
        "name": "main_table_stove_front_region",
        "body_name": "main_table_stove_front_region",
        "segmentable": False,
    },
    "basket_1_contain_region": {
        "name": "basket_1_main",
        "body_name": "basket_1_main",
        "body_ids": [0],
        "geom_ids": [],
        "segmentable": True,
    },
    "basket_2_contain_region": {
        "name": "basket_2_main",
        "body_name": "basket_2_main",
        "body_ids": [0],
        "geom_ids": [],
        "segmentable": True,
    },
    "microwave_1": {
        "name": "microwave_1_microdoorroot",
        "body_name": "microwave_1_microdoorroot",
        "body_ids": [0],
        "geom_ids": [],
        "segmentable": True,
    },
    "microwave_1_heating_region": {
        "name": "microwave_1_main",
        "body_name": "microwave_1_main",
        "body_ids": [0],
        "geom_ids": [],
        "segmentable": True,
    },
    "white_cabinet_1_bottom_region": {
        "name": "white_cabinet_1_cabinet_bottom",
        "body_name": "white_cabinet_1_cabinet_bottom",
        "body_ids": [0],
        "geom_ids": [],
        "segmentable": True,
    },
    "white_cabinet_1_middle_region": {
        "name": "white_cabinet_1_cabinet_middle",
        "body_name": "white_cabinet_1_cabinet_middle",
        "body_ids": [0],
        "geom_ids": [],
        "segmentable": True,
    },
    "white_cabinet_1_top_region": {
        "name": "white_cabinet_1_cabinet_top",
        "body_name": "white_cabinet_1_cabinet_top",
        "body_ids": [0],
        "geom_ids": [],
        "segmentable": True,
    },
    "white_cabinet_1_top_side": {
        "name": "white_cabinet_1_main",
        "body_name": "white_cabinet_1_main",
        "body_ids": [0],
        "geom_ids": [],
        "segmentable": True,
    },
    "desk_caddy_1_back_contain_region": {
        "name": "desk_caddy_1_main",
        "body_name": "desk_caddy_1_main",
        "body_ids": [0],
        "geom_ids": [],
        "segmentable": True,
    },
}


def find_source_episode(episodes: Dict[int, dict], task_description: str) -> dict:
    want = str(task_description or "").strip().lower()
    cands = [
        ep for ep in episodes.values()
        if str(ep.get("task_description", "")).strip().lower() == want
    ]
    if not cands:
        raise KeyError(f"No source episode found for task description: {task_description}")
    cands.sort(key=lambda ep: int(ep.get("episode_index", 10**9)))
    return cands[0]

ATOMIC_ACTION_PLAN: Dict[int, List[dict]] = {
    1: [
        {"action_type": "grasp", "primary_object_id": "akita_black_bowl_1", "target_object_id": "akita_black_bowl_1"},
        {"action_type": "release", "primary_object_id": "akita_black_bowl_1", "target_object_id": "plate_1"},
    ],
    2: [
        {"action_type": "grasp", "primary_object_id": "wine_bottle_1", "target_object_id": "wine_bottle_1"},
        {"action_type": "release", "primary_object_id": "wine_bottle_1", "target_object_id": "wine_rack_1_top_region"},
    ],
    3: [
        {"action_type": "open", "primary_object_id": "wooden_cabinet_1_top_region", "target_object_id": "wooden_cabinet_1_top_region"},
        {"action_type": "grasp", "primary_object_id": "akita_black_bowl_1", "target_object_id": "akita_black_bowl_1"},
        {"action_type": "release", "primary_object_id": "akita_black_bowl_1", "target_object_id": "wooden_cabinet_1_top_region"},
    ],
    4: [
        {"action_type": "grasp", "primary_object_id": "cream_cheese_1", "target_object_id": "cream_cheese_1"},
        {"action_type": "release", "primary_object_id": "cream_cheese_1", "target_object_id": "akita_black_bowl_1"},
    ],
    5: [
        {"action_type": "grasp", "primary_object_id": "wine_bottle_1", "target_object_id": "wine_bottle_1"},
        {"action_type": "release", "primary_object_id": "wine_bottle_1", "target_object_id": "wooden_cabinet_1_top_side"},
    ],
    6: [
        {"action_type": "approach", "primary_object_id": "plate_1", "target_object_id": "plate_1"},
        {"action_type": "push", "primary_object_id": "plate_1", "target_object_id": "main_table_stove_front_region"},
    ],
    7: [
        {"action_type": "turn_on", "primary_object_id": "flat_stove_1", "target_object_id": "flat_stove_1"},
    ],
    8: [
        {"action_type": "grasp", "primary_object_id": "akita_black_bowl_1", "target_object_id": "akita_black_bowl_1"},
        {"action_type": "release", "primary_object_id": "akita_black_bowl_1", "target_object_id": "flat_stove_1_cook_region"},
    ],
    9: [
        {"action_type": "grasp", "primary_object_id": "akita_black_bowl_1", "target_object_id": "akita_black_bowl_1"},
        {"action_type": "release", "primary_object_id": "akita_black_bowl_1", "target_object_id": "wooden_cabinet_1_top_side"},
    ],
    10: [
        {"action_type": "open", "primary_object_id": "wooden_cabinet_1_middle_region", "target_object_id": "wooden_cabinet_1_middle_region"},
    ],
    301: [
        {"action_type": "grasp", "primary_object_id": "alphabet_soup_1", "target_object_id": "alphabet_soup_1"},
        {"action_type": "release", "primary_object_id": "alphabet_soup_1", "target_object_id": "basket_1_contain_region"},
    ],
    302: [
        {"action_type": "grasp", "primary_object_id": "cream_cheese_1", "target_object_id": "cream_cheese_1"},
        {"action_type": "release", "primary_object_id": "cream_cheese_1", "target_object_id": "basket_1_contain_region"},
    ],
    303: [
        {"action_type": "grasp", "primary_object_id": "salad_dressing_1", "target_object_id": "salad_dressing_1"},
        {"action_type": "release", "primary_object_id": "salad_dressing_1", "target_object_id": "basket_1_contain_region"},
    ],
    304: [
        {"action_type": "grasp", "primary_object_id": "bbq_sauce_1", "target_object_id": "bbq_sauce_1"},
        {"action_type": "release", "primary_object_id": "bbq_sauce_1", "target_object_id": "basket_1_contain_region"},
    ],
    305: [
        {"action_type": "grasp", "primary_object_id": "ketchup_1", "target_object_id": "ketchup_1"},
        {"action_type": "release", "primary_object_id": "ketchup_1", "target_object_id": "basket_1_contain_region"},
    ],
    306: [
        {"action_type": "grasp", "primary_object_id": "tomato_sauce_1", "target_object_id": "tomato_sauce_1"},
        {"action_type": "release", "primary_object_id": "tomato_sauce_1", "target_object_id": "basket_1_contain_region"},
    ],
    307: [
        {"action_type": "grasp", "primary_object_id": "butter_1", "target_object_id": "butter_1"},
        {"action_type": "release", "primary_object_id": "butter_1", "target_object_id": "basket_1_contain_region"},
    ],
    308: [
        {"action_type": "grasp", "primary_object_id": "milk_1", "target_object_id": "milk_1"},
        {"action_type": "release", "primary_object_id": "milk_1", "target_object_id": "basket_1_contain_region"},
    ],
    309: [
        {"action_type": "grasp", "primary_object_id": "chocolate_pudding_1", "target_object_id": "chocolate_pudding_1"},
        {"action_type": "release", "primary_object_id": "chocolate_pudding_1", "target_object_id": "basket_1_contain_region"},
    ],
    310: [
        {"action_type": "grasp", "primary_object_id": "orange_juice_1", "target_object_id": "orange_juice_1"},
        {"action_type": "release", "primary_object_id": "orange_juice_1", "target_object_id": "basket_1_contain_region"},
    ],
}


class Task(NamedTuple):
    name: str
    language: str
    problem: str
    problem_folder: str
    bddl_file: str
    init_states_file: str
    task_id: str
    category: str
    anchor_suite: str
    source_prior_task_ids: List[int]
    max_steps: int


class LiberoEXBenchmark:
    def __init__(
        self,
        benchmark_root: str,
        categories: Optional[List[str]] = None,
    ) -> None:
        self.benchmark_root = os.path.abspath(benchmark_root)
        self.support_root = os.path.join(self.benchmark_root, "benchmark_support")
        self._meta_by_index: Dict[int, dict] = {}
        self._eval_by_index: Dict[int, dict] = {}
        self._action_plan_by_task_id: Dict[str, List[dict]] = {}
        self.tasks: List[Task] = []

        registry_entries: List[str] = []
        skip_default_registry = str(
            os.environ.get("LIBERO_EX_SKIP_DEFAULT_REGISTRY", "0")
        ).strip() in {"1", "true", "TRUE", "yes", "YES"}
        registry_path = Path(self.support_root) / "benchmark_registry_candidates.yaml"
        if registry_path.exists() and not skip_default_registry:
            with registry_path.open() as f:
                registry = yaml.safe_load(f) or {}
            registry_entries.extend(registry.get("registry_entries") or [])
        elif not skip_default_registry:
            config_path = Path(self.benchmark_root) / "benchmark_config.json"
            if config_path.exists():
                with config_path.open() as f:
                    config = json.load(f)
                public_index_rel = (
                    (config.get("registry_indexes") or {}).get("selected75")
                )
                if public_index_rel:
                    public_index_path = Path(self.benchmark_root) / str(public_index_rel)
                    if public_index_path.exists():
                        with public_index_path.open() as f:
                            public_registry = yaml.safe_load(f) or {}
                        registry_entries.extend(
                            public_registry.get("registry_entries") or []
                        )
        extra_registry_indexes = str(
            os.environ.get("LIBERO_EX_REGISTRY_INDEXES", "")
        ).strip()
        if extra_registry_indexes:
            raw_paths = re.split(r"[:,]", extra_registry_indexes)
            for raw_path in raw_paths:
                token = str(raw_path).strip()
                if not token:
                    continue
                index_path = Path(token)
                if not index_path.is_absolute():
                    index_path = Path(self.benchmark_root) / token
                with index_path.open() as f:
                    extra_registry = yaml.safe_load(f) or {}
                registry_entries.extend(extra_registry.get("registry_entries") or [])
        seen_registry_entries = set()
        registry_entries = [
            entry
            for entry in registry_entries
            if not (entry in seen_registry_entries or seen_registry_entries.add(entry))
        ]
        action_plan_registry_path = Path(self.support_root) / "action_plan_registry.yaml"
        if action_plan_registry_path.exists():
            with action_plan_registry_path.open() as f:
                action_plan_registry = yaml.safe_load(f) or {}
            raw_action_plans = action_plan_registry.get("action_plans") or {}
            self._action_plan_by_task_id = {
                str(task_id): [dict(step or {}) for step in (steps or [])]
                for task_id, steps in raw_action_plans.items()
            }
        wanted_categories = None
        if categories:
            wanted_categories = {str(c).strip().upper() for c in categories if str(c).strip()}

        for idx, rel_registry_path in enumerate(registry_entries):
            reg_path = Path(self.benchmark_root) / str(rel_registry_path)
            with reg_path.open() as f:
                reg = yaml.safe_load(f) or {}

            category = str(reg.get("category", "")).strip().upper()
            if wanted_categories and category not in wanted_categories:
                continue

            support_dir = reg_path.parent
            meta_path = support_dir / str(reg.get("metadata_file", "task_meta.yaml"))
            eval_path = support_dir / str(reg.get("eval_rules_file", "eval_rules.yaml"))
            with meta_path.open() as f:
                meta = yaml.safe_load(f) or {}
            with eval_path.open() as f:
                eval_rules = yaml.safe_load(f) or {}

            task = Task(
                name=str(reg["name"]),
                language=str(reg["language"]),
                problem=str(reg.get("problem", "Libero")),
                problem_folder=str(reg["problem_folder"]),
                bddl_file=str(reg.get("bddl_file", "task.bddl")),
                init_states_file=str(reg.get("init_states_file", "task.pruned_init")),
                task_id=str(reg["name"]),
                category=category,
                anchor_suite=str(reg.get("anchor_suite", "libero_goal")),
                source_prior_task_ids=[int(x) for x in meta.get("source_prior_task_ids", [])],
                max_steps=int(meta.get("max_steps", 520)),
            )
            out_idx = len(self.tasks)
            self.tasks.append(task)
            self._meta_by_index[out_idx] = meta
            self._eval_by_index[out_idx] = eval_rules

    def get_num_tasks(self) -> int:
        return len(self.tasks)

    def get_task(self, i: int) -> Task:
        return self.tasks[i]

    def get_task_meta(self, i: int) -> dict:
        return self._meta_by_index[i]

    def get_task_eval_rules(self, i: int) -> dict:
        return self._eval_by_index[i]

    def get_task_bddl_file_path(self, i: int) -> str:
        task = self.tasks[i]
        return os.path.join(self.benchmark_root, task.problem_folder, task.bddl_file)

    def get_task_init_states_path(self, i: int) -> str:
        task = self.tasks[i]
        return os.path.join(self.benchmark_root, task.problem_folder, task.init_states_file)

    def get_task_action_plan(self, i: int) -> List[dict]:
        task = self.tasks[i]
        return [dict(step or {}) for step in self._action_plan_by_task_id.get(task.task_id, [])]


def _copy_object_meta(object_id: str) -> dict:
    base = ACTION_OBJECTS.get(str(object_id), None)
    if base is None:
        segmentable = not str(object_id).endswith("_region")
        return {
            "name": str(object_id),
            "body_name": str(object_id),
            "body_ids": [0] if segmentable else [],
            "geom_ids": [],
            "segmentable": segmentable,
        }
    return deepcopy(base)


_ATOM_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$")


def _parse_goal_atom(text: str):
    raw = str(text or "").strip()
    if not raw:
        return None
    m = _ATOM_RE.match(raw)
    if not m:
        return None
    pred = str(m.group(1)).strip().lower()
    args = [part.strip() for part in m.group(2).split(",") if part.strip()]
    if not args:
        return None
    return (pred, args)


def _infer_single_task_action_plan(task_meta: dict) -> List[dict]:
    atoms = task_meta.get("canonical_goal_atoms") or []
    parsed = [_parse_goal_atom(atom) for atom in atoms]
    parsed = [item for item in parsed if item is not None]
    if len(parsed) != 1:
        return []

    pred, args = parsed[0]
    source_desc = str(
        task_meta.get("source_prior_description")
        or task_meta.get("task_name")
        or task_meta.get("language")
        or ""
    )
    if pred in {"on", "in"} and len(args) == 2:
        primary_oid, target_oid = args
        return [
            {
                "action_type": "grasp",
                "primary_object_id": primary_oid,
                "target_object_id": primary_oid,
                "source_task_id": 0,
                "source_task_description": source_desc,
                "source_subtask_index": 1,
            },
            {
                "action_type": "release",
                "primary_object_id": primary_oid,
                "target_object_id": target_oid,
                "source_task_id": 0,
                "source_task_description": source_desc,
                "source_subtask_index": 2,
            },
        ]
    if pred == "open" and len(args) == 1:
        target_oid = args[0]
        return [
            {
                "action_type": "open",
                "primary_object_id": target_oid,
                "target_object_id": target_oid,
                "source_task_id": 0,
                "source_task_description": source_desc,
                "source_subtask_index": 1,
            }
        ]
    if pred in {"turnon", "turn_on"} and len(args) == 1:
        target_oid = args[0]
        return [
            {
                "action_type": "turn_on",
                "primary_object_id": target_oid,
                "target_object_id": target_oid,
                "source_task_id": 0,
                "source_task_description": source_desc,
                "source_subtask_index": 1,
            }
        ]
    return []


def _canonical_goal_object_ids(task_meta: dict) -> set[str]:
    atoms = task_meta.get("canonical_goal_atoms") or []
    parsed = [_parse_goal_atom(atom) for atom in atoms]
    parsed = [item for item in parsed if item is not None]
    out: set[str] = set()
    for _, args in parsed:
        for arg in args:
            text = str(arg or "").strip()
            if text:
                out.add(text)
    return out


def _canonical_goal_action_plan(task_meta: dict) -> List[dict]:
    atoms = task_meta.get("canonical_goal_atoms") or []
    parsed = [_parse_goal_atom(atom) for atom in atoms]
    parsed = [item for item in parsed if item is not None]
    if not parsed:
        return []

    source_desc = str(
        task_meta.get("new_test_task_description")
        or task_meta.get("language")
        or task_meta.get("task_name")
        or task_meta.get("source_prior_description")
        or ""
    )
    out: List[dict] = []
    next_subtask_index = 1
    for pred, args in parsed:
        pred = str(pred).strip().lower()
        if pred in {"on", "in"} and len(args) == 2:
            primary_oid, target_oid = args
            out.append(
                {
                    "action_type": "grasp",
                    "primary_object_id": primary_oid,
                    "target_object_id": primary_oid,
                    "source_task_id": 0,
                    "source_task_description": source_desc,
                    "source_subtask_index": next_subtask_index,
                }
            )
            next_subtask_index += 1
            out.append(
                {
                    "action_type": "release",
                    "primary_object_id": primary_oid,
                    "target_object_id": target_oid,
                    "source_task_id": 0,
                    "source_task_description": source_desc,
                    "source_subtask_index": next_subtask_index,
                }
            )
            next_subtask_index += 1
            continue
        if pred == "open" and len(args) == 1:
            target_oid = args[0]
            out.append(
                {
                    "action_type": "open",
                    "primary_object_id": target_oid,
                    "target_object_id": target_oid,
                    "source_task_id": 0,
                    "source_task_description": source_desc,
                    "source_subtask_index": next_subtask_index,
                }
            )
            next_subtask_index += 1
            continue
        if pred == "close" and len(args) == 1:
            target_oid = args[0]
            out.append(
                {
                    "action_type": "close",
                    "primary_object_id": target_oid,
                    "target_object_id": target_oid,
                    "source_task_id": 0,
                    "source_task_description": source_desc,
                    "source_subtask_index": next_subtask_index,
                }
            )
            next_subtask_index += 1
            continue
        if pred in {"turnon", "turn_on"} and len(args) == 1:
            target_oid = args[0]
            out.append(
                {
                    "action_type": "turn_on",
                    "primary_object_id": target_oid,
                    "target_object_id": target_oid,
                    "source_task_id": 0,
                    "source_task_description": source_desc,
                    "source_subtask_index": next_subtask_index,
                }
            )
            next_subtask_index += 1
            continue
    return out


def _plan_object_ids(action_plan: List[dict]) -> set[str]:
    out: set[str] = set()
    for step in action_plan or []:
        for key in ("primary_object_id", "target_object_id"):
            oid = str(step.get(key, "") or "").strip()
            if oid:
                out.add(oid)
    return out


def build_action_plan(task_meta: dict, action_plan: Optional[List[dict]] = None) -> List[dict]:
    canonical_goal_ids = _canonical_goal_object_ids(task_meta)
    canonical_goal_plan = _canonical_goal_action_plan(task_meta)

    explicit = action_plan or task_meta.get("action_plan") or []
    if explicit:
        explicit_steps = [dict(step or {}) for step in explicit]
        if (
            canonical_goal_ids
            and canonical_goal_plan
            and not canonical_goal_ids.issubset(_plan_object_ids(explicit_steps))
        ):
            return canonical_goal_plan
        return explicit_steps

    source_prior_ids = [int(x) for x in task_meta.get("source_prior_task_ids", [])]
    if source_prior_ids and all(source_tid in ATOMIC_ACTION_PLAN for source_tid in source_prior_ids):
        out: List[dict] = []
        for source_tid in source_prior_ids:
            source_desc = SOURCE_PRIOR_TASKS[source_tid]
            for step in ATOMIC_ACTION_PLAN[source_tid]:
                out.append(
                    {
                        **dict(step),
                        "source_task_id": source_tid,
                        "source_task_description": source_desc,
                    }
                )
        if (
            canonical_goal_ids
            and canonical_goal_plan
            and not canonical_goal_ids.issubset(_plan_object_ids(out))
        ):
            return canonical_goal_plan
        return out

    inferred = _infer_single_task_action_plan(task_meta)
    if inferred:
        return inferred

    if canonical_goal_plan:
        return canonical_goal_plan

    out: List[dict] = []
    for source_tid in source_prior_ids:
        source_desc = SOURCE_PRIOR_TASKS[source_tid]
        for step in ATOMIC_ACTION_PLAN[source_tid]:
            out.append(
                {
                    **dict(step),
                    "source_task_id": source_tid,
                    "source_task_description": source_desc,
                }
            )
    return out


def synthesize_episode(task_meta: dict, action_plan: Optional[List[dict]] = None) -> dict:
    language = str(task_meta.get("language") or task_meta.get("task_name") or "")
    action_plan = build_action_plan(task_meta, action_plan=action_plan)

    objects_out: Dict[str, dict] = {}
    merged_segments: List[dict] = []
    next_subtask_id = 1

    for step in action_plan:
        primary_oid = str(step.get("primary_object_id", "")).strip()
        target_oid = str(step.get("target_object_id") or primary_oid).strip()
        for oid in (primary_oid, target_oid):
            if oid and oid not in objects_out:
                objects_out[oid] = _copy_object_meta(oid)

        source_tid = int(step.get("source_task_id", 0) or 0)
        source_desc = str(
            step.get("source_task_description")
            or SOURCE_PRIOR_TASKS.get(source_tid, "")
        )
        merged_segments.append(
            {
                "subtask_id": next_subtask_id,
                "action_type": str(step.get("action_type", "grasp")),
                "primary_object_id": primary_oid,
                "target_object_id": target_oid,
                "object": objects_out.get(primary_oid, {}).get("name", primary_oid),
                "target": objects_out.get(target_oid, {}).get("name", target_oid),
                "start_frame": 0,
                "end_frame": 0,
                "source_task_description": source_desc,
                "source_subtask_index": int(step.get("source_subtask_index", next_subtask_id)),
            }
        )
        next_subtask_id += 1

    return {
        "episode_index": -1,
        "task_description": language,
        "task_name": language,
        "task_index": -1,
        "objects": objects_out,
        "frames": {},
        "subtask_segments": merged_segments,
    }
