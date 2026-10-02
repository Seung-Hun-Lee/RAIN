from __future__ import annotations

import re
from typing import Optional


_ATOM_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$")


def normalize_predicate_name(name: str) -> str:
    text = str(name or "").strip().lower()
    alias = {
        "turn_on": "turnon",
        "turn_off": "turnoff",
    }
    return alias.get(text, text)


def parse_goal_atom(text: str):
    raw = str(text or "").strip()
    if not raw:
        return None
    m = _ATOM_RE.match(raw)
    if not m:
        return None
    pred = normalize_predicate_name(m.group(1))
    args = [part.strip() for part in m.group(2).split(",") if part.strip()]
    if len(args) not in (1, 2):
        return None
    return tuple([pred] + args)


def compile_eval_rules(eval_rules: Optional[dict]) -> dict:
    rules = dict(eval_rules or {})
    sequence_stages = []
    for raw_stage in rules.get("sequence_stages") or []:
        parsed_stage = [
            atom
            for atom in (parse_goal_atom(raw) for raw in (raw_stage or []))
            if atom is not None
        ]
        if parsed_stage:
            sequence_stages.append(parsed_stage)
    return {
        "category": str(rules.get("category", "")),
        "custom_eval_needed": bool(rules.get("custom_eval_needed", False)),
        "continue_after_success": bool(rules.get("continue_after_success", False)),
        "overshoot_policy": str(rules.get("overshoot_policy", "none")),
        "required_atoms": [
            atom
            for atom in (
                parse_goal_atom(raw) for raw in (rules.get("required_goal_atoms") or [])
            )
            if atom is not None
        ],
        "forbidden_atoms": [
            atom
            for atom in (
                parse_goal_atom(raw) for raw in (rules.get("forbidden_goal_atoms") or [])
            )
            if atom is not None
        ],
        "sequence_stages": sequence_stages,
    }


def eval_goal_atom(env, atom) -> bool:
    if atom is None:
        return False
    pred = str(atom[0]).strip().lower()
    args = [str(x).strip() for x in atom[1:]]
    try:
        return bool(env.env._eval_predicate(tuple([pred] + args)))
    except Exception:
        return False


def init_eval_tracker(compiled_rules: dict) -> dict:
    required_atoms = list(compiled_rules.get("required_atoms", []))
    return {
        "custom": bool(compiled_rules.get("custom_eval_needed", False)),
        "category": str(compiled_rules.get("category", "")).strip().upper(),
        "required_atoms": required_atoms,
        # These fields are diagnostic only.  They let post-processing
        # distinguish a genuine partial composition (for example, the first
        # two of three BDDL goals were reached) from a progress-head switch.
        # Keeping one slot per required atom also preserves the instruction
        # order encoded by the benchmark's eval_rules.yaml.
        "required_atom_first_steps": [None] * len(required_atoms),
        "required_atom_final": [False] * len(required_atoms),
        "required_prefix_first_steps": [None] * len(required_atoms),
        "required_combination_first_steps": {},
        "max_required_atoms_satisfied": 0,
        "max_required_atom_values": [False] * len(required_atoms),
        "max_required_atoms_first_step": None,
        "forbidden_atoms": list(compiled_rules.get("forbidden_atoms", [])),
        "overshoot_policy": str(compiled_rules.get("overshoot_policy", "none")),
        "continue_after_success": bool(compiled_rules.get("continue_after_success", False)),
        "required_ever": False,
        "forbidden_ever": False,
        "required_first_step": None,
        "forbidden_first_step": None,
        "sequence_stages": [
            list(stage) for stage in compiled_rules.get("sequence_stages", [])
        ],
        "sequence_stage_index": 0,
        "sequence_stage_first_steps": [],
    }


def is_task_decomposition(tracker: dict) -> bool:
    category = re.sub(r"[\s_-]+", "", str(tracker.get("category", "")).upper())
    return category in {"TD", "DECOMPOSITION", "TASKDECOMPOSITION"}


def custom_stops_on_goal(tracker: dict) -> bool:
    """Non-decomposition termination never depends on learned TC.

    Explicit continue-after-success/overshoot protocols remain respected.
    This does not affect intermediate subtask switching.
    """
    return (
        bool(tracker.get("custom", False))
        and not is_task_decomposition(tracker)
        and not bool(tracker.get("continue_after_success", False))
    )


def update_eval_tracker(env, tracker: dict, step_idx: int) -> None:
    required_atoms = tracker.get("required_atoms", [])
    forbidden_atoms = tracker.get("forbidden_atoms", [])
    required_values = [eval_goal_atom(env, atom) for atom in required_atoms]
    tracker["required_atom_final"] = list(required_values)
    first_steps = tracker.get("required_atom_first_steps", [])
    if len(first_steps) != len(required_values):
        first_steps = [None] * len(required_values)
    for index, value in enumerate(required_values):
        if value and first_steps[index] is None:
            first_steps[index] = int(step_idx)
    tracker["required_atom_first_steps"] = first_steps
    prefix_first_steps = tracker.get("required_prefix_first_steps", [])
    if len(prefix_first_steps) != len(required_values):
        prefix_first_steps = [None] * len(required_values)
    for index in range(len(required_values)):
        if all(required_values[: index + 1]) and prefix_first_steps[index] is None:
            prefix_first_steps[index] = int(step_idx)
    tracker["required_prefix_first_steps"] = prefix_first_steps

    if required_values:
        combination = "".join("1" if value else "0" for value in required_values)
        combination_steps = tracker.get("required_combination_first_steps", {})
        if combination not in combination_steps:
            combination_steps[combination] = int(step_idx)
        tracker["required_combination_first_steps"] = combination_steps
        satisfied = sum(bool(value) for value in required_values)
        if satisfied > int(tracker.get("max_required_atoms_satisfied", 0)):
            tracker["max_required_atoms_satisfied"] = int(satisfied)
            tracker["max_required_atom_values"] = list(required_values)
            tracker["max_required_atoms_first_step"] = int(step_idx)

    # Standard LIBERO tasks use env.check_success() for the outcome. Collect
    # atom-level diagnostics above, then skip custom temporal/forbidden scoring.
    if not tracker.get("custom", False):
        return

    required_now = bool(required_atoms) and all(required_values)
    forbidden_now = bool(forbidden_atoms) and any(
        eval_goal_atom(env, atom) for atom in forbidden_atoms
    )
    if required_now and not tracker["required_ever"]:
        tracker["required_ever"] = True
        tracker["required_first_step"] = int(step_idx)
    if forbidden_now and not tracker["forbidden_ever"]:
        tracker["forbidden_ever"] = True
        tracker["forbidden_first_step"] = int(step_idx)
    stages = tracker.get("sequence_stages", [])
    stage_index = int(tracker.get("sequence_stage_index", 0))
    # Advancing through all currently-satisfied stages is intentional for
    # nested monotonic milestones.  A true transition such as Turnon ->
    # Turnoff cannot collapse because the two predicates are mutually
    # exclusive in one simulator state.
    while stage_index < len(stages):
        stage = stages[stage_index]
        if not stage or not all(eval_goal_atom(env, atom) for atom in stage):
            break
        tracker["sequence_stage_first_steps"].append(int(step_idx))
        stage_index += 1
    tracker["sequence_stage_index"] = stage_index


def custom_eval_success(tracker: dict) -> bool:
    stages = tracker.get("sequence_stages", [])
    sequence_complete = not stages or int(
        tracker.get("sequence_stage_index", 0)
    ) >= len(stages)
    return (
        bool(tracker.get("required_ever", False))
        and sequence_complete
        and not bool(tracker.get("forbidden_ever", False))
    )


def custom_eval_failed(tracker: dict) -> bool:
    return (
        str(tracker.get("overshoot_policy", "none")) == "fail_on_forbidden_ever"
        and bool(tracker.get("forbidden_ever", False))
    )


def custom_eval_now(env, tracker: dict) -> tuple[bool, bool]:
    required_atoms = tracker.get("required_atoms", [])
    forbidden_atoms = tracker.get("forbidden_atoms", [])
    required_now = bool(required_atoms) and all(
        eval_goal_atom(env, atom) for atom in required_atoms
    )
    stages = tracker.get("sequence_stages", [])
    if stages and int(tracker.get("sequence_stage_index", 0)) < len(stages):
        required_now = False
    forbidden_now = bool(forbidden_atoms) and any(
        eval_goal_atom(env, atom) for atom in forbidden_atoms
    )
    return required_now, forbidden_now
