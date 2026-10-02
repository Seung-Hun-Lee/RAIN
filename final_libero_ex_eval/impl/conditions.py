from __future__ import annotations

from dataclasses import dataclass
from typing import List

from final_libero_ex_eval.impl.benchmark_support import synthesize_episode
from final_libero_ex_eval.impl.runtime import SubtaskCondition, build_subtask_conditions


@dataclass
class LiberoEXCondition(SubtaskCondition):
    source_task_description: str = ""
    primary_object_id: str = ""
    target_object_id: str = ""
    source_subtask_index: int = 0


def episode_with_task_desc(episode_data: dict, task_desc: str) -> dict:
    return {
        **episode_data,
        "task_description": task_desc,
    }


def build_libero_ex_conditions(task_meta: dict, action_plan: List[dict], dino_input_size: int):
    episode = synthesize_episode(task_meta, action_plan=action_plan)
    base_conditions = build_subtask_conditions(episode, dino_input_size=dino_input_size)
    conditions: List[LiberoEXCondition] = []
    for cond, seg in zip(base_conditions, episode.get("subtask_segments", [])):
        primary_object_id = str(seg.get("primary_object_id", ""))
        target_object_id = str(seg.get("target_object_id", ""))
        object_id = cond.object_id
        object_name = cond.object_name
        if str(seg.get("action_type", "")).strip().lower() == "push" and primary_object_id:
            object_id = primary_object_id
            object_name = str(
                episode.get("objects", {}).get(primary_object_id, {}).get("name", object_name)
            )
        cond_kwargs = dict(cond.__dict__)
        cond_kwargs["object_id"] = object_id
        cond_kwargs["object_name"] = object_name
        conditions.append(
            LiberoEXCondition(
                **cond_kwargs,
                source_task_description=str(seg.get("source_task_description", "")),
                primary_object_id=primary_object_id,
                target_object_id=target_object_id,
                source_subtask_index=int(seg.get("source_subtask_index", 0) or 0),
            )
        )
    return episode, conditions
