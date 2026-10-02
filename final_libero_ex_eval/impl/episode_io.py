"""Episode JSON loading and normalization helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable


def slugify_task_description(task_description: str) -> str:
    """Convert task language to the replay-pair directory naming scheme."""
    text = str(task_description or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown_task"


def _iter_object_aliases(value: Any) -> Iterable[str]:
    """Yield canonical + compatibility aliases for object lookup."""
    if value is None:
        return
    text = str(value)
    if not text:
        return
    yield text

    # Compatibility: some segments use `*_burner_plate` while object table uses
    # `*_burner`. Map both directions so release target resolution stays stable.
    if text.endswith("_burner_plate"):
        yield text[:-len("_plate")]
    if text.endswith("_burner"):
        yield f"{text}_plate"


def _normalize_object_map(objects_in: Dict[str, Dict[str, Any]]) -> tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    objects: Dict[str, Dict[str, Any]] = {}
    name_to_id: Dict[str, str] = {}

    for raw_oid, raw_obj in (objects_in or {}).items():
        oid = str(raw_oid)
        obj = dict(raw_obj or {})
        name = str(obj.get("name") or obj.get("body_name") or oid)
        body_name = str(obj.get("body_name") or name)

        obj["name"] = name
        obj["body_name"] = body_name

        if "body_ids" not in obj:
            body_id = obj.get("body_id")
            obj["body_ids"] = [] if body_id is None else [int(body_id)]
        else:
            obj["body_ids"] = [int(v) for v in (obj.get("body_ids") or [])]

        if "geom_ids" in obj and obj["geom_ids"] is not None:
            obj["geom_ids"] = [int(v) for v in obj.get("geom_ids") or []]

        objects[oid] = obj
        for alias in (oid, name, body_name):
            for candidate in _iter_object_aliases(alias):
                name_to_id[candidate] = oid

    return objects, name_to_id


def _resolve_object_id(
    value: Any,
    name_to_id: Dict[str, str],
) -> str:
    if value is None:
        return ""
    text = str(value)
    for candidate in _iter_object_aliases(text):
        resolved = name_to_id.get(candidate)
        if resolved is not None:
            return resolved
    return text


def _normalize_subtask_segments(
    segments_in: Iterable[Dict[str, Any]],
    name_to_id: Dict[str, str],
    objects: Dict[str, Dict[str, Any]],
) -> list[Dict[str, Any]]:
    segments = []
    for idx, raw_seg in enumerate(segments_in or []):
        seg = dict(raw_seg or {})
        seg["subtask_id"] = int(seg.get("subtask_id", idx))
        action_type = str(seg.get("action_type", "grasp"))

        manip_object_id = _resolve_object_id(
            seg.get("manip_object_id", seg.get("object", seg.get("primary_object_id"))),
            name_to_id,
        )
        target_object_id = _resolve_object_id(
            seg.get("target_object_id", seg.get("target")),
            name_to_id,
        )
        existing_primary_object_id = _resolve_object_id(seg.get("primary_object_id"), name_to_id)

        if action_type in ("release", "push") and target_object_id in objects:
            primary_object_id = target_object_id
        elif existing_primary_object_id in objects:
            primary_object_id = existing_primary_object_id
        elif existing_primary_object_id:
            primary_object_id = existing_primary_object_id
        elif manip_object_id in objects:
            primary_object_id = manip_object_id
        elif target_object_id in objects:
            primary_object_id = target_object_id
        else:
            primary_object_id = manip_object_id or target_object_id

        seg["primary_object_id"] = str(primary_object_id)
        seg["manip_object_id"] = str(manip_object_id)
        seg["target_object_id"] = str(target_object_id)

        if primary_object_id in objects:
            seg["primary_object_name"] = objects[primary_object_id].get("name", str(primary_object_id))
        if manip_object_id in objects:
            seg["manip_object_name"] = objects[manip_object_id].get("name", str(manip_object_id))
        if target_object_id in objects:
            seg["target_object_name"] = objects[target_object_id].get("name", str(target_object_id))

        segments.append(seg)
    return segments


def normalize_episode_record(raw_episode: Dict[str, Any]) -> Dict[str, Any]:
    episode = dict(raw_episode or {})
    episode["episode_index"] = int(episode.get("episode_index", -1))

    objects, name_to_id = _normalize_object_map(episode.get("objects") or {})
    episode["objects"] = objects

    frames_out: Dict[str, Dict[str, Any]] = {}
    for raw_frame_idx, raw_frame in (episode.get("frames") or {}).items():
        frame = dict(raw_frame or {})
        frame_objects = {
            str(raw_oid): dict(raw_obj or {})
            for raw_oid, raw_obj in (frame.get("objects") or {}).items()
        }
        frame["objects"] = frame_objects
        frames_out[str(raw_frame_idx)] = frame
    episode["frames"] = frames_out

    episode["subtask_segments"] = _normalize_subtask_segments(
        episode.get("subtask_segments") or [],
        name_to_id=name_to_id,
        objects=objects,
    )
    return episode


def load_episode_records(json_path: str) -> Dict[int, Dict[str, Any]]:
    path = Path(json_path)
    with open(path) as f:
        raw = json.load(f)

    if isinstance(raw, dict) and isinstance(raw.get("episodes"), dict):
        episodes_iter = raw["episodes"].values()
    elif isinstance(raw, list):
        episodes_iter = raw
    else:
        raise ValueError(f"Unsupported episode JSON format: {path}")

    episodes: Dict[int, Dict[str, Any]] = {}
    for raw_episode in episodes_iter:
        episode = normalize_episode_record(raw_episode)
        episodes[int(episode["episode_index"])] = episode
    return episodes
