from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np

from shared.eval_transition import subtask_completion_ready

from final_libero_ex_eval.impl.benchmark_support import find_source_episode
from final_libero_ex_eval.impl.conditions import episode_with_task_desc
from final_libero_ex_eval.impl.custom_eval import (
    compile_eval_rules,
    custom_stops_on_goal,
    custom_eval_failed,
    custom_eval_now,
    custom_eval_success,
    init_eval_tracker,
    is_task_decomposition,
    update_eval_tracker,
)
from final_libero_ex_eval.impl.runtime import (
    DUMMY_ACTION,
    GRIPPER_ACTIONS,
    LIBERO_ENV_RESOLUTION,
    NUM_STEPS_WAIT,
    _render_views_once,
    _resolve_object_ref_to_known_id,
    compose_vis_frame,
    downsample_mask_to_patches,
    extract_state,
    is_object_segmentable,
    load_subtask_mask,
    sim_mask_for_object_id,
    sim_region_mask_for_object_id,
)


def _mask_to_patches(mask, patch_grid: int) -> np.ndarray:
    return downsample_mask_to_patches(mask, grid=patch_grid).astype(np.float32)


def _mask_bbox(mask) -> list[int] | None:
    if mask is None:
        return None
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _load_source_mask(
    source_episodes,
    cond_like,
    object_id_override: Optional[str],
    view: str,
    patch_grid: int,
):
    if source_episodes is None:
        return None, None
    if not cond_like.source_task_description or cond_like.source_subtask_index <= 0:
        return None, None
    try:
        source_episode = find_source_episode(source_episodes, cond_like.source_task_description)
    except Exception:
        return None, None
    segs = sorted(
        source_episode.get("subtask_segments", []),
        key=lambda x: int(x.get("subtask_id", 0)),
    )
    idx = int(cond_like.source_subtask_index) - 1
    if idx < 0 or idx >= len(segs):
        return None, None
    source_seg = segs[idx]
    source_obj_id = str(object_id_override or "").strip()
    if not source_obj_id:
        source_obj_id = str(source_seg.get("primary_object_id", ""))
        if str(source_seg.get("action_type", "")).strip().lower() in ("release", "push"):
            target_ref = str(source_seg.get("target_object_id", ""))
            resolved_target_obj_id = _resolve_object_ref_to_known_id(source_episode, target_ref)
            if resolved_target_obj_id:
                source_obj_id = resolved_target_obj_id
    prefer_nonzero = view == "wrist"
    mask_raw, mask_patches = load_subtask_mask(
        source_episode,
        source_seg,
        "visible",
        mask_object_id=source_obj_id,
        patch_grid=patch_grid,
        view=view,
        prefer_nonzero=prefer_nonzero,
    )
    if mask_raw is None:
        mask_raw, mask_patches = load_subtask_mask(
            source_episode,
            source_seg,
            "bbox",
            mask_object_id=source_obj_id,
            patch_grid=patch_grid,
            view=view,
            prefer_nonzero=prefer_nonzero,
        )
    if mask_raw is None or mask_patches is None:
        return None, None
    return mask_raw.astype(np.uint8), mask_patches.astype(np.float32)


def _compute_place_masks(
    env,
    source_episodes,
    source_cond,
    target_episode,
    target_object_id: str,
    corrected_bids,
    patch_grid: int,
    image_size: int,
    bddl_path: str,
):
    place_patches = None
    place_wrist_patches = None
    place_raw = None
    place_wrist_raw = None
    place_source = "none"
    place_wrist_source = "none"
    target_object_id = str(target_object_id or "").strip()
    if not target_object_id:
        return (
            place_patches,
            place_wrist_patches,
            place_raw,
            place_wrist_raw,
            place_source,
            place_wrist_source,
        )

    if is_object_segmentable(target_episode, target_object_id):
        place_sm = sim_mask_for_object_id(
            env,
            target_episode,
            target_object_id,
            corrected_bids=corrected_bids,
        )
        if place_sm is not None:
            place_raw = place_sm
            place_patches = _mask_to_patches(place_sm, patch_grid)
            place_source = "sim_seg"
        place_wrist_sm = sim_mask_for_object_id(
            env,
            target_episode,
            target_object_id,
            corrected_bids=corrected_bids,
            camera_name="robot0_eye_in_hand",
        )
        if place_wrist_sm is not None:
            place_wrist_raw = place_wrist_sm
            place_wrist_patches = _mask_to_patches(place_wrist_sm, patch_grid)
            place_wrist_source = "sim_seg"
        return (
            place_patches,
            place_wrist_patches,
            place_raw,
            place_wrist_raw,
            place_source,
            place_wrist_source,
        )

    place_region_agent = sim_region_mask_for_object_id(
        env,
        target_episode,
        target_object_id,
        bddl_path=bddl_path,
        image_size=image_size,
        camera_name="agentview",
    )
    source_agent_raw, source_agent_patches = _load_source_mask(
        source_episodes,
        source_cond,
        object_id_override=target_object_id,
        view="agent",
        patch_grid=patch_grid,
    )
    if place_region_agent is not None:
        place_raw = place_region_agent
        place_patches = _mask_to_patches(place_region_agent, patch_grid)
        place_source = "sim_region_poly"
    elif source_agent_raw is not None:
        place_raw = source_agent_raw
        place_patches = source_agent_patches
        place_source = "json_region_fallback"

    place_region_wrist = sim_region_mask_for_object_id(
        env,
        target_episode,
        target_object_id,
        bddl_path=bddl_path,
        image_size=image_size,
        camera_name="robot0_eye_in_hand",
    )
    source_wrist_raw, source_wrist_patches = _load_source_mask(
        source_episodes,
        source_cond,
        object_id_override=target_object_id,
        view="wrist",
        patch_grid=patch_grid,
    )
    if place_region_wrist is not None:
        place_wrist_raw = place_region_wrist
        place_wrist_patches = _mask_to_patches(place_region_wrist, patch_grid)
        place_wrist_source = "sim_region_poly"
    elif source_wrist_raw is not None:
        place_wrist_raw = source_wrist_raw
        place_wrist_patches = source_wrist_patches
        place_wrist_source = "json_region_fallback"

    return (
        place_patches,
        place_wrist_patches,
        place_raw,
        place_wrist_raw,
        place_source,
        place_wrist_source,
    )


def _resolve_active_masks(
    env,
    cond,
    cond_episode,
    corrected_bids,
    patch_grid: int,
    num_patches: int,
    image_size: int,
    bddl_path: str,
    turn_on_sim_only: bool,
):
    active_mask_raw = cond.mask_raw
    active_mask_patches = cond.mask_patches
    active_wrist_mask_raw = cond.wrist_mask_raw
    active_wrist_mask_patches = cond.wrist_mask_patches
    active_mask_source = cond.mask_source

    if is_object_segmentable(cond_episode, cond.object_id):
        is_stove_turn = str(cond.action_type).strip().lower() in {"turn_on", "turn_off"}
        sm = sim_mask_for_object_id(
            env,
            cond_episode,
            cond.object_id,
            corrected_bids=corrected_bids,
            camera_name="agentview",
            action_type=cond.action_type,
        )
        if sm is not None:
            active_mask_raw = sm
            active_mask_patches = _mask_to_patches(sm, patch_grid)
            active_mask_source = "sim_seg_stove_knob" if is_stove_turn else "sim_seg"
        # Stove toggles are always knob-only.  If the live simulator cannot
        # resolve the button geom, never fall back to a cached whole-stove
        # JSON mask, regardless of the compatibility flag.
        elif is_stove_turn:
            active_mask_raw = None
            active_mask_patches = np.zeros(num_patches, dtype=np.float32)
            active_mask_source = "sim_missing_stove_knob"
        elif cond.mask_patches is not None and float(cond.mask_patches.sum()) > 0.0:
            active_mask_raw = cond.mask_raw
            active_mask_patches = cond.mask_patches.astype(np.float32)
            active_mask_source = "json_fallback"
        else:
            active_mask_raw = None
            active_mask_patches = np.zeros(num_patches, dtype=np.float32)
            active_mask_source = "sim_missing"

        wrist_sm = sim_mask_for_object_id(
            env,
            cond_episode,
            cond.object_id,
            corrected_bids=corrected_bids,
            camera_name="robot0_eye_in_hand",
            action_type=cond.action_type,
        )
        if wrist_sm is not None:
            active_wrist_mask_raw = wrist_sm
            active_wrist_mask_patches = _mask_to_patches(wrist_sm, patch_grid)
        elif is_stove_turn:
            active_wrist_mask_raw = None
            active_wrist_mask_patches = np.zeros(num_patches, dtype=np.float32)
        elif cond.wrist_mask_patches is not None and float(cond.wrist_mask_patches.sum()) > 0.0:
            active_wrist_mask_raw = cond.wrist_mask_raw
            active_wrist_mask_patches = cond.wrist_mask_patches.astype(np.float32)
        else:
            active_wrist_mask_raw = None
            active_wrist_mask_patches = np.zeros(num_patches, dtype=np.float32)
        return (
            active_mask_raw,
            active_mask_patches,
            active_wrist_mask_raw,
            active_wrist_mask_patches,
            active_mask_source,
        )

    region_agent = sim_region_mask_for_object_id(
        env,
        cond_episode,
        cond.object_id,
        bddl_path=bddl_path,
        image_size=image_size,
        camera_name="agentview",
    )
    if region_agent is not None:
        active_mask_raw = region_agent
        active_mask_patches = _mask_to_patches(region_agent, patch_grid)
        active_mask_source = "sim_region"
    elif cond.mask_patches is not None and float(cond.mask_patches.sum()) > 0.0:
        active_mask_raw = cond.mask_raw
        active_mask_patches = cond.mask_patches.astype(np.float32)
        active_mask_source = "json_region_fallback"
    else:
        active_mask_raw = None
        active_mask_patches = np.zeros(num_patches, dtype=np.float32)
        active_mask_source = "sim_region_missing"

    region_wrist = sim_region_mask_for_object_id(
        env,
        cond_episode,
        cond.object_id,
        bddl_path=bddl_path,
        image_size=image_size,
        camera_name="robot0_eye_in_hand",
    )
    if region_wrist is not None:
        active_wrist_mask_raw = region_wrist
        active_wrist_mask_patches = _mask_to_patches(region_wrist, patch_grid)
    elif cond.wrist_mask_patches is not None and float(cond.wrist_mask_patches.sum()) > 0.0:
        active_wrist_mask_raw = cond.wrist_mask_raw
        active_wrist_mask_patches = cond.wrist_mask_patches.astype(np.float32)
    else:
        active_wrist_mask_raw = None
        active_wrist_mask_patches = np.zeros(num_patches, dtype=np.float32)

    return (
        active_mask_raw,
        active_mask_patches,
        active_wrist_mask_raw,
        active_wrist_mask_patches,
        active_mask_source,
    )


def _resolve_prev_completion_patches(
    env,
    prev_episode,
    prev_oid: str,
    prev_action_type: str,
    corrected_bids,
    patch_grid: int,
    num_patches: int,
):
    prev_patches = np.zeros(num_patches, dtype=np.float32)
    prev_wrist_patches = np.zeros(num_patches, dtype=np.float32)
    if not is_object_segmentable(prev_episode, prev_oid):
        return prev_patches, prev_wrist_patches

    prev_sm = sim_mask_for_object_id(
        env,
        prev_episode,
        prev_oid,
        corrected_bids=corrected_bids,
        camera_name="agentview",
        action_type=prev_action_type,
    )
    if prev_sm is not None:
        prev_patches = _mask_to_patches(prev_sm, patch_grid)
    prev_wrist_sm = sim_mask_for_object_id(
        env,
        prev_episode,
        prev_oid,
        corrected_bids=corrected_bids,
        camera_name="robot0_eye_in_hand",
        action_type=prev_action_type,
    )
    if prev_wrist_sm is not None:
        prev_wrist_patches = _mask_to_patches(prev_wrist_sm, patch_grid)
    return prev_patches, prev_wrist_patches


def run_single_episode_libero_ex(
    gpu_worker,
    env,
    init_state,
    text_feat,
    episode_data,
    conditions,
    replan_steps=8,
    num_inference_steps=4,
    max_steps=520,
    feas_threshold=0.7,
    consecutive_stop=2,
    record_video=True,
    corrected_bids=None,
    dino_input_size=224,
    state_dim=8,
    bddl_path="",
    source_episodes=None,
    eval_rules=None,
    prepared_observation=None,
    benchmark_scorer=None,
    decomposition_tc_stop=False,
):
    H = W = LIBERO_ENV_RESOLUTION
    patch_grid = dino_input_size // 14
    num_patches = patch_grid * patch_grid

    if prepared_observation is None:
        obs = env.reset()
        obs = env.set_init_state(init_state)
    else:
        obs = prepared_observation
    compiled_eval_rules = compile_eval_rules(eval_rules)
    eval_tracker = init_eval_tracker(compiled_eval_rules)
    if prepared_observation is None:
        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = env.step(DUMMY_ACTION)
    update_eval_tracker(env, eval_tracker, step_idx=0)

    total_steps = 0
    def step_env(action):
        result = env.step(action)
        if benchmark_scorer is not None:
            benchmark_scorer.observe(total_steps + 1)
        return result

    def failed():
        if benchmark_scorer is not None:
            return benchmark_scorer.violation
        return custom_eval_failed(eval_tracker)

    from rain.transition_control import FinalConditionTCStop
    final_tc = FinalConditionTCStop()
    final_tc_stop_step = None
    sub_idx = 0
    consecutive_complete = 0
    success = False
    success_reason = ""
    custom_goal_stop = custom_stops_on_goal(eval_tracker)
    if benchmark_scorer is not None:
        eval_tracker["custom"] = True
        custom_goal_stop = not decomposition_tc_stop

    def finish_custom_goal_if_ready():
        nonlocal success, success_reason
        if not custom_goal_stop:
            return False
        if benchmark_scorer is not None:
            if benchmark_scorer.success():
                success = True
                success_reason = "benchmark_goal_order_physical"
                return True
            return False
        # Custom observers supply the existing goal/order/forbidden checks.
        # Do not use required_ever alone: current final goals must still hold.
        required_now, forbidden_now = custom_eval_now(env, eval_tracker)
        if required_now and not forbidden_now and not bool(eval_tracker.get("forbidden_ever", False)):
            success = True
            success_reason = "custom_goal_without_tc"
            return True
        return False

    frames = [] if record_video else None
    rollout_timeline = []
    just_switched = False
    prev_check_info = None
    rollout_started = time.monotonic()
    progress_interval = max(
        0, int(os.environ.get("RAIN_ROLLOUT_PROGRESS_STEPS", "64"))
    )
    next_progress_step = 0
    turn_on_sim_only = str(
        os.environ.get("RAIN_TURN_ON_SIM_ONLY", "1")
    ).strip().lower() not in {"0", "false", "off", "no"}

    while total_steps < max_steps:
        if finish_custom_goal_if_ready():
            break
        if progress_interval and total_steps >= next_progress_step:
            print(
                f"  [rollout] step={total_steps}/{max_steps} "
                f"subtask={sub_idx + 1}/{len(conditions)} "
                f"elapsed={time.monotonic() - rollout_started:.1f}s",
                flush=True,
            )
            next_progress_step = total_steps + progress_interval
        cond = conditions[min(sub_idx, len(conditions) - 1)]
        cond_episode = episode_with_task_desc(
            episode_data,
            cond.source_task_description or episode_data.get("task_description", ""),
        )

        (
            active_mask_raw,
            active_mask_patches,
            active_wrist_mask_raw,
            active_wrist_mask_patches,
            active_mask_source,
        ) = _resolve_active_masks(
            env,
            cond,
            cond_episode,
            corrected_bids,
            patch_grid=patch_grid,
            num_patches=num_patches,
            image_size=H,
            bddl_path=bddl_path,
            turn_on_sim_only=turn_on_sim_only,
        )

        img_t, img_w = _render_views_once(env, H, W)
        state = extract_state(obs, state_dim=state_dim)

        active_place_patches = cond.target_place_patches
        active_place_wrist_patches = cond.target_place_wrist_patches
        active_place_raw = None
        active_place_wrist_raw = None
        active_place_source = "none"
        active_place_wrist_source = "none"
        if str(cond.action_type).strip().lower() == "push" and cond.target_object_id:
            (
                active_place_patches,
                active_place_wrist_patches,
                active_place_raw,
                active_place_wrist_raw,
                active_place_source,
                active_place_wrist_source,
            ) = _compute_place_masks(
                env,
                source_episodes,
                cond,
                cond_episode,
                cond.target_object_id,
                corrected_bids,
                patch_grid=patch_grid,
                image_size=H,
                bddl_path=bddl_path,
            )
        elif (
            active_place_patches is not None
            and cond.action_type in ("grasp", "push")
            and sub_idx + 1 < len(conditions)
        ):
            next_cond = conditions[sub_idx + 1]
            next_episode = episode_with_task_desc(
                episode_data,
                next_cond.source_task_description or episode_data.get("task_description", ""),
            )
            next_place_object_id = (
                next_cond.target_object_id
                if str(next_cond.action_type).strip().lower() == "push"
                and next_cond.target_object_id
                else next_cond.object_id
            )
            (
                next_place_patches,
                next_place_wrist_patches,
                next_place_raw,
                next_place_wrist_raw,
                next_place_source,
                next_place_wrist_source,
            ) = _compute_place_masks(
                env,
                source_episodes,
                next_cond,
                next_episode,
                next_place_object_id,
                corrected_bids,
                patch_grid=patch_grid,
                image_size=H,
                bddl_path=bddl_path,
            )
            if next_place_patches is not None:
                active_place_patches = next_place_patches
            if next_place_wrist_patches is not None:
                active_place_wrist_patches = next_place_wrist_patches
            if next_place_raw is not None:
                active_place_raw = next_place_raw
                active_place_source = next_place_source
            else:
                active_place_raw = next_cond.mask_raw if next_cond.mask_raw is not None else active_place_raw
            if next_place_wrist_raw is not None:
                active_place_wrist_raw = next_place_wrist_raw
                active_place_wrist_source = next_place_wrist_source
            else:
                active_place_wrist_raw = next_cond.wrist_mask_raw
                if (
                    active_place_wrist_patches is None
                    or float(active_place_wrist_patches.sum()) <= 0.0
                ):
                    active_place_wrist_patches = next_cond.wrist_mask_patches.astype(np.float32)

        place_arg = None
        if active_place_patches is not None and float(active_place_patches.sum()) > 0:
            place_arg = active_place_patches[np.newaxis]
        place_wrist_arg = None
        if active_place_wrist_patches is not None and float(active_place_wrist_patches.sum()) > 0:
            place_wrist_arg = active_place_wrist_patches[np.newaxis]

        output = gpu_worker.infer(
            imgs_third=[img_t],
            imgs_wrist=[img_w],
            states=state[np.newaxis],
            masks=active_mask_patches[np.newaxis],
            wrist_masks=active_wrist_mask_patches[np.newaxis],
            text_feat=text_feat[np.newaxis],
            action_type=np.array([cond.action_type_id], dtype=np.int64),
            num_inference_steps=num_inference_steps,
            target_place_mask=place_arg,
            target_place_mask_wrist=place_wrist_arg,
        )

        actions = output["action"][0]
        tc_val = float(output["task_comp_prob"][0]) if "task_comp_prob" in output else 0.0
        rollout_timeline.append(
            {
                "replan_frame": len(rollout_timeline),
                "subtask_index": int(sub_idx),
                "total_steps": int(total_steps),
                "tc": tc_val,
                "gripper_qpos": np.asarray(
                    obs["robot0_gripper_qpos"], dtype=np.float64
                ).tolist(),
                "eef_pos": np.asarray(
                    obs["robot0_eef_pos"], dtype=np.float64
                ).tolist(),
                "agent_area": int(
                    0
                    if active_mask_raw is None
                    else np.asarray(active_mask_raw, dtype=bool).sum()
                ),
                "wrist_area": int(
                    0
                    if active_wrist_mask_raw is None
                    else np.asarray(active_wrist_mask_raw, dtype=bool).sum()
                ),
                "agent_bbox": _mask_bbox(active_mask_raw),
                "wrist_bbox": _mask_bbox(active_wrist_mask_raw),
                "place_agent_area": int(
                    0
                    if active_place_raw is None
                    else np.asarray(active_place_raw, dtype=bool).sum()
                ),
                "place_wrist_area": int(
                    0
                    if active_place_wrist_raw is None
                    else np.asarray(active_place_wrist_raw, dtype=bool).sum()
                ),
                "mask_source": active_mask_source,
                "place_mask_source": active_place_source,
                "place_wrist_mask_source": active_place_wrist_source,
            }
        )

        if (
            benchmark_scorer is None
            and eval_tracker["custom"]
            and is_task_decomposition(eval_tracker)
            and not bool(eval_tracker.get("continue_after_success", False))
            and tc_val > feas_threshold
        ):
            required_now, forbidden_now = custom_eval_now(env, eval_tracker)
            if required_now and not forbidden_now and not bool(eval_tracker.get("forbidden_ever", False)):
                success = True
                success_reason = "custom_tc_bddl"
                break

        if prev_check_info is not None:
            prev_oid, prev_atid, prev_action_type, prev_source_desc = prev_check_info
            prev_check_info = None
            prev_episode = episode_with_task_desc(episode_data, prev_source_desc)
            prev_patches, prev_wrist_patches = _resolve_prev_completion_patches(
                env,
                prev_episode,
                prev_oid,
                prev_action_type,
                corrected_bids,
                patch_grid=patch_grid,
                num_patches=num_patches,
            )
            prev_output = gpu_worker.infer(
                imgs_third=[img_t],
                imgs_wrist=[img_w],
                states=state[np.newaxis],
                masks=prev_patches[np.newaxis],
                wrist_masks=prev_wrist_patches[np.newaxis],
                text_feat=text_feat[np.newaxis],
                action_type=np.array([prev_atid], dtype=np.int64),
                num_inference_steps=num_inference_steps,
            )
            prev_tc = (
                float(prev_output["task_comp_prob"][0])
                if "task_comp_prob" in prev_output else 0.0
            )
            if prev_tc < feas_threshold:
                final_tc.streak = 0
                sub_idx -= 1
                consecutive_complete = 0
                # No action has been executed since the forward switch.  Keep
                # the guard set so the reverted subtask must execute one
                # action chunk before it may switch again; otherwise the two
                # completion checks can bounce forever at the same simulator
                # state while total_steps remains unchanged.
                just_switched = True
                continue

        if decomposition_tc_stop:
            ready = final_tc.observe(tc_val, final_condition=(sub_idx == len(conditions) - 1))
            rollout_timeline[-1]["rain_decomposition_tc_streak"] = final_tc.streak
            if ready:
                final_tc_stop_step = total_steps
                break

        consecutive_complete = consecutive_complete + 1 if tc_val > feas_threshold else 0
        can_switch = (
            sub_idx < len(conditions) - 1
            and not just_switched
            and subtask_completion_ready(
                consecutive_complete,
                consecutive_stop,
            )
        )

        if can_switch:
            gripper_qpos = obs["robot0_gripper_qpos"][0]
            if cond.action_type == "grasp" and gripper_qpos > 0.01:
                grip_action = GRIPPER_ACTIONS["grasp"]
                max_grip = min(16, max_steps - total_steps)
                for _ in range(max_grip):
                    obs, _, _, _ = step_env(grip_action.tolist())
                    total_steps += 1
                    if custom_goal_stop:
                        update_eval_tracker(env, eval_tracker, step_idx=total_steps)
                        if finish_custom_goal_if_ready():
                            break
                    if obs["robot0_gripper_qpos"][0] <= 0.01:
                        break
            elif cond.action_type == "release":
                grip_action = GRIPPER_ACTIONS["release"]
                max_grip = min(16, max_steps - total_steps)
                for _ in range(max_grip):
                    obs, _, _, _ = step_env(grip_action.tolist())
                    total_steps += 1
                    if custom_goal_stop:
                        update_eval_tracker(env, eval_tracker, step_idx=total_steps)
                        if finish_custom_goal_if_ready():
                            break
                    if obs["robot0_gripper_qpos"][0] >= 0.035:
                        break

            if success:
                break

            if not eval_tracker["custom"] and env.check_success():
                success = True
                success_reason = "env_success_before_switch"
                break

            prev_check_info = (
                cond.object_id,
                cond.action_type_id,
                cond.action_type,
                cond.source_task_description,
            )
            sub_idx += 1
            consecutive_complete = 0
            just_switched = True
            nxt = conditions[sub_idx]
            nxt_episode = episode_with_task_desc(
                episode_data,
                nxt.source_task_description or episode_data.get("task_description", ""),
            )

            if record_video:
                nxt_mask = None
                if is_object_segmentable(nxt_episode, nxt.object_id):
                    nxt_mask = sim_mask_for_object_id(
                        env,
                        nxt_episode,
                        nxt.object_id,
                        corrected_bids=corrected_bids,
                        action_type=nxt.action_type,
                    )
                if nxt_mask is None:
                    nxt_mask = nxt.mask_raw
                nxt_wrist_mask = None
                if is_object_segmentable(nxt_episode, nxt.object_id):
                    nxt_wrist_mask = sim_mask_for_object_id(
                        env,
                        nxt_episode,
                        nxt.object_id,
                        corrected_bids=corrected_bids,
                        camera_name="robot0_eye_in_hand",
                        action_type=nxt.action_type,
                    )
                if nxt_wrist_mask is None:
                    nxt_wrist_mask = sim_region_mask_for_object_id(
                        env,
                        nxt_episode,
                        nxt.object_id,
                        bddl_path=bddl_path,
                        image_size=H,
                        camera_name="robot0_eye_in_hand",
                    )
                if nxt_mask is None:
                    nxt_mask = sim_region_mask_for_object_id(
                        env,
                        nxt_episode,
                        nxt.object_id,
                        bddl_path=bddl_path,
                        image_size=H,
                        camera_name="agentview",
                    )
                frames.append(
                    compose_vis_frame(
                        img_t,
                        img_w,
                        nxt_mask,
                        [
                            f"SWITCH sub{cond.subtask_id} -> sub{nxt.subtask_id}",
                            f"obj: {cond.object_name} -> {nxt.object_name} [{nxt.action_type}]",
                            f"step={total_steps}/{max_steps} TC={tc_val:.2f}",
                        ],
                        tc_val=tc_val,
                        wrist_mask_raw=nxt_wrist_mask,
                    )
                )
            continue

        if record_video:
            frames.append(
                compose_vis_frame(
                    img_t,
                    img_w,
                    active_mask_raw,
                    [
                        f"sub{sub_idx + 1}/{len(conditions)} (id={cond.subtask_id}) [{cond.action_type}]",
                        f"obj={cond.object_name} mask={active_mask_source}",
                        f"place={active_place_source}/{active_place_wrist_source}",
                        f"step={total_steps}/{max_steps} TC={tc_val:.2f}",
                    ],
                    tc_val=tc_val,
                    wrist_mask_raw=active_wrist_mask_raw,
                    place_mask_raw=active_place_raw,
                    place_mask_wrist_raw=active_place_wrist_raw,
                )
            )

        steps_to_exec = min(replan_steps, max_steps - total_steps)
        for step_index in range(steps_to_exec):
            action = np.clip(actions[step_index], -1.0, 1.0)
            obs, _, _, _ = step_env(action.tolist())
            total_steps += 1
            update_eval_tracker(env, eval_tracker, step_idx=total_steps)
            if eval_tracker["custom"]:
                if failed():
                    success = False
                    success_reason = "custom_forbidden"
                    break
                if finish_custom_goal_if_ready():
                    break
            elif env.check_success():
                success = True
                success_reason = "env_success"
                break

        just_switched = False
        if success:
            break
        if eval_tracker["custom"] and failed():
            break

    if not success and not eval_tracker["custom"]:
        for _ in range(20):
            obs, _, _, _ = step_env([0.0] * 7)
            if env.check_success():
                success = True
                success_reason = "env_success_cooldown"
                break
    elif eval_tracker["custom"]:
        success = (benchmark_scorer.success(final=True) if benchmark_scorer is not None
                   else custom_eval_success(eval_tracker))
        if success and not success_reason:
            success_reason = "custom_eval"
        elif not success and not success_reason and bool(eval_tracker.get("forbidden_ever", False)):
            success_reason = "custom_forbidden"

    # Refresh atom-level diagnostics after the optional cooldown as well.
    # This does not participate in the standard LIBERO success decision.
    update_eval_tracker(env, eval_tracker, step_idx=total_steps)

    if record_video:
        fr_t, fr_w = _render_views_once(env, H, W)
        cond = conditions[min(sub_idx, len(conditions) - 1)]
        cond_episode = episode_with_task_desc(
            episode_data,
            cond.source_task_description or episode_data.get("task_description", ""),
        )
        final_mask = cond.mask_raw
        final_wrist_mask = cond.wrist_mask_raw
        if is_object_segmentable(cond_episode, cond.object_id):
            sm = sim_mask_for_object_id(
                env,
                cond_episode,
                cond.object_id,
                corrected_bids=corrected_bids,
                camera_name="agentview",
                action_type=cond.action_type,
            )
            if sm is not None:
                final_mask = sm
            wrist_sm = sim_mask_for_object_id(
                env,
                cond_episode,
                cond.object_id,
                corrected_bids=corrected_bids,
                camera_name="robot0_eye_in_hand",
                action_type=cond.action_type,
            )
            if wrist_sm is not None:
                final_wrist_mask = wrist_sm
        else:
            sm = sim_region_mask_for_object_id(
                env,
                cond_episode,
                cond.object_id,
                bddl_path=bddl_path,
                image_size=H,
                camera_name="agentview",
            )
            if sm is not None:
                final_mask = sm
            wrist_sm = sim_region_mask_for_object_id(
                env,
                cond_episode,
                cond.object_id,
                bddl_path=bddl_path,
                image_size=H,
                camera_name="robot0_eye_in_hand",
            )
            if wrist_sm is not None:
                final_wrist_mask = wrist_sm
        atom_values = list(eval_tracker.get("required_atom_final", []))
        atom_count = len(atom_values)
        atom_done = sum(bool(value) for value in atom_values)
        frames.append(
            compose_vis_frame(
                fr_t,
                fr_w,
                final_mask,
                [
                    f"DONE {'SUCCESS' if success else 'FAIL'}",
                    f"subs={min(sub_idx + 1, len(conditions))}/{len(conditions)}",
                    f"goals={atom_done}/{atom_count}" if atom_count else "goals=n/a",
                    f"steps={total_steps}/{max_steps}",
                ],
                tc_val=0.0,
                success=success,
                wrist_mask_raw=final_wrist_mask,
            )
        )

    return frames, success, {
        "subtasks_reached": min(sub_idx + 1, len(conditions)),
        "num_subtasks": len(conditions),
        "total_steps": total_steps,
        "custom_eval": bool(eval_tracker["custom"]),
        "success_reason": success_reason,
        "termination_reason": ("rain_decomposition_final_tc_ge_0_7_consecutive_2_v1"
                               if final_tc_stop_step is not None else success_reason or "max_steps"),
        "rain_decomposition_tc_stop_step": final_tc_stop_step,
        "success_termination_policy": (
            "goal_order_without_tc" if custom_goal_stop
            else "existing_category_protocol"
        ),
        "required_ever": bool(eval_tracker["required_ever"]),
        "forbidden_ever": bool(eval_tracker["forbidden_ever"]),
        "required_first_step": eval_tracker["required_first_step"],
        "forbidden_first_step": eval_tracker["forbidden_first_step"],
        "sequence_stage_index": int(eval_tracker.get("sequence_stage_index", 0)),
        "sequence_stage_count": len(eval_tracker.get("sequence_stages", [])),
        "sequence_stage_first_steps": list(
            eval_tracker.get("sequence_stage_first_steps", [])
        ),
        "required_goal_atoms": [
            f"{atom[0]}({', '.join(atom[1:])})"
            for atom in eval_tracker.get("required_atoms", [])
        ],
        "required_atom_first_steps": list(
            eval_tracker.get("required_atom_first_steps", [])
        ),
        "required_atom_final": list(
            eval_tracker.get("required_atom_final", [])
        ),
        "required_prefix_first_steps": list(
            eval_tracker.get("required_prefix_first_steps", [])
        ),
        "required_combination_first_steps": dict(
            eval_tracker.get("required_combination_first_steps", {})
        ),
        "max_required_atoms_satisfied": int(
            eval_tracker.get("max_required_atoms_satisfied", 0)
        ),
        "max_required_atom_values": list(
            eval_tracker.get("max_required_atom_values", [])
        ),
        "max_required_atoms_first_step": eval_tracker.get(
            "max_required_atoms_first_step"
        ),
        "rollout_timeline": rollout_timeline,
    }
