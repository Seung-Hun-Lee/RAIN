"""RAIN GT-mask rollout on the 60-task LIBERO-Analogy benchmark.

Environment preparation and success labels come from libero_analogy's
fixture, ordered-goal, and physical-observer API.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import yaml


def build_conditions(root, row):
    from final_libero_ex_eval.impl.benchmark_support import ACTION_OBJECTS
    from final_libero_ex_eval.impl.conditions import build_libero_ex_conditions
    from .object_bindings import register_object_bindings
    from libero_analogy._support.novel_feedback_object_bindings import register_object_bindings as feedback_bindings
    register_object_bindings(ACTION_OBJECTS)
    feedback_bindings(ACTION_OBJECTS)
    bundle = Path(root) / row["bundle"]
    metadata = yaml.safe_load((bundle / "task_meta.yaml").read_text())
    plan = []
    plan_path = bundle / "action_plan.yaml"
    if plan_path.is_file():
        plan = yaml.safe_load(plan_path.read_text()).get("steps", [])
    else:
        registry_path = Path(root) / "benchmark_support/action_plan_registry.yaml"
        if registry_path.is_file():
            registry = yaml.safe_load(registry_path.read_text()).get("action_plans", {})
            plan = registry.get(row["task_id"], [])
    episode, conditions = build_libero_ex_conditions(metadata, plan, 224)
    if not conditions:
        raise ValueError(f"No condition plan for {row['task_id']}")
    return episode, conditions


def exact_region_mask(env, episode_data, object_id, bddl_path="", image_size=256, camera_name="agentview"):
    from .region_geometry import render_region_mask
    binding = (episode_data or {}).get("objects", {}).get(str(object_id), {})
    if not binding or binding.get("segmentable", True):
        return None
    mask = render_region_mask(env, Path(bddl_path), binding["name"], image_size, camera_name)
    return None if mask is None else np.ascontiguousarray(mask[:, ::-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--task-ids", nargs="+")
    parser.add_argument("--episodes-per-task", type=int, default=50)
    parser.add_argument("--episode-ids", nargs="+", type=int)
    parser.add_argument("--save-dir", default="eval_results/analogy")
    parser.add_argument("--preflight", action="store_true", help="Validate all task/condition metadata without simulator/model initialization")
    parser.add_argument("--record-video", action="store_true")
    args = parser.parse_args()
    from libero_analogy.tasks import load_index, validate
    validate(args.benchmark_root)
    rows = load_index(args.benchmark_root)
    if args.task_ids:
        unknown = set(args.task_ids) - {row["task_id"] for row in rows}
        if unknown:
            parser.error(f"Unknown task IDs: {sorted(unknown)}")
        rows = [row for row in rows if row["task_id"] in args.task_ids]
    plans = {row["task_id"]: build_conditions(args.benchmark_root, row) for row in rows}
    if args.preflight:
        print(json.dumps({"tasks": len(rows), "conditions": {key: len(value[1]) for key, value in plans.items()}, "gpu_initialized": False}, indent=2))
        return
    if not args.checkpoint:
        parser.error("--checkpoint is required unless --preflight is used")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.gpu)
    from final_libero_ex_eval.impl import runtime, rollout
    runtime.sim_region_mask_for_object_id = exact_region_mask
    rollout.sim_region_mask_for_object_id = exact_region_mask
    # Must precede EGL simulator construction.
    # One checkpoint supplies both action and Transition Head weights.
    worker = runtime.GPUInferenceWorker(args.checkpoint, args.checkpoint, "rain", gpu_id=args.gpu, dino_input_size=224)
    from libero_analogy.runtime import load_task, prepared_env
    output = Path(args.save_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    episode_ids = args.episode_ids if args.episode_ids is not None else list(range(args.episodes_per_task))
    try:
        runtime._patch_robosuite_egl()
        for row in rows:
            task = load_task(row["task_id"], args.benchmark_root)
            episode_data, conditions = plans[row["task_id"]]
            for episode_id in episode_ids:
                env, obs, scorer, evidence = prepared_env(task, episode_id)
                try:
                    worker.set_seed(evidence["seed"])
                    corrections = runtime.verify_and_correct_body_ids(env, episode_data)
                    frames, success, policy = rollout.run_single_episode_libero_ex(
                        worker, env, None, np.zeros(768, np.float32), episode_data, conditions,
                        max_steps=task["max_steps"], record_video=args.record_video,
                        corrected_bids=corrections, dino_input_size=224, state_dim=8,
                        bddl_path=str(task["bddl_path"]), eval_rules=task["rules"],
                        prepared_observation=obs, benchmark_scorer=scorer,
                        decomposition_tc_stop=(row["category"] == "Decompose"),
                    )
                    result = {"task_id": row["task_id"], "episode": episode_id, "success": bool(success),
                              "protocol": "rain_gt_portable_analogy_v1", "preparation": evidence,
                              "policy": policy, "scoring": scorer.evidence()}
                    name = f"{row['task_id']}_episode_{episode_id:03d}"
                    (output / f"{name}.json").write_text(json.dumps(result, indent=2, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item()) + "\n")
                    if args.record_video:
                        runtime.save_video(frames, str(output / f"{name}.mp4"))
                    results.append({key: result[key] for key in ("task_id", "episode", "success", "protocol")})
                    print(json.dumps(results[-1]), flush=True)
                finally:
                    env.close()
    finally:
        worker.close()
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
