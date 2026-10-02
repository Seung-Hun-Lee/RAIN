"""Portable launch recipes for RAIN and one-factor ablations."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from rain.configs.config import TrainingConfig
from .transition_head import VARIANTS

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"
ABLATIONS = {
    "full": {},
    "no_view_blocks": {"third_encoder.use_transformer_blocks": False, "wrist_encoder.use_transformer_blocks": False},
    "no_multistage": {"dit.num_scales": 1},
    "no_mask_augmentation": {"data.mask_augmentation": "none"},
    "no_plan": {"dit.use_plan": False, "dit.goal_xyz_consistency_weight": 0.0},
    "no_goal_consistency": {"dit.goal_xyz_consistency_weight": 0.0},
    "bidirectional_plan": {"dit.plan_action_causal": False},
    "no_rrs": {"data.use_retarget_aug": False},
    "linear_rrs": {"data.retarget_rollout_mode": "linear_interp_v1"},
}


def merge(target, source):
    for key, value in source.items():
        if isinstance(value, dict):
            merge(target[key], value)
        else:
            target[key] = value
    return target


def set_value(config, key, value):
    parts = key.split(".")
    node = config
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def build_recipe(args):
    raw = (CONFIG_ROOT / f"{args.stage}.json").read_text()
    raw = raw.replace("${RAIN_DATA_ROOT}", str(Path(args.data_root).resolve()))
    raw = raw.replace("${RAIN_OUTPUT_ROOT}", str(Path(args.output_root).resolve()))
    config = merge(asdict(TrainingConfig()), json.loads(raw))
    if args.stage == "transition":
        if not args.action_checkpoint or not args.action_config:
            raise ValueError("Transition training requires --action-checkpoint and --action-config")
        action = json.loads(Path(args.action_config).read_text())
        for key in ("third_encoder", "wrist_encoder", "dit", "condition_mode", "cross_attn_mode"):
            config[key] = deepcopy(action[key])
        if args.ablation == "no_mask_augmentation":
            config["data"]["mask_augmentation"] = "none"
        # Architectural ablations must already be present in the loaded action config.
        for key, value in ABLATIONS[args.ablation].items():
            if key.split(".")[0] in ("third_encoder", "wrist_encoder", "dit"):
                node = config
                for part in key.split("."):
                    node = node[part]
                if node != value:
                    raise ValueError(f"Action config does not contain requested ablation: {key}")
    else:
        for key, value in ABLATIONS[args.ablation].items():
            set_value(config, key, value)
    if args.packed_features_dir:
        config["data"]["packed_features_dir"] = str(Path(args.packed_features_dir).resolve())
    if config["dit"]["num_scales"] == 1 and not args.packed_features_dir:
        raise ValueError("Single-stage ablation needs --packed-features-dir from rain.prepare_last_scale")
    if args.batch_size is not None:
        config["batch_size_per_gpu"] = args.batch_size
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    config["experiment_name"] = args.stage + "_" + args.ablation
    if args.stage == "transition":
        config["experiment_name"] += "_" + args.head_variant
    output = Path(config["output_dir"]) / config["experiment_name"]
    module = "rain.training.action" if args.stage == "action" else "rain.train_transition"
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", f"--nproc_per_node={args.nproc}", "-m", module]
    data = config["data"]
    fields = {
        "--episodes-json": data["episodes_json"], "--parquet-dir": data["parquet_dir"],
        "--packed-features-dir": data["packed_features_dir"], "--mask-augmentation": data["mask_augmentation"],
        "--train-ratio": data["train_ratio"], "--text-dropout-prob": data["text_dropout_prob"],
        "--batch-size": config["batch_size_per_gpu"], "--learning-rate": config["learning_rate"],
        "--warmup-steps": config["warmup_steps"], "--max-epochs": config["max_epochs"],
        "--gradient-accumulation-steps": config["gradient_accumulation_steps"],
        "--num-workers": config["num_workers"], "--seed": data["seed"],
        "--output-dir": config["output_dir"], "--experiment-name": config["experiment_name"],
        "--expected-config": str(output / "expected_config.json"),
    }
    if args.stage == "action":
        fields.update({
            "--max-steps": config["max_steps"], "--condition-mode": config["condition_mode"],
            "--cross-attn-mode": config["cross_attn_mode"], "--wrist-conditioning": "mask",
            "--encoder-hidden-dim": config["third_encoder"]["hidden_dim"],
            "--dino-feature-dim": config["third_encoder"]["dino_dim"],
            "--num-scales": config["dit"]["num_scales"], "--dino-input-size": config["dino_input_size"],
            "--clip-model-name": data["clip_model_name"], "--retarget-version": data["retarget_version"],
            "--retarget-rollout-mode": data["retarget_rollout_mode"],
            "--retarget-text-dropout-prob": data["retarget_text_dropout_prob"],
            "--goal-xyz-consistency-weight": config["dit"]["goal_xyz_consistency_weight"],
            "--consistency-group-size": data["consistency_group_size"],
            "--plan-attn-mode": "causal" if config["dit"]["plan_action_causal"] else "bidirectional",
        })
        command += ["--force-goal-pair-batch", "--use-target-place", "--use-wrist-goal-mask", "--use-wrist-target-place"]
        for flag, enabled in [("--use-plan", config["dit"]["use_plan"]), ("--use-view-transformer-blocks", config["third_encoder"]["use_transformer_blocks"]), ("--use-retarget-aug", data["use_retarget_aug"])]:
            command.append(flag if enabled else flag.replace("--", "--no-", 1))
    else:
        fields.update({
            "--action-checkpoint": str(Path(args.action_checkpoint).resolve()),
            "--action-config": str(Path(args.action_config).resolve()),
            "--tc-event-manifest": data["tc_event_manifest"],
            "--early-stop-patience": config["early_stop_patience"],
            "--tail-exclude-ratio": data["tail_exclude_ratio"],
            "--tc-mask-dist-thresh": config["progress"]["tc_mask_dist_thresh"],
            "--progress-lambda-dist": 0.0, "--progress-lambda-align": 0.0,
            "--progress-lambda-tc": config["progress"]["lambda_tc"],
        })
    for flag, value in fields.items():
        command += [flag, str(value)]
    if args.resume:
        command += ["--resume", args.resume]
    return config, command, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["action", "transition"])
    parser.add_argument("--data-root", default=os.environ.get("RAIN_DATA_ROOT"), required=not bool(os.environ.get("RAIN_DATA_ROOT")))
    parser.add_argument("--output-root", default=os.environ.get("RAIN_OUTPUT_ROOT", "outputs"))
    parser.add_argument("--action-checkpoint")
    parser.add_argument("--action-config")
    parser.add_argument("--ablation", choices=list(ABLATIONS), default="full")
    parser.add_argument("--head-variant", choices=VARIANTS, default="region_gated")
    parser.add_argument("--packed-features-dir")
    parser.add_argument("--nproc", type=int, default=4)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--execute", action="store_true", help="Without this flag, print the command without creating files or using a GPU")
    args = parser.parse_args()
    try:
        config, command, output = build_recipe(args)
    except ValueError as exc:
        parser.error(str(exc))
    print(shlex.join(command), flush=True)
    if args.execute:
        if args.nproc * config["batch_size_per_gpu"] != 1024:
            print("WARNING: effective batch differs from the published 1024-sample recipe", file=sys.stderr)
        output.mkdir(parents=True, exist_ok=True)
        (output / "expected_config.json").write_text(json.dumps(config, indent=2) + "\n")
        env = dict(os.environ, RAIN_POOLING_HEAD_VARIANT=args.head_variant)
        subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
