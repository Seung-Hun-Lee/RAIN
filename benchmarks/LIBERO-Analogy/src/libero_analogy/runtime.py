"""Public loader and scorer API shared by baseline and RAIN evaluation."""
import json
import yaml

from .tasks import benchmark_root, load_index, validate


def load_task(task_id, root=None):
    root = benchmark_root(root)
    validate(root)
    rows = {row["task_id"]: row for row in load_index(root)}
    if task_id not in rows:
        raise ValueError(f"Unknown task: {task_id}")
    from .assets import configure
    configure()
    import torch
    from .environment import install_support
    row = dict(rows[task_id])
    bundle = root / row["bundle"]
    row.update(bundle_path=bundle, bddl_path=bundle / "task.bddl",
               meta=yaml.safe_load((bundle / "task_meta.yaml").read_text()),
               rules=yaml.safe_load((bundle / "eval_rules.yaml").read_text()),
               initial_states=torch.load(bundle / "task.pruned_init", map_location="cpu", weights_only=False))
    row['replays'] = json.loads((bundle / 'FIXTURE_REPLAY.json').read_text())['rows'] if (bundle / 'FIXTURE_REPLAY.json').exists() else None
    if len(row["initial_states"]) != row["init_state_count"]:
        raise ValueError("Initial-state count differs from frozen inventory")
    install_support(row["legacy_task_id"])
    from libero.libero.envs.bddl_utils import robosuite_parse_problem
    language = robosuite_parse_problem(str(row["bddl_path"]))["language_instruction"]
    if isinstance(language, list):
        language = " ".join(language)
    if language.strip().casefold() != row["instruction"].strip().casefold():
        raise ValueError("BDDL instruction differs from frozen inventory")
    return row


def prepared_env(task, episode):
    from .scoring import prepared_env as prepare
    return prepare(task, episode)

