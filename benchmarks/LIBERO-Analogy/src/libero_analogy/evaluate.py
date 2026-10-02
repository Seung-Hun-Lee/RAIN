"""Policy-neutral evaluation with resumable records and technical-error isolation."""
from collections import deque
import dataclasses
import hashlib
import importlib
import json
from pathlib import Path
import time

import numpy as np

from .observations import policy_observation
from .policy import OpenPIClient, input_evidence, validate_actions
from .runtime import load_task, prepared_env
from .tasks import benchmark_root, validate


def jsonable(value):
    if dataclasses.is_dataclass(value):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def fingerprint(root):
    """Guard tasks AND runtime code. Outputs must live outside these paths."""
    root = benchmark_root(root)
    entries = []
    for prefix in (root / "tasks", root / "src"):
        entries.extend(p for p in prefix.rglob("*") if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith('.pyc'))
    entries.extend(root / name for name in ("TASK_INDEX.json", "SOURCE_MANIFEST.json", "ASSET_MANIFEST.json"))
    digest = hashlib.sha256()
    for path in sorted(entries):
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def policy_client(args):
    if args.policy_factory:
        module, name = args.policy_factory.split(":", 1)
        return getattr(importlib.import_module(module), name)(args)
    return OpenPIClient(args.host, args.port, args.timeout)


def run_episode(task, episode, client, replan_steps=5, record=False):
    if hasattr(client, "reset"):
        client.reset()
    env, obs, scorer, evidence = prepared_env(task, episode)
    started = time.monotonic()
    frames, actions, action_trace, receipts = [], deque(), [], []
    steps = 0
    requests = 0
    first_input = None
    try:
        while steps < task['max_steps']:
            if not actions:
                payload, frame = policy_observation(obs, task['instruction'])
                if record:
                    frames.append(frame)
                actual_input = input_evidence(payload)
                if first_input is None:
                    first_input = actual_input
                response = client.infer(payload)
                # Accept standard official replies containing just `actions`.
                chunk = validate_actions(response)
                actions.extend(chunk[:replan_steps])
                if response.get('input_evidence') is not None:
                    receipts.append(response['input_evidence'])
                requests += 1
            action = np.asarray(actions.popleft(), dtype=np.float32)
            obs, _, _, _ = env.step(action.tolist())
            action_trace.append(action.tolist())
            steps += 1
            scorer.observe(steps)
            if scorer.violation or scorer.success():
                break
        success = scorer.success(final=True)
        termination = 'rule_violation' if scorer.violation else 'goal' if success else 'timeout'
        cooldown = 0
        if not success and not scorer.tracker['custom']:
            for cooldown in range(1, 21):
                obs, _, _, _ = env.step([0.0] * 7)
                scorer.observe(steps + cooldown)
                if scorer.success():
                    success, termination = True, 'native_goal_cooldown'
                    break
        if record:
            frames.append(policy_observation(obs, task['instruction'])[1])
        return frames, dict(task_id=task['task_id'], category=task['category'], episode_idx=episode,
                            success=bool(success), technical_failure=False, steps=steps,
                            max_steps=task['max_steps'], cooldown_steps=cooldown, termination=termination,
                            policy_requests=requests, inference_payload_evidence=first_input,
                            model_input_receipts=receipts, model_consumption_verified=False,
                            actions=action_trace, scoring=scorer.evidence(), seconds=time.monotonic() - started,
                            **evidence)
    finally:
        env.close()


def evaluate_task(args):
    """Called in a fresh interpreter per task to isolate process-local geometry."""
    root = benchmark_root(args.benchmark_root)
    validate(root)
    revision = fingerprint(root)
    protocol = json.loads((Path(args.output) / 'protocol.json').read_text())
    if protocol['benchmark_revision'] != revision:
        raise RuntimeError('Benchmark/runtime changed between tasks; refusing mixed-version results')
    task = load_task(args.task_ids[0], root)
    output = Path(args.output) / task['category'] / task['task_id']
    output.mkdir(parents=True, exist_ok=True)
    client = None
    try:
        for episode in range(args.episodes):
            target = output / f'episode_{episode:03d}.json'
            if target.exists():
                previous = json.loads(target.read_text())
                if previous.get('benchmark_revision') != revision or previous.get('episode_idx') != episode or previous.get('technical_failure'):
                    raise ValueError(f'Invalid resumable record: {target}')
                continue
            existing = [json.loads(p.read_text()) for p in output.glob('episode_*.json')]
            counts = {True: sum(bool(r.get('video')) and r['success'] for r in existing),
                      False: sum(bool(r.get('video')) and not r['success'] for r in existing)}
            record = args.videos and (counts[True] < 1 or counts[False] < 5)
            for attempt in range(args.retries + 1):
                try:
                    if fingerprint(root) != revision:
                        raise RuntimeError('Frozen benchmark/runtime changed; refusing mixed-version results')
                    if client is None:
                        client = policy_client(args)
                    frames, result = run_episode(task, episode, client, args.replan_steps, record)
                    if fingerprint(root) != revision:
                        raise RuntimeError('Frozen benchmark/runtime changed during an episode')
                    result.update(benchmark_revision=revision, policy_name=args.policy_name,
                                  checkpoint=args.checkpoint, video=None)
                    cap = 1 if result['success'] else 5
                    if record and counts[result['success']] < cap:
                        import imageio.v2 as imageio
                        video = output / f"episode_{episode:03d}_{'success' if result['success'] else 'failure'}.mp4"
                        imageio.mimwrite(video, frames, fps=10, codec='libx264')
                        result['video'] = str(video.relative_to(Path(args.output)))
                    atomic_json(target, result)
                    print(f"{task['task_id']} episode={episode} success={result['success']}", flush=True)
                    break
                except Exception as error:
                    atomic_json(output / f'error_{episode:03d}_{attempt:02d}.json',
                                dict(episode=episode, attempt=attempt, technical_failure=True,
                                     error_type=type(error).__name__, error=str(error)))
                    if client is not None:
                        if hasattr(client, 'close'):
                            client.close()
                        client = None
                    if attempt >= args.retries or fingerprint(root) != revision:
                        raise
    finally:
        if client is not None and hasattr(client, 'close'):
            client.close()


def summarize(output, rows, episodes):
    output = Path(output)
    summaries = []
    for row in rows:
        directory = output / row['category'] / row['task_id']
        records = [json.loads(p.read_text()) for p in sorted(directory.glob('episode_*.json'))]
        good = [r for r in records if not r.get('technical_failure')]
        successes = sum(bool(r['success']) for r in good)
        summaries.append(dict(task_id=row['task_id'], category=row['category'], episodes=len(good),
                              expected_episodes=episodes, successes=successes,
                              success_rate=successes / len(good) if good else None,
                              complete=len(good) == episodes))
    count = sum(r['episodes'] for r in summaries)
    successes = sum(r['successes'] for r in summaries)
    categories = {}
    for category in ('Decompose', 'Adapt', 'Compose'):
        category_count = sum(r['episodes'] for r in summaries if r['category'] == category)
        category_successes = sum(r['successes'] for r in summaries if r['category'] == category)
        categories[category] = dict(episodes=category_count, successes=category_successes,
                                    success_rate=category_successes / category_count if category_count else None)
    result = dict(tasks=summaries, episodes=count, successes=successes,
                  success_rate=successes / count if count else None,
                  complete=all(r['complete'] for r in summaries),
                  categories=categories)
    atomic_json(output / 'summary.json', result)
    lines = ['# Evaluation summary', '', f"Completed episodes: {count}; successes: {successes}; complete: {result['complete']}", '',
             f"Overall success rate: {100 * successes / count:.2f}%" if count else 'Overall success rate: not evaluated', '',
             '| Category | Successes / episodes | Success rate |', '|---|---:|---:|']
    for category, row in categories.items():
        rate = f"{100 * row['success_rate']:.2f}%" if row['episodes'] else 'not evaluated'
        lines.append(f"| {category} | {row['successes']} / {row['episodes']} | {rate} |")
    lines += ['',
             '| Task | Successes / episodes | Success rate |', '|---|---:|---:|']
    for row in summaries:
        rate = f"{100 * row['success_rate']:.2f}%" if row['episodes'] else 'not evaluated'
        lines.append(f"| {row['task_id']} | {row['successes']} / {row['episodes']} | {rate} |")
    lines += ['', '## Saved videos', '']
    lines.extend(f'- [{p.stem}]({p.relative_to(output).as_posix()})' for p in sorted(output.rglob('*.mp4')))
    (output / 'README.md').write_text('\n'.join(lines) + '\n')
    return result
