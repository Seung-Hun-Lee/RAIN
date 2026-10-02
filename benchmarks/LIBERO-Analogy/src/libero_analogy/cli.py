import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .tasks import benchmark_root, load_index, validate


def parser():
    p = argparse.ArgumentParser(description='Frozen LIBERO-Analogy evaluation; no model weights included')
    p.add_argument('--benchmark-root', help='Root containing TASK_INDEX.json (also LIBERO_ANALOGY_ROOT)')
    commands = p.add_subparsers(dest='command', required=True)
    commands.add_parser('list', help='List the exact 60 task IDs and current instructions')
    commands.add_parser('validate', help='Verify task schema and immutable source-backed file hashes')
    commands.add_parser('verify-assets', help='Verify separately installed upstream meshes/textures/XML')
    smoke = commands.add_parser('smoke', help='Reset one environment; no policy inference or scoring claim')
    smoke.add_argument('--task-id', default='Decompose_001')
    smoke.add_argument('--episode', type=int, default=0)
    smoke.add_argument('--output', type=Path)
    for name in ('evaluate', '_evaluate-task'):
        e = commands.add_parser(name, help='Run independent policy evaluation' if name == 'evaluate' else argparse.SUPPRESS)
        e.add_argument('--task-ids', nargs='+')
        e.add_argument('--episodes', type=int, default=50)
        e.add_argument('--output', type=Path, required=True)
        e.add_argument('--host', default='127.0.0.1')
        e.add_argument('--port', type=int, default=8000)
        e.add_argument('--timeout', type=float, default=900)
        e.add_argument('--policy-name', required=True)
        e.add_argument('--checkpoint', required=True, help='Reproducible checkpoint URI/version (reporting only)')
        e.add_argument('--policy-factory', help='Optional module:function taking argparse args, returning infer/reset/close client')
        e.add_argument('--replan-steps', type=int, default=5)
        e.add_argument('--retries', type=int, default=2)
        e.add_argument('--no-videos', dest='videos', action='store_false')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    root = benchmark_root(args.benchmark_root)
    if args.command == 'list':
        for row in load_index(root):
            print(f"{row['task_id']}\t{row['instruction']}")
        return
    if args.command == 'validate':
        print(json.dumps(validate(root), indent=2))
        return
    if args.command == 'verify-assets':
        from .assets import verify_assets
        print(json.dumps(verify_assets(root), indent=2))
        return
    os.environ.setdefault('MUJOCO_GL', 'egl')
    if args.command == 'smoke':
        from .runtime import load_task, prepared_env
        from .evaluate import atomic_json
        task = load_task(args.task_id, root)
        env, obs, scorer, evidence = prepared_env(task, args.episode)
        try:
            result = dict(task_id=args.task_id, reset_ok=True, **evidence)
            if args.output:
                atomic_json(args.output, result)
            print(json.dumps(result, indent=2))
        finally:
            env.close()
        return
    if args.episodes < 1 or args.episodes > 50 or args.replan_steps < 1 or args.retries < 0:
        raise ValueError('Require 1 <= episodes <= 50, replan-steps >= 1, retries >= 0')
    from .evaluate import atomic_json, evaluate_task, fingerprint, summarize
    if args.command == '_evaluate-task':
        if not args.task_ids or len(args.task_ids) != 1:
            raise ValueError('Task worker expects exactly one task ID')
        evaluate_task(args)
        return
    validate(root)
    from .assets import verify_assets
    verify_assets(root)
    rows = load_index(root)
    known = {r['task_id'] for r in rows}
    if args.task_ids and set(args.task_ids) - known:
        raise ValueError(f'Unknown tasks: {set(args.task_ids) - known}')
    rows = [r for r in rows if not args.task_ids or r['task_id'] in args.task_ids]
    args.output = args.output.resolve()
    for frozen in (root / 'tasks', root / 'src'):
        if args.output == frozen or frozen in args.output.parents:
            raise ValueError('Output must not overwrite benchmark tasks or runtime')
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = dict(benchmark_revision=fingerprint(root), task_ids=[r['task_id'] for r in rows],
                    episodes=args.episodes, policy_name=args.policy_name, checkpoint=args.checkpoint,
                    policy_factory=args.policy_factory, replan_steps=args.replan_steps, videos=args.videos,
                    seed='7+(task.order-1)*100+episode', physical_cameras=['third_person', 'wrist'],
                    physical_camera_model_consumption='Requires independent model-side evidence; payload hashes alone do not prove consumption')
    protocol_path = args.output / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Output belongs to a different protocol. Choose a new output directory; nothing was deleted.')
    if not protocol_path.exists() and any(args.output.iterdir()):
        raise ValueError('Output contains files without a matching protocol; choose an empty directory')
    atomic_json(protocol_path, protocol)
    try:
        for row in rows:
            command = [sys.executable, '-m', 'libero_analogy', '--benchmark-root', str(root), '_evaluate-task',
                       '--task-ids', row['task_id'], '--episodes', str(args.episodes), '--output', str(args.output),
                       '--host', args.host, '--port', str(args.port), '--timeout', str(args.timeout),
                       '--policy-name', args.policy_name, '--checkpoint', args.checkpoint,
                       '--replan-steps', str(args.replan_steps), '--retries', str(args.retries)]
            if args.policy_factory:
                command += ['--policy-factory', args.policy_factory]
            if not args.videos:
                command += ['--no-videos']
            subprocess.run(command, check=True)
            summarize(args.output, rows, args.episodes)
    finally:
        summarize(args.output, rows, args.episodes)
