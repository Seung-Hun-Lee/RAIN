# LIBERO-Analogy

LIBERO-Analogy contains 60 tasks: 20 Decompose, 20 Adapt, and 20 Compose tasks. This package includes task definitions, initial states, evaluation rules, simulator support, and a policy-neutral evaluator.

[Explore all 60 tasks](https://seung-hun-lee.github.io/projects/RAIN/gallery.html) with task descriptions and videos.

This benchmark is bundled under `benchmarks/LIBERO-Analogy` in the RAIN
repository and remains an independently installable package. The commands below
are run inside this benchmark directory; from the RAIN root first run
`cd benchmarks/LIBERO-Analogy`.

See [NOTICE.md](NOTICE.md) for third-party attribution and license scope, and [VALIDATION.md](VALIDATION.md) for tests and environment limitations.

## Install

Use a separate Python 3.10 environment on Linux. Simulation and policy inference can run in different environments or on different machines.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[sim,test]'
```

The simulator environment does not need `openpi-client` or the model's JAX/PyTorch inference stack. Two unmodified official OpenPI helpers, the NumPy msgpack codec and image preprocessing, are included with source hashes and license notices. Optional tests compare their wire bytes and image outputs against an independently installed OpenPI client. See [VALIDATION.md](VALIDATION.md) for dependency checks and simulator installation limitations.

Obtain the upstream LIBERO meshes/textures separately (about 403 MiB in the reference asset set; not redistributed here):

```bash
hf download jadechoghari/libero-assets --revision 90001343cb134b7e26e18fde0fa2416f3ed6e6a3 --local-dir /path/to/libero-assets
export LIBERO_ANALOGY_ASSETS=/path/to/libero-assets
libero-analogy verify-assets
```

`ASSET_MANIFEST.json` records the required asset hashes. If verification fails, check that the asset directory matches the pinned revision; do not change task predicates or geometry to bypass the check. These assets have separate upstream licensing terms.

The evaluator creates a temporary process-local LIBERO configuration. It does not rewrite `~/.libero/config.yaml`, installed packages, meshes, or task bundles. The `task.pruned_init` files are PyTorch/pickle artifacts: load them only from a trusted source and verify their hashes.

## Inspect and smoke-test

```bash
libero-analogy list
libero-analogy validate
python -m pytest -q
MUJOCO_GL=egl libero-analogy smoke --task-id Decompose_001
MUJOCO_GL=egl libero-analogy smoke --task-id Compose_005
```

EGL needs a working OpenGL/EGL driver. `MUJOCO_GL=osmesa` may be used with a properly installed OSMesa software renderer; OSMesa was not available in the validation environment. The smoke command performs reset, exact initial-state replay, ten warmup controls, initial-rule checks, and extraction of both camera views. It does not evaluate a model. Offscreen rendering is 256×256, with both physical views rotated 180 degrees and resized/padded to 224×224 for the policy.

Use `--benchmark-root /path/to/LIBERO-Analogy` before the subcommand, or set `LIBERO_ANALOGY_ROOT`, when the data checkout is separate from the installed package. Editable installation is recommended: task assets intentionally remain in the GitHub checkout, not a Python wheel.

## Evaluate π₀.₅

In an independently installed [official OpenPI](https://github.com/Physical-Intelligence/openpi/tree/54cbaee6ae0c010a1ed431871cdaa8f4684ac709) environment, launch its LIBERO checkpoint server:

```bash
uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=gs://openpi-assets/checkpoints/pi05_libero
```

In the simulator environment:

```bash
MUJOCO_GL=egl libero-analogy evaluate \
  --host 127.0.0.1 --port 8000 \
  --policy-name pi0.5 --checkpoint gs://openpi-assets/checkpoints/pi05_libero \
  --episodes 50 --output outputs/pi05
```

For an integration trial, add `--task-ids Decompose_001 --episodes 1`. A short trial is not a benchmark score. The OpenPI transport accepts arbitrary server metadata and requires only `actions` in replies. Model sampling depends on the server's RNG configuration, so record the server configuration and seed for each evaluation.

Output contains a protocol fingerprint, per-episode actions and scoring evidence, per-task/category/overall summaries, and up to one success plus five failure videos per task. Rerunning the identical command resumes completed episodes; conflicting protocols are rejected without deleting output. Technical errors are recorded and retried (default two retries), never counted as policy failures. Persistent errors stop evaluation with an incomplete report. Every task uses a fresh simulator process so geometry-registration state cannot leak across tasks.

For comparisons, verify the checkpoint was fine-tuned on all four suites (Spatial, Object, Goal, Long/LIBERO-10) and save model-side evidence that both physical cameras actually reach the model. Client-side tensor hashes prove what was sent, not what the server consumed. This evaluator reports that distinction explicitly; a sketch/reference image does not count as a second physical camera.

## Other policies

Either implement the [OpenPI websocket server protocol](https://github.com/Physical-Intelligence/openpi/blob/54cbaee6ae0c010a1ed431871cdaa8f4684ac709/src/openpi/serving/websocket_policy_server.py), or pass `--policy-factory your_module:make_client`. The factory receives the parsed CLI arguments and returns an object with `infer(payload) -> dict`; optional `reset()` and `close()` are called when available.

| Input key | Shape/type | Meaning |
|---|---|---|
| `observation/image` | uint8 `[224,224,3]` | Physical third-person RGB |
| `observation/wrist_image` | uint8 `[224,224,3]` | Physical wrist RGB |
| `observation/state` | float32 `[8]` | XYZ, axis-angle, two gripper joint positions |
| `prompt` | string | Exact current task instruction |

Return `{"actions": array}` with finite, denormalized LIBERO controls shaped `[T,7]`, `T >= 1`: three translation, three rotation, and one native gripper control. The evaluator executes the first five controls (or fewer if `T < 5`) before replanning. It does not invert, clip, or renormalize model actions. Your adapter owns conversions from a different model's action convention and must preserve both physical camera views. Stateful policies should implement episode reset in a custom client; the official OpenPI wire protocol has no reset message.

The reusable Python interface is:

```python
from libero_analogy.runtime import load_task, prepared_env
task = load_task("Compose_005", root="/path/to/LIBERO-Analogy")
env, observation, scorer, evidence = prepared_env(task, episode=0)
# After each actual env.step(action), call scorer.observe(control_step).
# scorer.violation, scorer.success(final=False), scorer.evidence()
env.close()
```

## Frozen task semantics

[TASKS.md](TASKS.md) lists all task IDs, instructions, step caps and initial-state counts. Adapt_017's instruction says "left compartment", while its native target is `bowl_drainer_1_right_region`. Use the supplied instruction and target unchanged; region identifiers do not determine the wording of a task instruction.

The baseline protocol uses 50 episodes per task. Twenty-four tasks have 50 initial states; 36 have five states repeated by `episode % state_count`. Seed is `7 + (task_order - 1) * 100 + episode`. Each task's step cap is fixed in the task index. Ten warmup controls precede inference; native-only unsuccessful tasks receive a 20-control passive cooldown. Scoring includes Decompose's continue/forbidden rules, strict rising-event order, released/support checks, and Compose_005's direct-contact/door-clearance check. Learned RAIN TC-based termination is model-specific and is not applied to baseline policies.

`SOURCE_MANIFEST.json` records source and package file hashes. Descriptive source references use portable `source://` labels. Compose_005's runtime sweep data contain state-keyed geometric bounds used by its scorer.

## Assets and model checkpoints

All 60 task bundles and the evaluator are included in this directory. Download simulator assets separately using the pinned Hugging Face revision above. Training data and model checkpoints are separate from this package and have their own installation instructions and licensing terms.
