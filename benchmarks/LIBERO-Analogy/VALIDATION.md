# Testing

Run these commands from `benchmarks/LIBERO-Analogy` after installation:

```bash
python -m pip check
libero-analogy validate
python -m pytest -q
```

`validate` checks the 60-task inventory, file hashes, task identities, and instruction consistency. The CPU test suite covers action validation, ordered events, forbidden-goal handling, microwave contact-force and clearance boundaries, and a local websocket exchange with empty server metadata and actions-only replies. Two optional tests compare the included OpenPI codec and image preprocessing against an independently installed `openpi-client`; they are skipped when that package is unavailable.

## Simulator checks

After downloading and verifying the assets described in [README.md](README.md), check environment initialization with:

```bash
MUJOCO_GL=egl libero-analogy smoke --task-id Decompose_001
MUJOCO_GL=egl libero-analogy smoke --task-id Compose_005
```

Each smoke test resets one task, replays its initial state, performs ten warmup controls, initializes scoring, and extracts both 224×224 physical camera views. It does not run policy inference. By default it covers initial-state index 0; use `--episode` to select another episode. One reset per task does not cover every initial-state and episode combination.

## Reference environment and limitations

The reference simulator environment uses Linux, Python 3.10, `libero==0.1.1`, `robosuite==1.4.0`, `mujoco==3.5.0`, `numpy==2.2.6`, `torch==2.11.0+cu128`, `bddl==1.0.1`, `gymnasium==1.2.3`, `scipy==1.15.3`, and `imageio==2.37.3`. It uses EGL for offscreen rendering. Optional parity tests use `openpi-client==0.1.0`; the benchmark does not depend on that package.

A clean installation of the full simulator stack, software-only OSMesa rendering, and live π₀.₅ inference through the OpenPI server have not been tested. The installation, transport, and environment checks do not establish model performance or reproduce a 3,000-episode benchmark result. Evaluators must also verify that their policy consumes both physical camera views; client-side payload checks alone cannot establish this.
