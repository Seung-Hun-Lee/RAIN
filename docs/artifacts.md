# Data and checkpoints

The training dataset and model checkpoints are hosted on Hugging Face:
[Seunghun5688/RAIN-LIBERO](https://huggingface.co/datasets/Seunghun5688/RAIN-LIBERO)
and [Seunghun5688/RAIN](https://huggingface.co/Seunghun5688/RAIN).

| Artifact | Contents | Approximate size |
|---|---|---:|
| Replay dataset | Original and replay RGB, masks, subtasks, completion labels, actions, and robot states | 73.8 GB |
| RAIN model | LIBERO-trained action policy and Transition Head in one checkpoint | 0.70 GB |

We replayed LIBERO demonstrations, added target masks and subtask/completion
annotations, and excluded failed replays. The training set contains 1,628
demonstrations across 40 tasks. Original RGB Parquet retains the source episodes;
the supplied annotations select the successful replay subset for training.

## Download and unpack

Choose separate download and unpacking directories:

```bash
export RAIN_DATA_ROOT=/absolute/path/to/rain-data
export RAIN_MODEL_ROOT=/absolute/path/to/rain-model
export RAIN_DOWNLOAD_ROOT=/absolute/path/to/downloads/RAIN-LIBERO
hf download Seunghun5688/RAIN --local-dir "$RAIN_MODEL_ROOT"
hf download Seunghun5688/RAIN-LIBERO --repo-type dataset --local-dir "$RAIN_DOWNLOAD_ROOT"
python "$RAIN_DOWNLOAD_ROOT/tools/unpack.py" --destination "$RAIN_DATA_ROOT"
```

Use an empty unpacking destination. The tool verifies checksums and links the
original Parquet files, so keep the download directory. Add `--original-mode copy`
to copy them instead (about 35 GB extra).

```text
DATA/
  annotations/final_full.json
  annotations/tc_event_v3_manifest.json  # subtask-completion labels
  libero/parquet/                  # selected action/state records
  libero/original_parquet/         # original image-bearing source files
  replay/                         # RGB aligned with exported replay masks
  features/packed_dinov2_large_224_ms/
MODEL/
  checkpoint.pt
  config.json
```

## Model and encoders

`checkpoint.pt` contains the LIBERO-trained action policy and Transition Head.
Keep `config.json` beside it.

RGB inference requires DINOv2 weights; CLIP action features are stored in the
checkpoint. See
[encoder setup](core.md#data-and-feature-preparation) and
[pretrained.json](../configs/pretrained.json).

## Storage and inference requirements

Training requires a DINO feature cache (about 801 GiB), generated from original
Parquet RGB using the [feature preparation guide](core.md#data-and-feature-preparation).
The cache is excluded from the download and is not needed for inference.

LIBERO-Analogy task definitions and initial states are included in GitHub.

## Simulator setup

After installing both packages from the root README, download and verify the
shared LIBERO meshes and textures. Run these commands from the RAIN root:

```bash
export LIBERO_ANALOGY_ASSETS=/absolute/path/to/libero-assets
hf download jadechoghari/libero-assets \
  --revision 90001343cb134b7e26e18fde0fa2416f3ed6e6a3 \
  --local-dir "$LIBERO_ANALOGY_ASSETS"
libero-analogy --benchmark-root benchmarks/LIBERO-Analogy verify-assets
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

EGL requires a working NVIDIA/OpenGL driver. LIBERO-Analogy configures its own
process-local asset paths. You can check the simulator before loading RAIN:

```bash
MUJOCO_EGL_DEVICE_ID=0 libero-analogy \
  --benchmark-root benchmarks/LIBERO-Analogy smoke --task-id Decompose_001
```

### Standard LIBERO

Standard LIBERO needs its own configuration before the first import. In the
dedicated virtual environment, with `RAIN_DATA_ROOT` and
`LIBERO_ANALOGY_ASSETS` set above, run:

```bash
export LIBERO_CONFIG_PATH="$VIRTUAL_ENV/libero-config"
python - <<'PY'
import os
from importlib.metadata import distribution
from pathlib import Path
import yaml

root = Path(distribution("libero").locate_file("libero/libero")).resolve()
assets = Path(os.environ["LIBERO_ANALOGY_ASSETS"]).resolve(strict=True)
config_dir = Path(os.environ["LIBERO_CONFIG_PATH"])
config = {
    "benchmark_root": str(root),
    "bddl_files": str(root / "bddl_files"),
    "init_states": str(root / "init_files"),
    "datasets": str(Path(os.environ["RAIN_DATA_ROOT"]).resolve()),
    "assets": str(assets),
}
link = root / "assets"
if link.exists() or link.is_symlink():
    if link.resolve() != assets:
        raise RuntimeError(f"Existing asset path differs: {link}")
else:
    link.symlink_to(assets, target_is_directory=True)
config_dir.mkdir(parents=True, exist_ok=True)
config_file = config_dir / "config.yaml"
if config_file.exists():
    if yaml.safe_load(config_file.read_text()) != config:
        raise RuntimeError(f"Existing configuration differs: {config_file}")
else:
    config_file.write_text(yaml.safe_dump(config))
print(config_file)
PY
```

`libero==0.1.1` includes task definitions and initial states, but resolves meshes
through its package's `assets/` directory rather than the YAML assets entry.
The link above uses the verified download without copying it or modifying
simulator source. Keep `LIBERO_CONFIG_PATH`, `LIBERO_ANALOGY_ASSETS`, and the EGL
settings exported in each evaluation shell.
