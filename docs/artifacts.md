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
Follow the [benchmark setup guide](../benchmarks/LIBERO-Analogy/README.md#install)
for shared simulator assets; standard LIBERO uses its own asset paths.
