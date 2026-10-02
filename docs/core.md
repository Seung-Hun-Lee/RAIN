# Implementation guide

The released LIBERO model uses multi-stage visual features and a Transition
Head with target-region pooling and gated view fusion. This simulation
implementation uses third-person and wrist camera observations. See
[checkpoint and evaluation details](reproduction.md).

## Source and tensor names

| Paper component | Actual source | Checkpoint prefix |
| --- | --- | --- |
| TCE, Target-adaptive Cross-view Encoder | `rain/models/vision_encoder.py` | `dual_view_encoder.` |
| TarLN, Target-adaptive Layer Normalization | `TargetAdaptiveLayerNorm` / `TarLN` in the same file | Existing TCE submodule names |
| PlanDiT | `shared/plan_dit.py` | `dit.` |
| Transition Head | `rain/transition_head.py`; final model in `rain/model.py` | `fusion_branch.` |
| RRS, Reference-based Retargeting Strategy | `shared/data/dataset.py`: stateful selection, tangent junction, cubic Hermite approach and recorded-action replay | No learned tensors |

Import the components with
`from rain import RAIN, TCE, TarLN, PlanDiT, TransitionHead`.
The checkpoint loader validates tensor names, shapes, dtypes, and head type.
Model components are in `rain/models/`, with action and Transition Head trainers
in `rain/training/action.py` and `rain/training/transition.py`.

## Data and feature preparation

After unpacking the dataset, the training paths are:

```text
DATA/
  annotations/final_full.json
  annotations/tc_event_v3_manifest.json  # subtask-completion labels
  libero/parquet/                 # original numeric training columns
  libero/original_parquet/        # original image-bearing parquet files
  features/packed_dinov2_large_224_ms/text_features.npz
  replay/                        # inspectable simulator-replayed RGB
```

Training reads the supplied episode annotations and subtask-completion labels.

The approximately 801 GiB DINO feature cache is not distributed. Recompute it
from the original image-bearing Parquet, never substitute the replay PNGs:

```bash
python -m rain.prepare_data \
  --parquet-dir "$RAIN_DATA_ROOT/libero/original_parquet" \
  --output-dir "$RAIN_DATA_ROOT/features/packed_dinov2_large_224_ms" \
  --gpus 0,1,2,3 --batch-size 64
```

The extractor uses PIL RGB decoding, PIL bicubic resize to 224, ImageNet
normalization, CUDA FP16 autocast, normalized DINOv2-L/14-register patch tokens
from blocks 11/17/23, and raw float16 memmaps. Online inference uses tensor
resizing and BF16. The `dino_packed_s*.npy` files are raw
memmaps despite their `.npy` suffix; use the supplied loader, not `np.load`.

The supplied CLIP cache can be used directly. To regenerate it:

```bash
python -m rain.precompute_text \
  --episodes-json "$RAIN_DATA_ROOT/annotations/final_full.json" \
  --packed-features-dir "$RAIN_DATA_ROOT/features/packed_dinov2_large_224_ms"
```

Training initializes seven CLIP action-type features. Inference restores
those features from the checkpoint without
downloading CLIP again. Feature extraction/online RGB inference still require
DINOv2 code and pretrained weights, obtained by `torch.hub` or its local cache.
The actual backbone is `dinov2_vitl14_reg`, with **four register tokens** in the
backbone; the stored features contain only the 256 patch tokens, not registers.
`configs/pretrained.json` records weight URLs, checksums, source-file hashes,
and the CLIP revision. Verify your local DINOv2 installation with:

```bash
python -m rain.verify_backbones --dino-repo PATH_TO_DINOV2_SOURCE \
  --dino-weights PATH_TO_DINOV2_VITL14_REG4_PRETRAIN_PTH
```

The extractor uses the Torch Hub cache. The evaluator prefers the standard
`~/.cache/torch/hub/facebookresearch_dinov2_main` checkout when it exists.
Verification records identity but does not itself redirect either loader.

## Training and ablations

```bash
python -m rain.train action --data-root "$RAIN_DATA_ROOT" --output-root outputs --execute
python -m rain.train transition --data-root "$RAIN_DATA_ROOT" --output-root outputs \
  --action-checkpoint outputs/action_full/checkpoints/checkpoint_final.pt \
  --action-config outputs/action_full/config.json --execute
```

Omit `--execute` to print the exact command without writing files or initializing
a GPU. Defaults use four training processes, batch 256 per process, 100k action
steps, and up to 30 Transition Head epochs. Each launch validates its configuration
before loading data. Keep the global batch size at 1024 to match the experiments.

The default is `--ablation full`. Select any one-factor change explicitly:

| Option | Change from full |
|---|---|
| `--ablation no_view_blocks` | Remove cross-view/self-attention/FFN blocks; retain TarLN |
| `--ablation no_multistage` | Use the final visual stage only |
| `--ablation no_mask_augmentation` | Disable mask augmentation in both training stages |
| `--ablation no_plan` | Remove plan tokens and their endpoint consistency loss |
| `--ablation no_goal_consistency` | Remove only `goal_xyz_consistency_loss` |
| `--ablation bidirectional_plan` | Use bidirectional plan/action attention |
| `--ablation no_rrs` | Disable retarget augmentation |
| `--ablation linear_rrs` | Use linear interpolation instead of RRS |

Retrain the action model and pass its own config and checkpoint to Transition
Head training, using the same `--ablation` option in both stages.
`--head-variant` supports
`region_gated` (default), `region_concat`, `global_gated`, `global_concat`,
`global_region_gated`, and `global_region_concat`; evaluation of a nondefault
head requires the matching `RAIN_POOLING_HEAD_VARIANT` environment value.
All six heads use action-type features and subtask-completion supervision.

The single-stage ablation selects the existing last stage, without re-extraction:

```bash
python -m rain.prepare_last_scale "$RAIN_DATA_ROOT/features/packed_dinov2_large_224_ms" \
  "$RAIN_DATA_ROOT/features/last_scale"
python -m rain.train action --data-root "$RAIN_DATA_ROOT" --ablation no_multistage \
  --packed-features-dir "$RAIN_DATA_ROOT/features/last_scale" --execute
```

## Evaluation

Install the matching LIBERO simulator/assets separately. For LIBERO:

```bash
python -m rain.eval_libero \
  --checkpoint "$RAIN_MODEL_ROOT/checkpoint.pt" \
  --episodes-json "$RAIN_DATA_ROOT/annotations/final_full.json" \
  --benchmark libero_10 --gpus 0 --episodes-per-task 50 --save-dir results/libero_10
```

Run the other three suites with `libero_spatial`, `libero_object`, and
`libero_goal`. Evaluation uses simulator ground-truth masks, four flow steps,
and replanning every eight actions.

For LIBERO-Analogy, install the benchmark bundled in this repository:

```bash
pip install -e benchmarks/LIBERO-Analogy
export LIBERO_ANALOGY_ROOT="$PWD/benchmarks/LIBERO-Analogy"
python -m rain.eval_analogy --benchmark-root "$LIBERO_ANALOGY_ROOT" --preflight
python -m rain.eval_analogy --benchmark-root "$LIBERO_ANALOGY_ROOT" \
  --checkpoint "$RAIN_MODEL_ROOT/checkpoint.pt" \
  --gpu 0 --episodes-per-task 50 --save-dir results/analogy
```

LIBERO-Analogy uses the benchmark's task preparation and success criteria.
Decompose stops when the predicted completion
probability for the final subtask is at least 0.7 on two consecutive checks.
This prediction is separate from the benchmark's success label.
Intermediate subtask switching uses a strict `>` threshold.

## Transition Head training

The trainer selects the checkpoint by completion-prediction accuracy on eligible
validation samples in the primary training process.
Positive completion labels for release actions receive a loss weight of 1.5.
The loss includes samples with distance <= 0.8 or a positive completion label.

## Tests

Run `python -m pytest -q` to check model components, checkpoint loading, training
configurations, and evaluation interfaces. The benchmark's `--preflight` option
validates task metadata without running the simulator.

`requirements-tested.txt` records the tested simulation environment. Use a
dedicated environment to avoid package-name conflicts with other projects.
