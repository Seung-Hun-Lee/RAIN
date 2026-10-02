# Evaluation settings and results

## Selected model

The full model is the 100k-step multi-stage, mask-augmented DINOv2-L/224 action
policy and the target-region/gated Transition Head selected at epoch 29,
global step 7308. The complete policy is stored in one file:

| File | SHA256 |
|---|---|
| checkpoint.pt | `a747f1583294abee689e0a268b63c88535bfab1bc9fd4f1a95cc7b00a8134c3d` |

The file contains both the action policy and Transition Head. Keep the supplied
`config.json` beside it. See [encoder setup](core.md#data-and-feature-preparation)
for pretrained visual features.

## Paper results

Success rates (%) over 50 episodes per task:

| Mask source | Spatial | Object | Goal | Long | LIBERO average |
|---|---:|---:|---:|---:|---:|
| Simulator GT | 93.6 | 99.0 | 95.2 | 93.6 | 95.4 |
| Qwen3.5-4B + SAM | 92.4 | 96.8 | 91.2 | 82.4 | 90.7 |

| Mask source | Adapt | Compose | Decompose | Analogy average |
|---|---:|---:|---:|---:|
| Simulator GT | 62.4 | 37.1 | 82.7 | 60.7 |
| Qwen3.5-4B + SAM | 53.3 | 31.0 | 77.5 | 53.9 |

Values are rounded to one decimal. The supplied evaluators use simulator
ground-truth masks. The predicted-mask results use Qwen3.5-4B and SAM3.1.

## Evaluation protocol

- Same selected policy and pretrained encoder identities.
- Same original RGB preprocessing, proprioceptive representation and action scaling.
- Same target-place conditioning and mask-augmentation mode in the applicable stage.
- Same dataset samples, temporal labels and validation eligibility criteria.
- Same task definitions, initial-state IDs, seed and simulator versions.
- Same control horizon, replanning cadence and Transition Head stopping criterion.
- Benchmark-specific success/forbidden-goal rules; especially do not replace the
  LIBERO-Analogy scorer with an arbitrary simulator `done` flag.
- Report GT and predicted-mask runs separately. Evaluation GT must not feed the
  VLM provider or repair a failed localization.

## Ablations

"w/o cross-view" in the reported ablation refers to removing the
entire cross-view/self-attention/FFN block. It is not an isolated removal of
only one cross-attention operator. TarLN and the other full-model components
remain active unless explicitly ablated.

See [training and ablation options](core.md#training-and-ablations) for all settings.
