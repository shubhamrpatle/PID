# Corrected class-isolated checkpoint: training evidence

HF revision: `266d967359913eeaa9e184651e787b67341d744b`. Run: `class_isolated_full_8gpu_s1e2_s2e2_both_both_ft_ciso_ep2_v32_20260917T115718Z_TvBPX9`.

| Stage | Source records | Epochs | Updates | Selected update | Selected macro loss |
|---|---:|---:|---:|---:|---:|
| stage1 | 116386 | 2 | 7276 | 7276 | 0.534111795 |
| stage2 | 39851 | 2 | 2492 | 1246 | 0.514839234 |

Both complete training curves contain every update, finite logged values, and direct-only loss.
Full MoonViT + projector + Qwen + LM head: 3,517,975,280 trainable parameters, as recorded in configs.
Stage-2 best is update 1246 (macro validation loss 0.514839234), not last update 2492 (0.531022335).
Stage-2 initialization hash equals the pinned Stage-1 best checkpoint hash.

Recipe compared to: refwave, standard.
Exact compared fields, training/validation hashes and evidence hashes are in TRAINING_AUDIT.json.

## Scope of verification

- Logs/configs establish recorded execution, not semantic correctness of every annotation.
- No executed training-source snapshot was supplied; exact remote implementation is not independently proven.
- The full weight file and GPU inference are verified on the evaluation machine before full evaluation.
- Matched source episodes and optimizer schedule do not equate supervised token counts across formulations.
