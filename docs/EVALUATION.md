# Evaluation


[`paper_evaluation/`](../paper_evaluation/README.md) holds the code that scored the
paper's results under the frozen protocol: detection, grounding and pointing, the
paired sequential/packed efficiency comparison, the class-isolated control, and the
address ablation. Every source file there is byte-identical to the version that ran.
`evaluate.sh` in the repository root generates predictions from a trained checkpoint;
`paper_evaluation/` is what produced the reported numbers.

## 4. Evaluate Generated Predictions

Evaluation GPU count is independent of training GPU count. Default: all **747
validation queries**, without GT points/GT boxes passed to the model, with no NMS
or max-detections cap. This is validation evaluation, **not a full benchmark**.

```bash
RUN_ROOT="/absolute/path/to/the/refwave/run"  # Printed when training starts.
bash evaluate.sh 4 "$RUN_ROOT/stage2/best.pt"

METHOD=standard bash evaluate.sh 4 "/absolute/path/to/the/standard/run/stage2/best.pt"
bash evaluate.sh 4 baseline
```

The loader restores the full trained state of the checkpoint, including the
output head.

For independent sequential decoding rather than class/address waves:

```bash
CLASS_WAVE=1 BOX_WAVE=1 EVAL_OUTPUT="$RUN_ROOT/evaluation_sequential" \
  bash evaluate.sh 4 "$RUN_ROOT/stage2/best.pt"
```

RefWave defaults to class waves of 18 and box waves of 200. It reuses projected
global vision across rounds and shares Qwen prefixes within each round; it does
**not** claim cross-round Qwen KV reuse. Each Round-2 address generates one PBD6
box/None. Direct-box control uses the native LA generator. Existing decoding
rules are retained; these should not be presented as identical output grammars.

Round 2 now defaults to `ROUND2_EXECUTION=fused_one_pass`: each wave combines
its address prompts and six-position geometry inputs into **one Qwen forward**.
`BOX_WAVE=1` also uses one forward per address. Prefix preparation and Round 1
are separate work; this does not mean one forward for the entire image.
No retraining is needed. Training, prompts, point filtering and scoring are unchanged.
BF16 fusion can slightly change coordinates or `None` decisions, so historical
two-pass results must not be relabeled as one-pass results.

For the explicit two-pass reference only:

```bash
ROUND2_EXECUTION=two_pass bash evaluate.sh 4 "$RUN_ROOT/stage2/best.pt"
```

RefWave outputs default to `evaluation_fused_one_pass/` alongside the checkpoint
(`evaluation_two_pass/` for the reference). `EVAL_OUTPUT` overrides this directory.
Standard/baseline controls retain their existing output directory and decoding.
Outputs: `predictions.rank*.jsonl`, `metrics.json`, `config.json` and
`done.json` alongside the checkpoint. Each shard is resumable with identical
settings. Detection uses class-aware greedy one-to-one IoU@0.5 P/R/F1. Grounding
additionally reports first-box Acc@0.5, Acc@0.7 and mean IoU on single-GT queries.
Latency is per query excluding model loading and scoring; raw responses are kept.
First-query warmup is included; these are recorded timings, not a controlled
latency benchmark. No AP/mAP or statistically reliable gain is claimed.

`EVAL_LIMIT=100` selects the first 100 records; it is **not** the historical
DOTAv2 Balanced-100 set. To use another frozen evaluation set, supply
`EVAL_RECORDS=/path/to/episodes.jsonl` and its `DATA_ROOT`, retaining the same
episode schema. Detection benchmarks must list all protocol-requested classes,
including absent ones; never build the query list from GT-present labels.

Optional portable model export:

```bash
bash export.sh "$RUN_ROOT/stage2/best.pt" "$WORK_ROOT/exports/refwave_stage2"
```
