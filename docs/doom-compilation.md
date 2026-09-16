# Doom inference compilation experiment

The target is latency for a serial frame → action → next-frame controller.
All five action fields still run together for the current frame. Frames are not
batched across time, and no image detail is removed by this experiment.

## Workload and method

- Gemma 4 E2B, pinned project revision, BF16, physical GPU 0 (RTX 4090).
- Six distinct 640×480 RGB screenshots from `basic`, `defend_the_center`, and
  `deadly_corridor`; two frames per scenario, captured in separate headless games.
- The existing Doom five-field schema and the existing 280-token visual budget.
  The processor is unchanged. The shared prefix is 354 tokens for these frames.
- One shared prefill and one batch of five suffixes per screenshot, as before.
- Same loaded model for paired eager/compiled measurements, ten repetitions per
  screenshot, seeded shuffled order, three warmups, synchronized GPU timing.
- End-to-end inference includes image/prompt processing and scoring. It excludes
  model loading, compilation, frame capture, game advancement, and browser work.
- Eager outputs are the behavioral reference, not ground-truth optimal actions.
  This is a six-frame numerical/performance check, not a gameplay-quality test.

The benchmark records all action choices, label scores, per-stage timings, frame
pixel hashes, compiler counters, and peak allocated GPU memory. An action change,
unstable repeated predictions, or a new compilation graph during steady-state
measurement causes an exit code of 1; the report is still saved.

## Full-model compilation

`torch.compile(model.forward, mode="reduce-overhead", dynamic=False,
fullgraph=False)` reduces median inference latency from **89.0 ms to 66.6 ms**
(1.34× speedup, approximately 25% lower latency). There are no new compilation
graphs during the measured calls. The compiler captures multiple regions rather
than one full graph; some Python/multimodal processing remains outside them.

However, one of six frames changes its `fire` action. The largest candidate
probability change is about 0.85. Both eager and compiled outputs are stable over
repeated calls, so this is a reproducible behavioral difference.

Enabling `emulate_precision_casts` and `emulate_divison_rounding` (PyTorch 2.10's
actual option spelling) yields **89.0 ms → 66.9 ms**. This fixes the first changed
action but changes `move` on another frame. These settings therefore do not make
the whole-model path behaviorally equivalent on this fixture set.

The first trial took about 95 seconds to compile and warm up. A subsequent run
with compiler disk caches took about 35 seconds; the new rounding configuration
took about 93 seconds. These startup costs are separate from the warm latency.

Full-model reports:

- `results/doom-compile-benchmark.json`
- `results/doom-compile-preserve-rounding.json`

## Language-decoder compilation

Compiling only `model.model.language_model.forward` with rounding preservation
gives the most promising result in this experiment:

| Stage | Eager median | Compiled median |
| --- | ---: | ---: |
| Total frame inference | 88.46 ms | 74.12 ms |
| Shared prefill, including eager vision | 41.02 ms | 32.65 ms |
| Batched five-field suffix | 25.60 ms | 19.40 ms |

This is **16.2% lower end-to-end latency** (1.19× speedup). All five action
choices match on all six frames and stay stable across ten repetitions. Label
probabilities are not identical: the largest difference is 0.0488. No additional
graphs are compiled during the measured calls; this scope captures two graphs.
Compilation and warmup took 50.1 seconds. Peak allocated memory was about
9.65 GiB eager and 9.63 GiB compiled with the compiler's buffers already resident.

This is a candidate for further Doom testing, not proof of numerical equivalence
on arbitrary frames. It retains the existing vision path and mask preparation,
and does not shrink or replace the final output projection.

Report: `results/doom-compile-language.json`.

## Reproduce

From the project directory:

```bash
uv run python scripts/benchmark_doom_compile.py --gpu 0 --capture
uv run python scripts/benchmark_doom_compile.py --gpu 0 --preserve-rounding \
  --output results/doom-compile-preserve-rounding.json
uv run python scripts/benchmark_doom_compile.py --gpu 0 --scope language \
  --preserve-rounding --output results/doom-compile-language.json
```

`--scope language` compiles only the language decoder, leaving vision encoding,
multimodal assembly, attention-mask preparation and the vocabulary projection
outside compilation. `--mode default` is also available for further comparisons.

The benchmark is standalone. The browser controller now uses the tested
decoder-only recipe by default; use `typesafe-doom --no-compile` for eager mode.
The model worker completes three warmups before accepting frames and marks one
CUDA graph iteration per frame, spanning both the shared prefill and suffix batch.
The integrated path passed eight single steps and fourteen live autopilot
predictions on GPU 0, at about 78 ms median inference latency. See
`results/doom-compiled-integration.json` for the integration report.

## Implementation references

- [PyTorch 2.10 compiler configuration and precision-cast explanation](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/_inductor/config.py)
- [Transformers inference compilation documentation](https://huggingface.co/docs/transformers/v5.17.0/llm_optims)
