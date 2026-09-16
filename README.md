# Gemma Doom

Control Doom from pixels with **Gemma 4 E2B**, running locally on a single GPU.
Play in your browser, inspect the model's decisions, or hand it the controls.

The frozen model takes 640×480 RGB game frames through its standard vision
processor. There is no intermediate text description of the scene. One shared
image prefill and a batch of five action queries produce movement, strafing,
turning, firing, and interaction controls directly from next-token logits.
No fine-tuning or autoregressive JSON generation is required.

It sort of works: this is an experimental control loop, not a trained Doom agent.
The model can miss enemies, waste ammunition, and die quickly.

## Run the browser demo

Requirements: Linux, `uv`, Python 3.12+, and an NVIDIA GPU with enough memory for
the BF16 model. The experiment was tested on an RTX 4090 using roughly 10 GiB of
allocated GPU memory. The pinned weights download is approximately 10.25 GB.

```bash
git clone https://github.com/simedw/gemma-doom.git
cd gemma-doom
uv sync --locked
uv run typesafe-gemma download
uv run typesafe-doom --gpu 0
```

Open **http://127.0.0.1:8766**. The arena is available while Gemma loads and warms
up; model controls become available when it is ready. Cold compilation can take
about 50 seconds. Use `--no-compile` for eager inference, or `--no-model` for a
manual-only demo that does not need the model download or CUDA.

- **You:** play with WASD, left/right arrows to turn, Space to fire, and E to use.
- **Copilot:** you play while the model suggests controls.
- **Gemma:** the model controls the player.
- **Model step:** make one prediction, apply it for a chosen number of game
  ticks, and pause again. Four ticks is about 0.11 seconds of game time.
- **Escape / Take control:** pause or return to manual control.

The sidebar shows typed decisions, their observation frame, inference latency,
and the buttons being applied. The decision-rate setting controls the target
number of model calls per second during continuous play. The browser stream and
game loop run independently of inference.

See [browser controls and architecture](#doom-in-the-browser) and the
[compilation experiment](docs/doom-compilation.md) for details. The earlier email
and image experiments, including their benchmarks, are documented below.
Benchmark reports under `results/` are generated locally by the reproduction
commands and are not bundled in the repository. Model weights, environments,
and game runtime files are also excluded.

## Email classification experiment

A frozen multimodal **google/gemma-4-E2B-it** model classifies emails into:

```json
{"spam": false, "category": "work"}
```

Both fields are required. Categories are `personal`, `work`, `outreach`,
`automatic_response`, `newsletter`, `transactional`, and `other`.
Spam is independent of category: a phishing account notice can be both spam and
transactional. Personalized outreach is not automatically spam. Category rules
live in `src/typesafe_gemma/schema.py`; the output contract is in
`schema/email.schema.json`.

## Setup and GPU selection

```bash
uv sync
uv run typesafe-gemma download
uv run typesafe-gemma --gpu 0 classify --text 'From: Mum\nAre you coming for Sunday lunch?'
uv run typesafe-gemma --gpu 1 classify --file message.txt --details
```

Only physical GPUs **0, 1, and 2** are allowed. GPU **3 is broken**.
The CLI validates the index and resolves it to its NVIDIA UUID before importing
PyTorch, exposing only that device to CUDA. It defaults to GPU 0; this experiment
fits on one 24 GB 4090 and does not shard weights across cards. Inside the process,
the selected physical GPU is called `cuda:0`.

Python 3.12, CUDA 12.8 PyTorch wheels, Transformers and all resolved dependencies
are recorded in `uv.lock`. Model revision
`3e22461f65e89153144f8adb70e3b8c2cc9845a7` is pinned. The approximately 10.25 GB
weight file and processor assets are downloaded to the standard Hugging Face
cache, not copied into the repository. Classification loads only local files.
The full multimodal checkpoint is retained for the image phase.

## First measured run

On physical GPU 0 (RTX 4090), BF16, September 16, 2026:

| Method | Both fields correct | Valid schema | Median latency | p95 latency |
| --- | ---: | ---: | ---: | ---: |
| Typed cached probes | 28/28 | 28/28 | 65.1 ms | 81.6 ms |
| Greedy JSON generation | 27/28 | 28/28 | 236.8 ms | 259.7 ms |

Peak allocated GPU memory was 9.56 GiB. The typed path was 3.63× faster by
median latency in this run. This tiny, deliberately straightforward synthetic set
does not establish real-world accuracy or a general speedup. The raw report is
`results/email-evaluation.json`.

## Batched suffix optimization

A subsequent paired run on the same GPU used five repetitions per input, three
warmups per mode, and shuffled method order. Both email fields are now scored in
one batched suffix forward after the shared prefill.

| Method | Email median / p95 | Image median / p95 |
| --- | ---: | ---: |
| Batched cached probes (default) | 46.5 / 58.3 ms | 69.0 / 70.3 ms |
| Sequential cached probes | 67.0 / 86.5 ms | 91.8 / 93.2 ms |
| Uncached full-prompt probes | 43.0 / 54.7 ms | 88.2 / 94.5 ms |
| Greedy JSON generation | 247.0 / 299.5 ms | not measured |

Batching reduced median latency by **30.7% for emails** and **24.8% for images**
versus sequential cache reuse. It retained all 28/28 email and 3/3 image results,
and the same run's JSON baseline scored 27/28. There were 140 timed email calls
per mode and 15 timed image calls per mode. Peak allocated memory was 9.57 GiB
for batched email and 9.60 GiB for batched image classification.

Uncached probes remain about 3.5 ms faster on these short emails; the prefill and
batch-padding overhead outweigh the saved repeated context work there. Batched
reuse wins for the tested images. This is a small synthetic workload, not a
general workload crossover study. All output labels matched across probe modes;
floating-point logits are not bit-identical. The report preserves logits and
probability deltas as well as predictions.

Boundary checks at prefix lengths 511, 512, 513, and 1186 passed the prediction
and probability checks. Reversing field order preserved results, and a model
forward hook confirmed exactly two calls per batched classification. Full raw
measurements are in `results/probe-benchmark.json`.

## What the experiment does

1. Render a separate multiple-choice prompt for each field using Gemma's official
   chat template, with thinking disabled.
2. Find the exact shared **token** prefix and prefill it once.
3. Expand the prefix KV cache along the batch dimension and forward **all field
   questions together**. Each row owns an independent continuation. Question
   suffixes are right-padded and masked; each field is scored at its last real
   token. The whole classification takes two model forward calls: prefill plus
   the suffix batch.
4. Read the next-token logits at single-token labels `A`, `B`, etc.; take the
   maximum among the allowed labels and map it to a native boolean or enum.

There is no fine-tuning, new trained classification head, autoregressive answer
generation, or JSON parsing in the typed path. Python constructs the output from
the allowed values. This guarantees output shape, **not correct classification**.
It is a prototype of the approach in the project notes, not a reproduction of
TypeSafe's internal implementation.

The batched path is the default for email and image classification. Pass
`--sequential` to `classify`, `image`, or `evaluate` to reproduce the original
cached implementation, which deep-copies the cache for each field and runs one
forward per suffix. Pass `--uncached` to `classify` or `image` to run full prompts
individually without reusing the prefix. For short inputs with only two fields,
the extra prefix forward and cache handling can cost more than simply processing
the two complete prompts. Shared caching
reduces repeated prefix tokens; it does not automatically reduce latency. The
speedup above compares with JSON generation, not with uncached logit probes.

`--details` includes per-field softmax probabilities restricted to the allowed
labels, the probability mass assigned to those labels across the full vocabulary,
token counts, actual execution mode, model-call count, and synchronized GPU
timing. `batched_suffix` reports the whole batch duration; per-field times are
only reported for sequential execution. `tokens.evaluated` excludes padding;
`tokens.padding` records the additional padded suffix positions. These
probabilities are **not calibrated confidence**. Tokenization, cache expansion
or copying, and score extraction are
included in total latency; model loading is excluded. Emails beyond an
8192-token prompt limit are rejected rather than silently truncated.

## Reproduce the checks

```bash
uv run pytest -q
uv run typesafe-gemma --gpu 0 check-cache
uv run typesafe-gemma --gpu 0 evaluate --compare-json
uv run typesafe-gemma --gpu 0 benchmark-probes --repeats 5 --compare-json
```

The cache check compares batched, sequential cached, and full-prompt results and label probabilities
on both a short email and a prefix longer than the 512-token sliding window.
Small numerical differences are expected with BF16; a probability delta of 0.03
or more, or a changed answer, fails the check.

The evaluation uses 28 hand-authored synthetic emails (four per category),
including obvious phishing, unsolicited bulk mail, and an instruction-injection
example. This is an implementation smoke test, **not an estimate of real inbox
accuracy**. Ground-truth spam labels reflect the explicit policy above.

The optional JSON baseline uses the same model and field definitions, greedy
decoding, thinking disabled, and a 96-token limit. Invalid generated JSON counts
as incorrect. The prompt format differs because the baseline must generate a
whole object. Each mode gets one warmup; there is one measured run per email.
Results include per-example predictions, schema validity, per-field accuracy,
exact-match accuracy, median/p95 latency, and peak allocated GPU memory.
These are exploratory latency measurements, not a controlled throughput study.

`benchmark-probes` compares all three probe modes on the same loaded model,
using three warmups per mode and a seeded shuffle of input and method order for
each repetition. It records end-to-end latency samples, p95, per-mode peak GPU
allocation, prediction agreement, and accuracy on the existing synthetic set.
`--compare-json` includes ordinary JSON generation in the same run. It also
checks exact shared-prefix lengths of 511, 512, 513, and 1186 tokens, reverses
field order to check independence, verifies two actual model calls using a
forward hook, and measures the three image fixtures with the same protocol.
The report is written to `results/probe-benchmark.json`.

Reports are written under `results/`. To evaluate your own data, pass
`--dataset path/to/emails.jsonl`, with one object per line:

```json
{"id":"example-1","email":"From: ...\nSubject: ...\n...","expected":{"spam":false,"category":"personal"}}
```

## Initial image experiment

The same engine now accepts images and returns a demonstration schema:

```json
{"has_red": true, "shape": "circle"}
```

`shape` is one of `circle`, `square`, `triangle`, or `other`. This is a simple
check that pixels flow through the vision encoder into the shared prefix cache
and typed probes; it is not yet an image spam classifier or Doom controller.

```bash
uv run typesafe-gemma --gpu 0 check-images
uv run typesafe-gemma --gpu 0 image --file data/images/red-circle.png --details
```

All three deterministic fixtures (red circle, blue square, green triangle) passed
both the expected labels and the cached/full-prompt agreement check. Image
tokens are included in the common prefill, and branch suffixes contain only text.
The processor and image preprocessing remain unmodified. Image tests include
cold-start effects and should not be used as a latency benchmark.

## Doom compilation

The [Doom compilation experiment](docs/doom-compilation.md) compares compilation
on actual 640×480 Doom frames without reducing image detail. Compiling only the
language decoder reduced median latency from 88.5 ms to 74.1 ms and retained all
action choices on six test frames. The standalone benchmark is
`scripts/benchmark_doom_compile.py`. The browser controller now enables this
decoder-only recipe by default.

## Sources

- [Google's Gemma 4 E2B model and checkpoint](https://huggingface.co/google/gemma-4-E2B-it)
- [Transformers Gemma 4 implementation and documentation](https://huggingface.co/docs/transformers/model_doc/gemma4)

## Doom in the browser

```bash
uv run typesafe-doom --gpu 1 --host 127.0.0.1 --port 8766
```

Open `http://127.0.0.1:8766`. To use another network interface, pass its address
with `--host`. GPU 1 is the default for the Doom demo; the original classifier
defaults to GPU 0. The quick start explicitly chooses GPU 0 for machines with a
single GPU. Only physical
GPUs 0, 1, and 2 are accepted. GPU 3 remains blocked. `--no-model` runs a manual-only
demo without loading CUDA.

Doom compiles only the language decoder, preserving intermediate BF16 casts and
division rounding. Vision processing, screenshot resolution, and the visual-token
budget stay unchanged. Startup includes three warmup passes before model controls
become available; the UI shows **Preparing model…** during this stage. A cold
startup can take about 50 seconds, depending on compiler caches.

Use `--no-compile` to run the original eager decoder. Compilation failures keep
model controls unavailable and expose the error; they do not silently switch
execution mode. `GET /api/state` includes `model.decoder_compiled` and
`model.warmup_seconds`, and each prediction records `decoder_compiled`.

The UI starts paused. Click **Play**, then use **WASD** to move/strafe, **left/right
arrows** to turn, **Space** to fire, and **E** to use a door or switch. **Escape**
pauses. Click the game to restore keyboard focus.

- **You:** keyboard control, model idle.
- **Copilot:** keyboard control while Gemma predicts actions from screenshots.
- **Gemma:** the model's typed predictions control the game.
- **Model step:** pause, infer once, apply the prediction for 1–12 ticks, and stay
  paused. The default is four ticks.
- **Take control:** switch to manual and pause. Press Play to resume.
- **Reset:** start a new paused episode. The arena menu switches between Defend
  the center, Shooting gallery, and Deadly corridor.

The first connected browser owns controls. Other tabs can watch or explicitly
claim controls. Closing the controlling tab pauses the game. Keyboard focus loss
releases buttons, and missing keyboard heartbeats expire after 400 ms.

### Frame and action loop

The game lives in a dedicated CPU thread using synchronous ViZDoom `PLAYER`
mode, paced at a target of 35 game ticks per second. A FastAPI WebSocket publishes
state and JPEG frames at up to 20 updates per second. A separate inference thread
keeps Gemma resident on its chosen GPU. All five control fields share one image
prefill and one batched suffix forward pass; there is no generated JSON to parse.

```typescript
{
  move: "none" | "forward" | "backward",
  strafe: "none" | "left" | "right",
  turn: "none" | "left" | "right",
  fire: boolean,
  use: boolean
}
```

The model receives the RGB image from the same captured frame sent to the
browser, including the visible HUD. Health/ammo/game variables are used for the
web display only; they are not added to the model prompt. The inspector shows
the observation's frame ID and game tick, selected values, candidate scores,
inference latency, and the buttons actually being applied.

Only one inference request can be in flight. The next request samples the latest
frame; screenshots cannot build up in a work queue. Live results older than
750 ms are discarded. Model actions expire after 400 ms without a new result.
Reset, pause, takeover, and mode changes invalidate in-flight predictions. A
paused single step may take longer because its observation is held still.

Read-only integration endpoints: `GET /api/state` (JSON) and `GET /api/frame`
(JPEG). `WS /ws` emits a JSON `state` message followed by a binary JPEG whenever
the frame ID changes. Controller commands include:

```json
{"type":"mode","value":"autopilot"}
{"type":"pause","value":false}
{"type":"step"}
{"type":"reset"}
{"type":"keys","keys":["ArrowLeft","Space"]}
{"type":"settings","hz":5,"tics":4}
```

`keys` commands are only applied during manual/copilot play and need a heartbeat
at least every 400 ms while held. The default model decision target is 5 Hz and
can be set from 1–10 Hz; actual rate depends on inference and game timing. The
initial five-field decisions measured roughly 97–120 ms after warmup on a 4090.
With the compiled decoder integrated, a live GPU 0 check measured about 78 ms
median for both single steps and autopilot. Eight single steps and fourteen live
predictions passed; raw results are in `results/doom-compiled-integration.json`.
This is a working control loop, not a trained Doom policy. The frozen model can
move and fire but may miss enemies, waste ammunition, or die quickly.

The demo uses ViZDoom's packaged scenarios and bundled Freedoom assets.
[ViZDoom documentation](https://vizdoom.farama.org/) describes the game API.
No commercial Doom WAD is required for these scenarios.

Validation: unit tests cover action mapping, invalid commands, GPU restrictions,
and stale-result rejection. Browser checks exercised manual keys, copilot,
autopilot, stepping, takeover, arena switching, reset, desktop/mobile layout, and
JavaScript errors. Live protocol checks cover controller ownership, input expiry,
reset during inference, and disconnect behavior.
Compilation tests cover decoder-only targeting, rounding configuration, per-frame
CUDA graph boundaries, readiness after warmup, startup failure, and eager/manual
fallback options.
