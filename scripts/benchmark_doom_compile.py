"""Compare eager and torch.compile on unchanged 640x480 Doom screenshots.

Run from the project directory with:
  uv run python scripts/benchmark_doom_compile.py --gpu 0
"""

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import time
from pathlib import Path


def capture_frames(folder):
    """Use separate headless games, never the running interactive Doom session."""
    import vizdoom as v
    from PIL import Image

    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for scenario in ("basic", "defend_the_center", "deadly_corridor"):
        game = v.DoomGame()
        try:
            game.load_config(str(Path(v.scenarios_path) / f"{scenario}.cfg"))
            game.set_window_visible(False)
            game.set_sound_enabled(False)
            game.set_screen_format(v.ScreenFormat.RGB24)
            game.set_screen_resolution(v.ScreenResolution.RES_640X480)
            game.set_render_hud(True)
            game.set_render_crosshair(True)
            game.set_seed(7)
            game.init()
            for index in range(2):
                path = folder / f"{scenario}-{index}.png"
                Image.fromarray(game.get_state().screen_buffer.copy()).save(path)
                paths.append(path)
                game.make_action([False] * game.get_available_buttons_size(), 35)
        finally:
            game.close()
    return paths


def summary(samples):
    result = {}
    for stage in ("total", "prefill", "batched_suffix"):
        values = sorted(sample[stage] for sample in samples)
        result[stage] = {
            "median_ms": statistics.median(values),
            "mean_ms": statistics.mean(values),
            "p95_ms": values[math.ceil(len(values) * 0.95) - 1],
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--mode", choices=("default", "reduce-overhead"), default="reduce-overhead")
    parser.add_argument("--scope", choices=("model", "language"), default="model", help="compile the full model forward or only the language decoder")
    parser.add_argument("--preserve-rounding", action="store_true", help="preserve intermediate BF16 casts and eager division rounding")
    parser.add_argument("--frames", type=Path, default=Path("data/doom-compile"))
    parser.add_argument("--capture", action="store_true", help="regenerate the six deterministic Doom frames")
    parser.add_argument("--output", type=Path, default=Path("results/doom-compile-benchmark.json"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")

    from typesafe_gemma.runtime import configure_gpu
    uuid = configure_gpu(args.gpu)
    import torch
    if args.preserve_rounding:
        torch._inductor.config.emulate_precision_casts = True
        torch._inductor.config.emulate_divison_rounding = True
    from PIL import Image
    from typesafe_gemma import MODEL_ID, MODEL_REVISION
    from typesafe_gemma.engine import TypedGemma
    from typesafe_gemma.doom.actions import DOOM_FIELDS, DOOM_SYSTEM

    paths = capture_frames(args.frames) if args.capture or not args.frames.exists() else sorted(args.frames.glob("*.png"))
    if not paths:
        parser.error("No PNG frames found")
    frames = []
    for path in paths:
        with Image.open(path) as im:
            image = im.convert("RGB")
        if image.size != (640, 480):
            parser.error(f"{path} is {image.size}; expected the fixed 640x480 benchmark shape")
        frames.append((path, image))

    engine = TypedGemma()
    target = engine.model if args.scope == "model" else engine.model.model.language_model
    original_forward = target.forward

    def run(image):
        # Mark the frame boundary, not the boundary between prefill and suffix.
        torch.compiler.cudagraph_mark_step_begin()
        return engine.classify_image(image, fields=DOOM_FIELDS, system_prompt=DOOM_SYSTEM)

    for _ in range(3):
        run(frames[0][1])
    eager_references = [run(im) for _, im in frames]
    compiled_forward = torch.compile(original_forward, mode=args.mode, fullgraph=False, dynamic=False)
    target.forward = compiled_forward
    cold_started = time.perf_counter()
    warmups = []
    for index in range(3):
        report = run(frames[index % len(frames)][1])
        warmups.append(report["timing_ms"])
        print(f"Compiled warmup {index+1}/3: {report['timing_ms']['total']:.1f} ms", flush=True)
    cold_seconds = time.perf_counter() - cold_started
    compiled_references = [run(im) for _, im in frames]

    rng = random.Random(20260916)
    samples = {"eager": [], "compiled": []}
    peaks = {"eager": 0.0, "compiled": 0.0}
    stable = True
    graphs_before = dict(torch._dynamo.utils.counters["stats"])
    records = []
    for repeat in range(args.repeats):
        order = list(range(len(frames)))
        rng.shuffle(order)
        for index in order:
            path, im = frames[index]
            modes = ["eager", "compiled"]
            rng.shuffle(modes)
            for mode in modes:
                target.forward = original_forward if mode == "eager" else compiled_forward
                torch.cuda.reset_peak_memory_stats()
                report = run(im)
                peak = torch.cuda.max_memory_allocated() / 1024**3
                peaks[mode] = max(peaks[mode], peak)
                samples[mode].append(report["timing_ms"])
                reference = (eager_references if mode == "eager" else compiled_references)[index]
                stable &= report["result"] == reference["result"]
                records.append({"frame": str(path), "repeat": repeat, "mode": mode, "result": report["result"], "timing_ms": report["timing_ms"]})
        print(f"Measured repetition {repeat+1}/{args.repeats}", flush=True)
    target.forward = original_forward

    comparisons = []
    for (path, im), eager, compiled in zip(frames, eager_references, compiled_references):
        probability_delta = max(
            abs(value - compiled["scores"][field]["label_probabilities"][label])
            for field, score in eager["scores"].items()
            for label, value in score["label_probabilities"].items()
        )
        comparisons.append({
            "frame": str(path), "pixel_sha256": hashlib.sha256(im.tobytes()).hexdigest(),
            "same_action": eager["result"] == compiled["result"],
            "max_probability_delta": probability_delta, "eager": eager, "compiled": compiled,
        })
    summaries = {mode: summary(values) for mode, values in samples.items()}
    graphs_after = dict(torch._dynamo.utils.counters["stats"])
    recompiles = graphs_after.get("unique_graphs", 0) - graphs_before.get("unique_graphs", 0)
    report = {
        "model": MODEL_ID, "revision": MODEL_REVISION, "physical_gpu": args.gpu,
        "gpu_uuid": uuid, "torch": torch.__version__, "compile_mode": args.mode,
        "compile_scope": args.scope,
        "preserve_rounding": args.preserve_rounding,
        "image_resolution": [640, 480], "visual_token_budget": engine.model.config.vision_soft_tokens_per_image,
        "field_count": len(DOOM_FIELDS), "frames": len(frames), "repeats": args.repeats,
        "method": "Serial one-frame requests; same BF16 weights, processor, image detail, schema and shared-prefix probes; three warmups; seeded shuffled eager/compiled pairs; total includes preprocessing and score extraction but excludes model loading and compilation.",
        "compile_and_warmup_seconds": cold_seconds, "warmups": warmups,
        "steady_state_new_graphs": recompiles, "compiler_stats": graphs_after,
        "graph_breaks": dict(torch._dynamo.utils.counters["graph_break"]),
        "stable_predictions": stable, "all_actions_match": all(c["same_action"] for c in comparisons),
        "peak_allocated_gib": peaks, "summary": summaries,
        "speedup": summaries["eager"]["total"]["median_ms"] / summaries["compiled"]["total"]["median_ms"],
        "comparisons": comparisons, "measurements": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("comparisons", "measurements", "graph_breaks")}, indent=2))
    print(f"Full report: {args.output}")
    if not stable or not report["all_actions_match"] or recompiles:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
