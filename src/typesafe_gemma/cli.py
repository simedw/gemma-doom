import argparse
import json
import math
import statistics
import sys
from pathlib import Path

from . import MODEL_ID, MODEL_REVISION
from .runtime import configure_gpu


def emit(value, output=None):
    rendered = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered)
    print(rendered, end="")


def main():
    parser = argparse.ArgumentParser(description="Frozen Gemma 4 typed email classifier")
    parser.add_argument("--gpu", type=int, choices=(0, 1, 2), default=0, help="physical GPU index; GPU 3 is never allowed")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download")
    classify = sub.add_parser("classify")
    source = classify.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--file", type=Path, help="UTF-8 email text; use - for stdin")
    classify.add_argument("--details", action="store_true")
    classify.add_argument("--uncached", action="store_true")
    classify.add_argument("--sequential", action="store_true", help="run cached field probes one at a time")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--dataset", type=Path, default=Path("data/emails.synthetic.jsonl"))
    evaluate.add_argument("--output", default="results/email-evaluation.json")
    evaluate.add_argument("--compare-json", action="store_true")
    evaluate.add_argument("--sequential", action="store_true")
    check = sub.add_parser("check-cache")
    check.add_argument("--output", default="results/cache-check.json")
    vision = sub.add_parser("image", help="vision smoke test: has_red boolean and shape enum")
    vision.add_argument("--file", required=True, type=Path)
    vision.add_argument("--details", action="store_true")
    vision.add_argument("--uncached", action="store_true")
    vision.add_argument("--sequential", action="store_true")
    vision_check = sub.add_parser("check-images", help="generate deterministic geometric test fixtures and classify them")
    vision_check.add_argument("--output", default="results/image-check.json")
    benchmark = sub.add_parser("benchmark-probes", help="compare batched, sequential, and uncached probes")
    benchmark.add_argument("--dataset", type=Path, default=Path("data/emails.synthetic.jsonl"))
    benchmark.add_argument("--output", default="results/probe-benchmark.json")
    benchmark.add_argument("--repeats", type=int, default=5)
    benchmark.add_argument("--compare-json", action="store_true")
    args = parser.parse_args()

    if args.command == "download":
        from huggingface_hub import snapshot_download
        path = snapshot_download(MODEL_ID, revision=MODEL_REVISION, allow_patterns=["*.json", "*.jinja", "*.safetensors", "README.md"])
        emit({"model": MODEL_ID, "revision": MODEL_REVISION, "path": path})
        return

    gpu_uuid = configure_gpu(args.gpu)
    from .engine import TypedGemma
    engine = TypedGemma()
    if args.command == "image":
        from PIL import Image
        with Image.open(args.file) as im:
            report = engine.classify_image(im.convert("RGB"), cached=not args.uncached, batched=not args.sequential)
        emit(report if args.details else report["result"])
        return
    if args.command == "classify":
        email = args.text if args.text is not None else (sys.stdin.read() if str(args.file) == "-" else args.file.read_text())
        report = engine.classify(email, cached=not args.uncached, batched=not args.sequential)
        emit(report if args.details else report["result"])
        return

    metadata = {"model": MODEL_ID, "revision": MODEL_REVISION, "physical_gpu": args.gpu, "gpu_uuid": gpu_uuid, "torch": engine.torch.__version__}
    if args.command == "benchmark-probes":
        from .benchmark import benchmark_probes
        if args.repeats < 1:
            parser.error("--repeats must be positive")
        report = benchmark_probes(engine, args.dataset, args.repeats, args.compare_json)
        emit({**metadata, **report}, args.output)
        if not report["checks_passed"]:
            raise SystemExit(1)
        return
    if args.command == "check-images":
        from PIL import Image, ImageDraw
        fixture_dir = Path("data/images")
        fixture_dir.mkdir(parents=True, exist_ok=True)
        checks = []
        for shape, color in (("circle", "red"), ("square", "blue"), ("triangle", "green")):
            im = Image.new("RGB", (384, 384), "white")
            draw = ImageDraw.Draw(im)
            if shape == "circle":
                draw.ellipse((72, 72, 312, 312), fill=color)
            elif shape == "square":
                draw.rectangle((72, 72, 312, 312), fill=color)
            else:
                draw.polygon([(192, 60), (60, 312), (324, 312)], fill=color)
            path = fixture_dir / f"{color}-{shape}.png"
            im.save(path)
            cached = engine.classify_image(im)
            sequential = engine.classify_image(im, batched=False)
            uncached = engine.classify_image(im, cached=False)
            expected = {"has_red": color == "red", "shape": shape}
            delta = max(abs(a-b) for reference in (sequential, uncached) for name in cached["scores"] for a,b in zip(cached["scores"][name]["label_probabilities"].values(), reference["scores"][name]["label_probabilities"].values()))
            checks.append({"file": str(path), "expected": expected, "passed": cached["result"] == sequential["result"] == uncached["result"] == expected and delta < 0.03, "max_probability_delta": delta, "cached": cached, "sequential": sequential, "uncached": uncached})
        passed = all(c["passed"] for c in checks)
        emit({**metadata, "passed": passed, "checks": checks}, args.output)
        if not passed:
            raise SystemExit(1)
        return
    if args.command == "check-cache":
        cases = [
            "From: mom@example.org\nSubject: Sunday lunch\nCome over at noon. Love, Mum.",
            "From: colleague@example.org\nSubject: Project update\n" + "We reviewed the implementation and discussed the next project milestones. " * 100 + "\nPlease review the attached plan before our meeting.",
        ]
        checks = []
        for email in cases:
            cached = engine.classify(email)
            sequential = engine.classify(email, batched=False)
            uncached = engine.classify(email, cached=False)
            delta = max(abs(a-b) for reference in (sequential, uncached) for name in cached["scores"] for a,b in zip(cached["scores"][name]["label_probabilities"].values(), reference["scores"][name]["label_probabilities"].values()))
            checks.append({"prefix_tokens": cached["tokens"]["shared_prefix"], "same_result": cached["result"] == sequential["result"] == uncached["result"], "max_probability_delta": delta, "cached": cached, "sequential": sequential, "uncached": uncached})
        passed = all(c["same_result"] and c["max_probability_delta"] < 0.03 for c in checks)
        emit({**metadata, "passed": passed, "checks": checks}, args.output)
        if not passed:
            raise SystemExit(1)
        return

    from .schema import validate_email_result
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    if not rows:
        parser.error("Dataset must contain at least one example")
    for row in rows:
        validate_email_result(row["expected"])
    # Exclude one warmup from latency measurements for each execution mode.
    engine.classify(rows[0]["email"], batched=not args.sequential)
    if args.compare_json:
        engine.json_baseline(rows[0]["email"])
    engine.torch.cuda.reset_peak_memory_stats()
    predictions = []
    for i, row in enumerate(rows):
        prediction = {"id": row["id"], "expected": row["expected"], "typed": engine.classify(row["email"], batched=not args.sequential)}
        if args.compare_json:
            prediction["json_baseline"] = engine.json_baseline(row["email"])
        predictions.append(prediction)
        print(f"[{i+1}/{len(rows)}] {row['id']}: {prediction['typed']['result']}", file=sys.stderr)
    summary = {}
    for mode in ("typed", "json_baseline") if args.compare_json else ("typed",):
        n = len(rows)
        valid = [p for p in predictions if p[mode]["result"] is not None]
        latencies = sorted(p[mode]["timing_ms"]["total"] for p in predictions)
        summary[mode] = {
            "schema_valid_rate": len(valid) / n,
            "spam_accuracy": sum(p[mode]["result"]["spam"] == p["expected"]["spam"] for p in valid) / n,
            "category_accuracy": sum(p[mode]["result"]["category"] == p["expected"]["category"] for p in valid) / n,
            "exact_match": sum(p[mode]["result"] == p["expected"] for p in valid) / n,
            "median_ms": statistics.median(latencies), "mean_ms": statistics.mean(latencies),
            "p95_ms": latencies[min(n-1, math.ceil(0.95*n)-1)],
        }
    if args.compare_json:
        summary["median_speedup_vs_json"] = summary["json_baseline"]["median_ms"] / summary["typed"]["median_ms"]
    report = {**metadata, "dataset": str(args.dataset), "dataset_kind": "hand-authored synthetic smoke test; not a real-world accuracy benchmark", "examples": len(rows), "peak_gpu_allocated_gib": engine.torch.cuda.max_memory_allocated()/1024**3, "summary": summary, "predictions": predictions}
    emit(report, args.output)


if __name__ == "__main__":
    main()
