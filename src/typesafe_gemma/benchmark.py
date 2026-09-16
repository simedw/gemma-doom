"""Paired measurements of probe implementations on the same loaded model."""

import hashlib
import json
import math
import random
import statistics
import sys
import time

from .engine import common_prefix_length
from .schema import EMAIL_FIELDS, validate_email_result


PROBE_MODES = {
    "batched": {"cached": True, "batched": True},
    "sequential": {"cached": True, "batched": False},
    "uncached": {"cached": False},
}


def agreement(candidate, reference):
    delta = max(
        abs(probability - reference["scores"][field]["label_probabilities"][label])
        for field, scores in candidate["scores"].items()
        for label, probability in scores["label_probabilities"].items()
    )
    logit_delta = max(
        abs(a - b)
        for field, scores in candidate["scores"].items()
        for a, b in zip(scores["label_logits"], reference["scores"][field]["label_logits"])
    )
    same = candidate["result"] == reference["result"]
    return {"passed": same and delta < 0.03, "same_result": same, "max_probability_delta": delta, "max_logit_delta": logit_delta}


def _measure_group(engine, cases, run, repeats, compare_json=False):
    modes = list(PROBE_MODES) + (["json_baseline"] if compare_json else [])
    # Warm every implementation, using the same model instance, before measuring.
    for _ in range(3):
        for mode in modes:
            run(cases[0], mode)
    rng = random.Random(20260916)
    measurements = {case["id"]: {mode: {"samples_ms": [], "peak_allocated_gib": 0.0} for mode in modes} for case in cases}
    for repeat in range(repeats):
        order = list(cases)
        rng.shuffle(order)
        for case in order:
            mode_order = list(modes)
            rng.shuffle(mode_order)
            for mode in mode_order:
                engine.torch.cuda.reset_peak_memory_stats()
                report = run(case, mode)
                record = measurements[case["id"]][mode]
                record["samples_ms"].append(report["timing_ms"])
                record["peak_allocated_gib"] = max(record["peak_allocated_gib"], engine.torch.cuda.max_memory_allocated() / 1024**3)
                if "first_report" not in record:
                    record["first_report"] = report
                    record["stable_result"] = True
                else:
                    record["stable_result"] &= record["first_report"]["result"] == report["result"]
        print(f"Completed repetition {repeat+1}/{repeats} on {len(cases)} inputs", file=sys.stderr)
    summary = {}
    for mode in modes:
        records = [measurements[case["id"]][mode] for case in cases]
        latencies = sorted(sample["total"] for record in records for sample in record["samples_ms"])
        summary[mode] = {
            "measurements": len(latencies), "median_ms": statistics.median(latencies),
            "mean_ms": statistics.mean(latencies), "p95_ms": latencies[math.ceil(0.95 * len(latencies))-1],
            "peak_allocated_gib": max(r["peak_allocated_gib"] for r in records),
            "correct_examples": sum(measurements[c["id"]][mode]["first_report"]["result"] == c["expected"] for c in cases),
            "valid_examples": sum(r["first_report"]["result"] is not None for r in records),
        }
    predictions, checks = [], []
    for case in cases:
        records = measurements[case["id"]]
        candidate = records["batched"]["first_report"]
        comparisons = {mode: agreement(candidate, records[mode]["first_report"]) for mode in ("sequential", "uncached")}
        stable = all(records[mode]["stable_result"] for mode in PROBE_MODES)
        checks.append(stable and all(c["passed"] for c in comparisons.values()))
        predictions.append({"id": case["id"], "expected": case["expected"], "comparisons": comparisons, "modes": records})
    return {
        "examples": len(cases), "repeats": repeats, "summary": summary,
        "speedup_vs_sequential": summary["sequential"]["median_ms"] / summary["batched"]["median_ms"],
        "latency_reduction_vs_sequential_percent": 100 * (1 - summary["batched"]["median_ms"] / summary["sequential"]["median_ms"]),
        "checks_passed": all(checks), "predictions": predictions,
    }


def _check_boundaries(engine, email):
    """Stress exact sliding-window boundaries and branch order, independent of labels."""
    base = engine._prompts(email)
    base_len = common_prefix_length(base)
    # This artificial filler is for cache equivalence, not an accuracy example.
    filler = engine.tokenizer.encode(" note", add_special_tokens=False)[0]
    checks = []
    for target in (511, 512, 513, 1186):
        if base_len > target:
            raise ValueError("Boundary fixture is too long")
        sequences = [ids[:base_len] + [filler] * (target - base_len) + ids[base_len:] for ids in base]
        reports = {}
        for mode, options in PROBE_MODES.items():
            engine._sync()
            reports[mode] = engine._probe(sequences, EMAIL_FIELDS, started=time.perf_counter(), **options)
        engine._sync()
        reordered = engine._probe(list(reversed(sequences)), tuple(reversed(EMAIL_FIELDS)), cached=True, batched=True, started=time.perf_counter())
        comparisons = {mode: agreement(reports["batched"], reports[mode]) for mode in ("sequential", "uncached")}
        comparisons["reordered_fields"] = agreement(reports["batched"], reordered)
        checks.append({"prefix_tokens": target, "passed": all(c["passed"] for c in comparisons.values()), "comparisons": comparisons, "reports": reports})
    # Instrument an actual call to verify the advertised model-call count.
    calls = []
    handle = engine.model.register_forward_pre_hook(lambda _model, _inputs: calls.append(1))
    try:
        report = engine.classify(email)
    finally:
        handle.remove()
    checks.append({"actual_model_forward_calls": len(calls), "passed": len(calls) == report["model_forward_calls"] == 2})
    return checks


def benchmark_probes(engine, dataset, repeats, compare_json):
    content = dataset.read_bytes()
    rows = [json.loads(line) for line in content.decode().splitlines() if line.strip()]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Dataset must be nonempty with unique example IDs")
    for row in rows:
        validate_email_result(row["expected"])

    def run_email(case, mode):
        if mode == "json_baseline":
            return engine.json_baseline(case["email"])
        return engine.classify(case["email"], **PROBE_MODES[mode])

    emails = _measure_group(engine, rows, run_email, repeats, compare_json)
    boundaries = _check_boundaries(engine, rows[0]["email"])

    from PIL import Image, ImageDraw
    cases = []
    for shape, color in (("circle", "red"), ("square", "blue"), ("triangle", "green")):
        im = Image.new("RGB", (384, 384), "white")
        draw = ImageDraw.Draw(im)
        if shape == "circle":
            draw.ellipse((72, 72, 312, 312), fill=color)
        elif shape == "square":
            draw.rectangle((72, 72, 312, 312), fill=color)
        else:
            draw.polygon([(192, 60), (60, 312), (324, 312)], fill=color)
        cases.append({"id": f"{color}_{shape}", "image": im, "expected": {"has_red": color == "red", "shape": shape}})
    images = _measure_group(engine, cases, lambda case, mode: engine.classify_image(case["image"], **PROBE_MODES[mode]), repeats)
    return {
        "dataset": str(dataset), "dataset_sha256": hashlib.sha256(content).hexdigest(),
        "method": "Same loaded BF16 model and GPU; 3 warmups per mode; seeded shuffled input/mode order for each repetition; total latency includes preprocessing, cache expansion/copies, forward passes, and scoring; model loading excluded.",
        "checks_passed": emails["checks_passed"] and images["checks_passed"] and all(c["passed"] for c in boundaries),
        "emails": emails, "images": images, "boundary_checks": boundaries,
    }
