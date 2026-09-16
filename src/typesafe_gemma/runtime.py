"""Restrict CUDA to one explicitly allowed physical GPU before importing torch."""

import os
import subprocess
import sys

_configured_uuid = None


def configure_gpu(gpu: int = 0) -> str:
    global _configured_uuid
    if gpu not in (0, 1, 2):
        raise ValueError("Only physical GPUs 0, 1, and 2 are allowed. GPU 3 is broken.")
    if "torch" in sys.modules:
        raise RuntimeError("configure_gpu must run before importing torch")
    uuid = subprocess.check_output(
        ["nvidia-smi", f"--id={gpu}", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    if not uuid.startswith("GPU-") or "\n" in uuid:
        raise RuntimeError("Could not identify the requested physical GPU")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _configured_uuid = uuid
    return uuid


def require_configured_gpu():
    if _configured_uuid is None or os.environ.get("CUDA_VISIBLE_DEVICES") != _configured_uuid:
        raise RuntimeError("Call configure_gpu(0), configure_gpu(1), or configure_gpu(2) before loading the model")
