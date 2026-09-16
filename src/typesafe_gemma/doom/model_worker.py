import logging
import queue
import threading
import time
from dataclasses import dataclass

from .actions import DOOM_FIELDS, DOOM_SYSTEM, Action

logger = logging.getLogger(__name__)


@dataclass
class Observation:
    image: object
    frame_id: int
    tic: int
    generation: int
    captured_at: float
    single_step: bool = False


class ModelWorker:
    """One model, one request in flight. No inference backlog can accumulate."""

    def __init__(self, enabled=True, *, compile_decoder=True):
        self.requests = queue.Queue(maxsize=1)
        self.results = queue.Queue(maxsize=1)
        self.status = "loading" if enabled else "disabled"
        self.error = None
        self.busy = False
        self.enabled = enabled
        self.compile_enabled = compile_decoder
        self.decoder_compiled = False
        self.warmup_seconds = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="gemma-doom", daemon=True)

    def start(self):
        if self.enabled:
            self.thread.start()

    def submit(self, observation):
        if self.status != "ready" or self.busy:
            return False
        self.busy = True
        self.requests.put_nowait(observation)
        return True

    def poll(self):
        try:
            result = self.results.get_nowait()
        except queue.Empty:
            return None
        self.busy = False
        return result

    def _run(self):
        try:
            from ..engine import TypedGemma

            model = TypedGemma()
            warmup_started = time.monotonic()
            self.status = "compiling" if self.compile_enabled else "warming"
            if self.compile_enabled:
                model.compile_decoder()
            # Warm both graph shapes before accepting an observation. Vision and
            # image detail remain unchanged, and the output contract is checked.
            from PIL import Image

            warmup_image = Image.new("RGB", (640, 480), "black")
            for _ in range(3 if self.compile_enabled else 1):
                if self.stop.is_set():
                    return
                report = model.classify_image(
                    warmup_image, fields=DOOM_FIELDS, system_prompt=DOOM_SYSTEM,
                )
                Action(**report["result"])
            if self.stop.is_set():
                return
            self.decoder_compiled = model.decoder_compiled
            self.warmup_seconds = time.monotonic() - warmup_started
            self.status = "ready"
            logger.info(
                "Doom model ready: decoder_compiled=%s, warmup=%.1fs",
                self.decoder_compiled, self.warmup_seconds,
            )
            while not self.stop.is_set():
                try:
                    obs = self.requests.get(timeout=0.2)
                except queue.Empty:
                    continue
                try:
                    report = model.classify_image(
                        obs.image, fields=DOOM_FIELDS, system_prompt=DOOM_SYSTEM
                    )
                    action = Action(**report["result"])
                    result = {
                        "observation": obs,
                        "action": action,
                        "report": report,
                        "completed_at": time.monotonic(),
                    }
                except Exception as exc:
                    logger.exception("Doom inference failed")
                    result = {"observation": obs, "error": str(exc)}
                self.results.put(result)
        except Exception as exc:
            logger.exception("Could not load Doom model")
            self.status, self.error = "error", str(exc)

    def close(self):
        self.stop.set()
        if self.thread.is_alive():
            self.thread.join(timeout=15)
