import io
import logging
import queue
import threading
import time
from pathlib import Path

from PIL import Image

from .actions import BUTTON_NAMES, Action, result_is_current
from .model_worker import ModelWorker, Observation

logger = logging.getLogger(__name__)

SCENARIOS = {
    "defend_the_center": "Defend the center",
    "basic": "Shooting gallery",
    "deadly_corridor": "Deadly corridor",
}


class DoomSession:
    """All ViZDoom calls belong to this thread; the web loop only exchanges messages."""

    def __init__(
        self, *, gpu=1, model_enabled=True, scenario="defend_the_center",
        compile_decoder=True,
    ):
        self.gpu = gpu
        self.scenario = scenario
        self.commands = queue.Queue(maxsize=128)
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name="doom-game", daemon=True)
        self.model = ModelWorker(enabled=model_enabled, compile_decoder=compile_decoder)
        self.latest = {"status": "starting", "paused": True, "mode": "manual"}
        self.jpeg = None
        self._jpeg = None
        self.frame_id = 0
        self.generation = 0
        self.episode = 0
        self.mode = "manual"
        self.paused = True
        self.manual = Action()
        self.manual_until = 0.0
        self.ai_action = Action()
        self.ai_until = 0.0
        self.applied = Action()
        self.last_prediction = None
        self.last_request = 0.0
        self.model_hz = 5.0
        self.step_tics = 4
        self.pending_step = False
        self.step_remaining = 0
        self.discarded = 0
        self.error = None
        self.event = "Ready for manual play"
        self.game = None
        self.last_frame = None
        self.last_capture = 0.0

    def start(self):
        self.model.start()
        self.thread.start()

    def close(self):
        self.stop.set()
        self.model.close()
        self.thread.join(timeout=5)

    def command(self, command):
        try:
            self.commands.put_nowait(command)
            return True
        except queue.Full:
            return False

    def snapshot(self):
        with self.lock:
            return dict(self.latest), self.jpeg

    def _invalidate(self):
        self.generation += 1
        self.ai_action, self.manual, self.applied = Action(), Action(), Action()
        self.ai_until = self.manual_until = 0.0
        self.pending_step = False
        self.step_remaining = 0

    def _open_game(self):
        import vizdoom as v

        if self.game is not None:
            self.game.close()
        self.game = v.DoomGame()
        self.game.load_config(str(Path(v.scenarios_path) / f"{self.scenario}.cfg"))
        self.game.set_mode(v.Mode.PLAYER)
        self.game.set_window_visible(False)
        self.game.set_sound_enabled(False)
        self.game.set_screen_format(v.ScreenFormat.RGB24)
        self.game.set_screen_resolution(v.ScreenResolution.RES_640X480)
        self.game.set_render_hud(True)
        self.game.set_render_crosshair(True)
        self.game.set_available_buttons([getattr(v.Button, b) for b in BUTTON_NAMES])
        self.game.set_episode_timeout(35 * 180)
        self.game.set_seed(7)
        self.game.init()
        self.episode += 1
        self._invalidate()
        self.paused = True
        self.last_prediction = None
        self.error = None
        self._capture()

    def _capture(self):
        state = self.game.get_state()
        if state is None:
            return
        self.last_frame = Image.fromarray(state.screen_buffer.copy())
        self.frame_id += 1
        self.frame_tic = state.tic
        self.last_capture = time.monotonic()
        buf = io.BytesIO()
        self.last_frame.save(buf, format="JPEG", quality=82)
        self._jpeg = buf.getvalue()

    def _submit(self, single_step=False):
        if self.last_frame is None:
            return False
        obs = Observation(
            self.last_frame.copy(),
            self.frame_id,
            self.frame_tic,
            self.generation,
            self.last_capture,
            single_step,
        )
        if self.model.submit(obs):
            self.last_request = time.monotonic()
            self.pending_step = single_step
            return True
        return False

    def _handle(self, command):
        kind = command["type"]
        if kind == "keys":
            if self.mode in ("manual", "copilot") and not self.paused:
                self.manual = Action.from_keys(command["keys"])
                self.manual_until = time.monotonic() + 0.4
        elif kind == "mode":
            self._invalidate()
            self.mode = command["value"]
            self.event = f"Control: {self.mode}"
        elif kind == "pause":
            self._invalidate()
            self.paused = command["value"]
            self.event = "Paused" if self.paused else "Running"
        elif kind in ("takeover", "disconnect"):
            self._invalidate()
            self.mode, self.paused = "manual", True
            self.event = (
                "Manual control — press Play"
                if kind == "takeover"
                else "Controller disconnected; paused"
            )
        elif kind == "reset":
            self._invalidate()
            self.game.new_episode()
            self.episode += 1
            self.paused = True
            self.last_prediction = None
            self.error = None
            self.event = "New episode — press Play or Model step"
            self._capture()
        elif kind == "scenario":
            self.scenario = command["value"]
            self._open_game()
            self.event = "Arena loaded — paused"
        elif kind == "step":
            if (
                self.model.status != "ready"
                or self.model.busy
                or self.game.is_episode_finished()
            ):
                self.event = (
                    "Model step unavailable: wait for the model or reset the episode"
                )
                return
            self._invalidate()
            self.paused = True
            self._capture()
            self._submit(single_step=True)
            self.event = "Reading this frame…"
        elif kind == "settings":
            self.model_hz = command["hz"]
            self.step_tics = command["tics"]

    def _read_prediction(self):
        result = self.model.poll()
        if result is None:
            return
        obs = result["observation"]
        now = time.monotonic()
        if not result_is_current(
            result_generation=obs.generation,
            generation=self.generation,
            result_age=now - obs.captured_at,
            single_step=obs.single_step,
        ):
            self.discarded += 1
            return
        if "error" in result:
            self.error = result["error"]
            self._invalidate()
            self.paused = True
            return
        self.last_prediction = {
            **result["report"],
            "frame_id": obs.frame_id,
            "tic": obs.tic,
            "age_ms": (now - obs.captured_at) * 1000,
            "received_at": now,
            "single_step": obs.single_step,
        }
        if obs.single_step or (self.mode == "autopilot" and not self.paused):
            self.ai_action = result["action"]
            self.ai_until = now + 0.4
        if obs.single_step:
            self.pending_step = False
            self.step_remaining = self.step_tics
            self.event = f"Applying model action for {self.step_tics} ticks"

    def _publish(self):
        import vizdoom as v

        finished = self.game.is_episode_finished()
        prediction = dict(self.last_prediction) if self.last_prediction else None
        if prediction:
            prediction["since_ms"] = (
                time.monotonic() - prediction.pop("received_at")
            ) * 1000
        payload = {
            "status": "finished" if finished else "ready",
            "frame_id": self.frame_id,
            "tic": self.game.get_episode_time(),
            "episode": self.episode,
            "scenario": self.scenario,
            "mode": self.mode,
            "paused": self.paused,
            "health": self.game.get_game_variable(v.GameVariable.HEALTH),
            "ammo": self.game.get_game_variable(v.GameVariable.SELECTED_WEAPON_AMMO),
            "kills": self.game.get_game_variable(v.GameVariable.KILLCOUNT),
            "reward": round(self.game.get_total_reward(), 2),
            "applied": self.applied.json(),
            "prediction": prediction,
            "model": {
                "status": self.model.status,
                "busy": self.model.busy,
                "gpu": self.gpu,
                "error": self.model.error,
                "decoder_compiled": self.model.decoder_compiled,
                "warmup_seconds": self.model.warmup_seconds,
            },
            "model_hz": self.model_hz,
            "step_tics": self.step_tics,
            "pending_step": self.pending_step or self.step_remaining > 0,
            "discarded": self.discarded,
            "event": self.event,
            "error": self.error,
        }
        with self.lock:
            self.latest = payload
            self.jpeg = self._jpeg

    def _run(self):
        try:
            self._open_game()
            next_tick = time.monotonic()
            last_publish = 0.0
            while not self.stop.is_set():
                for _ in range(128):
                    try:
                        command = self.commands.get_nowait()
                    except queue.Empty:
                        break
                    self._handle(command)
                self._read_prediction()
                now = time.monotonic()
                finished = self.game.is_episode_finished()
                if finished:
                    if not self.paused or self.pending_step or self.step_remaining:
                        self._invalidate()
                    self.paused = True
                    self.applied = Action()
                    self.event = "Episode finished — Reset to play again"
                can_advance = not finished and (
                    not self.paused or self.step_remaining > 0
                )
                if now >= next_tick:
                    if can_advance:
                        if self.step_remaining:
                            action = self.ai_action
                            self.step_remaining -= 1
                            if not self.step_remaining:
                                self.event = "Model step complete — paused"
                        elif self.mode == "autopilot":
                            action = self.ai_action if now < self.ai_until else Action()
                        else:
                            action = (
                                self.manual if now < self.manual_until else Action()
                            )
                        self.applied = action
                        self.game.make_action(action.buttons(), 1)
                        if self.paused and not self.step_remaining:
                            self._capture()
                    else:
                        self.applied = Action()
                    next_tick = now + 1 / 35
                if now - last_publish >= 1 / 20:
                    if can_advance:
                        self._capture()
                    if (
                        not self.paused
                        and self.mode in ("copilot", "autopilot")
                        and not finished
                        and now - self.last_request >= 1 / self.model_hz
                    ):
                        self._submit()
                    self._publish()
                    last_publish = now
                self.stop.wait(0.003)
        except Exception as exc:
            logger.exception("Doom game worker failed")
            with self.lock:
                self.latest = {
                    "status": "error",
                    "error": str(exc),
                    "paused": True,
                    "mode": "manual",
                }
        finally:
            if self.game:
                self.game.close()
