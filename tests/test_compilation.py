import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from typesafe_gemma.doom.model_worker import ModelWorker, Observation
from typesafe_gemma.engine import TypedGemma


def test_compile_recipe_targets_only_decoder_and_marks_each_request():
    engine = TypedGemma.__new__(TypedGemma)
    original = Mock()
    decoder = SimpleNamespace(forward=original)
    vision = Mock()
    engine.model = SimpleNamespace(model=SimpleNamespace(language_model=decoder, vision_tower=vision))
    engine.decoder_compiled = False
    engine.torch = SimpleNamespace(
        _inductor=SimpleNamespace(config=SimpleNamespace()),
        compile=Mock(return_value=Mock()),
        compiler=SimpleNamespace(cudagraph_mark_step_begin=Mock()),
    )
    engine._begin_request()
    engine.torch.compiler.cudagraph_mark_step_begin.assert_not_called()
    engine.compile_decoder()
    engine.compile_decoder()
    engine.torch.compile.assert_called_once_with(
        original, mode="reduce-overhead", fullgraph=False, dynamic=False,
    )
    assert decoder.forward is engine.torch.compile.return_value
    assert engine.model.model.vision_tower is vision
    config = engine.torch._inductor.config
    assert config.emulate_precision_casts and config.emulate_divison_rounding
    engine._begin_request()
    engine._begin_request()
    assert engine.torch.compiler.cudagraph_mark_step_begin.call_count == 2


def wait_for(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail("Worker did not reach expected state")
        time.sleep(0.005)


def test_worker_accepts_frames_only_after_all_compiled_warmups(monkeypatch):
    import threading

    entered, release = threading.Event(), threading.Event()
    worker = ModelWorker()
    calls = []

    class FakeModel:
        decoder_compiled = False

        def compile_decoder(self):
            self.decoder_compiled = True

        def classify_image(self, image, **kwargs):
            calls.append((image.size, kwargs))
            if len(calls) == 1:
                entered.set()
                assert release.wait(3)
            return {"result": {"move": "none", "strafe": "none", "turn": "none", "fire": False, "use": False}}

    monkeypatch.setattr("typesafe_gemma.engine.TypedGemma", FakeModel)
    worker.start()
    try:
        assert entered.wait(3)
        assert worker.status == "compiling"
        assert not worker.submit(None)
        release.set()
        wait_for(lambda: worker.status == "ready")
        assert len(calls) == 3
        assert all(size == (640, 480) and len(kwargs["fields"]) == 5 for size, kwargs in calls)
        assert worker.decoder_compiled and worker.warmup_seconds is not None
        from PIL import Image
        observation = Observation(Image.new("RGB", (640, 480)), 1, 0, 0, time.monotonic())
        assert worker.submit(observation)
        assert not worker.submit(observation)
        wait_for(lambda: not worker.results.empty())
        assert worker.poll()["observation"] is observation
        assert not worker.busy
    finally:
        release.set()
        worker.close()


def test_compilation_failure_does_not_enable_model_controls(monkeypatch):
    model = Mock()
    model.compile_decoder.side_effect = RuntimeError("compile failed")
    monkeypatch.setattr("typesafe_gemma.engine.TypedGemma", Mock(return_value=model))
    worker = ModelWorker()
    worker.start()
    try:
        wait_for(lambda: worker.status == "error")
        assert "compile failed" in worker.error
        assert not worker.submit(None)
        model.classify_image.assert_not_called()
    finally:
        worker.close()


def test_eager_option_and_disabled_model_skip_compilation(monkeypatch):
    model = Mock(decoder_compiled=False)
    model.classify_image.return_value = {"result": {"move": "none", "strafe": "none", "turn": "none", "fire": False, "use": False}}
    factory = Mock(return_value=model)
    monkeypatch.setattr("typesafe_gemma.engine.TypedGemma", factory)
    disabled = ModelWorker(enabled=False)
    disabled.start()
    factory.assert_not_called()
    worker = ModelWorker(compile_decoder=False)
    worker.start()
    try:
        wait_for(lambda: worker.status == "ready")
        model.compile_decoder.assert_not_called()
        model.classify_image.assert_called_once()
        assert not worker.decoder_compiled
    finally:
        worker.close()
