import pytest
from pydantic import ValidationError

from typesafe_gemma.doom.actions import Action, result_is_current
from typesafe_gemma.doom.server import COMMAND


def test_simultaneous_controls_match_vizdoom_button_order():
    action = Action(move="forward", strafe="right", turn="left", fire=True)
    assert action.buttons() == [True, False, False, True, True, False, True, False]


def test_opposite_keys_cancel_without_cancelling_fire():
    action = Action.from_keys(
        {"KeyW", "KeyS", "KeyA", "KeyD", "ArrowLeft", "ArrowRight", "Space"}
    )
    assert action == Action(fire=True)


@pytest.mark.parametrize(
    "payload",
    [
        {"move": "fly"},
        {"turn": "backward"},
        {"fire": "false"},
        {"use": 1},
    ],
)
def test_invalid_model_controls_are_rejected(payload):
    with pytest.raises(ValueError):
        Action(**payload)


def test_old_inference_never_survives_a_reset_or_takeover():
    assert not result_is_current(result_generation=1, generation=2, result_age=0.1)
    assert not result_is_current(
        result_generation=1, generation=2, result_age=0.1, single_step=True
    )
    assert not result_is_current(result_generation=2, generation=2, result_age=0.8)
    # Paused single-step observations cannot become stale through game advancement.
    assert result_is_current(
        result_generation=2, generation=2, result_age=2, single_step=True
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "pause", "value": "false"},
        {"type": "settings", "hz": 100, "tics": 4},
        {"type": "settings", "hz": 5, "tics": 0},
        {"type": "settings", "hz": 5, "tics": True},
        {"type": "scenario", "value": "../../anything"},
        {"type": "keys", "keys": ["Escape"]},
        {"type": "mode", "value": "autopilot", "surprise": True},
    ],
)
def test_web_commands_cannot_bypass_the_control_contract(payload):
    with pytest.raises(ValidationError):
        COMMAND.validate_python(payload)
