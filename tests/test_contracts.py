import sys
from unittest.mock import patch

import pytest

from typesafe_gemma.engine import common_prefix_length, padded_suffix_batch
from typesafe_gemma.runtime import configure_gpu, require_configured_gpu
from typesafe_gemma.schema import validate_email_result


def test_shared_prefix_at_divergence_and_end():
    assert common_prefix_length([[1, 2, 3], [1, 2, 4]]) == 2
    assert common_prefix_length([[1, 2], [1, 2, 3]]) == 2
    assert common_prefix_length([[1], [2]]) == 0


def test_suffix_padding_preserves_positions_and_masks_only_trailing_tokens():
    rows, masks, last = padded_suffix_batch([[10, 11, 21], [10, 11, 31, 32, 33]], 2, 0)
    assert rows == [[21, 0, 0], [31, 32, 33]]
    assert masks == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
    assert [row[pos] for row, pos in zip(rows, last)] == [21, 33]


def test_equal_suffix_lengths_need_no_padding_and_share_answer_position():
    rows, masks, last = padded_suffix_batch([[10, 20], [10, 30]], 1, 0)
    assert rows == [[20], [30]]
    assert masks == [[1, 1], [1, 1]]
    assert last == [0, 0]


def test_empty_suffix_is_rejected():
    with pytest.raises(ValueError, match="suffix token"):
        padded_suffix_batch([[10], [10, 20]], 1, 0)


def test_broken_gpu_rejected_before_any_hardware_access():
    with patch("subprocess.check_output") as query:
        with pytest.raises(ValueError, match="broken"):
            configure_gpu(3)
        query.assert_not_called()


def test_allowed_gpu_is_selected_by_uuid(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    with patch("subprocess.check_output", return_value="GPU-test\n") as query:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
        assert configure_gpu(1) == "GPU-test"
        import os
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-test"
        assert "--id=1" in query.call_args.args[0]


def test_model_cannot_load_after_gpu_visibility_is_changed(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    with pytest.raises(RuntimeError, match="configure_gpu"):
        require_configured_gpu()


@pytest.mark.parametrize("invalid", [
    {"category": "work"}, {"spam": "false", "category": "work"},
    {"spam": 0, "category": "work"}, {"spam": False, "category": "unknown"},
    {"spam": False, "category": "work", "explanation": "ok"},
])
def test_schema_rejects_missing_or_coerced_values(invalid):
    with pytest.raises(ValueError):
        validate_email_result(invalid)
