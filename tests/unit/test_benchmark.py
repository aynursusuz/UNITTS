"""Benchmark lifecycle tests without optional engines or model weights."""

import json
from unittest.mock import Mock

import pytest

from unitts.benchmarks import runner


def _engine(result):
    engine = Mock()
    engine.get_vram_usage_mb.return_value = None
    engine.synthesize_to_file.return_value = result
    return engine


def _run(tmp_path, names):
    return runner.run_benchmark(
        engine_names=names,
        text="A benchmark test.",
        results_dir=tmp_path / "results",
        samples_dir=tmp_path / "samples",
        device="cpu",
    )


@pytest.mark.parametrize("failure_stage", [None, "ensure_loaded", "synthesize_to_file"])
def test_unloads_after_success_or_failure(monkeypatch, tmp_path, sample_tts_result, failure_stage):
    engine = _engine(sample_tts_result)
    if failure_stage:
        getattr(engine, failure_stage).side_effect = RuntimeError("model failed")
    monkeypatch.setattr(runner, "get_engine", Mock(return_value=engine))

    results = _run(tmp_path, ["test"])

    engine.unload_model.assert_called_once_with()
    expected_status = "error" if failure_stage else "success"
    assert results[0]["status"] == expected_status
    if failure_stage:
        assert results[0]["error"] == "model failed"
    saved = json.loads((tmp_path / "results" / "test.json").read_text())
    assert saved == results[0]


def test_constructor_failure_does_not_unload_previous_engine_again(
    monkeypatch, tmp_path, sample_tts_result
):
    first = _engine(sample_tts_result)
    last = _engine(sample_tts_result)
    monkeypatch.setattr(
        runner,
        "get_engine",
        Mock(side_effect=[first, ValueError("unknown engine"), last]),
    )

    results = _run(tmp_path, ["first", "missing", "last"])

    assert [entry["status"] for entry in results] == ["success", "error", "success"]
    assert results[1]["error"] == "unknown engine"
    first.unload_model.assert_called_once_with()
    last.unload_model.assert_called_once_with()
    assert (tmp_path / "results" / "missing.json").is_file()


@pytest.mark.parametrize("synthesis_fails", [False, True])
def test_cleanup_failure_preserves_result_and_continues(
    monkeypatch, tmp_path, sample_tts_result, synthesis_fails
):
    first = _engine(sample_tts_result)
    first.unload_model.side_effect = RuntimeError("cleanup failed")
    if synthesis_fails:
        first.synthesize_to_file.side_effect = RuntimeError("synthesis failed")
    second = _engine(sample_tts_result)
    monkeypatch.setattr(runner, "get_engine", Mock(side_effect=[first, second]))
    console = Mock()
    monkeypatch.setattr(runner, "console", console)

    results = _run(tmp_path, ["first", "second"])

    assert results[0]["status"] == ("error" if synthesis_fails else "success")
    if synthesis_fails:
        assert results[0]["error"] == "synthesis failed"
    assert results[1]["status"] == "success"
    first.unload_model.assert_called_once_with()
    second.unload_model.assert_called_once_with()
    assert any("Could not unload first" in str(call) for call in console.print.call_args_list)
    assert json.loads((tmp_path / "results" / "first.json").read_text()) == results[0]
