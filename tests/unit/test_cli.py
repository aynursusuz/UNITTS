"""CLI setup diagnostics and foreground server lifecycle."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from unitts.cli import app
from unitts.setup_errors import EngineSetupError

runner = CliRunner()


def test_higgs_launcher_uses_installed_server_and_client_default_port(monkeypatch, tmp_path):
    from unitts import server
    from unitts.engines.higgs_tts_engine import HiggsTTSEngine

    monkeypatch.delenv("HIGGS_TTS_BASE_URL", raising=False)
    monkeypatch.setattr(server.sys, "prefix", str(tmp_path / ".venv-higgs-tts"))
    executable = tmp_path / ".venv-higgs-server" / "bin" / "sgl-omni"
    executable.parent.mkdir(parents=True)
    executable.touch()
    calls = []
    monkeypatch.setattr(server.os, "execv", lambda *args: calls.append(args))

    result = runner.invoke(app, ["serve", "--engine", "higgs-tts"])

    assert result.exit_code == 0, result.output
    program, args = calls[0]
    assert program == str(executable)
    assert args[:2] == [program, "serve"]
    assert args[args.index("--model-name") + 1] == "bosonai/higgs-tts-3-4b"
    host = args[args.index("--host") + 1]
    port = args[args.index("--port") + 1]
    assert HiggsTTSEngine().base_url == f"http://{host}:{port}"
    assert args[args.index("--mem-fraction-static") + 1] == "0.55"


def test_higgs_launcher_supports_explicit_environment_with_spaces(monkeypatch, tmp_path):
    from unitts import server

    server_python = tmp_path / "server environment" / "bin" / "python"
    server_python.parent.mkdir(parents=True)
    server_python.with_name("sgl-omni").touch()
    calls = []
    monkeypatch.setattr(server.os, "execv", lambda *args: calls.append(args))

    result = runner.invoke(
        app,
        ["serve", "-e", "higgs-tts", "--server-python", str(server_python), "--port", "8001"],
    )

    assert result.exit_code == 0, result.output
    assert Path(calls[0][0]).parent == server_python.parent
    args = calls[0][1]
    assert args[args.index("--port") + 1] == "8001"


def test_missing_higgs_server_reports_setup_without_traceback(monkeypatch, tmp_path):
    from unitts import server

    monkeypatch.setattr(server.sys, "prefix", str(tmp_path / ".venv-higgs-tts"))
    result = runner.invoke(app, ["serve", "--engine", "higgs-tts"])

    assert result.exit_code == 1
    assert "Higgs server is not installed" in result.output
    assert "scripts/setup.py" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    "args",
    [["--engine", "moss-tts"], ["--engine", "higgs-tts", "--port", "0"]],
)
def test_invalid_server_options_do_not_launch(monkeypatch, args):
    from unitts import server

    def unexpected_launch(**kwargs):
        pytest.fail("Invalid server options reached the launcher")

    monkeypatch.setattr(server, "serve_higgs", unexpected_launch)
    result = runner.invoke(app, ["serve", *args])
    assert result.exit_code == 2


def test_synthesis_setup_failure_is_readable_and_cleanup_cannot_mask_it(monkeypatch, tmp_path):
    import unitts.engines

    unloaded = []

    class BrokenEngine:
        def ensure_loaded(self):
            raise EngineSetupError(
                "Missing dependency; run python3 scripts/setup.py --engine kokoro"
            )

        def unload_model(self):
            unloaded.append(True)
            raise RuntimeError("cleanup failed")

    monkeypatch.setattr(unitts.engines, "get_engine", lambda *a, **k: BrokenEngine())
    output = tmp_path / "out.wav"
    result = runner.invoke(app, ["synthesize", "Hello", "-e", "kokoro", "-o", str(output)])

    assert result.exit_code == 1
    assert "Missing dependency" in result.output
    assert "scripts/setup.py" in result.output
    assert "cleanup failed" in result.output
    assert "Traceback" not in result.output
    assert unloaded == [True]
    assert not output.exists()
