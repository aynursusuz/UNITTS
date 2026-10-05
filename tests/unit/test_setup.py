"""Installer routing and failure safety, without downloads or subprocess execution."""

import importlib.metadata
import importlib.util
import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from unitts.engines.registry import ENGINE_REGISTRY

_SPEC = importlib.util.spec_from_file_location(
    "unitts_setup", Path(__file__).resolve().parents[2] / "scripts" / "setup.py"
)
setup = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(setup)


def test_installer_covers_every_registered_engine():
    assert set(setup.IMPORTS) == set(ENGINE_REGISTRY)


@pytest.fixture
def commands(monkeypatch):
    recorded = []

    def completed(command, **kwargs):
        recorded.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(setup, "run", recorded.append)
    monkeypatch.setattr(setup.subprocess, "run", completed)
    monkeypatch.setattr(setup.shutil, "which", lambda name: f"/tools/{name}")
    monkeypatch.setattr(setup.platform, "system", lambda: "Linux")
    monkeypatch.setattr(setup.platform, "libc_ver", lambda: ("glibc", "2.35"))
    monkeypatch.setattr(setup, "cuda_version", lambda: (13, 2))
    return recorded


def test_moss_uses_isolated_pinned_cuda_stack_and_preserves_spaces(tmp_path, commands):
    root = tmp_path / "checkout with spaces"
    root.mkdir()

    setup.install("moss-tts", "auto", root)

    python = str(root / ".venv-moss-tts" / "bin" / "python")
    install = next(command for command in commands if command[1:3] == ["pip", "install"])
    assert install[install.index("--python") + 1] == python
    assert install[install.index("-e") + 1] == f"{root}[moss-tts]"
    assert install[install.index("--torch-backend") + 1] == "cu128"
    assert {"torch==2.9.1", "torchaudio==2.9.1", "torchcodec==0.8.1"} <= set(install)
    assert commands[-2] == ["/tools/uv", "pip", "check", "--python", python]
    assert commands[-1][:2] == [python, "-c"]
    assert "torch.cuda.is_available()" in commands[-1][2]
    assert "from_pretrained" not in commands[-1][2]
    marker = json.loads((root / ".venv-moss-tts" / setup.MARKER).read_text())
    assert marker["root"] == str(root)
    assert marker["engine"] == "moss-tts"


def test_higgs_installs_separate_client_and_pinned_server(tmp_path, commands, capsys):
    setup.install("higgs-tts", "cuda", tmp_path)

    installs = [command for command in commands if command[1:3] == ["pip", "install"]]
    assert len(installs) == 2
    client, server = installs
    assert str(tmp_path / ".venv-higgs-tts" / "bin" / "python") in client
    assert str(tmp_path / ".venv-higgs-server" / "bin" / "python") in server
    assert client[client.index("--torch-backend") + 1] == "cpu"
    assert client[client.index("-e") + 1] == str(tmp_path)
    assert server[server.index("--torch-backend") + 1] == "cu130"
    assert "--prerelease=allow" in server
    assert server[-1].endswith("@3d4eb6e49096aa42e7f8f4e72f028823b2d85b73")
    assert "-e" not in server
    assert "higgs_audio_v2_tokenizer.json" in commands[-1][2]
    assert "unitts serve --engine higgs-tts" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("engine", "version", "backend"),
    [("chatterbox", "2.6.0", "cu124"), ("echo-tts", "2.9.1", "cu128")],
)
def test_cuda_install_keeps_torch_and_torchaudio_coherent(
    tmp_path, commands, engine, version, backend
):
    setup.install(engine, "cuda", tmp_path)
    install = commands[1]
    assert f"torch=={version}" in install
    assert f"torchaudio=={version}" in install
    assert install[install.index("--torch-backend") + 1] == backend


@pytest.mark.parametrize("system", ["Linux", "Darwin"])
def test_cpu_backend_does_not_install_cuda_wheels(tmp_path, commands, monkeypatch, system):
    monkeypatch.setattr(setup.platform, "system", lambda: system)
    probe = Mock(side_effect=AssertionError("Explicit CPU must not probe CUDA"))
    monkeypatch.setattr(setup, "cuda_version", probe)

    setup.install("moss-tts", "cpu", tmp_path)

    install = commands[1]
    if system == "Linux":
        assert install[install.index("--torch-backend") + 1] == "cpu"
    else:
        assert "--torch-backend" not in install
    assert "cuda.is_available" not in commands[-1][2]
    probe.assert_not_called()


@pytest.mark.parametrize("engine", ["echo-tts", "higgs-tts"])
@pytest.mark.parametrize(("system", "device"), [("Darwin", "cuda"), ("Linux", "cpu")])
def test_gpu_only_engines_reject_unsupported_platforms(
    commands, monkeypatch, engine, system, device
):
    monkeypatch.setattr(setup.platform, "system", lambda: system)
    with pytest.raises(RuntimeError, match="Linux and an NVIDIA CUDA GPU"):
        setup.preflight(engine, device)
    assert commands == []


@pytest.mark.parametrize(
    ("engine", "version", "minimum"),
    [("moss-tts", (12, 7), "12.8"), ("higgs-tts", (12, 8), "13.0")],
)
def test_old_driver_fails_before_creating_environment(
    tmp_path, commands, monkeypatch, engine, version, minimum
):
    monkeypatch.setattr(setup, "cuda_version", lambda: version)
    with pytest.raises(RuntimeError, match=minimum):
        setup.install(engine, "cuda", tmp_path)
    assert list(tmp_path.iterdir()) == []
    assert commands == []


def test_missing_ffmpeg_has_actionable_error_without_mutation(tmp_path, commands, monkeypatch):
    monkeypatch.setattr(setup.shutil, "which", lambda name: None if name == "ffmpeg" else "/uv")
    with pytest.raises(RuntimeError, match="apt-get install ffmpeg"):
        setup.install("moss-tts", "cpu", tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("libc", [("glibc", "2.31"), ("", ""), ("musl", "1.2")])
def test_higgs_rejects_incompatible_libc_before_install(tmp_path, commands, monkeypatch, libc):
    monkeypatch.setattr(setup.platform, "libc_ver", lambda: libc)
    with pytest.raises(RuntimeError, match="glibc>=2.34"):
        setup.install("higgs-tts", "cuda", tmp_path)
    assert commands == []


def test_supertonic_auto_uses_cpu_on_gpu_hosts(tmp_path, commands):
    setup.install("supertonic", "auto", tmp_path)
    install = commands[1]
    assert install[install.index("--torch-backend") + 1] == "cpu"
    assert "cuda.is_available" not in commands[-1][2]


def test_missing_uv_has_actionable_error(monkeypatch, capsys):
    monkeypatch.setattr(setup.shutil, "which", lambda name: None)
    assert setup.main(["--engine", "moss-tts"]) == 1
    output = capsys.readouterr()
    assert "python3 -m pip install uv" in output.err
    assert "Installed and checked" not in output.out


def test_unowned_existing_environment_is_untouched(tmp_path, commands):
    environment = tmp_path / ".venv-moss-tts"
    environment.mkdir()
    sentinel = environment / "user-data.txt"
    sentinel.write_text("preserve")

    with pytest.raises(RuntimeError, match="Refusing to modify"):
        setup.install("moss-tts", "cpu", tmp_path)

    assert sentinel.read_text() == "preserve"
    assert not (environment / setup.MARKER).exists()
    assert commands == []


@pytest.mark.parametrize(
    ("field", "value"),
    [("root", "/other/checkout"), ("engine", "echo-tts"), ("device", "cuda"), ("version", 2)],
)
def test_marker_must_match_checkout_engine_device_and_version(tmp_path, commands, field, value):
    environment = tmp_path / ".venv-moss-tts"
    environment.mkdir()
    marker = setup.identity(tmp_path, "moss-tts", "cpu", "engine")
    marker[field] = value
    marker_path = environment / setup.MARKER
    original = json.dumps(marker)
    marker_path.write_text(original)

    with pytest.raises(RuntimeError, match="Refusing to modify"):
        setup.install("moss-tts", "cpu", tmp_path)

    assert marker_path.read_text() == original
    assert commands == []


def test_symlink_environment_is_never_followed(tmp_path, commands):
    target = tmp_path / "other"
    target.mkdir()
    (tmp_path / ".venv-moss-tts").symlink_to(target, target_is_directory=True)
    with pytest.raises(RuntimeError, match="Refusing to modify"):
        setup.install("moss-tts", "cpu", tmp_path)
    assert list(target.iterdir()) == []


def test_partial_environment_can_be_retried(tmp_path, commands, monkeypatch):
    failing = Mock(side_effect=subprocess.CalledProcessError(1, ["uv", "venv"]))
    monkeypatch.setattr(setup, "run", failing)
    with pytest.raises(subprocess.CalledProcessError):
        setup.install("moss-tts", "cpu", tmp_path)
    assert (tmp_path / ".venv-moss-tts" / setup.MARKER).is_file()

    monkeypatch.setattr(setup, "run", commands.append)
    setup.install("moss-tts", "cpu", tmp_path)
    assert commands[0][1] == "venv"
    assert commands[-1][1] == "-c"


def test_higgs_checks_both_ownership_records_before_mutation(tmp_path, commands):
    (tmp_path / ".venv-higgs-server").mkdir()
    with pytest.raises(RuntimeError, match="Refusing to modify"):
        setup.install("higgs-tts", "cuda", tmp_path)
    assert not (tmp_path / ".venv-higgs-tts").exists()
    assert commands == []


def test_failed_check_never_reports_success(tmp_path, commands, monkeypatch, capsys):
    def fail_check(command):
        if command[1:3] == ["pip", "check"]:
            raise subprocess.CalledProcessError(1, command)
        commands.append(command)

    monkeypatch.setattr(setup, "run", fail_check)
    with pytest.raises(subprocess.CalledProcessError):
        setup.install("moss-tts", "cpu", tmp_path)
    assert "Installed and checked" not in capsys.readouterr().out
    assert not any(command[1] == "-c" for command in commands)


def test_main_preserves_process_error_and_returns_failure(monkeypatch, capsys):
    failure = Mock(side_effect=subprocess.CalledProcessError(7, ["uv", "pip", "install"]))
    monkeypatch.setattr(setup, "install", failure)
    assert setup.main(["--engine", "echo-tts"]) == 1
    output = capsys.readouterr()
    assert "exit status 7" in output.err
    assert "FFmpeg libraries" in output.err
    assert "Installed and checked" not in output.out


@pytest.mark.parametrize(
    ("output", "code", "expected"),
    [
        ("NVIDIA-SMI 595.71.05  CUDA Version: 13.2", 0, (13, 2)),
        ("NVIDIA-SMI failed", 1, None),
        ("CUDA Version: N/A", 0, None),
    ],
)
def test_driver_probe_handles_unavailable_gpu(monkeypatch, output, code, expected):
    monkeypatch.setattr(setup.shutil, "which", lambda name: "/tools/nvidia-smi")
    process = Mock(return_value=subprocess.CompletedProcess([], code, stdout=output))
    monkeypatch.setattr(setup.subprocess, "run", process)
    assert setup.cuda_version() == expected
    assert process.call_args.args[0] == ["/tools/nvidia-smi"]


_PROTOBUF_MISMATCH = (
    "The package `descript-audiotools` requires `protobuf>=3.9.2,<3.20`, "
    "but `6.33.0` is installed\n"
)


def test_only_upstream_protobuf_metadata_exception_is_allowed(tmp_path, commands, monkeypatch):
    monkeypatch.setattr(
        setup.subprocess,
        "run",
        Mock(
            return_value=subprocess.CompletedProcess(
                [],
                1,
                stdout="",
                stderr="Found 1 incompatibility\n" + _PROTOBUF_MISMATCH,
            )
        ),
    )
    setup.check_packages("uv", tmp_path / "python", "higgs-tts")
    assert "SpecifierSet('>=6.31.1,<7.0.0')" in commands[-1][2]


@pytest.mark.parametrize(
    "diagnostics",
    [
        "Found 2 incompatibilities\n" + _PROTOBUF_MISMATCH + "Unexpected package mismatch\n",
        "Found 1 incompatibility\nThe package `torch` requires a different dependency\n",
    ],
)
def test_other_package_check_errors_remain_fatal(tmp_path, commands, monkeypatch, diagnostics):
    monkeypatch.setattr(
        setup.subprocess,
        "run",
        Mock(
            return_value=subprocess.CompletedProcess(
                [],
                1,
                stdout="",
                stderr=diagnostics,
            )
        ),
    )
    with pytest.raises(subprocess.CalledProcessError):
        setup.check_packages("uv", tmp_path / "python", "higgs-tts")
    assert commands == []


@pytest.mark.parametrize("version", ["6.31.0", "7.0.0"])
def test_override_version_is_checked_even_after_clean_package_check(
    tmp_path, commands, monkeypatch, version
):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: version)
    monkeypatch.setattr(setup, "run", lambda command: exec(command[2], {}))
    with pytest.raises(AssertionError, match="Installed Protobuf"):
        setup.check_packages("uv", tmp_path / "python", "higgs-tts")


def test_fish_installer_applies_its_own_upstream_override(tmp_path, commands):
    setup.install("fish-audio", "cuda", tmp_path)
    install = next(command for command in commands if command[1:3] == ["pip", "install"])
    path = Path(install[install.index("--override") + 1])
    assert path.name == "fish-audio.txt"
    assert "protobuf>=3.20.0,<6.0.0" in path.read_text()
    assert any("SpecifierSet('>=3.20.0,<6.0.0')" in command[-1] for command in commands)
