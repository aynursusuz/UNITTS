"""eSpeak NG selection for Kokoro and NeuTTS, without loading eSpeak or the engines."""

import sys
import types

import pytest

from unitts.engines import _espeak
from unitts.setup_errors import EngineSetupError


@pytest.fixture
def wrapper(monkeypatch):
    """A fake phonemizer EspeakWrapper that records what it was pointed at."""
    calls = {}

    class EspeakWrapper:
        @staticmethod
        def set_library(path):
            calls["library"] = path

        @staticmethod
        def set_data_path(path):
            calls["data"] = path

    module = types.ModuleType("phonemizer.backend.espeak.wrapper")
    module.EspeakWrapper = EspeakWrapper
    for name in ("phonemizer", "phonemizer.backend", "phonemizer.backend.espeak"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "phonemizer.backend.espeak.wrapper", module)
    monkeypatch.delenv("UNITTS_ESPEAK_LIB", raising=False)
    monkeypatch.delenv("UNITTS_ESPEAK_DATA", raising=False)
    return calls


@pytest.fixture
def loader(monkeypatch):
    module = types.ModuleType("espeakng_loader")
    module.get_library_path = lambda: "/bundled/libespeak-ng.dylib"
    module.get_data_path = lambda: "/bundled/espeak-ng-data"
    monkeypatch.setitem(sys.modules, "espeakng_loader", module)
    return module


def _homebrew(tmp_path, monkeypatch):
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "libespeak-ng.dylib").write_bytes(b"")
    (tmp_path / "share" / "espeak-ng-data").mkdir(parents=True)
    monkeypatch.setattr(_espeak, "HOMEBREW_PREFIXES", (str(tmp_path),))


def test_other_platforms_are_left_alone(monkeypatch, wrapper):
    monkeypatch.setattr(_espeak.platform, "system", lambda: "Linux")
    monkeypatch.setattr(_espeak, "bundled_espeak_works", lambda engine: pytest.fail("no probe"))
    _espeak.ensure_working_espeak("kokoro")
    assert wrapper == {}


def test_macos_kokoro_uses_homebrew(tmp_path, monkeypatch, wrapper, loader):
    monkeypatch.setattr(_espeak.platform, "system", lambda: "Darwin")
    _homebrew(tmp_path, monkeypatch)
    _espeak.ensure_working_espeak("kokoro")
    assert wrapper["library"] == str(tmp_path / "lib" / "libespeak-ng.dylib")
    assert loader.get_library_path() == wrapper["library"]
    assert loader.get_data_path() == str(tmp_path / "share" / "espeak-ng-data")


def test_macos_neutts_uses_homebrew(tmp_path, monkeypatch, wrapper):
    monkeypatch.setattr(_espeak.platform, "system", lambda: "Darwin")
    monkeypatch.setenv("ESPEAK_DATA_PATH", "")  # restored after the test
    monkeypatch.setitem(sys.modules, "neutts", types.ModuleType("neutts"))
    monkeypatch.setitem(sys.modules, "neutts.phonemizers", types.ModuleType("neutts.phonemizers"))
    _homebrew(tmp_path, monkeypatch)
    _espeak.ensure_working_espeak("neutts")
    assert wrapper["library"] == str(tmp_path / "lib" / "libespeak-ng.dylib")
    assert _espeak.os.environ["ESPEAK_DATA_PATH"] == str(tmp_path / "share" / "espeak-ng-data")


def test_macos_without_homebrew_keeps_a_working_bundled_build(tmp_path, monkeypatch, wrapper):
    monkeypatch.setattr(_espeak.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(_espeak, "HOMEBREW_PREFIXES", (str(tmp_path),))
    monkeypatch.setattr(_espeak, "bundled_espeak_works", lambda engine: True)
    _espeak.ensure_working_espeak("kokoro")
    assert wrapper == {}


def test_macos_broken_bundled_build_is_a_setup_error(tmp_path, monkeypatch, wrapper):
    monkeypatch.setattr(_espeak.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(_espeak, "HOMEBREW_PREFIXES", (str(tmp_path),))
    monkeypatch.setattr(_espeak, "bundled_espeak_works", lambda engine: False)
    with pytest.raises(EngineSetupError, match="brew install espeak-ng"):
        _espeak.ensure_working_espeak("neutts")


def test_environment_override_applies_everywhere(monkeypatch, wrapper, loader):
    monkeypatch.setattr(_espeak.platform, "system", lambda: "Linux")
    monkeypatch.setenv("UNITTS_ESPEAK_LIB", "/custom/libespeak-ng.so")
    monkeypatch.setenv("UNITTS_ESPEAK_DATA", "/custom/espeak-ng-data")
    _espeak.ensure_working_espeak("kokoro")
    assert wrapper == {"library": "/custom/libespeak-ng.so", "data": "/custom/espeak-ng-data"}
