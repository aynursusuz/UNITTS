"""Make sure the eSpeak NG phonemizer used by Kokoro and NeuTTS can actually load.

On macOS the eSpeak NG builds bundled with ``espeakng-loader`` (used by Kokoro) and
with ``neutts`` look for their data at the path they were built in, print
"Error processing file ... phontab" and end the whole Python process. On macOS we
therefore switch to Homebrew's eSpeak NG when it is installed, and otherwise test the
bundled build in a child process first, so a broken build becomes a clear
:class:`EngineSetupError` instead of a silent exit. Other platforms are left alone.

``UNITTS_ESPEAK_LIB`` and ``UNITTS_ESPEAK_DATA`` select a specific eSpeak NG library
and data directory on any platform.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

from unitts.setup_errors import EngineSetupError

HOMEBREW_PREFIXES = ("/opt/homebrew", "/usr/local")

# Code that makes each engine's own package configure its bundled eSpeak NG.
_BUNDLED_SETUP = {
    "kokoro": (
        "import espeakng_loader\n"
        "from phonemizer.backend.espeak.wrapper import EspeakWrapper\n"
        "EspeakWrapper.set_library(espeakng_loader.get_library_path())\n"
        "EspeakWrapper.set_data_path(espeakng_loader.get_data_path())\n"
    ),
    "neutts": "import neutts.phonemizers\n",
}
_PROBE = "from phonemizer.backend import EspeakBackend\nEspeakBackend('en-us').phonemize(['a'])\n"


def homebrew_espeak() -> tuple[str, str] | None:
    """Homebrew's eSpeak NG library and data directory, if installed."""
    for prefix in HOMEBREW_PREFIXES:
        library = Path(prefix, "lib", "libespeak-ng.dylib")
        data = Path(prefix, "share", "espeak-ng-data")
        if library.is_file() and data.is_dir():
            return str(library), str(data)
    return None


def bundled_espeak_works(engine: str) -> bool:
    """Load the engine's bundled eSpeak NG in a child process, which it may end."""
    result = subprocess.run(
        [sys.executable, "-c", _BUNDLED_SETUP[engine] + _PROBE],
        capture_output=True,
        timeout=300,
        check=False,
    )
    return result.returncode == 0


def _use(engine: str, library: str, data: str) -> None:
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    if engine == "kokoro":
        # misaki configures eSpeak from espeakng_loader when it is first imported.
        import espeakng_loader

        espeakng_loader.get_library_path = lambda: library
        espeakng_loader.get_data_path = lambda: data
    else:
        import neutts.phonemizers  # noqa: F401 - configures its bundled build on import

        os.environ["ESPEAK_DATA_PATH"] = data
    EspeakWrapper.set_library(library)
    if hasattr(EspeakWrapper, "set_data_path"):
        EspeakWrapper.set_data_path(data)


def ensure_working_espeak(engine: str) -> None:
    """Point ``engine`` at an eSpeak NG that loads; call before loading the model."""
    library, data = os.environ.get("UNITTS_ESPEAK_LIB"), os.environ.get("UNITTS_ESPEAK_DATA")
    if library and data:
        _use(engine, library, data)
        return
    if platform.system() != "Darwin":
        return
    found = homebrew_espeak()
    if found:
        _use(engine, *found)
        return
    if not bundled_espeak_works(engine):
        raise EngineSetupError(
            f"Cannot load {engine!r}: on macOS the eSpeak NG bundled with its Python "
            "package cannot find its data and stops the process. Install eSpeak NG with "
            "Homebrew (brew install espeak-ng) and retry, or set UNITTS_ESPEAK_LIB and "
            "UNITTS_ESPEAK_DATA to a working eSpeak NG library and data directory."
        )
