"""Foreground launcher for the separately installed Higgs server."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from unitts.setup_errors import EngineSetupError


def serve_higgs(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    server_python: Path | None = None,
) -> None:
    """Replace this process with SGLang-Omni so Ctrl+C reaches the server."""
    if server_python is None:
        server_python = Path(sys.prefix).parent / ".venv-higgs-server" / "bin" / "python"
    executable = server_python.parent / "sgl-omni"
    if not executable.is_file():
        raise EngineSetupError(
            "Higgs server is not installed. From the UNITTS checkout run "
            "`python3 scripts/setup.py --engine higgs-tts`, then activate "
            "`.venv-higgs-tts` and retry `unitts serve --engine higgs-tts`."
        )
    args = [
        str(executable),
        "serve",
        "--model-path",
        "bosonai/higgs-tts-3-4b",
        "--model-name",
        "bosonai/higgs-tts-3-4b",
        "--host",
        host,
        "--port",
        str(port),
        "--mem-fraction-static",
        "0.55",
        "--max-running-requests",
        "8",
        "--cuda-graph-max-bs",
        "8",
    ]
    os.execv(str(executable), args)
