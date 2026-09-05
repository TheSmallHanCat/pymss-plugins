"""pymss plugin: OpenVPI GAME vocal to MIDI transcription.

Core model sources are adapted from https://github.com/openvpi/GAME at
tag v1.0.0 (commit e66c31251605e334b1bf0f565252d4987a9065c0), under the MIT license.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from pymss.plugins import register_capability


def vocal_to_midi(
    audio_path: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    tempo: float = 120.0,
    language: str | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> Any:
    try:
        from .infer import infer
    except (ImportError, ValueError):
        from game.infer import infer  # type: ignore

    return infer(
        model_path=Path(model_path),
        audio_path=Path(audio_path),
        output_dir=Path(output_dir),
        tempo=tempo,
        language=language,
        progress_callback=progress_callback,
    )


register_capability(
    "vocal_to_midi",
    vocal_to_midi,
    source="game",
    description="OpenVPI GAME vocal to MIDI transcription model",
)
