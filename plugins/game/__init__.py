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


def setup(plugin):
    """Protocol v1: expose vocal_to_midi as an Operation (audio file in, MIDI out)."""

    def run(ctx, inputs, params):
        audio_in = inputs["audio"]
        source = getattr(audio_in, "source_path", None) or str(inputs.get("audio_path", ""))
        if not source:
            raise ValueError("vocal_to_midi requires an audio file input")
        output_dir = str(params.get("output_dir") or Path(source).parent)
        result = vocal_to_midi(
            audio_path=source,
            model_path=str(params["model_path"]),
            output_dir=output_dir,
            tempo=float(params.get("tempo", 120.0)),
            language=params.get("language") or None,
        )
        return {"report": {"result": str(result) if result is not None else "done", "output_dir": output_dir}}

    plugin.register_operation(
        id="pymss.game.vocal_to_midi",
        title="Vocal to MIDI (GAME)",
        description="OpenVPI GAME vocal to MIDI transcription",
        inputs={"audio": {"type": "audio"}},
        params={"type": "object", "properties": {
            "model_path": {"type": "string"},
            "tempo": {"type": "number", "default": 120.0, "minimum": 30.0, "maximum": 300.0},
            "output_dir": {"type": "string"},
            "language": {"type": "string"},
        }, "required": ["model_path"], "additionalProperties": False},
        outputs={"report": {"type": "object"}},
        run=run,
    )
