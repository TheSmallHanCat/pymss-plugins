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
    precision: str = "fp32",
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> Any:
    # Apple Silicon: prefer the pure-MLX backend (gamemlx) — no torch needed.
    from .mlx_backend import have_gamemlx, infer_mlx, is_apple_silicon

    if is_apple_silicon() and have_gamemlx():
        return infer_mlx(
            audio_path=audio_path,
            model_path=model_path,
            output_dir=output_dir,
            tempo=tempo,
            language=language,
            precision=precision,
            progress_callback=progress_callback,
        )
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
        # Apple Silicon: model_path is optional - the gamemlx alias (default
        # 'medium') auto-downloads on first use. Other platforms require an
        # explicit .pt checkpoint path.
        model_path = str(params.get("model_path") or "").strip()
        from .mlx_backend import have_gamemlx, is_apple_silicon

        if not model_path:
            if not (is_apple_silicon() and have_gamemlx()):
                raise ValueError("model_path is required (GAME .pt checkpoint)")
            model_path = str(params.get("model_alias") or "medium")
        result = vocal_to_midi(
            audio_path=source,
            model_path=model_path,
            output_dir=output_dir,
            tempo=float(params.get("tempo", 120.0)),
            language=params.get("language") or None,
            precision=str(params.get("precision") or "fp32"),
        )
        # Both backends expose the same result shape; normalize to the report dict.
        return {"report": {
            "result": str(getattr(result, "output_path", result)) if result is not None else "done",
            "output_midi": str(getattr(result, "output_path", "") or ""),
            "output_dir": output_dir,
            "note_count": int(getattr(result, "note_count", 0) or 0),
            "input_duration": float(getattr(result, "input_duration", 0.0) or 0.0),
            "first_note_at": getattr(result, "first_note_at", None),
            "last_note_at": getattr(result, "last_note_at", None),
            "warnings": list(getattr(result, "warnings", ()) or ()),
            "language": getattr(result, "language", None),
        }}

    plugin.register_operation(
        id="pymss.game.vocal_to_midi",
        title="Vocal to MIDI (GAME)",
        description="OpenVPI GAME vocal to MIDI transcription",
        inputs={"audio": {"type": "audio"}},
        params={"type": "object", "properties": {
            "model_path": {"type": "string", "description": "optional .pt checkpoint or local gamemlx dir; empty on Apple Silicon = auto-download"},
            "model_alias": {"type": "string", "enum": ["small", "medium", "large"], "default": "medium"},
            "tempo": {"type": "number", "default": 120.0, "minimum": 30.0, "maximum": 300.0},
            "output_dir": {"type": "string"},
            "language": {"type": "string"},
            "precision": {"type": "string", "enum": ["fp32", "fp16"], "default": "fp32", "description": "gamemlx precision (Apple Silicon only)"},
        }, "additionalProperties": False},
        outputs={"report": {"type": "object"}},
        run=run,
    )


# ---------------------------------------------------------------------------
# Workflow graph node: vocal audio -> MIDI file path (STRING output)
# ---------------------------------------------------------------------------

def _register_graph_node():
    from pymss.graph.core import AUDIO, STRING, NodeSignature, PortSpec
    from pymss.plugins import register_node

    def signature(node):
        return NodeSignature(
            inputs=[PortSpec(name="audio", type=AUDIO)],
            output_names=["midi_path"],
            output_types=[STRING],
        )

    def execute(ctx, inputs):
        from pathlib import Path as _P

        from pymss.graph.core import DAGError, NodeResult, StringArtifact, audio_to_numpy

        audio_in = inputs.get("audio")
        source = getattr(audio_in, "source_path", None)
        if not source:
            raise DAGError("GameVocalToMidi requires a file-backed AUDIO input")
        node = ctx.nodes_by_id[ctx.current_node_id]
        widgets = node.data.get("widgets_values", [])
        # widgets: [tempo, language, model_path_or_alias]
        tempo = float(widgets[0]) if len(widgets) > 0 and widgets[0] else 120.0
        language = str(widgets[1]) if len(widgets) > 1 and widgets[1] else None
        model_path = str(widgets[2]) if len(widgets) > 2 and widgets[2] else "medium"
        result = vocal_to_midi(
            audio_path=str(source),
            model_path=model_path,
            output_dir=str(_P(source).parent),
            tempo=tempo,
            language=language or None,
        )
        return NodeResult(outputs={0: StringArtifact(str(getattr(result, "output_path", "")))})

    try:
        register_node("GameVocalToMidi", execute, signature=signature, source="game")
    except Exception:
        pass


_register_graph_node()
