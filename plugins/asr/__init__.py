"""pymss plugin: speech-to-text transcription.

Protocol v1. Two backends behind one Operation:

* Apple Silicon (macOS arm64) — ``funasr-mlx`` native MLX ports
  (sensevoice / paraformer offline, numerically verified, no PyTorch).
* Everything else (CUDA or CPU) — upstream ``funasr`` AutoModel with the same
  model zoo ids (Paraformer / SenseVoice).

The backend is chosen at runtime by availability, not by platform sniffing
alone: funasr-mlx wins if importable, else funasr. Both return the same
result shape (``text`` plus optional timestamps), so hosts and workflow nodes
see one contract regardless of machine.

Legacy capability registration (``asr_transcribe``) is kept for the DAG /
capability pool; the v1 Operation adds Studio form-driven usage.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from pymss.plugins import register_capability


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and sys.platform == "darwin" and _arm64()


def _arm64() -> bool:
    import platform

    return platform.machine() == "arm64"


def _have_funasr_mlx() -> bool:
    try:
        import funasr_mlx  # noqa: F401

        return True
    except ImportError:
        return False


def _have_funasr() -> bool:
    try:
        import funasr  # noqa: F401

        return True
    except ImportError:
        return False


def _ensure_audio_on_disk(audio_in: Any) -> Path:
    """Accept a path string or an audio artifact with source_path; else fail clearly."""
    source = getattr(audio_in, "source_path", None)
    if source and Path(str(source)).is_file():
        return Path(str(source))
    if isinstance(audio_in, (str, Path)) and Path(audio_in).is_file():
        return Path(audio_in)
    raise ValueError(
        "asr requires an audio file input; pass a file-backed audio input "
        "(e.g. a load node or file picker), not an in-memory-only buffer"
    )


# ---------------------------------------------------------------------------
# MLX backend
# ---------------------------------------------------------------------------


def _transcribe_mlx(audio_path: Path, model: str, language: str | None, model_dir: str | None) -> dict[str, Any]:
    from funasr_mlx.api import download_model, transcribe

    families = {
        "sensevoice-small": "sensevoice-mlx-fp16",
        "paraformer-zh": "paraformer-mlx-fp16",
        "paraformer-en": "paraformer-en-mlx-fp16",
    }
    artifact = families.get(model)
    if artifact is None:
        raise ValueError(
            f"model {model!r} has no MLX port; available: {sorted(families)}"
        )
    if model_dir:
        checkpoint = Path(model_dir)
    else:
        cache = Path.home() / ".pymss" / "models" / "funasr-mlx"
        cache.mkdir(parents=True, exist_ok=True)
        checkpoint = download_model(artifact, dest=str(cache))
    result = transcribe(str(audio_path), model=str(checkpoint))
    text = str(result.get("text", "")).strip()
    return {"text": text, "backend": "funasr-mlx", "model": model}


# ---------------------------------------------------------------------------
# funasr (CUDA/CPU) backend
# ---------------------------------------------------------------------------

_MLX_TO_FUNASR = {
    "sensevoice-small": "iic/SenseVoiceSmall",
    "paraformer-zh": "paraformer-zh",
    "paraformer-en": "paraformer-en",
}

_model_cache: dict[str, Any] = {}


def _transcribe_funasr(audio_path: Path, model: str, language: str | None) -> dict[str, Any]:
    from funasr import AutoModel

    model_id = _MLX_TO_FUNASR.get(model, model)
    key = f"{model_id}:{language or 'auto'}"
    asr = _model_cache.get(key)
    if asr is None:
        asr = AutoModel(model=model_id, disable_update=True)
        _model_cache[key] = asr
    result = asr.generate(input=str(audio_path))
    first = result[0] if result else {}
    text = str(first.get("text", "")).strip()
    timestamps: list[dict[str, float]] = []
    sentence_info = first.get("sentence_info") or []
    for item in sentence_info:
        start, end = item.get("start"), item.get("end")
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            timestamps.append({"start": float(start) / 1000.0, "end": float(end) / 1000.0, "text": str(item.get("text", ""))})
    out: dict[str, Any] = {"text": text, "backend": "funasr", "model": model}
    if timestamps:
        out["timestamps"] = timestamps
    return out


def asr_transcribe(audio_path, sample_rate=None, model: str = "sensevoice-small",
                   language: str | None = None, model_dir: str | None = None):
    """Transcribe an audio file to text. Returns {'text', 'backend', 'model', ...}."""
    path = _ensure_audio_on_disk(audio_path)
    if _is_apple_silicon() and _have_funasr_mlx():
        return _transcribe_mlx(path, model, language, model_dir)
    if _have_funasr():
        return _transcribe_funasr(path, model, language)
    raise ImportError(
        "no ASR backend available: install funasr-mlx (Apple Silicon) or funasr (CUDA/CPU)"
    )


register_capability(
    "asr_transcribe",
    asr_transcribe,
    source="asr",
    description="Speech-to-text via funasr-mlx (Apple Silicon) or funasr (CUDA/CPU)",
)


def setup(plugin):
    """Protocol v1: expose transcription as an Operation for Studio forms."""
    plugin.register_operation(
        id="pymss.asr.transcribe",
        title="Speech to Text",
        description="Transcribe audio to text (funasr-mlx on Apple Silicon, funasr elsewhere)",
        inputs={"audio": {"type": "audio"}},
        params={"type": "object", "properties": {
            "model": {"type": "string", "enum": ["sensevoice-small", "paraformer-zh", "paraformer-en"],
                       "default": "sensevoice-small"},
            "language": {"type": "string", "enum": ["auto", "zh", "en", "yue", "ja", "ko"],
                          "default": "auto"},
            "model_dir": {"type": "string"},
        }, "additionalProperties": False},
        outputs={"report": {"type": "object"}},
        run=lambda ctx, inputs, params: {"report": asr_transcribe(
            inputs["audio"],
            model=str(params.get("model", "sensevoice-small")),
            language=params.get("language") or None,
            model_dir=params.get("model_dir") or None,
        )},
    )
