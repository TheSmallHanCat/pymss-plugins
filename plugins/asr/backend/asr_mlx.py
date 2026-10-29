"""Apple Silicon FunASR MLX adapter kept separate from the Torch backend."""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any


MLX_MODELS = {
    "sensevoice-small": "sensevoice-mlx-fp16",
    "paraformer-zh": "paraformer-mlx-fp16",
    "paraformer-en": "paraformer-en-mlx-fp16",
}


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def transcribe(audio_path: Path, model: str, local_model_dir: str | None, model_root: str | None = None) -> dict[str, Any]:
    from funasr_mlx.api import download_model, transcribe as mlx_transcribe

    if local_model_dir:
        checkpoint = Path(local_model_dir).expanduser()
        if not checkpoint.is_dir():
            raise ValueError("Local MLX ASR model directory does not exist")
    else:
        artifact = MLX_MODELS.get(model)
        if not artifact:
            raise ValueError(f"ASR preset {model!r} is unavailable on Apple Silicon")
        cache = Path(model_root).expanduser() / "_tool_models" / "asr" / "funasr-mlx" if model_root else Path.home() / ".pymss" / "models" / "funasr-mlx"
        cache.mkdir(parents=True, exist_ok=True)
        checkpoint = Path(download_model(artifact, dest=str(cache)))
    result = mlx_transcribe(str(audio_path), model=str(checkpoint))
    return result if isinstance(result, dict) else {"text": str(result or "")}
