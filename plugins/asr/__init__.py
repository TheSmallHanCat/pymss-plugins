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


# CTC families run through funasr_mlx.pipeline (load_model + greedy_decode);
# LLM families (fun-asr-nano etc.) run through funasr_mlx.api.transcribe.
_MLX_CTC_ARTIFACTS = {
    "sensevoice-small": ("sensevoice-mlx-fp16", "sensevoice"),
    "paraformer-zh": ("paraformer-mlx-fp16", "paraformer"),
    "paraformer-en": ("paraformer-en-mlx-fp16", "paraformer"),
}

# Detokenizer files for the CTC families (SentencePiece BPE, id space = CTC
# vocab). Sourced from the upstream FunASR model repos; cached beside the
# checkpoint on first use. funasr-mlx ships weights only.
_BPE_SOURCES = {
    "sensevoice": "https://www.modelscope.cn/models/iic/SenseVoiceSmall/resolve/master/chn_jpn_yue_eng_ko_spectok.bpe.model",
    "paraformer": "https://www.modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/resolve/master/tokens.txt",
}

_LID_FOR_FAMILY = {"sensevoice": {"auto": "auto", "zh": "zh", "en": "en", "yue": "yue", "ja": "ja", "ko": "ko"}}


def _download_to(url: str, destination: Path) -> Path:
    import urllib.request

    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file():
        tmp = destination.with_suffix(destination.suffix + ".part")
        with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as out:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        tmp.rename(destination)
    return destination


def _detok_pieces(family: str, checkpoint: Path, token_ids: list[int]) -> str:
    """Decode CTC token ids to text with the family's vocabulary file."""
    cache = Path.home() / ".pymss" / "models" / "funasr-mlx"
    ids = [int(i) for i in token_ids if int(i) >= 0]
    if family == "sensevoice":
        import sentencepiece as spm

        local = checkpoint / "chn_jpn_yue_eng_ko_spectok.bpe.model"
        if not local.is_file():
            local = _download_to(_BPE_SOURCES[family], cache / "chn_jpn_yue_eng_ko_spectok.bpe.model")
        sp = spm.SentencePieceProcessor(model_file=str(local))
        import re

        decoded = str(sp.decode(ids))
        # SenseVoice emits metadata tags (<|zh|><|NEUTRAL|><|Speech|><|woitn|> ...)
        # before the transcript; strip them from the text output.
        decoded = re.sub(r"<\|[^|<>]*\|>", "", decoded)
        return decoded.strip()
    # paraformer: plain token-per-line vocabulary
    local = checkpoint / "tokens.txt"
    if not local.is_file():
        local = _download_to(_BPE_SOURCES[family], cache / "paraformer_tokens.txt")
    pieces = [line.rstrip("\n") for line in local.read_text(encoding="utf-8").splitlines()]
    return "".join(
        pieces[i] if 0 <= i < len(pieces) and pieces[i] not in {"<unk>", "<sos>", "<eos>"} else ""
        for i in ids
    ).strip()


def _transcribe_mlx(audio_path: Path, model: str, language: str | None, model_dir: str | None) -> dict[str, Any]:
    from funasr_mlx.api import download_model

    ctc = _MLX_CTC_ARTIFACTS.get(model)
    if ctc is None:
        raise ValueError(
            f"model {model!r} has no MLX port; available: {sorted(_MLX_CTC_ARTIFACTS)}"
        )
    artifact, family = ctc
    if model_dir:
        checkpoint = Path(model_dir)
    else:
        cache = Path.home() / ".pymss" / "models" / "funasr-mlx"
        cache.mkdir(parents=True, exist_ok=True)
        checkpoint = download_model(artifact, dest=str(cache))
    text = _transcribe_mlx_ctc(audio_path, checkpoint, family, language)
    return {"text": text, "backend": "funasr-mlx", "model": model}


def _transcribe_mlx_ctc(audio_path: Path, checkpoint: Path, family: str, language: str | None) -> str:
    """sensevoice: load_model(family='sensevoice') + greedy_decode + sp detok.

    funasr_mlx.api.transcribe only covers LLM-ASR families (qwen3/glm/moss/
    fun-asr-nano); passing a sensevoice checkpoint makes it fail with
    'Cannot infer family from checkpoint name; pass --family'.
    """
    from funasr_mlx.frontend import FbankConfig, WavFrontend, read_audio
    from funasr_mlx.pipeline import load_model

    model, config = load_model(checkpoint, family=family)
    options = dict(config.get("frontend_conf") or {})
    cmvn = options.pop("cmvn_file", None)
    options.setdefault("lfr_m", 7)
    options.setdefault("lfr_n", 6)
    if cmvn and not Path(cmvn).is_absolute():
        base = checkpoint if checkpoint.is_dir() else checkpoint.parent
        cmvn = str(base / cmvn)
    samples = read_audio(str(audio_path))
    frontend = WavFrontend(FbankConfig.from_dict(options), cmvn=cmvn)
    features = frontend(samples)
    if features.shape[0] == 0:
        return ""
    import mlx.core as mx

    lengths = mx.array([features.shape[0]], dtype=mx.int32)
    token_ids = model.greedy_decode(features[None], lengths)
    ids = token_ids[0] if token_ids and isinstance(token_ids[0], (list, tuple)) else token_ids
    return _detok_pieces(family, checkpoint, list(ids))


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


# ---------------------------------------------------------------------------
# Workflow graph node: speech audio -> transcript text (STRING output)
# ---------------------------------------------------------------------------

def _register_graph_node():
    from pymss.graph.core import AUDIO, STRING, NodeSignature, PortSpec
    from pymss.plugins import register_node

    def signature(node):
        return NodeSignature(
            inputs=[PortSpec(name="audio", type=AUDIO)],
            output_names=["text", "backend"],
            output_types=[STRING, STRING],
        )

    def execute(ctx, inputs):
        from pymss.graph.core import DAGError, NodeResult, StringArtifact

        audio_in = inputs.get("audio")
        source = getattr(audio_in, "source_path", None)
        if not source:
            raise DAGError("AsrTranscribe requires a file-backed AUDIO input")
        node = ctx.nodes_by_id[ctx.current_node_id]
        widgets = node.data.get("widgets_values", [])
        # widgets: [model, language, model_dir]
        model = str(widgets[0]) if len(widgets) > 0 and widgets[0] else "sensevoice-small"
        language = str(widgets[1]) if len(widgets) > 1 and widgets[1] else None
        model_dir = str(widgets[2]) if len(widgets) > 2 and widgets[2] else None
        result = asr_transcribe(
            str(source),
            model=model,
            language=language or None,
            model_dir=model_dir or None,
        )
        return NodeResult(outputs={
            0: StringArtifact(str(result.get("text", ""))),
            1: StringArtifact(str(result.get("backend", ""))),
        })

    try:
        register_node("AsrTranscribe", execute, signature=signature, source="asr")
    except Exception:
        pass


_register_graph_node()
