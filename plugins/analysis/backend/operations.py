"""pymss plugin: audio analysis and measurement.

Registers read-only capabilities that return measurement dicts rather than
audio. Backed by pyloudnorm (LUFS/true-peak) and numpy (RMS/dynamic range/FFT).
"""

from __future__ import annotations

import numpy as np


def measure_lufs(audio, sample_rate):
    """Integrated loudness in LUFS (ITU-R BS.1770-4). Returns {'lufs': float}.

    Returns lufs=-inf for silent input. Audio is mono or channels-first.
    """
    import pyloudnorm as pyln

    audio = np.asarray(audio, dtype=np.float32)
    measure = audio if audio.ndim == 2 else audio[None, :]
    if measure.shape[0] > 5:
        measure = measure.mean(axis=0, keepdims=True)
    meter = pyln.Meter(int(sample_rate))
    loudness = meter.integrated_loudness(measure.T)  # pyloudnorm wants (samples, channels)
    return {"lufs": float(loudness) if np.isfinite(loudness) else float("-inf")}


def measure_true_peak(audio, sample_rate):
    """True peak in dBTP (oversampled). Returns {'true_peak_dbtp': float, 'sample_peak_dbfs': float}."""
    audio = np.asarray(audio, dtype=np.float32)
    measure = audio if audio.ndim == 2 else audio[None, :]
    # Peak measurement treats channels independently, including multichannel audio.
    from scipy.signal import resample_poly

    oversampled = resample_poly(measure, up=4, down=1, axis=-1)
    tp = float(np.max(np.abs(oversampled)))
    sp = float(np.max(np.abs(measure)))
    def to_db(x):
        return 20.0 * np.log10(max(x, 1e-12))
    return {"true_peak_dbtp": to_db(tp), "sample_peak_dbfs": to_db(sp)}


def measure_rms(audio, sample_rate=None):
    """RMS level in dBFS. Returns {'rms_dbfs': float, 'rms_linear': float}."""
    audio = np.asarray(audio, dtype=np.float32)
    rms = float(np.sqrt(np.mean(audio ** 2)))
    return {
        "rms_linear": rms,
        "rms_dbfs": 20.0 * np.log10(max(rms, 1e-12)),
    }


def measure_dynamic_range(audio, sample_rate, block_seconds=0.4):
    """Crude dynamic-range estimate: difference between loud and quiet blocks.

    Splits into blocks, computes per-block LUFS, returns DR = Lmax - Lmin (LU).
    Returns {'dr_lu': float, 'lufs_max': float, 'lufs_min': float}.
    """
    import pyloudnorm as pyln

    audio = np.asarray(audio, dtype=np.float32)
    measure = audio if audio.ndim == 2 else audio[None, :]
    if measure.shape[0] > 5:
        measure = measure.mean(axis=0, keepdims=True)

    sr = int(sample_rate)
    block = max(int(sr * block_seconds), 1)
    meter = pyln.Meter(sr)
    loud = []
    for i in range(0, measure.shape[1] - block, block):
        seg = measure[:, i:i + block]
        L = meter.integrated_loudness(seg.T)
        if np.isfinite(L):
            loud.append(L)
    if len(loud) < 2:
        return {"dr_lu": 0.0, "lufs_max": float("-inf"), "lufs_min": float("-inf")}
    return {
        "dr_lu": float(max(loud) - min(loud)),
        "lufs_max": float(max(loud)),
        "lufs_min": float(min(loud)),
    }


def spectrum_analyze(audio, sample_rate, n_fft=2048):
    """Compute the average power spectrum. Returns {'freqs': [...], 'magnitudes_db': [...]}.

    Uses a single FFT on the mean of the audio across channels. For a proper
    spectrogram you'd want STFT; this is a quick spectrum readout.
    """
    audio = np.asarray(audio, dtype=np.float32)
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    n = min(n_fft, len(mono))
    win = mono[:n] * np.hanning(n)
    spectrum = np.abs(np.fft.rfft(win))
    mag_db = 20.0 * np.log10(np.maximum(spectrum, 1e-12))
    freqs = np.fft.rfftfreq(n, d=1.0 / int(sample_rate))
    return {
        "freqs": freqs.tolist(),
        "magnitudes_db": mag_db.tolist(),
        "nyquist_hz": float(int(sample_rate) / 2),
    }


_CAPABILITIES = [
    ("measure_lufs", measure_lufs),
    ("measure_true_peak", measure_true_peak),
    ("measure_rms", measure_rms),
    ("measure_dynamic_range", measure_dynamic_range),
    ("spectrum_analyze", spectrum_analyze),
]


def setup(plugin):
    """Register the measurement capabilities and expose them as read-only Operations."""
    from pymss.graph.core import audio_to_numpy

    for name, fn in _CAPABILITIES:
        plugin.register_capability(name, fn)

    def _measure(fn, param_defs):
        def run(ctx, inputs, params):
            waveform, sr = audio_to_numpy(inputs["audio"])
            kwargs = {k: params[k] for k in param_defs if k in params}
            result = fn(waveform, sr, **kwargs)
            return {"report": {k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                               for k, v in result.items()}}
        return run

    plugin.register_operation(
        id="pymss.analysis.measure_lufs",
        title="Measure Loudness (LUFS)",
        description="Integrated loudness per ITU-R BS.1770-4",
        inputs={"audio": {"type": "audio"}},
        params={"type": "object", "properties": {}, "additionalProperties": False},
        outputs={"report": {"type": "object"}},
        run=_measure(measure_lufs, {}),
    )
    plugin.register_operation(
        id="pymss.analysis.measure_true_peak",
        title="Measure True Peak",
        description="True peak (dBTP) and sample peak (dBFS)",
        inputs={"audio": {"type": "audio"}},
        params={"type": "object", "properties": {}, "additionalProperties": False},
        outputs={"report": {"type": "object"}},
        run=_measure(measure_true_peak, {}),
    )
    plugin.register_operation(
        id="pymss.analysis.measure_rms",
        title="Measure RMS",
        description="RMS level in dBFS and linear",
        inputs={"audio": {"type": "audio"}},
        params={"type": "object", "properties": {}, "additionalProperties": False},
        outputs={"report": {"type": "object"}},
        run=_measure(measure_rms, {}),
    )
    plugin.register_operation(
        id="pymss.analysis.spectrum",
        title="Spectrum Analyze",
        description="Average power spectrum (dB magnitudes)",
        inputs={"audio": {"type": "audio"}},
        params={"type": "object", "properties": {
            "n_fft": {"type": "integer", "default": 2048, "minimum": 64, "maximum": 32768},
        }, "additionalProperties": False},
        outputs={"report": {"type": "object"}},
        run=_measure(spectrum_analyze, {"n_fft"}),
    )

    _register_graph_nodes(plugin)


_ANALYSIS_NODE_SPECS = [
    ("AnalysisMeasureLUFS", "Measure Loudness (LUFS)", "measure_lufs", "lufs"),
    ("AnalysisTruePeak", "Measure True Peak", "measure_true_peak", "true_peak_dbtp"),
    ("AnalysisRMS", "Measure RMS", "measure_rms", "rms_dbfs"),
]


def _register_graph_nodes(plugin):
    from pymss.graph.core import AUDIO, STRING, NodeSignature, PortSpec

    for node_type, _title, cap, key in _ANALYSIS_NODE_SPECS:
        def signature(node, _cap=cap):
            return NodeSignature(
                inputs=[PortSpec(name="audio", type=AUDIO)],
                output_names=["value"],
                output_types=[STRING],
            )

        def execute(ctx, inputs, _cap=cap, _key=key, _type=node_type):
            from pymss.graph.core import DAGError, NodeResult, StringArtifact, audio_to_numpy

            audio_in = inputs.get("audio")
            if audio_in is None:
                raise DAGError(f"{_type} requires an AUDIO input")
            waveform, sr = audio_to_numpy(audio_in)
            result = ctx.require(_cap)(waveform, sr)
            value = result.get(_key) if isinstance(result, dict) else result
            return NodeResult(outputs={0: StringArtifact(str(value))})

        plugin.register_node(node_type, execute, signature=signature)
