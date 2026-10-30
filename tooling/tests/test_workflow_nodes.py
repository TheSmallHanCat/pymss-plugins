from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import tomllib
import unittest
from unittest.mock import Mock, patch

import numpy as np

from pymss.graph.core import AudioArtifact, DAGError, DAGNode, NodeContext, SeparatorCache
from pymss.operations import OperationError, create_runtime
from pymss.plugins import get_registry, require_capability


ROOT = Path(__file__).resolve().parents[2]
NODE_TYPES = {
    "SpectralHighpass", "SpectralLowpass", "SpectralPeakEQ",
    "DynamicsCompressor", "DynamicsLimiter", "DynamicsDeEsser",
    "TimingPitchShift", "TimingTimeStretch", "NormalizeLUFS",
    "AsrTranscribe", "GameVocalToMidi", "AnalysisMeasureLUFS", "AnalysisTruePeak", "AnalysisRMS",
}


class WorkflowNodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = create_runtime()
        cls.reports = cls.runtime.load_plugins([ROOT / "plugins"])

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output_dir = Path(directory.name)

    def context(self, widgets=(), require=require_capability, node_type="AsrTranscribe"):
        node = DAGNode("node", node_type, data={"widgets_values": list(widgets)})
        context = NodeContext(
            output_dir=self.output_dir, logger=None, debug=False,
            progress_callback=None, separator_cache=SeparatorCache(),
            current_node_id=node.id, nodes_by_id={node.id: node},
        )
        context.require = require
        return context

    def audio(self, source=""):
        return AudioArtifact(np.full((2, 44100), 0.1, dtype=np.float32), 44100, source_path=source)

    def test_all_upstream_nodes_have_matching_runtime_signatures(self):
        self.assertTrue(all(row["loaded"] for row in self.reports), self.reports)
        declarations = {}
        for path in (ROOT / "plugins").glob("*/pyproject.toml"):
            plugin = tomllib.loads(path.read_text(encoding="utf-8"))["tool"]["pymss"]["plugin"]
            for node in plugin.get("contributions", {}).get("nodes", []):
                declarations[node["type"]] = node
        self.assertEqual(set(declarations), NODE_TYPES)
        for name, declaration in declarations.items():
            with self.subTest(node=name):
                registration = get_registry().nodes[name]
                self.assertTrue(registration.source.startswith("plugin:pymss."))
                signature = registration.signature(SimpleNamespace(data={}))
                self.assertEqual([port.name for port in signature.inputs], [port["name"] for port in declaration["inputs"]])
                self.assertEqual(signature.output_names, [port["name"] for port in declaration["outputs"]])
                self.assertEqual(signature.output_types, [port["type"] for port in declaration["outputs"]])

    def test_processor_nodes_forward_widget_values_and_defaults(self):
        cases = [
            ("SpectralHighpass", "highpass", [200.0], {"cutoff_hz": 200.0, "Q": 0.707}),
            ("SpectralLowpass", "lowpass", [], {"cutoff_hz": 8000.0, "Q": 0.707}),
            ("SpectralPeakEQ", "peak_eq", [500.0, 0.0], {"center_hz": 500.0, "gain_db": 0.0, "Q": 1.0}),
            ("DynamicsCompressor", "compressor", [], {"threshold_db": 0.0, "ratio": 4.0, "attack_ms": 1.0, "release_ms": 100.0}),
            ("DynamicsLimiter", "limiter", [-6.0, 50.0], {"threshold_db": -6.0, "release_ms": 50.0}),
            ("DynamicsDeEsser", "deesser", [], {"threshold_db": 0.0, "ratio": 3.0, "crossover_hz": 6000.0}),
            ("TimingPitchShift", "pitch_shift", [], {"n_steps": 0.0, "bins_per_octave": 12}),
            ("TimingTimeStretch", "time_stretch", [0.5], {"rate": 0.5}),
        ]
        for name, capability, widgets, expected in cases:
            with self.subTest(node=name):
                processor = Mock(side_effect=lambda audio, sample_rate, **kwargs: audio)
                resolve = Mock(return_value=processor)
                execute = get_registry().nodes[name].func
                result = execute(self.context(widgets, resolve, name), {"audio": self.audio()})
                resolve.assert_called_once_with(capability)
                self.assertEqual(processor.call_args.kwargs, expected)
                self.assertEqual(result.outputs[0].audio.shape, (2, 44100))
                with self.assertRaises(DAGError):
                    execute(self.context(node_type=name), {})

    def test_analysis_nodes_capture_each_measurement_key(self):
        report = {"lufs": -14.0, "true_peak_dbtp": -1.0, "rms_dbfs": -20.0}
        for node, capability, expected in [
            ("AnalysisMeasureLUFS", "measure_lufs", "-14.0"),
            ("AnalysisTruePeak", "measure_true_peak", "-1.0"),
            ("AnalysisRMS", "measure_rms", "-20.0"),
        ]:
            with self.subTest(node=node):
                resolve = Mock(return_value=Mock(return_value=report))
                result = get_registry().nodes[node].func(self.context(require=resolve, node_type=node), {"audio": self.audio()})
                resolve.assert_called_once_with(capability)
                self.assertEqual(result.outputs[0].value, expected)

    def test_asr_node_preserves_backend_and_widget_contract(self):
        execute = get_registry().nodes["AsrTranscribe"].func
        transcribe = Mock(return_value={"text": "recognized", "backend": "funasr-mlx"})
        handler = self.runtime._operations["pymss.asr.transcribe"].run
        audio = self.audio("/audio/voice.wav")
        with patch.dict(handler.__globals__, {"asr_transcribe": transcribe}):
            result = execute(self.context(["paraformer-en", "en", "/models/asr"]), {"audio": audio})
        transcribe.assert_called_once_with(audio, model="paraformer-en", language="en", model_dir="/models/asr")
        self.assertEqual(result.outputs[0].value, "recognized")
        self.assertEqual(result.outputs[1].value, "funasr-mlx")

    def test_game_node_preserves_model_alias_and_midi_output(self):
        execute = get_registry().nodes["GameVocalToMidi"].func
        output = self.output_dir / "voice.mid"
        transcribe = Mock(return_value=SimpleNamespace(output_path=output))
        handler = self.runtime._operations["pymss.game.vocal_to_midi"].run
        with patch.dict(handler.__globals__, {"vocal_to_midi": transcribe}):
            result = execute(self.context([128, "zh", "small"], node_type="GameVocalToMidi"),
                             {"audio": self.audio("/audio/voice.wav")})
        self.assertEqual(transcribe.call_args.kwargs["model_path"], "small")
        self.assertEqual(transcribe.call_args.kwargs["tempo"], 128)
        self.assertEqual(transcribe.call_args.kwargs["language"], "zh")
        self.assertEqual(transcribe.call_args.kwargs["output_dir"], str(self.output_dir))
        self.assertEqual(result.outputs[0].value, str(output))
        self.assertEqual(result.saved_paths, [str(output)])
        self.assertEqual(result.saved_records[0].node_type, "GameVocalToMidi")

    def test_game_node_and_operation_reject_zero_tempo_before_inference(self):
        execute = get_registry().nodes["GameVocalToMidi"].func
        handler = self.runtime._operations["pymss.game.vocal_to_midi"].run
        transcribe = Mock()
        audio = self.audio()
        with patch.dict(handler.__globals__, {"vocal_to_midi": transcribe}):
            with self.assertRaisesRegex(DAGError, "tempo"):
                execute(self.context([0, "en", "small"], node_type="GameVocalToMidi"), {"audio": audio})
            with self.assertRaisesRegex(OperationError, "tempo.*minimum"):
                self.runtime.run("pymss.game.vocal_to_midi", inputs={"audio": audio},
                                 params={"model_path": "small", "tempo": 0})
        transcribe.assert_not_called()
