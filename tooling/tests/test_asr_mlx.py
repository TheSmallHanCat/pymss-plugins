from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from plugins.asr.backend import asr_mlx


class MlxModelLocationTests(unittest.TestCase):
    def test_selected_model_download_stays_in_the_studio_model_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / 'models/selected'
            download = Mock(return_value=str(checkpoint))
            transcribe = Mock(return_value={'text': 'transcript'})
            api = types.SimpleNamespace(download_model=download, transcribe=transcribe)
            with patch.dict(sys.modules, {'funasr_mlx': types.ModuleType('funasr_mlx'), 'funasr_mlx.api': api}):
                result = asr_mlx.transcribe(root / 'audio.wav', 'paraformer-zh', None, str(root / 'models'))
            self.assertEqual(result['text'], 'transcript')
            download.assert_called_once()
            self.assertEqual(download.call_args.args, ('paraformer-mlx-fp16',))
            self.assertTrue(Path(download.call_args.kwargs['dest']).is_relative_to(root / 'models'))
            transcribe.assert_called_once_with(str(root / 'audio.wav'), model=str(checkpoint))

    def test_explicit_local_model_never_downloads_or_uses_the_cache_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / 'local'
            checkpoint.mkdir()
            download = Mock()
            transcribe = Mock(return_value={'text': 'local transcript'})
            api = types.SimpleNamespace(download_model=download, transcribe=transcribe)
            with patch.dict(sys.modules, {'funasr_mlx': types.ModuleType('funasr_mlx'), 'funasr_mlx.api': api}):
                asr_mlx.transcribe(root / 'audio.wav', 'local', str(checkpoint), str(root / 'unused-cache'))
            download.assert_not_called()
            self.assertFalse((root / 'unused-cache').exists())
            transcribe.assert_called_once_with(str(root / 'audio.wav'), model=str(checkpoint))
