from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import threading
import time
import unittest

from plugins.asr.backend.asr import _ModelLoadOutput
from pymss.operations import OperationContext, OperationError
from pymss.plugins.kit import current_operation_context, isolate_stdout, operation_context


class AsrDownloadProgressTests(unittest.TestCase):
    def test_download_thread_keeps_the_operation_and_protocol_stream(self):
        protocol = io.StringIO()
        logs = io.StringIO()
        observed = []

        def progress(fraction, message, event):
            observed.append((current_operation_context(), fraction, message))
            print(json.dumps(event))

        context = OperationContext(progress=progress)
        with redirect_stdout(protocol), operation_context(context):
            output = _ModelLoadOutput(logs)
            with redirect_stderr(output), isolate_stdout(), ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(output.write, "model.pt: 20%|##        | 200M/1000M").result(timeout=5)
                self.assertIsNone(pool.submit(current_operation_context).result(timeout=5))

        self.assertEqual(observed, [(context, 0.2, "model.pt")])
        event = json.loads(protocol.getvalue())
        self.assertEqual(event["type"], "audio_tool_progress")
        self.assertEqual(event["payload"]["completed"], 20)
        self.assertEqual(logs.getvalue(), "")
        self.assertIsNone(current_operation_context())

    def test_parallel_downloads_serialize_reporting_without_losing_events(self):
        observed = []
        barrier = threading.Barrier(4)

        def progress(fraction, message, event):
            # Keep the callback active while the other download threads write.
            time.sleep(0.005)
            observed.append((current_operation_context(), message, fraction))

        context = OperationContext(progress=progress)
        with operation_context(context):
            output = _ModelLoadOutput(io.StringIO())

            def write(index):
                barrier.wait(timeout=5)
                output.write(f"file-{index}.pt: 50%|#####     | 1M/2M")

            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(write, index) for index in range(4)]
                for future in futures:
                    future.result(timeout=5)

        self.assertEqual(len(observed), 4)
        self.assertEqual({message for _, message, _ in observed}, {f"file-{index}.pt" for index in range(4)})
        self.assertTrue(all(owner is context and fraction == 0.5 for owner, _, fraction in observed))

    def test_download_thread_preserves_cancellation_and_can_report_after_an_error(self):
        cancelled = threading.Event()
        observed = []
        context = OperationContext(cancelled=cancelled.is_set, progress=lambda *args: observed.append(args))
        with operation_context(context):
            output = _ModelLoadOutput(io.StringIO())
            with ThreadPoolExecutor(max_workers=1) as pool:
                cancelled.set()
                with self.assertRaisesRegex(OperationError, "operation cancelled"):
                    pool.submit(output.write, "model.pt: 20%|##        | 200M/1000M").result(timeout=5)
                self.assertIsNone(pool.submit(current_operation_context).result(timeout=5))
                cancelled.clear()
                pool.submit(output.write, "model.pt: 20%|##        | 200M/1000M").result(timeout=5)
                self.assertIsNone(pool.submit(current_operation_context).result(timeout=5))

        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0][:2], (0.2, "model.pt"))
