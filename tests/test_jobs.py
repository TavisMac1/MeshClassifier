import tempfile
import threading
import time
import unittest
from pathlib import Path

from service.jobs import Job, JobManager


class BlockingClassifier:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def classify(self, image, progress):
        progress("classifying", "Generating a result.")
        self.started.set()
        if not self.release.wait(5):
            raise RuntimeError("Test classifier timed out")
        return {"name": "Coral", "category": "Nature", "description": "Coral mesh."}


class JobTests(unittest.TestCase):
    def test_running_job_reports_progress_rejects_overlap_and_finishes(self):
        classifier = BlockingClassifier()
        manager = JobManager(classifier)
        job = manager.submit(object())
        try:
            self.assertTrue(classifier.started.wait(5))
            self.assertEqual(job.snapshot()["state"], "classifying")
            self.assertIs(manager.get(job.id), job)
            self.assertIsNone(manager.get("unknown"))
            with self.assertRaisesRegex(RuntimeError, "already running"):
                manager.submit(object())
        finally:
            classifier.release.set()
        for _ in range(100):
            state = job.snapshot()
            if state["state"] == "done":
                break
            time.sleep(0.01)
        self.assertEqual(state["state"], "done")
        self.assertEqual(state["result"]["name"], "Coral")
        self.assertGreaterEqual(state["elapsed_seconds"], 0)

    def test_cache_size_is_sampled_during_download(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Job()
            job.update("download", "Downloading", files_done=1, files_total=3, cache_dir=directory)
            file = Path(directory) / "model.incomplete"
            file.write_bytes(b"1234")
            self.assertEqual(job.snapshot()["cache_bytes"], 4)
            file.write_bytes(b"1234567")
            snapshot = job.snapshot()
            self.assertEqual(snapshot["cache_bytes"], 7)
            self.assertEqual((snapshot["files_done"], snapshot["files_total"]), (1, 3))
            self.assertNotIn("cache_dir", snapshot)

    def test_classifier_error_is_reported(self):
        class FailingClassifier:
            def classify(self, image, progress):
                raise ValueError("model unavailable")

        manager = JobManager(FailingClassifier())
        job = manager.submit(object())
        for _ in range(100):
            state = job.snapshot()
            if state["state"] == "error":
                break
            time.sleep(0.01)
        self.assertEqual(state["state"], "error")
        self.assertIn("model unavailable", state["message"])


if __name__ == "__main__":
    unittest.main()
