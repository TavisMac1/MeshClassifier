import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from service.classifier import _download_assets, _extract_runtime


class ModelProgressTests(unittest.TestCase):
    @staticmethod
    def response(contents):
        response = io.BytesIO(contents)
        response.status = 206
        response.headers = {"Content-Range": "bytes 0-%d/%d" % (len(contents) - 1, len(contents))}
        return response

    def test_download_progress_and_cache(self):
        contents = b"a" * 2048
        assets = (("model.gguf", "https://example.test/model", len(contents), hashlib.sha256(contents).hexdigest()),)
        events = []
        with tempfile.TemporaryDirectory() as directory:
            with patch("service.classifier.FILES", assets), patch("urllib.request.urlopen", return_value=self.response(contents)) as download:
                _download_assets(Path(directory), lambda state, message, **details: events.append(details))
                self.assertEqual(download.call_count, 1)
                self.assertEqual(events[-1]["downloaded_bytes"], len(contents))
                self.assertEqual(events[-1]["total_bytes"], len(contents))
                _download_assets(Path(directory), lambda *args, **kwargs: None)
                self.assertEqual(download.call_count, 1)

    def test_rejects_bad_hash(self):
        assets = (("model.gguf", "https://example.test/model", 3, "bad-hash"),)
        with tempfile.TemporaryDirectory() as directory:
            with patch("service.classifier.FILES", assets), patch("urllib.request.urlopen", return_value=self.response(b"abc")):
                with self.assertRaisesRegex(ValueError, "integrity"):
                    _download_assets(Path(directory), lambda *args, **kwargs: None)
                self.assertFalse((Path(directory) / "model.gguf").exists())

    def test_resumes_partial_download(self):
        contents = b"abcdef"
        assets = (("model.gguf", "https://example.test/model", len(contents), hashlib.sha256(contents).hexdigest()),)
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "model.gguf.part").write_bytes(b"abc")
            response = io.BytesIO(b"def")
            response.status = 206
            response.headers = {"Content-Range": "bytes 3-5/6"}
            with patch("service.classifier.FILES", assets), patch("urllib.request.urlopen", return_value=response) as download:
                _download_assets(Path(directory), lambda *args, **kwargs: None)
                self.assertEqual(download.call_args.args[0].get_header("Range"), "bytes=3-5")
                self.assertEqual((Path(directory) / "model.gguf").read_bytes(), contents)

    def test_rejects_wrong_range(self):
        assets = (("model.gguf", "https://example.test/model", 3, "unused"),)
        response = self.response(b"abc")
        response.headers = {"Content-Range": "bytes 1-3/3"}
        with tempfile.TemporaryDirectory() as directory:
            with patch("service.classifier.FILES", assets), patch("urllib.request.urlopen", return_value=response):
                with self.assertRaisesRegex(RuntimeError, "did not honor"):
                    _download_assets(Path(directory), lambda *args, **kwargs: None)

    def test_rejects_unsafe_archive_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "runtime.zip"
            with zipfile.ZipFile(archive, "w") as file:
                file.writestr("../outside.dll", b"unsafe")
            with self.assertRaisesRegex(ValueError, "Unexpected path"):
                _extract_runtime(archive, Path(directory) / "runtime")


if __name__ == "__main__":
    unittest.main()
