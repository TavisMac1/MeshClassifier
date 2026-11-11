import io
import json
import socket
import threading
import time
import unittest
import urllib.error
import urllib.request

import uvicorn
from PIL import Image

from service import app as api
from service.jobs import JobManager


class APIClassifier:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.last_image_count = None

    def classify(self, images, progress):
        self.last_image_count = len(images)
        progress("download", "Downloading model", files_done=1, files_total=2)
        self.started.set()
        if not self.release.wait(10):
            raise RuntimeError("Test classifier timed out")
        return {"name": "Sphere", "category": "Shape", "description": "A sphere."}


class ProgressAPITests(unittest.TestCase):
    def test_submit_poll_and_complete(self):
        classifier = APIClassifier()
        previous_jobs = api.jobs
        api.jobs = JobManager(classifier)
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(16)
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(api.app, log_level="error"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        try:
            for _ in range(100):
                if server.started:
                    break
                time.sleep(0.02)
            self.assertTrue(server.started)
            url = "http://127.0.0.1:%s" % port
            image = io.BytesIO()
            Image.new("RGB", (4, 4), "red").save(image, format="PNG")
            boundary = "test-boundary"
            body = (
                b"--test-boundary\r\n"
                b'Content-Disposition: form-data; name="files"; filename="mesh.png"\r\n'
                b"Content-Type: image/png\r\n\r\n"
                + image.getvalue()
                + b"\r\n--test-boundary\r\n"
                + b'Content-Disposition: form-data; name="files"; filename="mesh_side.png"\r\n'
                + b"Content-Type: image/png\r\n\r\n"
                + image.getvalue()
                + b"\r\n--test-boundary--\r\n"
            )

            def submit():
                request = urllib.request.Request(
                    url + "/jobs",
                    data=body,
                    headers={"Content-Type": "multipart/form-data; boundary=" + boundary},
                )
                with urllib.request.urlopen(request, timeout=5) as response:
                    self.assertEqual(response.status, 202)
                    return json.loads(response.read())

            job_id = submit()["id"]
            self.assertTrue(classifier.started.wait(5))
            self.assertEqual(classifier.last_image_count, 2)
            with urllib.request.urlopen(url + "/jobs/" + job_id, timeout=5) as response:
                status = json.loads(response.read())
            self.assertEqual(status["state"], "download")
            self.assertEqual((status["files_done"], status["files_total"]), (1, 2))
            with self.assertRaises(urllib.error.HTTPError) as error:
                submit()
            self.assertEqual(error.exception.code, 409)

            classifier.release.set()
            for _ in range(100):
                with urllib.request.urlopen(url + "/jobs/" + job_id, timeout=5) as response:
                    status = json.loads(response.read())
                if status["state"] == "done":
                    break
                time.sleep(0.02)
            self.assertEqual(status["state"], "done")
            self.assertEqual(status["result"]["name"], "Sphere")
        finally:
            classifier.release.set()
            server.should_exit = True
            thread.join(10)
            listener.close()
            api.jobs = previous_jobs
        self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
