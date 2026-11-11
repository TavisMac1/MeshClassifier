"""In-memory classification jobs and progress."""

import threading
import time
import uuid
import logging
from pathlib import Path


class Job:
    def __init__(self):
        self.id = uuid.uuid4().hex
        self._lock = threading.Lock()
        self._started = time.monotonic()
        self._data = {"state": "queued", "message": "Waiting for the classifier."}
        self._cache_dir = None

    def update(self, state, message, **details):
        with self._lock:
            self._cache_dir = details.pop("cache_dir", None)
            self._data = {"state": state, "message": message, **details}

    def snapshot(self):
        with self._lock:
            data = {**self._data, "elapsed_seconds": int(time.monotonic() - self._started)}
            cache_dir = self._cache_dir
        if cache_dir:
            total = 0
            directory = Path(cache_dir)
            if directory.is_dir():
                for entry in directory.iterdir():
                    try:
                        if entry.is_file():
                            total += entry.stat().st_size
                    except FileNotFoundError:
                        continue
            data["cache_bytes"] = total
        return data


class JobManager:
    def __init__(self, classifier):
        self.classifier = classifier
        self._lock = threading.Lock()
        self._jobs = {}
        self._active = None

    def submit(self, image):
        with self._lock:
            if self._active is not None:
                raise RuntimeError("A classification is already running. Wait for it to finish.")
            if len(self._jobs) >= 32:
                oldest = next(iter(self._jobs))
                del self._jobs[oldest]
            job = Job()
            self._jobs[job.id] = job
            self._active = job.id
        threading.Thread(target=self._run, args=(job, image), daemon=True).start()
        return job

    def _run(self, job, image):
        try:
            result = self.classifier.classify(image, progress=job.update)
        except Exception as exc:
            logging.exception("Classification job failed")
            with self._lock:
                job.update("error", "Classification failed: %s" % exc)
                self._active = None
        else:
            with self._lock:
                job.update("done", "Classification complete.", result=result)
                self._active = None

    def get(self, job_id):
        with self._lock:
            return self._jobs.get(job_id)
