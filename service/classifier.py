"""Vulkan-backed local vision model inference."""

import atexit
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import re
import secrets
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
import zipfile
from typing import Iterable

from PIL import Image

from .windows_job import KillOnCloseJob


MODEL_ID = "ggml-org/SmolVLM-500M-Instruct-GGUF (Q8_0)"
REVISION = "72e986006ef53e37cdd3f6d4241c90b0f01df376"
MODEL_BASE = "https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/" + REVISION + "/"
FILES = (
    ("llama-b12274-bin-win-vulkan-x64.zip", "https://github.com/ggml-org/llama.cpp/releases/download/b12274/llama-b12274-bin-win-vulkan-x64.zip", 38020318, "3345e0ba454483e77dcd454c2da7517b1a29dbf3caa12c950ee764a7d416f47b"),
    ("SmolVLM-500M-Instruct-Q8_0.gguf", MODEL_BASE + "SmolVLM-500M-Instruct-Q8_0.gguf", 436806912, "9d4612de6a42214499e301494a3ecc2be0abdd9de44e663bda63f1152fad1bf4"),
    ("mmproj-SmolVLM-500M-Instruct-Q8_0.gguf", MODEL_BASE + "mmproj-SmolVLM-500M-Instruct-Q8_0.gguf", 108783360, "d1eb8b6b23979205fdf63703ed10f788131a3f812c7b1f72e0119d5d81295150"),
)
PROMPT = (
    "Name the object in these rendered views. Respond in JSON. The name is a few words; "
    "category is a broad group. Use all views when uncertain. The description is a short phrase describing "
    "the visible object's appearance, not an explanation or definition."
)


def parse_classification(response):
    candidate = response.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1)
    else:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
    try:
        result = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError("The model did not return valid JSON.") from exc
    if not isinstance(result, dict):
        raise ValueError("The model response must be a JSON object.")
    classification = {}
    for field in ("name", "category", "description"):
        value = result.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError("The model response is missing a non-empty %s field." % field)
        classification[field] = value.strip()[:500]
    return classification


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_assets(directory, progress):
    directory.mkdir(parents=True, exist_ok=True)
    total = sum(item[2] for item in FILES)
    completed = 0
    for name, url, size, expected_hash in FILES:
        target = directory / name
        progress("download", "Checking %s." % name, downloaded_bytes=completed, total_bytes=total)
        if target.is_file() and target.stat().st_size == size and _sha256(target) == expected_hash:
            completed += size
            continue
        partial = directory / (name + ".part")
        digest = hashlib.sha256()
        downloaded = 0
        if partial.is_file() and partial.stat().st_size <= size:
            with partial.open("rb") as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b""):
                    digest.update(chunk)
                    downloaded += len(chunk)
        else:
            partial.unlink(missing_ok=True)
        progress("download", "Downloading %s." % name, downloaded_bytes=completed + downloaded, total_bytes=total)
        with partial.open("ab") as file:
            while downloaded < size:
                end = min(downloaded + 8 * 1024 * 1024, size) - 1
                request = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (downloaded, end)})
                with urllib.request.urlopen(request, timeout=60) as response:
                    content_range = response.headers.get("Content-Range", "")
                    expected_range = "bytes %d-%d/%d" % (downloaded, end, size)
                    if response.status != 206 or content_range != expected_range:
                        raise RuntimeError("The download server did not honor the requested range for %s." % name)
                    received = 0
                    while received < end - downloaded + 1:
                        chunk = response.read(min(1024 * 1024, end - downloaded + 1 - received))
                        if not chunk:
                            raise OSError("Download interrupted while receiving %s." % name)
                        file.write(chunk)
                        digest.update(chunk)
                        received += len(chunk)
                        progress("download", "Downloading %s." % name, downloaded_bytes=completed + downloaded + received, total_bytes=total)
                downloaded += received
        if downloaded != size or digest.hexdigest() != expected_hash:
            partial.unlink(missing_ok=True)
            raise ValueError("Download failed integrity check for %s; retry setup." % name)
        partial.replace(target)
        completed += size
    progress("download", "Downloads verified.", downloaded_bytes=total, total_bytes=total)


def _extract_runtime(archive_path, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            name = member.filename
            if "/" in name or "\\" in name or name in (".", ".."):
                raise ValueError("Unexpected path in Vulkan runtime archive: %s" % name)
            if name.lower().endswith((".dll", ".exe")):
                with archive.open(member) as source, (destination / name).open("wb") as output:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(chunk)
    if not (destination / "llama-server.exe").is_file():
        raise ValueError("Vulkan runtime archive does not contain llama-server.exe.")


class MeshClassifier:
    def __init__(self):
        self.model_id = MODEL_ID
        self._directory = Path(os.environ.get("MESH_CLASSIFIER_DATA", str(Path.home() / ".mesh_classifier"))) / "vulkan-b12274"
        self._process = None
        self._port = None
        self._device = None
        self._job = None
        self._api_key = None
        self._lock = threading.Lock()
        atexit.register(self.close)

    @property
    def loaded(self):
        return self._process is not None and self._process.poll() is None

    def close(self):
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        self._process = None
        self._api_key = None
        if self._job is not None:
            self._job.close()
            self._job = None

    def _load(self, progress):
        if self.loaded:
            return
        progress("loading", "Preparing Vulkan runtime and model.")
        if os.name != "nt":
            raise RuntimeError("The bundled Vulkan runtime currently supports Windows x64 only.")
        _download_assets(self._directory, progress)
        runtime = self._directory / "runtime"
        executable = runtime / "llama-server.exe"
        marker = runtime / "extract-complete"
        if not executable.is_file() or not marker.is_file():
            progress("loading", "Installing Vulkan inference runtime.")
            _extract_runtime(self._directory / FILES[0][0], runtime)
            marker.write_text("b12274\n", encoding="utf-8")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        result = subprocess.run([str(executable), "--list-devices"], capture_output=True, text=True, timeout=30, creationflags=flags)
        if result.returncode:
            raise RuntimeError("Vulkan device detection failed: %s" % (result.stderr or result.stdout))
        devices = re.findall(r"^\s*(Vulkan\d+):\s*(.+)$", result.stdout + "\n" + result.stderr, re.MULTILINE)
        amd_devices = [item for item in devices if "amd" in item[1].lower() or "radeon" in item[1].lower()]
        if not amd_devices:
            raise RuntimeError("No AMD Vulkan GPU detected. Update your AMD graphics driver. Detected: %s" % (result.stdout + result.stderr))
        device, self._device = amd_devices[0]
        progress("loading", "Loading model on %s." % self._device)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self._port = sock.getsockname()[1]
        command = [
            str(executable), "--model", str(self._directory / FILES[1][0]),
            "--mmproj", str(self._directory / FILES[2][0]), "--device", device,
            "--gpu-layers", "all", "--mmproj-device", device, "--mmproj-offload",
            "--ctx-size", "4096", "--host", "127.0.0.1", "--port", str(self._port),
        ]
        self._api_key = secrets.token_urlsafe(32)
        command.extend(["--api-key", self._api_key, "--no-webui"])
        log_path = self._directory / "vulkan-service.log"
        self._job = KillOnCloseJob()
        with log_path.open("a", encoding="utf-8") as log:
            try:
                self._process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
                self._job.assign(self._process)
            except Exception:
                self.close()
                raise
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                self.close()
                raise RuntimeError("Vulkan model server exited. See %s." % log_path)
            try:
                with urllib.request.urlopen("http://127.0.0.1:%s/health" % self._port, timeout=2) as response:
                    if response.status == 200:
                        return
            except (urllib.error.URLError, TimeoutError):
                time.sleep(0.5)
        self.close()
        raise RuntimeError("Vulkan model server startup timed out. See %s." % log_path)

    def classify(self, images: Iterable[Image.Image], progress=None):
        if progress is None:
            progress = lambda state, message, **details: None
        image_list = list(images)
        if not image_list:
            raise ValueError("At least one rendered image is required.")
        with self._lock:
            self._load(progress)
            progress("preparing", "Preparing %s rendered view(s) for the Vulkan model." % len(image_list))
            content = []
            for image in image_list:
                buffer = io.BytesIO()
                image.convert("RGB").save(buffer, format="PNG")
                encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}})
            content.append({"type": "text", "text": PROMPT})
            schema = {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1, "maxLength": 50},
                    "category": {"type": "string", "minLength": 1, "maxLength": 40},
                    "description": {"type": "string", "minLength": 1, "maxLength": 80},
                },
                "required": ["name", "category", "description"],
            }
            payload = {
                "messages": [{"role": "user", "content": content}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "asset", "schema": schema}},
                "max_tokens": 192,
                "temperature": 0,
            }
            request = urllib.request.Request(
                "http://127.0.0.1:%s/v1/chat/completions" % self._port,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": "Bearer " + self._api_key},
            )
            progress("classifying", "Classifying with Vulkan on %s." % self._device)
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    result = json.loads(response.read())
            except urllib.error.HTTPError as exc:
                raise RuntimeError("Vulkan inference failed: %s" % exc.read().decode("utf-8", errors="replace")) from exc
            progress("parsing", "Processing the model response.")
            return parse_classification(result["choices"][0]["message"]["content"])
