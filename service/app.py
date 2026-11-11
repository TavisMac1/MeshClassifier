"""HTTP API for local mesh-image classification."""

from io import BytesIO
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from .classifier import MeshClassifier
from .jobs import JobManager


MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_COUNT = 6
classifier = MeshClassifier()
jobs = JobManager(classifier)
app = FastAPI(title="Mesh Classifier", version="0.1.0")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": classifier.model_id,
        "loaded": classifier.loaded,
        "progress_api": 3,
    }


def read_image(file):
    contents = file.file.read(MAX_IMAGE_BYTES + 1)
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB limit.")
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")

    try:
        with Image.open(BytesIO(contents)) as source:
            source.verify()
        with Image.open(BytesIO(contents)) as source:
            image = source.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail="Upload a valid image file.") from exc

    return image


def read_images(files: List[UploadFile]):
    if not files:
        raise HTTPException(status_code=400, detail="At least one image is required.")
    if len(files) > MAX_IMAGE_COUNT:
        raise HTTPException(status_code=400, detail="No more than %s images are supported." % MAX_IMAGE_COUNT)
    return [read_image(file) for file in files]


@app.post("/classify")
def classify(files: List[UploadFile] = File(...)):
    images = read_images(files)
    try:
        return classifier.classify(images)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Model classification failed: %s" % exc) from exc


@app.post("/jobs", status_code=202)
def create_job(files: List[UploadFile] = File(...)):
    images = read_images(files)
    try:
        job = jobs.submit(images)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"id": job.id}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Classification job not found.")
    return job.snapshot()
