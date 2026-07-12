"""FastAPI service exposing the docint pipeline.

Run locally (after ``pip install -r requirements.txt``)::

    uvicorn api.main:app --reload

Endpoints:
    GET  /health   — liveness probe.
    POST /extract  — multipart image upload -> ExtractResponse JSON.

The :class:`~docint.pipeline.Pipeline` is built once at startup and shared
across requests; the OCR models inside it load lazily on the first request
and stay loaded.
"""

from __future__ import annotations

import base64
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import numpy as np
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse

from docint.pipeline import Pipeline
from docint.preprocess import Image
from docint.config import load_config
from docint.schemas import ExtractError, ExtractResponse, Language


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the singleton Pipeline once at startup (models load lazily)."""
    app.state.config = load_config()
    app.state.pipeline = Pipeline(config=app.state.config)
    yield


app = FastAPI(
    title="docint — Document Intelligence Pipeline",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


def _decode_image(data: bytes) -> Image | None:
    """Decode upload bytes to a BGR ndarray; None when it isn't an image."""
    import cv2

    if not data:
        return None
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def _png_b64(image: Image) -> str:
    """Encode an image (BGR or grayscale) as a base64 PNG string."""
    import cv2

    ok, encoded = cv2.imencode(".png", image)
    return base64.b64encode(encoded.tobytes()).decode("ascii") if ok else ""


def _error(status_code: int, error: str, **extra: Any) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": error, **extra})


@app.post(
    "/extract",
    response_model=ExtractResponse,
    responses={
        400: {"model": ExtractError, "description": "Upload could not be decoded as an image"},
        413: {"model": ExtractError, "description": "Upload exceeds api.max_upload_mb"},
        422: {"model": ExtractError, "description": "Photo too blurry to OCR — retake it"},
    },
)
def extract(
    file: UploadFile = File(..., description="Document photo (JPEG/PNG/WebP)."),
    lang: Language = Query("en", description="Recognition language."),
    return_debug_images: bool = Query(
        False, description="Include the deskewed page + per-stage debug images as base64 PNGs."
    ),
) -> Any:
    """Extract structured text (regions, questions, full_text) from a photo.

    Whether an upload is an image is decided by decoding it, not by its
    content-type header. Rejections: undecodable uploads -> 400, oversize
    uploads -> 413, and photos whose blur score falls below
    ``preprocess.blur.laplacian_threshold`` -> 422 with the measured score.

    Sync endpoint on purpose: the pipeline is CPU-bound, so FastAPI runs it
    on the threadpool instead of blocking the event loop.
    """
    api_cfg = app.state.config["api"]

    data = file.file.read()
    max_bytes = int(float(api_cfg["max_upload_mb"]) * 1024 * 1024)
    if len(data) > max_bytes:
        return _error(413, "upload_too_large", detail=f"limit is {api_cfg['max_upload_mb']} MB")

    image = _decode_image(data)
    if image is None:
        return _error(400, "invalid_image", detail="upload could not be decoded as an image")

    result = app.state.pipeline.run(image, lang=lang)
    document = result.document
    if document.retake_photo:
        return _error(422, "blurry_image", blur_score=document.blur_score)

    response = ExtractResponse(result=document)
    if return_debug_images:
        if result.deskewed_image is not None:
            response.deskewed_image_png_b64 = _png_b64(result.deskewed_image)
        response.debug_images_png_b64 = {
            name: _png_b64(stage_image) for name, stage_image in result.intermediates.items()
        }
    return response
