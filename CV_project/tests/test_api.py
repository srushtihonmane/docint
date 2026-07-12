"""API tests over FastAPI's TestClient with mocked OCR models (no weights).

The TestClient context runs the lifespan, so ``app.state.pipeline`` is the
real singleton; fakes are seeded into its lazy-load caches exactly like in
test_pipeline.py.
"""

from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app


class _FakeDetModel:
    def __init__(self, polys: list | None) -> None:
        self._polys = None if polys is None else np.asarray(polys, dtype=np.float32)

    def text_detector(self, img):
        return self._polys, 0.01


class _FakeRecModel:
    def __init__(self, results: list[tuple[str, float]]) -> None:
        self._results = list(results)

    def ocr(self, img, det=True, rec=True, cls=False):
        assert rec and not det
        return [[self._results.pop(0)]]


@pytest.fixture()
def client():
    with TestClient(app) as test_client:  # enters lifespan -> builds Pipeline
        yield test_client


def _mock_models(client: TestClient, polys, rec_results) -> None:
    pipeline = client.app.state.pipeline
    pipeline.detector._model = _FakeDetModel(polys)
    pipeline.recognizer._models["en"] = _FakeRecModel(rec_results)


def _png_bytes(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def _sharp_page() -> np.ndarray:
    """High-frequency checkerboard — safely above the blur threshold."""
    tile = (((np.indices((400, 300)).sum(axis=0) // 8) % 2) * 255).astype(np.uint8)
    return np.dstack([tile] * 3)


_RECT = [[10, 10], [200, 10], [200, 30], [10, 30]]


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_extract_returns_document_json(client: TestClient) -> None:
    _mock_models(client, [_RECT], [("hello api", 0.95)])

    response = client.post(
        "/extract?lang=en",
        files={"file": ("page.png", _png_bytes(_sharp_page()), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result"]["full_text"] == "hello api"
    assert body["result"]["language"] == "en"
    assert body["result"]["pages"][0]["regions"][0]["text"] == "hello api"
    assert body["result"]["timings_ms"]["detect"] >= 0
    # debug images are opt-in
    assert body["deskewed_image_png_b64"] is None
    assert body["debug_images_png_b64"] is None


def test_extract_debug_images_are_valid_base64_pngs(client: TestClient) -> None:
    _mock_models(client, [_RECT], [("x", 0.9)])

    response = client.post(
        "/extract?lang=en&return_debug_images=true",
        files={"file": ("page.png", _png_bytes(_sharp_page()), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    debug_images = body["debug_images_png_b64"]
    for key in ("boundary_overlay", "warped", "deskewed", "shadow_free", "enhanced", "detections_overlay"):
        assert key in debug_images
    decoded = base64.b64decode(debug_images["deskewed"])
    assert decoded.startswith(b"\x89PNG")
    assert body["deskewed_image_png_b64"]


def test_extract_rejects_non_image_with_400(client: TestClient) -> None:
    response = client.post(
        "/extract",
        files={"file": ("notes.txt", b"definitely not an image", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_image"


def test_extract_rejects_blurry_image_with_422(client: TestClient) -> None:
    _mock_models(client, None, [])
    flat = np.full((300, 400, 3), 128, dtype=np.uint8)  # laplacian variance ~0

    response = client.post(
        "/extract",
        files={"file": ("flat.png", _png_bytes(flat), "image/png")},
    )

    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "blurry_image"
    assert body["blur_score"] < 120.0


def test_extract_rejects_oversize_upload_with_413(client: TestClient) -> None:
    oversized = b"\x00" * (16 * 1024 * 1024)  # api.max_upload_mb is 15

    response = client.post("/extract", files={"file": ("big.bin", oversized, "image/png")})

    assert response.status_code == 413
    assert response.json()["error"] == "upload_too_large"
