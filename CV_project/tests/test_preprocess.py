"""Unit tests for docint.preprocess, all on synthetic numpy/OpenCV images.

Covers the spec-mandated suites — corner ordering, blur detector (sharp
checkerboard vs Gaussian-blurred), and the fallback path when no contour is
found — plus warp, deskew, shadow removal and the orchestrator.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from docint import preprocess

# ---------------------------------------------------------------------------
# synthetic scenes
# ---------------------------------------------------------------------------

CORNERS_ORDERED = np.array([[10, 10], [290, 12], [288, 390], [8, 388]], dtype=np.float32)


def _checkerboard(size: int = 320, square: int = 8) -> np.ndarray:
    """High-frequency BGR checkerboard (maximally sharp)."""
    tile = (((np.indices((size, size)).sum(axis=0) // square) % 2) * 255).astype(np.uint8)
    return np.dstack([tile] * 3)


def _page_scene() -> tuple[np.ndarray, np.ndarray]:
    """A bright, slightly tilted 'page' on a dark table + its true corners."""
    img = np.full((480, 640, 3), 40, dtype=np.uint8)
    quad = np.array([[90, 70], [560, 90], [540, 410], [110, 430]], dtype=np.int32)
    cv2.fillPoly(img, [quad], (235, 235, 235))
    return img, quad.astype(np.float32)


def _stripes(size: int = 400, period: int = 60, thickness: int = 10) -> np.ndarray:
    """White page with horizontal black bars (text-line stand-in), grayscale."""
    img = np.full((size, size), 255, dtype=np.uint8)
    for y in range(period // 2, size - thickness, period):
        img[y : y + thickness, 20 : size - 20] = 0
    return img


def _rotate(img: np.ndarray, angle_deg: float) -> np.ndarray:
    height, width = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(img, matrix, (width, height), borderValue=255)


# ---------------------------------------------------------------------------
# order_corners (spec-mandated)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("perm", [(0, 1, 2, 3), (2, 0, 3, 1), (3, 2, 1, 0), (1, 3, 0, 2)])
def test_order_corners_is_permutation_invariant(perm: tuple[int, ...]) -> None:
    ordered = preprocess.order_corners(CORNERS_ORDERED[list(perm)])
    np.testing.assert_allclose(ordered, CORNERS_ORDERED)


def test_order_corners_output_shape_and_dtype() -> None:
    out = preprocess.order_corners(CORNERS_ORDERED.copy())
    assert out.shape == (4, 2)
    assert out.dtype == np.float32


def test_order_corners_accepts_contour_shaped_input() -> None:
    contour_style = CORNERS_ORDERED.reshape(4, 1, 2)  # as cv2.approxPolyDP emits
    np.testing.assert_allclose(preprocess.order_corners(contour_style), CORNERS_ORDERED)


def test_order_corners_rejects_wrong_point_count() -> None:
    with pytest.raises(ValueError, match="4 corner points"):
        preprocess.order_corners(np.zeros((3, 2), dtype=np.float32))


# ---------------------------------------------------------------------------
# is_blurry (spec-mandated: sharp checkerboard vs gaussian-blurred)
# ---------------------------------------------------------------------------


def test_is_blurry_checkerboard_vs_gaussian(default_cfg: dict) -> None:
    blur_cfg = default_cfg["preprocess"]["blur"]
    sharp = _checkerboard()
    blurred = cv2.GaussianBlur(sharp, (21, 21), 8.0)

    sharp_flag, sharp_score = preprocess.is_blurry(sharp, blur_cfg)
    blurred_flag, blurred_score = preprocess.is_blurry(blurred, blur_cfg)

    assert not sharp_flag
    assert blurred_flag
    assert sharp_score > blur_cfg["laplacian_threshold"] > blurred_score


def test_is_blurry_uses_default_config_when_none_given() -> None:
    flagged, score = preprocess.is_blurry(_checkerboard())
    assert not flagged
    assert score > 0


# ---------------------------------------------------------------------------
# find_document_contour
# ---------------------------------------------------------------------------


def test_find_document_contour_on_synthetic_page(default_cfg: dict) -> None:
    img, quad = _page_scene()
    corners = preprocess.find_document_contour(img, default_cfg["preprocess"]["boundary"])
    assert corners is not None
    np.testing.assert_allclose(corners, preprocess.order_corners(quad), atol=6)


def test_find_document_contour_hough_fallback_on_broken_edges(default_cfg: dict) -> None:
    """Corners missing -> no closed contour -> the Hough line path must recover."""
    img = np.full((480, 640, 3), 30, dtype=np.uint8)
    color, thickness = (200, 200, 200), 3
    cv2.line(img, (140, 60), (500, 60), color, thickness)  # top edge, corners cut off
    cv2.line(img, (140, 420), (500, 420), color, thickness)  # bottom
    cv2.line(img, (100, 100), (100, 380), color, thickness)  # left
    cv2.line(img, (540, 100), (540, 380), color, thickness)  # right

    corners = preprocess.find_document_contour(img, default_cfg["preprocess"]["boundary"])

    assert corners is not None
    expected = np.array([[100, 60], [540, 60], [540, 420], [100, 420]], dtype=np.float32)
    np.testing.assert_allclose(corners, expected, atol=8)


def test_find_document_contour_returns_none_when_featureless(default_cfg: dict) -> None:
    flat = np.full((300, 400, 3), 128, dtype=np.uint8)
    assert preprocess.find_document_contour(flat, default_cfg["preprocess"]["boundary"]) is None


# ---------------------------------------------------------------------------
# warp_perspective
# ---------------------------------------------------------------------------


def test_warp_perspective_size_matches_quad_edges() -> None:
    img, quad = _page_scene()
    warped = preprocess.warp_perspective(img, quad)
    height, width = warped.shape[:2]
    # Longest opposing edges of the quad: ~470 px wide, ~360 px tall.
    assert abs(width - 470) <= 3
    assert abs(height - 360) <= 3
    assert warped.mean() > 200  # page fills the output frame


def test_warp_perspective_accepts_unordered_corners() -> None:
    img, quad = _page_scene()
    shuffled = quad[[2, 0, 3, 1]]
    assert preprocess.warp_perspective(img, shuffled).shape == preprocess.warp_perspective(img, quad).shape


# ---------------------------------------------------------------------------
# deskew
# ---------------------------------------------------------------------------


def test_estimate_and_deskew_straightens_rotated_stripes(default_cfg: dict) -> None:
    deskew_cfg = default_cfg["preprocess"]["deskew"]
    rotated = _rotate(_stripes(), 7.0)

    estimate = preprocess.estimate_skew_angle(rotated, deskew_cfg)
    assert 5.0 <= abs(estimate) <= 9.0  # magnitude recovered

    straightened = preprocess.deskew(rotated, deskew_cfg)
    residual = preprocess.estimate_skew_angle(straightened, deskew_cfg)
    assert abs(residual) < 1.2  # sign was right: rotation was undone


def test_deskew_is_noop_on_straight_image(default_cfg: dict) -> None:
    stripes = _stripes()
    out = preprocess.deskew(stripes, default_cfg["preprocess"]["deskew"])
    assert out.shape == stripes.shape
    np.testing.assert_array_equal(out, stripes)


# ---------------------------------------------------------------------------
# remove_shadows
# ---------------------------------------------------------------------------


def test_remove_shadows_flattens_illumination_gradient(default_cfg: dict) -> None:
    page = np.full((200, 300), 255, dtype=np.uint8)
    page[90:110, 30:270] = 0  # text bar
    gradient = np.linspace(0.35, 1.0, 300)[None, :]  # dark left, lit right
    shaded = (np.dstack([page] * 3) * gradient[..., None]).astype(np.uint8)

    result = preprocess.remove_shadows(shaded, default_cfg["preprocess"]["shadow"])

    dark_side = float(result[20:70, 10:60].mean())
    lit_side = float(result[20:70, 240:290].mean())
    assert dark_side > 200 and lit_side > 200  # background restored to white
    assert abs(dark_side - lit_side) < 25  # gradient flattened
    assert float(result[92:108, 100:200].mean()) < 120  # ink still dark


# ---------------------------------------------------------------------------
# preprocess() orchestrator (spec-mandated fallback path)
# ---------------------------------------------------------------------------


def test_preprocess_falls_back_to_full_frame_with_warning(default_cfg: dict) -> None:
    flat = np.full((300, 400, 3), 128, dtype=np.uint8)
    result = preprocess.preprocess(flat, default_cfg)

    assert result.corners is None
    assert any("full frame" in warning for warning in result.warnings)  # degraded, not crashed
    assert result.image is not None and result.image.ndim == 2
    for key in preprocess.INTERMEDIATE_KEYS:
        assert key in result.images
    for step in preprocess.PREPROCESS_STEPS:
        assert step in result.timings_ms


def test_preprocess_end_to_end_on_synthetic_page(default_cfg: dict) -> None:
    img, _ = _page_scene()
    result = preprocess.preprocess(img, default_cfg)

    assert result.corners is not None
    assert not any("full frame" in warning for warning in result.warnings)
    assert result.images["warped"].shape[0] < img.shape[0]  # cropped down to the page
    assert result.image.ndim == 2
    assert abs(result.skew_angle_deg) <= default_cfg["preprocess"]["deskew"]["max_abs_angle_deg"]


def test_preprocess_rejects_non_image_input() -> None:
    with pytest.raises(ValueError):
        preprocess.preprocess(np.zeros((0, 0, 3), dtype=np.uint8))
