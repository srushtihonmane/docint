"""Stage 1 — image preprocessing (OpenCV).

Every step is a separate pure function — image in, image (or value) out, no
global state — with all thresholds coming from the ``preprocess`` section of
``configs/default.yaml``. Each function takes an optional ``cfg`` mapping
(its own config subsection); when omitted, the shipped ``default.yaml`` is
used, so every step stays independently callable::

    from docint import preprocess
    corners = preprocess.find_document_contour(photo)

:func:`preprocess` chains all steps with per-step timing and graceful
degradation and returns a :class:`PreprocessResult`.

``cv2`` is imported lazily inside functions so importing this module needs
numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt

from docint.timing import stage_timer

#: BGR or grayscale uint8 image, as returned by ``cv2.imread``.
Image = npt.NDArray[np.uint8]

#: Four corner points, float32, shape (4, 2), ordered TL, TR, BR, BL.
Corners = npt.NDArray[np.float32]

#: Keys of ``PreprocessResult.timings_ms``, in execution order.
PREPROCESS_STEPS: tuple[str, ...] = (
    "resize",
    "blur_check",
    "boundary",
    "warp",
    "deskew",
    "shadow",
    "enhance",
)

#: Keys of ``PreprocessResult.images``, in pipeline order.
INTERMEDIATE_KEYS: tuple[str, ...] = (
    "boundary_overlay",
    "warped",
    "deskewed",
    "shadow_free",
    "enhanced",
)


# --------------------------------------------------------------------------
# config plumbing
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _default_preprocess_cfg() -> dict[str, Any]:
    """The ``preprocess`` section of configs/default.yaml (loaded once)."""
    from docint.config import load_config

    return load_config()["preprocess"]


def _step_cfg(cfg: Mapping[str, Any] | None, step: str) -> Mapping[str, Any]:
    """Return ``cfg`` itself, or the default config's ``step`` subsection."""
    return cfg if cfg is not None else _default_preprocess_cfg()[step]


def _preprocess_section(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Accept the full docint config or just its ``preprocess`` section."""
    if config is None:
        return _default_preprocess_cfg()
    return config.get("preprocess", config)


# --------------------------------------------------------------------------
# small shared helpers
# --------------------------------------------------------------------------


def _to_gray(img: Image) -> Image:
    """Grayscale view of a BGR or already-gray image."""
    import cv2

    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _to_bgr(img: Image) -> Image:
    """3-channel view of a gray or already-BGR image."""
    import cv2

    return img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _line_intersection(
    seg_a: tuple[float, float, float, float],
    seg_b: tuple[float, float, float, float],
) -> tuple[float, float] | None:
    """Intersection of the two infinite lines through the given segments."""
    x1, y1, x2, y2 = seg_a
    x3, y3, x4, y4 = seg_b
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-9:  # parallel
        return None
    cross_a = x1 * y2 - y1 * x2
    cross_b = x3 * y4 - y3 * x4
    px = (cross_a * (x3 - x4) - (x1 - x2) * cross_b) / denom
    py = (cross_a * (y3 - y4) - (y1 - y2) * cross_b) / denom
    return float(px), float(py)


# --------------------------------------------------------------------------
# boundary detection
# --------------------------------------------------------------------------


def order_corners(pts: npt.NDArray[np.floating]) -> Corners:
    """Order four arbitrary corner points as TL, TR, BR, BL.

    The top-left corner has the smallest ``x + y`` sum, the bottom-right the
    largest; the top-right has the smallest ``y - x`` difference, the
    bottom-left the largest. Permutation-invariant: any shuffle of the same
    four points yields the same ordering. (Degenerate quads rotated ~45°
    can tie; document photos never get close.)

    Args:
        pts: ``(4, 2)`` array of points in any order (any float/int dtype).

    Returns:
        ``(4, 2)`` float32 array ordered TL, TR, BR, BL.

    Raises:
        ValueError: If ``pts`` does not contain exactly four 2-D points.
    """
    arr = np.asarray(pts, dtype=np.float32)
    try:
        arr = arr.reshape(4, 2)
    except ValueError as exc:
        raise ValueError(f"expected 4 corner points, got array of shape {np.asarray(pts).shape}") from exc

    sums = arr.sum(axis=1)
    diffs = arr[:, 1] - arr[:, 0]  # y - x
    return np.array(
        [arr[np.argmin(sums)], arr[np.argmin(diffs)], arr[np.argmax(sums)], arr[np.argmax(diffs)]],
        dtype=np.float32,
    )


def _contour_quad(edges: Image, shape: tuple[int, int], cfg: Mapping[str, Any]) -> Corners | None:
    """Largest convex 4-point contour covering enough of the frame, or None."""
    import cv2

    height, width = shape
    min_area = float(cfg["min_area_frac"]) * height * width
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < min_area:
            break  # sorted descending — nothing bigger remains
        epsilon = float(cfg["approx_epsilon_frac"]) * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4 and cv2.isContourConvex(approx) and cv2.contourArea(approx) >= min_area:
            return order_corners(approx.reshape(4, 2))
    return None


def _hough_quad(edges: Image, shape: tuple[int, int], cfg: Mapping[str, Any]) -> Corners | None:
    """Fallback: intersect extreme Hough lines into a quad, or None."""
    import cv2

    height, width = shape
    hough = cfg["hough"]
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=int(hough["threshold"]),
        minLineLength=int(float(hough["min_line_length_frac"]) * min(height, width)),
        maxLineGap=int(hough["max_line_gap_px"]),
    )
    if lines is None:
        return None

    horizontal: list[tuple[float, float, float, float]] = []
    vertical: list[tuple[float, float, float, float]] = []
    for x1, y1, x2, y2 in lines[:, 0, :].astype(np.float64):
        (horizontal if abs(x2 - x1) >= abs(y2 - y1) else vertical).append((x1, y1, x2, y2))
    if len(horizontal) < 2 or len(vertical) < 2:
        return None

    horizontal.sort(key=lambda s: (s[1] + s[3]) / 2.0)  # by mean y
    vertical.sort(key=lambda s: (s[0] + s[2]) / 2.0)  # by mean x
    top, bottom = horizontal[0], horizontal[-1]
    left, right = vertical[0], vertical[-1]

    quad: list[tuple[float, float]] = []
    for edge_a, edge_b in ((top, left), (top, right), (bottom, right), (bottom, left)):
        point = _line_intersection(edge_a, edge_b)
        if point is None:
            return None
        quad.append(point)

    pts = np.array(quad, dtype=np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
    if cv2.contourArea(pts) < float(cfg["min_area_frac"]) * height * width:
        return None
    return order_corners(pts)


def find_document_contour(img: Image, cfg: Mapping[str, Any] | None = None) -> Corners | None:
    """Find the document's four corners in a phone photo, or None.

    Two strategies, in order (config: ``preprocess.boundary``):

    1. **Canny + contours** — grayscale, Gaussian blur (``gaussian_ksize``),
       Canny (``canny_low``/``canny_high``), morphological close
       (``morph_close_ksize``) to bridge small edge gaps, then the largest
       convex quadrilateral from ``cv2.approxPolyDP``
       (``approx_epsilon_frac`` × perimeter) covering at least
       ``min_area_frac`` of the frame.
    2. **Hough fallback** — probabilistic Hough segments (``hough.*``) split
       into near-horizontal / near-vertical; the topmost, bottommost,
       leftmost and rightmost lines are intersected into a quad (clipped to
       the frame, same area check).

    Args:
        img: Input photo (BGR or grayscale uint8).
        cfg: The ``preprocess.boundary`` config section; None loads the
            shipped default.

    Returns:
        Corners ordered TL, TR, BR, BL in ``img``'s pixel coordinates, or
        None when both strategies fail (caller should fall back to the full
        frame).
    """
    import cv2

    cfg = _step_cfg(cfg, "boundary")
    gray = _to_gray(img)

    ksize = int(cfg["gaussian_ksize"]) | 1  # Gaussian kernel must be odd
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    edges = cv2.Canny(blurred, int(cfg["canny_low"]), int(cfg["canny_high"]))
    close = int(cfg["morph_close_ksize"])
    if close > 1:
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))

    corners = _contour_quad(edges, gray.shape[:2], cfg)
    if corners is None:
        corners = _hough_quad(edges, gray.shape[:2], cfg)
    return corners


# --------------------------------------------------------------------------
# perspective + deskew
# --------------------------------------------------------------------------


def warp_perspective(img: Image, corners: npt.NDArray[np.floating]) -> Image:
    """Apply a 4-point perspective correction ("scan" the page).

    The output rectangle size is taken from the longer of each pair of
    opposing edges, so no content is squeezed.

    Args:
        img: Input photo (BGR or grayscale).
        corners: Four page corners in any order (ordered internally).

    Returns:
        The top-down view of the quadrilateral.
    """
    import cv2

    quad = order_corners(corners)
    tl, tr, br, bl = quad
    width = max(int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))), 1)
    height = max(int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))), 1)
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img, matrix, (width, height))


def _normalize_rect_angle(raw: float) -> float:
    """Map a ``cv2.minAreaRect`` angle to a correction angle in (-45, 45]."""
    angle = float(raw)
    if angle > 45.0:
        angle -= 90.0
    elif angle <= -45.0:
        angle += 90.0
    return angle


def _skew_from_hough(ink: Image, cfg: Mapping[str, Any]) -> float | None:
    """Median angle of long near-horizontal segments in an ink mask, or None."""
    import cv2

    max_abs = float(cfg["max_abs_angle_deg"])
    lines = cv2.HoughLinesP(
        ink,
        rho=1,
        theta=np.deg2rad(float(cfg["angle_resolution_deg"])),
        threshold=int(cfg["hough_threshold"]),
        minLineLength=int(float(cfg["min_line_length_frac"]) * ink.shape[1]),
        maxLineGap=int(cfg["max_line_gap_px"]),
    )
    if lines is None:
        return None
    angles = []
    for x1, y1, x2, y2 in lines[:, 0, :].astype(np.float64):
        if x1 == x2 and y1 == y2:
            continue
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle > 90.0:
            angle -= 180.0
        elif angle <= -90.0:
            angle += 180.0
        if abs(angle) <= max_abs:
            angles.append(angle)
    return float(np.median(angles)) if angles else None


def estimate_skew_angle(img: Image, cfg: Mapping[str, Any] | None = None) -> float:
    """Estimate the residual text-line skew of a page, in degrees.

    The returned value is the **correction angle**: passing it to
    ``cv2.getRotationMatrix2D`` (as :func:`deskew` does) straightens the
    page. Primary estimator: ``cv2.minAreaRect`` over Otsu-thresholded ink
    pixels (subsampled beyond ``max_ink_px``). If its normalized angle
    exceeds ``max_abs_angle_deg`` (unreliable), a Hough-segment median is
    tried; if that also fails, 0.0 is returned.

    Args:
        img: Warped page (BGR or grayscale).
        cfg: The ``preprocess.deskew`` config section; None loads the
            shipped default.

    Returns:
        Correction angle in degrees, always within ``±max_abs_angle_deg``.
    """
    import cv2

    cfg = _step_cfg(cfg, "deskew")
    gray = _to_gray(img)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    ys, xs = np.nonzero(ink)
    if len(xs) < int(cfg["min_ink_px"]):
        return 0.0
    pts = np.column_stack([xs, ys]).astype(np.float32)
    max_pts = int(cfg["max_ink_px"])
    if len(pts) > max_pts:
        pts = pts[:: len(pts) // max_pts + 1]

    max_abs = float(cfg["max_abs_angle_deg"])
    angle = _normalize_rect_angle(cv2.minAreaRect(pts)[-1])
    if abs(angle) <= max_abs:
        return angle

    fallback = _skew_from_hough(ink, cfg)
    if fallback is not None and abs(fallback) <= max_abs:
        return fallback
    return 0.0


def deskew(img: Image, cfg: Mapping[str, Any] | None = None, *, angle: float | None = None) -> Image:
    """Rotate the page by its estimated skew correction angle.

    Uses ``cv2.warpAffine`` with border replication so no black wedges are
    introduced; rotations below 0.001° return an unmodified copy.

    Args:
        img: Warped page (BGR or grayscale).
        cfg: The ``preprocess.deskew`` config section; None loads the
            shipped default.
        angle: Pre-computed correction angle (skips re-estimation); None
            calls :func:`estimate_skew_angle`.

    Returns:
        The straightened image (same size as the input).
    """
    import cv2

    cfg = _step_cfg(cfg, "deskew")
    if angle is None:
        angle = estimate_skew_angle(img, cfg)
    if abs(angle) < 1e-3:
        return img.copy()
    height, width = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), float(angle), 1.0)
    return cv2.warpAffine(
        img, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


# --------------------------------------------------------------------------
# illumination + contrast
# --------------------------------------------------------------------------


def remove_shadows(img: Image, cfg: Mapping[str, Any] | None = None) -> Image:
    """Remove soft shadows via background estimation and division.

    Per channel: dilate (``dilate_kernel``) to erase the (dark) ink, then
    median-blur (``median_ksize``) — the result approximates the
    illumination background; dividing the channel by it (scaled to 255)
    flattens the lighting while keeping ink dark.

    Args:
        img: Page image (BGR or grayscale).
        cfg: The ``preprocess.shadow`` config section; None loads the
            shipped default.

    Returns:
        Shadow-flattened image (same shape as the input).
    """
    import cv2

    cfg = _step_cfg(cfg, "shadow")
    kernel = np.ones((int(cfg["dilate_kernel"]),) * 2, np.uint8)
    median_ksize = int(cfg["median_ksize"]) | 1  # medianBlur requires odd

    channels = cv2.split(img) if img.ndim == 3 else [img]
    flattened = []
    for channel in channels:
        background = cv2.medianBlur(cv2.dilate(channel, kernel), median_ksize)
        flattened.append(cv2.divide(channel, background, scale=255))
    return cv2.merge(flattened) if img.ndim == 3 else flattened[0]


def enhance(img: Image, mode: str | None = None, cfg: Mapping[str, Any] | None = None) -> Image:
    """Boost text/background contrast for OCR.

    Modes (default from config ``preprocess.enhance.mode``):

    * ``"clahe"`` — CLAHE on the grayscale image (``clahe_clip_limit``,
      ``clahe_tile_grid``); keeps natural grayscale, best for deep OCR.
    * ``"adaptive"`` — Gaussian adaptive threshold
      (``adaptive_block_size``, ``adaptive_c``); hard black/white.
    * ``"both"`` — CLAHE first, then adaptive threshold.

    Args:
        img: Page image (BGR or grayscale).
        mode: Override the configured mode.
        cfg: The ``preprocess.enhance`` config section; None loads the
            shipped default.

    Returns:
        Enhanced 2-D grayscale (or binary) image.

    Raises:
        ValueError: On an unknown ``mode``.
    """
    import cv2

    cfg = _step_cfg(cfg, "enhance")
    mode = (mode or str(cfg["mode"])).lower()
    if mode not in ("clahe", "adaptive", "both"):
        raise ValueError(f"unknown enhance mode {mode!r}; expected clahe | adaptive | both")

    out = _to_gray(img)
    if mode in ("clahe", "both"):
        grid = int(cfg["clahe_tile_grid"])
        clahe = cv2.createCLAHE(clipLimit=float(cfg["clahe_clip_limit"]), tileGridSize=(grid, grid))
        out = clahe.apply(out)
    if mode in ("adaptive", "both"):
        block = int(cfg["adaptive_block_size"]) | 1  # neighborhood must be odd
        out = cv2.adaptiveThreshold(
            out, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, float(cfg["adaptive_c"])
        )
    return out


# --------------------------------------------------------------------------
# blur check
# --------------------------------------------------------------------------


def is_blurry(img: Image, cfg: Mapping[str, Any] | None = None) -> tuple[bool, float]:
    """Decide whether a photo is too blurry to OCR (the "retake photo" flag).

    The sharpness score is the variance of the Laplacian of the grayscale
    image (higher = sharper), compared against
    ``preprocess.blur.laplacian_threshold``.

    Args:
        img: Input photo (BGR or grayscale).
        cfg: The ``preprocess.blur`` config section; None loads the shipped
            default.

    Returns:
        ``(blurry, score)`` — ``blurry`` is True when ``score`` falls below
        the threshold.
    """
    import cv2

    cfg = _step_cfg(cfg, "blur")
    score = float(cv2.Laplacian(_to_gray(img), cv2.CV_64F).var())
    return score < float(cfg["laplacian_threshold"]), score


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------


# eq=False: holds ndarrays (element-wise __eq__ would be ambiguous).
@dataclass(eq=False)
class PreprocessResult:
    """Everything stage 1 produces for one photo.

    Attributes:
        image: Final enhanced image handed to text detection (2-D grayscale).
        images: Intermediate images keyed by :data:`INTERMEDIATE_KEYS`, in
            pipeline order. All coordinates/pixels refer to the working copy
            (after the optional ``max_side_px`` downscale).
        corners: Detected page corners (TL, TR, BR, BL) in working-copy
            coordinates, or None when detection failed and the full frame
            was used.
        skew_angle_deg: Correction angle applied by the deskew step.
        blur_score: Variance of the Laplacian of the working copy.
        retake_photo: True when ``blur_score`` is below
            ``preprocess.blur.laplacian_threshold``.
        warnings: Degradation notes (fallbacks and recovered step errors).
        timings_ms: Per-step wall-clock milliseconds, keyed by
            :data:`PREPROCESS_STEPS`.
    """

    image: Image
    images: dict[str, Image] = field(default_factory=dict)
    corners: Corners | None = None
    skew_angle_deg: float = 0.0
    blur_score: float = 0.0
    retake_photo: bool = False
    warnings: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)


def preprocess(img: Image, config: Mapping[str, Any] | None = None) -> PreprocessResult:
    """Run the full preprocessing chain with timing and graceful degradation.

    Steps, each timed under its :data:`PREPROCESS_STEPS` key: downscale to
    ``max_side_px``, blur check, boundary detection, perspective warp,
    deskew, shadow removal (``shadow.enabled``-gated), enhancement.

    Degradation contract: a failed boundary detection falls back to the full
    frame with a warning; any other step error is caught, noted as a
    warning, and the previous image is carried forward — a *bad photo* never
    raises. (A non-image argument is a caller bug and raises ValueError.)

    Args:
        img: Document photo (BGR or grayscale uint8).
        config: Full docint config mapping, or just its ``preprocess``
            section; None loads ``configs/default.yaml``.

    Returns:
        A fully populated :class:`PreprocessResult`.
    """
    import cv2

    if not isinstance(img, np.ndarray) or img.size == 0 or img.ndim not in (2, 3):
        raise ValueError("expected a non-empty HxW or HxWxC uint8 image")
    cfg = _preprocess_section(config)

    warnings: list[str] = []
    timings: dict[str, float] = {}
    images: dict[str, Image] = {}

    with stage_timer(timings, "resize"):
        work = img
        max_side = int(cfg.get("max_side_px", 0) or 0)
        longest = max(img.shape[:2])
        if 0 < max_side < longest:
            scale = max_side / float(longest)
            new_size = (round(img.shape[1] * scale), round(img.shape[0] * scale))
            work = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    with stage_timer(timings, "blur_check"):
        retake, blur = is_blurry(work, cfg["blur"])
        if retake:
            warnings.append(
                f"image appears blurry (laplacian variance {blur:.1f} < "
                f"{cfg['blur']['laplacian_threshold']}); consider retaking the photo"
            )

    corners: Corners | None = None
    boundary_error: str | None = None
    with stage_timer(timings, "boundary"):
        try:
            corners = find_document_contour(work, cfg["boundary"])
        except Exception as exc:  # noqa: BLE001 — degradation contract: never raise
            boundary_error = f"boundary detection error ({exc}); using full frame"
        if corners is None:
            warnings.append(boundary_error or "document boundary not found; using full frame")
        overlay = _to_bgr(work).copy()
        if corners is not None:
            thickness = max(2, min(overlay.shape[:2]) // 250)
            cv2.polylines(overlay, [corners.astype(np.int32)], True, (0, 255, 0), thickness)
        images["boundary_overlay"] = overlay

    with stage_timer(timings, "warp"):
        warped = work
        if corners is not None:
            try:
                warped = warp_perspective(work, corners)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"perspective warp failed ({exc}); using unwarped frame")
                warped = work
        images["warped"] = warped

    angle = 0.0
    with stage_timer(timings, "deskew"):
        deskewed = warped
        try:
            angle = estimate_skew_angle(warped, cfg["deskew"])
            deskewed = deskew(warped, cfg["deskew"], angle=angle)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"deskew failed ({exc}); keeping warped frame")
            angle = 0.0
        images["deskewed"] = deskewed

    with stage_timer(timings, "shadow"):
        shadow_free = deskewed
        if cfg["shadow"].get("enabled", True):
            try:
                shadow_free = remove_shadows(deskewed, cfg["shadow"])
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"shadow removal failed ({exc}); keeping deskewed frame")
        images["shadow_free"] = shadow_free

    with stage_timer(timings, "enhance"):
        try:
            final = enhance(shadow_free, cfg=cfg["enhance"])
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"enhancement failed ({exc}); using plain grayscale")
            final = _to_gray(shadow_free)
        images["enhanced"] = final

    return PreprocessResult(
        image=final,
        images=images,
        corners=corners,
        skew_angle_deg=float(angle),
        blur_score=blur,
        retake_photo=retake,
        warnings=warnings,
        timings_ms=timings,
    )
