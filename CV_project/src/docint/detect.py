"""Stage 2 — text detection.

Defines the :class:`TextDetector` interface plus the default
:class:`PaddleDetector` (DBNet via PaddleOCR). Alternative backends (CRAFT,
EAST) plug in by subclassing :class:`TextDetector` and registering in
:data:`_DETECTORS`; the pipeline only ever talks to the interface.

Boxes come back in **reading order** (top-to-bottom, left-to-right) via
:func:`reading_order_sort`, which groups boxes into lines by vertical
overlap first — so ragged single-column scans don't zigzag on small
x/y jitter.

Heavy dependencies (paddleocr / paddle) are imported only inside
``_load_model``, so importing this module stays cheap; each backend
instance creates its model once and reuses it across calls.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, ClassVar, Mapping, Sequence

import numpy as np

from docint.preprocess import Image


# eq=False: ndarray equality is element-wise, which would break the
# generated __eq__; identity comparison (and identity hashing) is what
# callers actually want.
@dataclass(frozen=True, eq=False)
class TextBox:
    """One detected text region.

    Attributes:
        polygon: ``(N, 2)`` float32 array of vertices in image coordinates
            (DBNet emits quadrilaterals, N == 4).
        confidence: Detector score in ``[0, 1]``.
    """

    polygon: np.ndarray
    confidence: float = 1.0

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Axis-aligned ``(x1, y1, x2, y2)`` envelope of :attr:`polygon`."""
        xs, ys = self.polygon[:, 0], self.polygon[:, 1]
        return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


class TextDetector(ABC):
    """Interface every detection backend implements."""

    #: Registry key and human-readable backend name.
    name: ClassVar[str]

    @abstractmethod
    def detect(self, image: Image) -> list[TextBox]:
        """Detect text regions in a preprocessed page image.

        Args:
            image: Deskewed page (BGR or grayscale).

        Returns:
            Text boxes in **reading order** (top-to-bottom, left-to-right);
            implementations run :func:`reading_order_sort` before returning.
        """


# ---------------------------------------------------------------------------
# reading order
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _default_detect_cfg() -> dict[str, Any]:
    """The ``detect`` section of configs/default.yaml (loaded once)."""
    from docint.config import load_config

    return load_config()["detect"]


def group_into_lines(boxes: Sequence[TextBox], line_overlap_frac: float) -> list[list[TextBox]]:
    """Group boxes into text lines by vertical-interval overlap.

    A box joins an existing line when its vertical overlap with the line's
    running extent is at least ``line_overlap_frac`` of the smaller of the
    two heights. Lines come back top-to-bottom, boxes within a line
    left-to-right.

    Args:
        boxes: Detected boxes in any order.
        line_overlap_frac: Minimum relative overlap to share a line
            (``detect.reading_order.line_overlap_frac`` in the config).

    Returns:
        Lines of boxes; the input is not mutated.
    """
    lines: list[list[TextBox]] = []
    bounds: list[tuple[float, float]] = []  # running (top, bottom) per line

    for box in sorted(boxes, key=lambda b: b.bbox[1]):
        x1, y1, x2, y2 = box.bbox
        height = max(y2 - y1, 1e-6)
        for index, (top, bottom) in enumerate(bounds):
            overlap = min(bottom, y2) - max(top, y1)
            if overlap >= line_overlap_frac * min(height, max(bottom - top, 1e-6)):
                lines[index].append(box)
                bounds[index] = (min(top, y1), max(bottom, y2))
                break
        else:
            lines.append([box])
            bounds.append((y1, y2))

    ordered = sorted(zip(bounds, lines), key=lambda pair: pair[0][0])
    return [sorted(line, key=lambda b: b.bbox[0]) for _, line in ordered]


def reading_order_sort(
    boxes: Sequence[TextBox], line_overlap_frac: float | None = None
) -> list[TextBox]:
    """Sort detected boxes into reading order (top-to-bottom, left-to-right).

    Boxes are grouped into lines by vertical overlap first
    (:func:`group_into_lines`), which keeps ragged single-column layouts
    stable: a line that starts a few pixels lower than its neighbour still
    sorts as one line instead of interleaving.

    Args:
        boxes: Detected boxes in any order.
        line_overlap_frac: Minimum relative vertical overlap to share a
            line; None reads ``detect.reading_order.line_overlap_frac`` from
            the shipped default config.

    Returns:
        A new, flat list in reading order (the input is not mutated).
    """
    if line_overlap_frac is None:
        line_overlap_frac = float(_default_detect_cfg()["reading_order"]["line_overlap_frac"])
    return [box for line in group_into_lines(boxes, line_overlap_frac) for box in line]


# ---------------------------------------------------------------------------
# PaddleOCR backend
# ---------------------------------------------------------------------------


def _ensure_bgr(image: Image) -> Image:
    """PaddleOCR expects 3-channel BGR input."""
    import cv2

    return image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


class PaddleDetector(TextDetector):
    """DBNet text detector via PaddleOCR — the default backend.

    Config section ``detect``: ``reading_order.line_overlap_frac`` plus
    ``paddle.*`` keys passed straight to ``PaddleOCR`` (``det_db_thresh``,
    ``det_db_box_thresh``, ``det_db_unclip_ratio``, ``det_limit_side_len``).
    The model is created lazily on the first :meth:`detect` call (first run
    downloads weights to ``~/.paddleocr``) and reused afterwards.
    """

    name: ClassVar[str] = "paddle"

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        """Store the ``detect`` config section; defer model creation.

        Args:
            cfg: The full ``detect`` config section.
        """
        self._cfg = dict(cfg)
        self._model: Any | None = None  # lazy PaddleOCR handle, loaded once

    def _load_model(self) -> Any:
        """Create the PaddleOCR handle (patched out in unit tests)."""
        from paddleocr import PaddleOCR  # heavy import stays lazy

        params = dict(self._cfg.get("paddle", {}))
        return PaddleOCR(lang="en", use_angle_cls=False, show_log=False, **params)

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def detect(self, image: Image) -> list[TextBox]:
        """Run DBNet and return reading-ordered boxes ([] when none found)."""
        page = self._detect_page(self._get_model(), _ensure_bgr(image))
        # The page result is an (N, 4, 2) ndarray (or None / empty) — explicit
        # None/len checks only; ndarray truthiness raises.
        if page is None or len(page) == 0:
            return []
        boxes = [TextBox(polygon=np.asarray(poly, dtype=np.float32)) for poly in page]
        overlap_frac = self._cfg.get("reading_order", {}).get("line_overlap_frac")
        return reading_order_sort(boxes, overlap_frac)

    @staticmethod
    def _detect_page(model: Any, image: Image) -> Any:
        """Det-only inference for one page.

        PaddleOCR 2.7's ``ocr(det=True, rec=False)`` wrapper crashes on
        multi-box pages (``if not dt_boxes:`` on an ndarray — upstream bug),
        so the underlying ``text_detector`` engine is called directly when
        present; the wrapper stays as a fallback for API variants without it.
        """
        text_detector = getattr(model, "text_detector", None)
        if callable(text_detector):
            dt_boxes, _elapse = text_detector(image)
            return dt_boxes
        raw = model.ocr(image, det=True, rec=False, cls=False)
        return raw[0] if raw is not None and len(raw) > 0 else None


#: Backend registry — add new detectors here (spec: CRAFT / EAST later).
_DETECTORS: dict[str, type[TextDetector]] = {
    PaddleDetector.name: PaddleDetector,
    # "craft": CraftDetector,
    # "east": EastDetector,
}


def get_detector(name: str, cfg: Mapping[str, Any]) -> TextDetector:
    """Instantiate a detection backend by registry name.

    Args:
        name: Registry key (``detect.backend`` in the config).
        cfg: The ``detect`` config section, passed to the constructor.

    Returns:
        A ready-to-use (lazily initialized) detector.

    Raises:
        KeyError: If ``name`` is not a registered backend.
    """
    try:
        cls = _DETECTORS[name]
    except KeyError as exc:
        raise KeyError(f"unknown detector {name!r}; available: {sorted(_DETECTORS)}") from exc
    return cls(cfg)
