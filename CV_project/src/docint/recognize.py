"""Stage 3 — text recognition.

Defines the :class:`TextRecognizer` interface with the default
:class:`PaddleRecognizer`; :class:`TesseractRecognizer` and
:class:`TrOCRRecognizer` are alternate backends behind the same interface.

Language support: every recognizer's **constructor takes**
``lang in {"en", "mar", "hin"}`` as its default language, and
:meth:`TextRecognizer.recognize` accepts a per-call override. Each backend
maps these codes to its own models via config
(``recognize.<backend>.lang_map``) — e.g. PaddleOCR serves both ``mar`` and
``hin`` with its ``devanagari`` recognition model — so callers never see
backend specifics.

Heavy dependencies (paddleocr, pytesseract, torch/transformers) are imported
only inside ``_load_model``; models are created once per language and reused
across calls.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping, Sequence, get_args

from docint.detect import TextBox, group_into_lines
from docint.preprocess import Image, warp_perspective
from docint.schemas import Language

#: The language codes every recognizer must accept.
SUPPORTED_LANGS: tuple[str, ...] = get_args(Language)


@dataclass(frozen=True)
class RecognizedSpan:
    """A detected text box together with its recognized content."""

    box: TextBox
    text: str
    confidence: float


def join_spans(spans: Sequence[RecognizedSpan], line_overlap_frac: float) -> str:
    """Reading-ordered plain text: spaces within a line, newlines between.

    Args:
        spans: Recognition output (any order; grouped internally).
        line_overlap_frac: Vertical-overlap fraction for line grouping
            (``detect.reading_order.line_overlap_frac`` in the config).

    Returns:
        The page text with empty spans dropped.
    """
    if not spans:
        return ""
    span_by_box = {span.box: span for span in spans}  # TextBox hashes by identity
    lines = group_into_lines([span.box for span in spans], line_overlap_frac)
    text_lines = (
        " ".join(span_by_box[box].text for box in line if span_by_box[box].text)
        for line in lines
    )
    return "\n".join(line for line in text_lines if line)


class TextRecognizer(ABC):
    """Interface every recognition backend implements.

    Args:
        cfg: The ``recognize`` config section.
        lang: Default language for :meth:`recognize` — one of
            :data:`SUPPORTED_LANGS`.

    Raises:
        ValueError: If ``lang`` is not supported.
    """

    #: Registry key and human-readable backend name.
    name: ClassVar[str]

    def __init__(self, cfg: Mapping[str, Any], lang: Language = "en") -> None:
        _validate_lang(lang)
        self._cfg = dict(cfg)
        self.lang: Language = lang

    @abstractmethod
    def recognize(
        self, image: Image, boxes: Sequence[TextBox], lang: Language | None = None
    ) -> list[RecognizedSpan]:
        """Recognize the text inside each detected box.

        Implementations crop each ``box`` from ``image`` (perspective-
        rectified for non-rectangular quads), run the model for ``lang``,
        and return spans in the same order as ``boxes``.

        Args:
            image: Deskewed page the boxes were detected on.
            boxes: Detection output, already in reading order.
            lang: Per-call language override; None uses the constructor
                default.

        Returns:
            One :class:`RecognizedSpan` per input box (empty text allowed).

        Raises:
            ValueError: If ``lang`` is not supported.
        """


def _validate_lang(lang: str) -> None:
    if lang not in SUPPORTED_LANGS:
        raise ValueError(f"unsupported lang {lang!r}; expected one of {SUPPORTED_LANGS}")


def _ensure_bgr(image: Image) -> Image:
    """PaddleOCR expects 3-channel BGR input."""
    import cv2

    return image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def _parse_rec_result(raw: Any) -> tuple[str, float]:
    """Unwrap a PaddleOCR ``det=False`` result to ``(text, confidence)``.

    PaddleOCR 2.7 returns ``[[(text, score)]]`` for a single crop; empty or
    failed crops come back as ``[]`` / ``[None]``. Nested single-element
    lists are unwrapped until the ``(text, score)`` pair (or nothing) is
    found, so minor shape drift between versions doesn't crash the pipeline.
    """

    def _is_pair(node: Any) -> bool:
        return (
            isinstance(node, (list, tuple))
            and len(node) == 2
            and isinstance(node[0], str)
            and isinstance(node[1], (int, float))
        )

    node = raw
    while isinstance(node, (list, tuple)) and node and not _is_pair(node):
        node = node[0]
    if _is_pair(node):
        return str(node[0]), float(node[1])
    return "", 0.0


class PaddleRecognizer(TextRecognizer):
    """PaddleOCR recognition — the default backend.

    ``mar`` and ``hin`` map to PaddleOCR's ``devanagari`` recognition model
    via ``recognize.paddle.lang_map``; other ``paddle.*`` keys (e.g.
    ``rec_batch_num``) pass straight to ``PaddleOCR``. One model is created
    per mapped language code, lazily, and cached for the lifetime of the
    recognizer — ``mar`` and ``hin`` therefore share a single model.
    """

    name: ClassVar[str] = "paddle"

    def __init__(self, cfg: Mapping[str, Any], lang: Language = "en") -> None:
        super().__init__(cfg, lang)
        self._models: dict[str, Any] = {}  # paddle lang code -> loaded-once model

    def _lang_code(self, lang: Language) -> str:
        """Map a docint language to a PaddleOCR model code via config."""
        lang_map = self._cfg.get("paddle", {}).get("lang_map", {})
        code = lang_map.get(lang)
        if code is None:
            raise ValueError(
                f"no PaddleOCR model mapped for lang {lang!r} — "
                "add it to recognize.paddle.lang_map in the config"
            )
        return str(code)

    def _load_model(self, code: str) -> Any:
        """Create the PaddleOCR handle for one language code (patched in tests)."""
        from paddleocr import PaddleOCR  # heavy import stays lazy

        params = {k: v for k, v in self._cfg.get("paddle", {}).items() if k != "lang_map"}
        return PaddleOCR(lang=code, use_angle_cls=False, show_log=False, **params)

    def _get_model(self, code: str) -> Any:
        if code not in self._models:
            self._models[code] = self._load_model(code)
        return self._models[code]

    def recognize(
        self, image: Image, boxes: Sequence[TextBox], lang: Language | None = None
    ) -> list[RecognizedSpan]:
        """Perspective-crop each box and run PaddleOCR recognition on it."""
        lang = lang or self.lang
        _validate_lang(lang)
        if not boxes:
            return []

        model = self._get_model(self._lang_code(lang))
        page = _ensure_bgr(image)
        spans: list[RecognizedSpan] = []
        for box in boxes:
            crop = warp_perspective(page, box.polygon)  # rectifies rotated quads
            text, confidence = _parse_rec_result(model.ocr(crop, det=False, rec=True, cls=False))
            spans.append(RecognizedSpan(box=box, text=text, confidence=confidence))
        return spans


class TesseractRecognizer(TextRecognizer):
    """pytesseract backend (stub).

    Requires the ``tesseract`` binary plus ``mar`` / ``hin`` traineddata
    (installed in the Dockerfile). Config: ``recognize.tesseract`` —
    ``lang_map``, ``psm``, ``oem``.
    """

    name: ClassVar[str] = "tesseract"

    def recognize(
        self, image: Image, boxes: Sequence[TextBox], lang: Language | None = None
    ) -> list[RecognizedSpan]:
        """See :meth:`TextRecognizer.recognize`."""
        raise NotImplementedError("TODO: pytesseract.image_to_data over box crops")


class TrOCRRecognizer(TextRecognizer):
    """TrOCR (transformers) backend (stub) — strongest on handwriting.

    ``mar`` / ``hin`` need fine-tuned checkpoints configured in
    ``recognize.trocr.lang_map``; only an English checkpoint is mapped by
    default. Requires the optional torch/transformers pins in
    requirements.txt.
    """

    name: ClassVar[str] = "trocr"

    def recognize(
        self, image: Image, boxes: Sequence[TextBox], lang: Language | None = None
    ) -> list[RecognizedSpan]:
        """See :meth:`TextRecognizer.recognize`."""
        raise NotImplementedError("TODO: TrOCR generate() over box crops")


#: Backend registry — mirrors docint.detect._DETECTORS.
_RECOGNIZERS: dict[str, type[TextRecognizer]] = {
    PaddleRecognizer.name: PaddleRecognizer,
    TesseractRecognizer.name: TesseractRecognizer,
    TrOCRRecognizer.name: TrOCRRecognizer,
}


def get_recognizer(
    name: str, cfg: Mapping[str, Any], lang: Language | None = None
) -> TextRecognizer:
    """Instantiate a recognition backend by registry name.

    Args:
        name: Registry key (``recognize.backend`` in the config).
        cfg: The ``recognize`` config section, passed to the constructor.
        lang: Default language for the recognizer; None reads
            ``recognize.default_lang`` from ``cfg`` (falling back to "en").

    Returns:
        A ready-to-use (lazily initialized) recognizer.

    Raises:
        KeyError: If ``name`` is not a registered backend.
        ValueError: If the language is not supported.
    """
    try:
        cls = _RECOGNIZERS[name]
    except KeyError as exc:
        raise KeyError(f"unknown recognizer {name!r}; available: {sorted(_RECOGNIZERS)}") from exc
    return cls(cfg, lang=lang or cfg.get("default_lang", "en"))
