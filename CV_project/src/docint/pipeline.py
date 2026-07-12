"""Stage orchestration — composes preprocess, detect, recognize and layout.

::

    photo -> preprocess -> detect -> recognize -> layout -> DocumentResult
              |  (boundary, warp, deskew, shadow, enhance, blur check)
              +-- every intermediate image is kept for the demo UI

Contracts enforced HERE, not in the stages:

* **Graceful degradation** — with ``pipeline.fail_fast: false`` a stage
  error downgrades to a warning on the result and the run continues with
  that stage's fallback; the pipeline never raises for a bad image.
* **Timing** — each stage's wall-clock duration lands in
  ``DocumentResult.timings_ms`` under a stable name from :data:`STAGES`
  (preprocess sub-steps are namespaced as ``preprocess.<step>``).

CLI::

    python -m docint.pipeline photo.jpg --lang en --out out.json
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from docint import preprocess as preprocess_stage
from docint.config import load_config
from docint.detect import TextBox, TextDetector, get_detector
from docint.preprocess import Image
from docint.recognize import RecognizedSpan, TextRecognizer, get_recognizer, join_spans
from docint.schemas import BBox, DocumentResult, Language, Page, Question, Region, RegionType
from docint.timing import stage_timer

__all__ = ["STAGES", "Pipeline", "PipelineResult", "main", "stage_timer"]

#: Stable stage names — the timing keys of ``DocumentResult.timings_ms``.
STAGES: tuple[str, ...] = ("preprocess", "detect", "recognize", "layout", "output")


# eq=False: holds ndarrays (element-wise __eq__ would be ambiguous).
@dataclass(eq=False)
class PipelineResult:
    """Everything one pipeline run produces.

    Attributes:
        document: The structured JSON payload (spec output format), including
            warnings, the retake_photo flag and per-stage timings.
        deskewed_image: The corrected page (BGR), or ``None`` if
            preprocessing never got that far.
        intermediates: Per-stage visualization images for the demo UI, in
            pipeline order. Keys (when ``pipeline.return_intermediates``):
            ``boundary_overlay``, ``warped``, ``deskewed``, ``shadow_free``,
            ``enhanced``, ``detections_overlay``.
    """

    document: DocumentResult
    deskewed_image: Image | None = None
    intermediates: dict[str, Image] = field(default_factory=dict)


def _span_region(span: RecognizedSpan) -> Region:
    """One pass-through Region per recognized span (pre-layout fallback)."""
    x1, y1, x2, y2 = (max(value, 0.0) for value in span.box.bbox)
    return Region(
        type=RegionType.OTHER,
        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
        polygon=[(float(x), float(y)) for x, y in span.box.polygon],
        text=span.text,
        confidence=min(max(float(span.confidence), 0.0), 1.0),
    )


def _draw_detections(image: Image, boxes: list[TextBox]) -> Image:
    """Page copy with the detected polygons drawn (for the demo gallery)."""
    import cv2

    canvas = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    thickness = max(1, min(canvas.shape[:2]) // 400)
    for box in boxes:
        cv2.polylines(canvas, [box.polygon.astype(np.int32)], True, (0, 0, 255), thickness)
    return canvas


class Pipeline:
    """End-to-end document intelligence pipeline.

    Stages are resolved once from config (detector / recognizer backends via
    their registries) and reused across :meth:`run` calls; the underlying
    models load lazily on first use and stay loaded.
    """

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        config_path: str | Path | None = None,
    ) -> None:
        """Build the pipeline from configuration.

        Args:
            config: Pre-loaded config mapping (takes precedence).
            config_path: YAML file to load instead; defaults to
                ``configs/default.yaml``.
        """
        self.cfg: dict[str, Any] = dict(config) if config is not None else load_config(config_path)
        self.detector: TextDetector = get_detector(self.cfg["detect"]["backend"], self.cfg["detect"])
        self.recognizer: TextRecognizer = get_recognizer(
            self.cfg["recognize"]["backend"], self.cfg["recognize"]
        )

    def run(self, image: Image, lang: Language | None = None) -> PipelineResult:
        """Process one document photo end to end.

        Stage sequence (each timed under its :data:`STAGES` key):

        1. ``preprocess`` — :func:`docint.preprocess.preprocess`: blur check
           (``retake_photo`` + ``blur_score``), boundary detection with
           full-frame fallback, perspective warp, deskew, shadow removal,
           enhancement. Its warnings and namespaced step timings are merged
           into the result.
        2. ``detect`` — ``self.detector`` on the enhanced page; boxes arrive
           in reading order.
        3. ``recognize`` — ``self.recognizer`` over box crops from the
           shadow-free page; spans below ``recognize.min_confidence`` are
           dropped with a warning.
        4. ``layout`` — rule-based ``classify_regions`` (heading / paragraph
           / table / question_block) plus ``parse_questions`` filling
           ``document.questions``; a layout error degrades to raw
           ``RegionType.OTHER`` span regions with a warning.
        5. ``output`` — assemble the :class:`~docint.schemas.DocumentResult`.

        Degradation: with ``pipeline.fail_fast: false`` (default) a detect /
        recognize error becomes a warning and the run continues with empty
        results for that stage; ``fail_fast: true`` re-raises for debugging.

        Args:
            image: Document photo (BGR or grayscale uint8).
            lang: Recognition language; None uses the recognizer's
                constructor default (``recognize.default_lang``).

        Returns:
            A fully populated :class:`PipelineResult`.
        """
        lang = lang or self.recognizer.lang
        fail_fast = bool(self.cfg["pipeline"].get("fail_fast", False))
        timings: dict[str, float] = {}
        warnings: list[str] = []

        with stage_timer(timings, "preprocess"):
            pre = preprocess_stage.preprocess(image, self.cfg)
        warnings.extend(pre.warnings)

        boxes: list[TextBox] = []
        with stage_timer(timings, "detect"):
            try:
                boxes = self.detector.detect(pre.image)
            except Exception as exc:  # noqa: BLE001 — degradation contract
                if fail_fast:
                    raise
                warnings.append(f"text detection failed ({exc}); no regions extracted")

        spans: list[RecognizedSpan] = []
        with stage_timer(timings, "recognize"):
            if boxes:
                rec_source = pre.images.get("shadow_free", pre.image)
                try:
                    spans = self.recognizer.recognize(rec_source, boxes, lang)
                except Exception as exc:  # noqa: BLE001
                    if fail_fast:
                        raise
                    warnings.append(f"text recognition failed ({exc}); regions have no text")
            min_confidence = float(self.cfg["recognize"]["min_confidence"])
            kept = [span for span in spans if span.confidence >= min_confidence]
            if len(kept) != len(spans):
                warnings.append(
                    f"dropped {len(spans) - len(kept)} low-confidence span(s) (< {min_confidence})"
                )
            spans = kept

        with stage_timer(timings, "layout"):
            full_text = self._join_spans(spans)
            regions, questions = self._layout(
                spans, full_text, pre.image.shape[:2], warnings, fail_fast
            )

        with stage_timer(timings, "output"):
            height, width = pre.image.shape[:2]
            document = DocumentResult(
                pages=[Page(page_number=1, width=int(width), height=int(height), regions=regions)],
                questions=questions,
                full_text=full_text,
                language=lang,
                warnings=warnings,
                retake_photo=pre.retake_photo,
                blur_score=pre.blur_score,
            )
            intermediates: dict[str, Image] = {}
            if self.cfg["pipeline"].get("return_intermediates", True):
                intermediates = dict(pre.images)
                intermediates["detections_overlay"] = _draw_detections(
                    pre.images.get("deskewed", pre.image), boxes
                )

        document.timings_ms = {
            **timings,
            **{f"preprocess.{step}": ms for step, ms in pre.timings_ms.items()},
        }
        return PipelineResult(
            document=document,
            deskewed_image=pre.images.get("deskewed"),
            intermediates=intermediates,
        )

    def run_path(self, image_path: str | Path, lang: Language | None = None) -> PipelineResult:
        """Convenience wrapper: read ``image_path`` with cv2 and call :meth:`run`.

        Args:
            image_path: Path to a JPEG/PNG/WebP photo.
            lang: Recognition language (see :meth:`run`).

        Returns:
            See :meth:`run`.

        Raises:
            FileNotFoundError: If the path does not exist or cannot be
                decoded (a missing *file* is a caller bug, not a bad image).
        """
        import cv2

        path = Path(image_path)
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"could not read image: {path}")
        return self.run(image, lang=lang)

    # -- internals ----------------------------------------------------------

    def _join_spans(self, spans: list[RecognizedSpan]) -> str:
        """Reading-ordered plain text via :func:`docint.recognize.join_spans`."""
        return join_spans(spans, float(self.cfg["detect"]["reading_order"]["line_overlap_frac"]))

    def _layout(
        self,
        spans: list[RecognizedSpan],
        full_text: str,
        image_shape: tuple[int, int],
        warnings: list[str],
        fail_fast: bool,
    ) -> tuple[list[Region], list[Question]]:
        """Run the layout stage; degrade to raw span regions on error."""
        from docint import layout as layout_stage

        try:
            regions = layout_stage.classify_regions(spans, image_shape, self.cfg["layout"])
            questions = layout_stage.parse_questions(full_text, self.cfg["layout"])
            return regions, questions
        except Exception as exc:  # noqa: BLE001 — degradation contract
            if fail_fast:
                raise
            warnings.append(f"layout parsing failed ({exc}); emitting raw detection regions")
            return [_span_region(span) for span in spans], []


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: one image in, DocumentResult JSON out."""
    import argparse
    import sys

    from docint.recognize import SUPPORTED_LANGS

    parser = argparse.ArgumentParser(
        prog="python -m docint.pipeline",
        description="Run the docint pipeline on one document photo and emit DocumentResult JSON.",
    )
    parser.add_argument("image", type=Path, help="Path to the photo (JPEG/PNG/WebP)")
    parser.add_argument("--lang", choices=SUPPORTED_LANGS, default="en", help="Recognition language")
    parser.add_argument("--out", type=Path, default=None, help="Write JSON here instead of stdout")
    parser.add_argument("--config", type=Path, default=None, help="Alternative config YAML")
    args = parser.parse_args(argv)

    pipeline = Pipeline(config_path=args.config)
    result = pipeline.run_path(args.image, lang=args.lang)
    payload = result.document.model_dump_json(indent=2)

    if args.out is not None:
        args.out.write_text(payload, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        try:  # Windows consoles may default to a non-UTF-8 code page
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
        print(payload)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
