"""Pipeline wiring + mocked end-to-end runs (no Paddle weights needed).

The fakes are seeded straight into the backends' lazy-load caches
(``detector._model`` / ``recognizer._models``), so the real caching code
paths are exercised while nothing heavy is imported.
"""

from __future__ import annotations

import numpy as np
import pytest

from docint.pipeline import STAGES, Pipeline, stage_timer


class _FakeDetModel:
    """Stands in for PaddleOCR in det-only mode.

    Like the real model, the page result is an (N, 4, 2) float ndarray —
    never rely on its truthiness.
    """

    def __init__(self, polys: list | None) -> None:
        self._polys = None if polys is None else np.asarray(polys, dtype=np.float32)

    def ocr(self, img, det=True, rec=True, cls=False):
        assert det and not rec
        return [self._polys]


class _FakeRecModel:
    """Stands in for PaddleOCR in rec-only mode; yields queued (text, conf)."""

    def __init__(self, results: list[tuple[str, float]]) -> None:
        self._results = list(results)

    def ocr(self, img, det=True, rec=True, cls=False):
        assert rec and not det
        return [[self._results.pop(0)]]


class _BoomModel:
    def ocr(self, *args, **kwargs):
        raise RuntimeError("boom")


def _rect(x1: float, y1: float, x2: float, y2: float) -> list[list[float]]:
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _mocked_pipeline(det_polys, rec_results, config=None) -> Pipeline:
    pipeline = Pipeline(config)
    pipeline.detector._model = _FakeDetModel(det_polys)
    pipeline.recognizer._models["en"] = _FakeRecModel(rec_results)
    return pipeline


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------


def test_pipeline_constructs_from_default_config() -> None:
    pipeline = Pipeline()
    assert pipeline.cfg["detect"]["backend"] == "paddle"
    assert pipeline.detector.name == "paddle"
    assert pipeline.recognizer.name == "paddle"
    assert pipeline.recognizer.lang == "en"  # recognize.default_lang


def test_unknown_backend_raises_helpful_keyerror() -> None:
    from docint.detect import get_detector

    with pytest.raises(KeyError, match="unknown detector"):
        get_detector("nonexistent", {})


def test_stage_timer_records_milliseconds() -> None:
    timings: dict[str, float] = {}
    with stage_timer(timings, "preprocess"):
        sum(range(1000))
    assert "preprocess" in timings
    assert timings["preprocess"] >= 0.0


def test_stage_names_are_stable() -> None:
    assert STAGES == ("preprocess", "detect", "recognize", "layout", "output")


# ---------------------------------------------------------------------------
# end-to-end with mocked models
# ---------------------------------------------------------------------------


def test_run_end_to_end_with_mocked_models(white_page: np.ndarray) -> None:
    pipeline = _mocked_pipeline(
        det_polys=[
            _rect(120, 12, 220, 32),  # line 1, right (deliberately unsorted)
            _rect(10, 50, 200, 70),  # line 2
            _rect(10, 10, 100, 30),  # line 1, left
        ],
        # queued in reading order: crops happen after sorting
        rec_results=[("hello", 0.97), ("world", 0.95), ("again", 0.99)],
    )

    result = pipeline.run(white_page, lang="en")
    document = result.document

    assert document.full_text == "hello world\nagain"
    regions = document.pages[0].regions
    # layout merges the two adjacent lines into one paragraph block
    assert len(regions) == 1
    assert regions[0].type.value == "paragraph"
    assert regions[0].text == "hello world\nagain"
    assert regions[0].confidence == pytest.approx((0.97 + 0.95 + 0.99) / 3)
    assert regions[0].bbox.x1 == pytest.approx(10.0)
    assert document.questions == []
    assert document.language == "en"
    assert document.pages[0].width == 300 and document.pages[0].height == 400
    assert set(STAGES) <= set(document.timings_ms)
    assert any(key.startswith("preprocess.") for key in document.timings_ms)
    assert "detections_overlay" in result.intermediates
    assert not any("layout" in warning for warning in document.warnings)


def test_run_drops_low_confidence_spans(white_page: np.ndarray) -> None:
    pipeline = _mocked_pipeline(
        det_polys=[_rect(10, 10, 100, 30), _rect(10, 50, 200, 70)],
        rec_results=[("keep", 0.9), ("noise", 0.1)],
    )
    document = pipeline.run(white_page).document

    regions = document.pages[0].regions
    assert len(regions) == 1
    assert regions[0].text == "keep"
    assert document.full_text == "keep"
    assert any("low-confidence" in warning for warning in document.warnings)


def test_run_extracts_questions_from_question_paper(white_page: np.ndarray) -> None:
    pipeline = _mocked_pipeline(
        det_polys=[_rect(10, 10, 280, 30), _rect(10, 44, 290, 64)],
        rec_results=[
            ("Q1. What is the SI unit of force?", 0.98),
            ("(a) joule (b) newton (c) watt (d) pascal", 0.96),
        ],
    )

    document = pipeline.run(white_page).document

    assert len(document.questions) == 1
    question = document.questions[0]
    assert question.question_number == "1"
    assert question.question == "What is the SI unit of force?"
    assert question.options == ["joule", "newton", "watt", "pascal"]
    regions = document.pages[0].regions
    assert len(regions) == 1
    assert regions[0].type.value == "question_block"


def test_run_never_raises_on_featureless_image() -> None:
    flat = np.full((300, 400, 3), 128, dtype=np.uint8)
    pipeline = _mocked_pipeline(det_polys=None, rec_results=[])  # detector finds nothing

    document = pipeline.run(flat).document

    assert document.pages[0].regions == []
    assert document.full_text == ""
    assert any("full frame" in warning for warning in document.warnings)


def test_run_detection_failure_degrades_to_warning(white_page: np.ndarray) -> None:
    pipeline = Pipeline()
    pipeline.detector._model = _BoomModel()

    document = pipeline.run(white_page).document

    assert document.pages[0].regions == []
    assert any("text detection failed" in warning for warning in document.warnings)


def test_run_fail_fast_reraises_stage_errors(white_page: np.ndarray) -> None:
    from docint.config import load_config

    config = load_config(overrides={"pipeline": {"fail_fast": True}})
    pipeline = Pipeline(config)
    pipeline.detector._model = _BoomModel()

    with pytest.raises(RuntimeError, match="boom"):
        pipeline.run(white_page)
