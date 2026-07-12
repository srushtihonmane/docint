"""Tests for the (mocked) PaddleRecognizer and language handling."""

from __future__ import annotations

import numpy as np
import pytest

from docint.detect import TextBox
from docint.recognize import PaddleRecognizer, _parse_rec_result


def _box(x1: float, y1: float, x2: float, y2: float) -> TextBox:
    polygon = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    return TextBox(polygon=polygon)


class _FakeRecPaddle:
    """Stands in for PaddleOCR in rec-only mode; emits word1, word2, ..."""

    def __init__(self) -> None:
        self.calls = 0

    def ocr(self, img, det=True, rec=True, cls=False):
        assert rec and not det
        self.calls += 1
        return [[(f"word{self.calls}", 0.9)]]


@pytest.fixture()
def page() -> np.ndarray:
    return np.full((100, 300, 3), 255, dtype=np.uint8)


def test_recognizer_constructor_validates_lang(default_cfg: dict) -> None:
    with pytest.raises(ValueError, match="unsupported lang"):
        PaddleRecognizer(default_cfg["recognize"], lang="fr")


def test_recognize_rejects_unsupported_lang_override(default_cfg: dict, page) -> None:
    recognizer = PaddleRecognizer(default_cfg["recognize"])
    with pytest.raises(ValueError, match="unsupported lang"):
        recognizer.recognize(page, [_box(0, 0, 10, 10)], lang="xx")


def test_recognize_empty_boxes_needs_no_model(default_cfg: dict, page) -> None:
    recognizer = PaddleRecognizer(default_cfg["recognize"])
    assert recognizer.recognize(page, []) == []
    assert recognizer._models == {}  # nothing was loaded


def test_paddle_recognizer_spans_in_box_order(monkeypatch, default_cfg: dict, page) -> None:
    fake = _FakeRecPaddle()
    requested_codes: list[str] = []

    def fake_load(self, code: str) -> _FakeRecPaddle:
        requested_codes.append(code)
        return fake

    monkeypatch.setattr(PaddleRecognizer, "_load_model", fake_load)
    recognizer = PaddleRecognizer(default_cfg["recognize"], lang="en")
    boxes = [_box(10, 10, 100, 30), _box(120, 10, 220, 30)]

    spans = recognizer.recognize(page, boxes)

    assert [span.text for span in spans] == ["word1", "word2"]
    assert spans[0].box is boxes[0] and spans[1].box is boxes[1]
    assert all(span.confidence == pytest.approx(0.9) for span in spans)
    assert requested_codes == ["en"]


def test_mar_and_hin_share_one_devanagari_model(monkeypatch, default_cfg: dict, page) -> None:
    fake = _FakeRecPaddle()
    requested_codes: list[str] = []

    def fake_load(self, code: str) -> _FakeRecPaddle:
        requested_codes.append(code)
        return fake

    monkeypatch.setattr(PaddleRecognizer, "_load_model", fake_load)
    recognizer = PaddleRecognizer(default_cfg["recognize"], lang="mar")
    boxes = [_box(10, 10, 100, 30)]

    recognizer.recognize(page, boxes)  # constructor default: mar
    recognizer.recognize(page, boxes, lang="hin")  # per-call override

    assert requested_codes == ["devanagari"]  # mapped once, cached, shared
    assert fake.calls == 2


def test_parse_rec_result_tolerates_shape_variants() -> None:
    assert _parse_rec_result([[("hello", 0.98)]]) == ("hello", 0.98)
    assert _parse_rec_result([("hi", 0.5)]) == ("hi", 0.5)
    assert _parse_rec_result([None]) == ("", 0.0)
    assert _parse_rec_result([]) == ("", 0.0)
    assert _parse_rec_result([[]]) == ("", 0.0)
