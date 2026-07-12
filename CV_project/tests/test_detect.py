"""Tests for reading-order sorting and the (mocked) PaddleDetector."""

from __future__ import annotations

import numpy as np
import pytest

from docint.detect import PaddleDetector, TextBox, group_into_lines, reading_order_sort


def _box(x1: float, y1: float, x2: float, y2: float) -> TextBox:
    polygon = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    return TextBox(polygon=polygon)


class _FakePaddle:
    """Stands in for PaddleOCR in det-only mode.

    Like the real model, the page result is an (N, 4, 2) float ndarray —
    never rely on its truthiness.
    """

    def __init__(self, polys: list | None) -> None:
        self._polys = None if polys is None else np.asarray(polys, dtype=np.float32)
        self.calls = 0

    def ocr(self, img, det=True, rec=True, cls=False):
        assert det and not rec
        self.calls += 1
        return [self._polys]


# ---------------------------------------------------------------------------
# TextBox
# ---------------------------------------------------------------------------


def test_textbox_bbox_is_polygon_envelope() -> None:
    box = _box(10, 20, 110, 45)
    assert box.bbox == (10.0, 20.0, 110.0, 45.0)


# ---------------------------------------------------------------------------
# reading_order_sort / group_into_lines
# ---------------------------------------------------------------------------


def test_reading_order_sort_single_column_ignores_x_jitter() -> None:
    """A ragged single column must stay top-to-bottom, never x-sorted."""
    first = _box(40, 10, 300, 40)
    second = _box(10, 55, 280, 85)  # starts further left than `first`
    third = _box(60, 100, 310, 130)
    assert reading_order_sort([third, first, second], 0.5) == [first, second, third]


def test_reading_order_sort_groups_jittered_lines() -> None:
    line1_left = _box(10, 10, 100, 30)
    line1_right = _box(120, 13, 220, 33)  # 3 px vertical jitter, same line
    line2_left = _box(12, 50, 100, 70)
    line2_right = _box(130, 48, 220, 68)

    shuffled = [line2_right, line1_right, line2_left, line1_left]
    ordered = reading_order_sort(shuffled, 0.5)

    # TextBox uses identity equality (eq=False), so this asserts exact objects.
    assert ordered == [line1_left, line1_right, line2_left, line2_right]


def test_group_into_lines_structure() -> None:
    a, b = _box(10, 10, 100, 30), _box(120, 12, 220, 32)
    c = _box(10, 60, 200, 80)
    lines = group_into_lines([c, b, a], 0.5)
    assert lines == [[a, b], [c]]


def test_reading_order_sort_empty_input() -> None:
    assert reading_order_sort([], 0.5) == []


def test_reading_order_sort_reads_default_config_when_frac_omitted() -> None:
    box = _box(0, 0, 10, 10)
    assert reading_order_sort([box]) == [box]


# ---------------------------------------------------------------------------
# PaddleDetector (mocked model — no weights, no paddle import)
# ---------------------------------------------------------------------------


def test_paddle_detector_lazy_loads_once_and_sorts(monkeypatch, default_cfg: dict) -> None:
    polys = [
        [[120, 12], [220, 12], [220, 32], [120, 32]],  # line 1, right
        [[10, 50], [100, 50], [100, 70], [10, 70]],  # line 2
        [[10, 10], [100, 10], [100, 30], [10, 30]],  # line 1, left
    ]
    fake = _FakePaddle(polys)
    loads = {"count": 0}

    def fake_load(self) -> _FakePaddle:
        loads["count"] += 1
        return fake

    monkeypatch.setattr(PaddleDetector, "_load_model", fake_load)
    detector = PaddleDetector(default_cfg["detect"])

    first = detector.detect(np.zeros((100, 240, 3), dtype=np.uint8))
    second = detector.detect(np.zeros((100, 240), dtype=np.uint8))  # grayscale input ok

    assert loads["count"] == 1  # model created once, reused
    assert fake.calls == 2
    assert len(first) == len(second) == 3
    assert all(isinstance(box, TextBox) for box in first)
    # reading order: line 1 left, line 1 right, line 2
    assert [box.bbox[:2] for box in first] == [(10.0, 10.0), (120.0, 12.0), (10.0, 50.0)]


def test_paddle_detector_returns_empty_when_nothing_found(monkeypatch, default_cfg: dict) -> None:
    monkeypatch.setattr(PaddleDetector, "_load_model", lambda self: _FakePaddle(None))
    detector = PaddleDetector(default_cfg["detect"])
    assert detector.detect(np.zeros((50, 50, 3), dtype=np.uint8)) == []


def test_paddle_detector_prefers_text_detector_engine(monkeypatch, default_cfg: dict) -> None:
    """Real PaddleOCR exposes .text_detector; the buggy .ocr wrapper is bypassed."""

    class _FakeEngineModel:
        def __init__(self) -> None:
            self.engine_calls = 0

        def text_detector(self, img):
            self.engine_calls += 1
            boxes = np.asarray([[[10, 10], [100, 10], [100, 30], [10, 30]]], dtype=np.float32)
            return boxes, 0.01

        def ocr(self, *args, **kwargs):
            raise AssertionError("wrapper must not be used when text_detector exists")

    fake = _FakeEngineModel()
    monkeypatch.setattr(PaddleDetector, "_load_model", lambda self: fake)
    detector = PaddleDetector(default_cfg["detect"])

    boxes = detector.detect(np.zeros((50, 120, 3), dtype=np.uint8))

    assert fake.engine_calls == 1
    assert len(boxes) == 1
    assert boxes[0].bbox == (10.0, 10.0, 100.0, 30.0)
