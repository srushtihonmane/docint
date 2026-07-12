"""Tests for the benchmark harness — discovery, ablation, report writing.

``benchmark`` and ``make_gt`` are importable via pyproject's pytest
``pythonpath = ["src", "eval", "."]``. OCR models are faked through the
class-level ``_load_model`` seams, so no weights are needed.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

import benchmark
import make_gt
from docint.detect import PaddleDetector
from docint.recognize import PaddleRecognizer


class _FakeDetModel:
    def text_detector(self, img):
        boxes = np.asarray([[[10, 10], [190, 10], [190, 30], [10, 30]]], dtype=np.float32)
        return boxes, 0.01


class _FakeRecModel:
    """Always recognizes the same text (cyclic — ON and OFF paths both call it)."""

    def ocr(self, img, det=True, rec=True, cls=False):
        assert rec and not det
        return [[("hello world", 0.95)]]


@pytest.fixture()
def fake_models(monkeypatch):
    monkeypatch.setattr(PaddleDetector, "_load_model", lambda self: _FakeDetModel())
    monkeypatch.setattr(PaddleRecognizer, "_load_model", lambda self, code: _FakeRecModel())


def _write_png(path) -> None:
    tile = (((np.indices((200, 200)).sum(axis=0) // 8) % 2) * 255).astype(np.uint8)
    assert cv2.imwrite(str(path), np.dstack([tile] * 3))


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def test_discover_pairs_and_skips_unlabelled(tmp_path) -> None:
    condition = tmp_path / "clean_scan"
    condition.mkdir()
    _write_png(condition / "ok.png")
    (condition / "ok.gt.txt").write_text("some text", encoding="utf-8")
    _write_png(condition / "missing.png")  # no gt file
    _write_png(condition / "empty.png")
    (condition / "empty.gt.txt").write_text("   \n", encoding="utf-8")

    samples, warnings = benchmark.discover_samples(tmp_path, ["clean_scan"])

    assert [s.image_path.name for s in samples] == ["ok.png"]
    assert samples[0].condition == "clean_scan"
    assert len(warnings) == 2
    assert any("no missing.gt.txt" in w for w in warnings)
    assert any("ground truth is empty" in w for w in warnings)


def test_discover_flags_unknown_condition_dir(tmp_path) -> None:
    condition = tmp_path / "weird_condition"
    condition.mkdir()
    _write_png(condition / "a.png")
    (condition / "a.gt.txt").write_text("x", encoding="utf-8")

    samples, warnings = benchmark.discover_samples(tmp_path, ["clean_scan"])

    assert len(samples) == 1  # included anyway
    assert any("not in eval.conditions" in w for w in warnings)


def test_discover_missing_root(tmp_path) -> None:
    samples, warnings = benchmark.discover_samples(tmp_path / "nope", ["clean_scan"])
    assert samples == []
    assert len(warnings) == 1


def test_normalize_collapses_whitespace() -> None:
    assert benchmark._normalize("a\n  b\tc ") == "a b c"


# ---------------------------------------------------------------------------
# end-to-end (mocked models)
# ---------------------------------------------------------------------------


def test_benchmark_end_to_end_writes_report(tmp_path, fake_models, capsys) -> None:
    root = tmp_path / "bench"
    for condition, gt_text in (
        ("clean_scan", "hello world"),  # matches the fake -> CER 0
        ("angled_photo", "entirely different reference text"),  # mismatch -> CER > 0
    ):
        condition_dir = root / condition
        condition_dir.mkdir(parents=True)
        _write_png(condition_dir / "img1.png")
        (condition_dir / "img1.gt.txt").write_text(gt_text, encoding="utf-8")
    out = tmp_path / "RESULTS.md"

    rc = benchmark.main(["--data-dir", str(root), "--out", str(out)])

    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert "| clean_scan | 1 | 0.0% | 0.0% |" in text  # perfect ON scores
    assert "angled_photo" in text
    assert "**overall** | 2" in text
    assert "## Latency per stage" in text
    assert "detect (no preprocess)" in text
    assert out.name in capsys.readouterr().err  # "report written to ..."


def test_benchmark_without_data_fails_helpfully(tmp_path, capsys) -> None:
    rc = benchmark.main(["--data-dir", str(tmp_path / "void"), "--out", str(tmp_path / "r.md")])

    assert rc == 1
    assert "make_gt" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# make_gt
# ---------------------------------------------------------------------------


def test_make_gt_creates_empty_templates_and_preserves_existing(tmp_path, capsys) -> None:
    condition = tmp_path / "low_light"
    condition.mkdir()
    _write_png(condition / "new.png")
    _write_png(condition / "done.png")
    (condition / "done.gt.txt").write_text("already transcribed", encoding="utf-8")

    rc = make_gt.main(["--data-dir", str(tmp_path)])

    assert rc == 0
    assert (condition / "new.gt.txt").read_text(encoding="utf-8") == ""
    assert (condition / "done.gt.txt").read_text(encoding="utf-8") == "already transcribed"
    out = capsys.readouterr().out
    assert "new.gt.txt" in out and "done.gt.txt" not in out


def test_make_gt_reports_nothing_to_do(tmp_path, capsys) -> None:
    condition = tmp_path / "clean_scan"
    condition.mkdir()
    _write_png(condition / "a.png")
    (condition / "a.gt.txt").write_text("x", encoding="utf-8")

    rc = make_gt.main(["--data-dir", str(tmp_path)])

    assert rc == 0
    assert "already have" in capsys.readouterr().out
