"""Tests for layout classification and the question-paper parser."""

from __future__ import annotations

import numpy as np
import pytest

from docint import layout
from docint.detect import TextBox
from docint.recognize import RecognizedSpan
from docint.schemas import BBox, Region, RegionType


def _span(x1: float, y1: float, x2: float, y2: float, text: str, conf: float = 0.9) -> RecognizedSpan:
    polygon = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    return RecognizedSpan(box=TextBox(polygon=polygon), text=text, confidence=conf)


@pytest.fixture()
def layout_cfg(default_cfg: dict) -> dict:
    return default_cfg["layout"]


# ===========================================================================
# parse_questions
# ===========================================================================

SAMPLE_EN = """PHYSICS - SECTION A

Q1. What is the SI unit of force?
(a) joule (b) newton (c) watt (d) pascal

Q2. Light travels fastest in
(a) glass
(b) water
(c) vacuum
(d) diamond
"""

SAMPLE_MR = """प्र.१ महाराष्ट्राची राजधानी कोणती?
(अ) पुणे (ब) मुंबई (क) नागपूर (ड) नाशिक
"""


def test_parse_questions_english_mcq(layout_cfg: dict) -> None:
    questions = layout.parse_questions(SAMPLE_EN, layout_cfg)

    assert [q.question_number for q in questions] == ["1", "2"]

    q1 = questions[0]
    assert q1.question == "What is the SI unit of force?"
    assert q1.options == ["joule", "newton", "watt", "pascal"]

    q2 = questions[1]
    assert q2.question == "Light travels fastest in"
    assert q2.options == ["glass", "water", "vacuum", "diamond"]


def test_parse_questions_devanagari_numerals_and_options(layout_cfg: dict) -> None:
    questions = layout.parse_questions(SAMPLE_MR, layout_cfg)

    assert len(questions) == 1
    assert questions[0].question_number == "१"  # kept as printed
    assert questions[0].question == "महाराष्ट्राची राजधानी कोणती?"
    assert questions[0].options == ["पुणे", "मुंबई", "नागपूर", "नाशिक"]


def test_parse_questions_bare_number_prefix(layout_cfg: dict) -> None:
    text = "1. First question?\n(a) x (b) y\n2. Second question?\n(a) p (b) q\n"
    questions = layout.parse_questions(text, layout_cfg)

    assert [q.question_number for q in questions] == ["1", "2"]
    assert questions[0].options == ["x", "y"]
    assert questions[1].question == "Second question?"


def test_parse_questions_spaced_prefix_and_bare_paren_options(layout_cfg: dict) -> None:
    text = "Q 1) Choose the correct answer\na) one\nb) two\nc) three\n"
    questions = layout.parse_questions(text, layout_cfg)

    assert len(questions) == 1
    assert questions[0].question_number == "1"
    assert questions[0].question == "Choose the correct answer"
    assert questions[0].options == ["one", "two", "three"]


def test_parse_questions_uppercase_dot_options(layout_cfg: dict) -> None:
    text = "Q1. Pick one\nA. alpha\nB. beta\nC. gamma\nD. delta\n"
    questions = layout.parse_questions(text, layout_cfg)

    assert len(questions) == 1
    assert questions[0].options == ["alpha", "beta", "gamma", "delta"]


def test_parse_questions_multiline_question_text_is_collapsed(layout_cfg: dict) -> None:
    text = "Q3. Which of the following\nis a scalar\nquantity?\n(a) force (b) speed\n"
    questions = layout.parse_questions(text, layout_cfg)

    assert len(questions) == 1
    assert questions[0].question == "Which of the following is a scalar quantity?"
    assert questions[0].options == ["force", "speed"]


def test_parse_questions_without_options(layout_cfg: dict) -> None:
    text = "Q7. Explain Newton's second law with an example.\n"
    questions = layout.parse_questions(text, layout_cfg)

    assert len(questions) == 1
    assert questions[0].question_number == "7"
    assert questions[0].options == []


def test_parse_questions_plain_document_yields_empty_list(layout_cfg: dict) -> None:
    plain = "This is an ordinary paragraph with no question numbering at all."
    assert layout.parse_questions(plain, layout_cfg) == []


def test_parse_questions_empty_and_whitespace_input(layout_cfg: dict) -> None:
    assert layout.parse_questions("", layout_cfg) == []
    assert layout.parse_questions("   \n\t  ", layout_cfg) == []


def test_parse_questions_malformed_input_gives_partial_results(layout_cfg: dict) -> None:
    # Truncated option marker, stray marker with no text, dangling question number.
    text = "Q1. What is X? (a) foo (b\nQ2. (a) (b) bar\nQ3."
    questions = layout.parse_questions(text, layout_cfg)  # must not raise

    assert [q.question_number for q in questions] == ["1", "2"]  # Q3 has no content -> dropped
    assert questions[0].question == "What is X?"
    assert questions[0].options[0].startswith("foo")  # best-effort tail
    assert questions[1].options == ["bar"]  # empty (a) dropped, (b) kept


def test_parse_questions_truncates_to_max_options(layout_cfg: dict) -> None:
    options = " ".join(f"({letter}) opt{letter}" for letter in "abcdabc")  # 7 markers
    questions = layout.parse_questions(f"Q1. Too many options {options}", layout_cfg)

    assert len(questions) == 1
    assert len(questions[0].options) == layout_cfg["question_parser"]["max_options"]


def test_parse_questions_accepts_region_objects(layout_cfg: dict) -> None:
    regions = [
        Region(
            type=RegionType.QUESTION_BLOCK,
            bbox=BBox(x1=0, y1=0, x2=100, y2=20),
            text="Q1. Capital of India?",
            confidence=0.9,
        ),
        Region(
            type=RegionType.QUESTION_BLOCK,
            bbox=BBox(x1=0, y1=30, x2=100, y2=50),
            text="(a) Mumbai (b) Delhi",
            confidence=0.9,
        ),
    ]
    questions = layout.parse_questions(regions, layout_cfg)

    assert len(questions) == 1
    assert questions[0].question == "Capital of India?"
    assert questions[0].options == ["Mumbai", "Delhi"]


# ===========================================================================
# classify_regions
# ===========================================================================


def test_classify_empty_spans(layout_cfg: dict) -> None:
    assert layout.classify_regions([], (400, 300), layout_cfg) == []


def test_classify_heading_by_relative_height(layout_cfg: dict) -> None:
    spans = [
        _span(10, 10, 290, 44, "ANNUAL REPORT"),  # 34 px tall
        _span(10, 60, 290, 74, "This is the first body line of text."),  # 14 px
        _span(10, 80, 290, 94, "And this is the second body line."),  # 14 px
    ]
    regions = layout.classify_regions(spans, (400, 300), layout_cfg)

    assert [r.type for r in regions] == [RegionType.HEADING, RegionType.PARAGRAPH]
    assert regions[0].text == "ANNUAL REPORT"
    # adjacent same-type body lines merged into one paragraph block
    assert regions[1].text == "This is the first body line of text.\nAnd this is the second body line."


def test_classify_table_by_column_alignment(layout_cfg: dict) -> None:
    spans = [
        _span(10, 10, 80, 26, "Name"), _span(150, 10, 240, 26, "Marks"),
        _span(12, 34, 82, 50, "Asha"), _span(149, 34, 242, 50, "92"),
        _span(11, 58, 80, 74, "Ravi"), _span(151, 58, 238, 74, "88"),
    ]
    regions = layout.classify_regions(spans, (400, 300), layout_cfg)

    assert len(regions) == 1
    assert regions[0].type == RegionType.TABLE
    assert regions[0].text == "Name Marks\nAsha 92\nRavi 88"


def test_classify_question_block_lines_merge(layout_cfg: dict) -> None:
    spans = [
        _span(10, 10, 200, 28, "Q1. What is force?"),
        _span(10, 36, 220, 54, "(a) push (b) pull"),
    ]
    regions = layout.classify_regions(spans, (400, 300), layout_cfg)

    assert len(regions) == 1
    assert regions[0].type == RegionType.QUESTION_BLOCK
    assert regions[0].text == "Q1. What is force?\n(a) push (b) pull"


def test_classify_uniform_page_has_no_headings(layout_cfg: dict) -> None:
    spans = [
        _span(10, 10 + i * 30, 290, 26 + i * 30, f"Body line number {i} of the page.")
        for i in range(4)
    ]
    regions = layout.classify_regions(spans, (400, 300), layout_cfg)

    assert all(region.type == RegionType.PARAGRAPH for region in regions)
