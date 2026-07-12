"""Stage 4 — layout parsing.

Rule-based region classification plus a question-paper parser.

Classification rules (:func:`classify_regions`), in precedence order:

1. **question_block** — the line starts with a question number
   (``question_parser.question_start``) or carries option markers.
2. **table** — a run of at least ``table.min_rows`` consecutive lines whose
   column x-starts align (>= ``table.min_columns`` columns within
   ``table.x_align_tol_px``).
3. **heading** — line height at or above the ``heading.height_percentile``
   percentile of all line heights AND at least ``heading.min_height_ratio``
   x the median height, with at most ``heading.max_words`` words.
4. **paragraph** — everything else.

Consecutive same-type lines merge into one region when their vertical gap is
at most ``block_merge_gap_ratio`` x the median line height.

The question parser (:func:`parse_questions`) segments exam-paper text like::

    Q1. What is the SI unit of force?
    (a) joule (b) newton (c) watt (d) pascal

into :class:`~docint.schemas.Question` objects. Supported: ``Q1.`` / ``1.`` /
``Q 1)`` / ``Question 3:`` / ``प्र.४`` starts (Devanagari numerals kept as
printed), options as ``(a)``..``(d)``, ``a)``..``d)``, ``A.``..``D.`` or
``(अ)``..``(ड)``, multi-line question text, and malformed input —
best-effort partial results, never an exception.

All patterns and thresholds live in the ``layout`` section of
``configs/default.yaml``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping, Sequence

import numpy as np

from docint.detect import group_into_lines
from docint.recognize import RecognizedSpan
from docint.schemas import BBox, Question, Region, RegionType

#: Anything with a ``.text`` attribute (Region, RecognizedSpan, ...) or raw text.
TextSource = "str | Sequence[Any]"


# ---------------------------------------------------------------------------
# config plumbing
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _default_layout_cfg() -> dict[str, Any]:
    """The ``layout`` section of configs/default.yaml (loaded once)."""
    from docint.config import load_config

    return load_config()["layout"]


def _layout_section(cfg: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return cfg if cfg is not None else _default_layout_cfg()


def _parser_cfg(cfg: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Accept the ``layout`` section or its ``question_parser`` subsection."""
    section = _layout_section(cfg)
    return section.get("question_parser", section)


@lru_cache(maxsize=8)
def _compile_start(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE | re.MULTILINE)


@lru_cache(maxsize=8)
def _compile_option(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)  # case-sensitive by design — see default.yaml


def _first_group(match: re.Match[str]) -> str | None:
    """The regexes use alternation groups; take whichever one matched."""
    return next((group for group in match.groups() if group), None)


def _collapse(text: str) -> str:
    """Collapse runs of whitespace (incl. newlines) to single spaces."""
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# region classification
# ---------------------------------------------------------------------------


@dataclass
class _Line:
    """One text line assembled from reading-ordered spans."""

    spans: list[RecognizedSpan]
    text: str
    top: float
    bottom: float
    left: float
    right: float

    @property
    def height(self) -> float:
        return max(self.bottom - self.top, 1e-6)


def _build_lines(spans: Sequence[RecognizedSpan], overlap_frac: float) -> list[_Line]:
    span_by_box = {span.box: span for span in spans}  # TextBox hashes by identity
    lines: list[_Line] = []
    for boxes in group_into_lines([span.box for span in spans], overlap_frac):
        line_spans = [span_by_box[box] for box in boxes]
        bboxes = [box.bbox for box in boxes]
        lines.append(
            _Line(
                spans=line_spans,
                text=" ".join(span.text for span in line_spans if span.text).strip(),
                top=min(b[1] for b in bboxes),
                bottom=max(b[3] for b in bboxes),
                left=min(b[0] for b in bboxes),
                right=max(b[2] for b in bboxes),
            )
        )
    return lines


def _aligned_column_count(starts_a: list[float], starts_b: list[float], tol: float) -> int:
    """How many column x-starts two (sorted) lines share, within ``tol`` px."""
    i = j = count = 0
    while i < len(starts_a) and j < len(starts_b):
        delta = starts_a[i] - starts_b[j]
        if abs(delta) <= tol:
            count += 1
            i += 1
            j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    return count


def _table_line_indices(lines: list[_Line], cfg: Mapping[str, Any]) -> set[int]:
    """Indices of lines that belong to a column-aligned run (a table)."""
    table_cfg = cfg["table"]
    min_columns = int(table_cfg["min_columns"])
    min_rows = int(table_cfg["min_rows"])
    tol = float(table_cfg["x_align_tol_px"])

    starts = [sorted(span.box.bbox[0] for span in line.spans) for line in lines]
    aligned_with_next = [
        len(starts[i]) >= min_columns
        and len(starts[i + 1]) >= min_columns
        and _aligned_column_count(starts[i], starts[i + 1], tol) >= min_columns
        for i in range(len(lines) - 1)
    ]

    table_lines: set[int] = set()
    run_start: int | None = None
    for index, aligned in enumerate(aligned_with_next + [False]):
        if aligned and run_start is None:
            run_start = index
        elif not aligned and run_start is not None:
            run_line_count = index - run_start + 1  # flags i..index-1 cover lines run_start..index
            if run_line_count >= min_rows:
                table_lines.update(range(run_start, index + 1))
            run_start = None
    return table_lines


def _block_region(block: list[_Line], region_type: RegionType) -> Region:
    """Union a run of same-type lines into one Region (lines joined by \\n)."""
    spans = [span for line in block for span in line.spans]
    confidences = [span.confidence for span in spans] or [0.0]
    return Region(
        type=region_type,
        bbox=BBox(
            x1=max(min(line.left for line in block), 0.0),
            y1=max(min(line.top for line in block), 0.0),
            x2=max(max(line.right for line in block), 0.0),
            y2=max(max(line.bottom for line in block), 0.0),
        ),
        polygon=None,  # block-level region, not a single detector quad
        text="\n".join(line.text for line in block if line.text),
        confidence=min(max(float(np.mean(confidences)), 0.0), 1.0),
    )


def classify_regions(
    spans: Sequence[RecognizedSpan],
    image_shape: tuple[int, int],
    cfg: Mapping[str, Any] | None = None,
) -> list[Region]:
    """Group recognized spans into lines, tag each line, merge into regions.

    Rules and precedence are documented in the module docstring; every
    threshold comes from the ``layout`` config section.

    Args:
        spans: Recognition output in reading order (boxes with text).
        image_shape: ``(height, width)`` of the page — reserved for future
            position-based rules.
        cfg: The ``layout`` config section; None loads the shipped default.

    Returns:
        Classified regions in reading order; empty-text blocks are dropped.
    """
    cfg = _layout_section(cfg)
    if not spans:
        return []

    lines = _build_lines(spans, float(cfg["line_overlap_frac"]))
    if not lines:
        return []

    heights = np.array([line.height for line in lines])
    median_height = max(float(np.median(heights)), 1e-6)
    heading_cfg = cfg["heading"]
    percentile_height = float(np.percentile(heights, float(heading_cfg["height_percentile"])))

    start_re = _compile_start(_parser_cfg(cfg)["question_start"])
    option_re = _compile_option(_parser_cfg(cfg)["option_marker"])
    table_lines = _table_line_indices(lines, cfg)

    def _line_type(index: int, line: _Line) -> RegionType:
        if start_re.match(line.text) or option_re.search(line.text):
            return RegionType.QUESTION_BLOCK
        if index in table_lines:
            return RegionType.TABLE
        if (
            line.text
            and line.height >= percentile_height
            and line.height >= float(heading_cfg["min_height_ratio"]) * median_height
            and len(line.text.split()) <= int(heading_cfg["max_words"])
        ):
            return RegionType.HEADING
        return RegionType.PARAGRAPH

    types = [_line_type(index, line) for index, line in enumerate(lines)]

    gap_limit = float(cfg["block_merge_gap_ratio"]) * median_height
    regions: list[Region] = []
    block: list[_Line] = [lines[0]]
    block_type = types[0]
    for line, line_type in zip(lines[1:], types[1:]):
        if line_type == block_type and (line.top - block[-1].bottom) <= gap_limit:
            block.append(line)
        else:
            regions.append(_block_region(block, block_type))
            block, block_type = [line], line_type
    regions.append(_block_region(block, block_type))

    return [region for region in regions if region.text]


# ---------------------------------------------------------------------------
# question parsing
# ---------------------------------------------------------------------------


def parse_questions(
    source: str | Sequence[Any], cfg: Mapping[str, Any] | None = None
) -> list[Question]:
    """Segment question-paper content into structured questions.

    Accepts either plain reading-ordered text or a sequence of objects with
    a ``.text`` attribute (:class:`~docint.schemas.Region`,
    :class:`~docint.recognize.RecognizedSpan`, ...) whose texts are joined
    line-wise. The text is split at question starts
    (``question_parser.question_start``); each question's tail is split at
    option markers (``question_parser.option_marker``), keeping at most
    ``question_parser.max_options`` non-empty options.

    Malformed input degrades to partial results — entries with neither
    question text nor options are dropped, stray markers become best-effort
    option texts, and nothing here raises for bad text.

    Args:
        source: Full text, or regions/spans to take text from.
        cfg: The ``layout`` section (or its ``question_parser`` subsection);
            None loads the shipped default.

    Returns:
        Questions in document order; ``[]`` when nothing question-like is
        found (ordinary documents).
    """
    parser_cfg = _parser_cfg(cfg)
    if isinstance(source, str):
        text = source
    else:
        text = "\n".join(getattr(item, "text", "") for item in source if getattr(item, "text", ""))
    if not text or not text.strip():
        return []

    start_re = _compile_start(parser_cfg["question_start"])
    option_re = _compile_option(parser_cfg["option_marker"])
    max_options = int(parser_cfg["max_options"])

    starts = list(start_re.finditer(text))
    questions: list[Question] = []
    for index, start in enumerate(starts):
        number = _first_group(start) or ""
        segment_end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        body = text[start.end() : segment_end]

        markers = list(option_re.finditer(body))
        if markers:
            question_text = _collapse(body[: markers[0].start()])
            options = []
            for j, marker in enumerate(markers):
                option_end = markers[j + 1].start() if j + 1 < len(markers) else len(body)
                option_text = _collapse(body[marker.end() : option_end]).rstrip(",;")
                if option_text:
                    options.append(option_text)
            options = options[:max_options]
        else:
            question_text = _collapse(body)
            options = []

        if question_text or options:
            questions.append(
                Question(question_number=number, question=question_text, options=options)
            )
    return questions
