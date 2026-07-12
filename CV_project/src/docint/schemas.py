"""Pydantic models for the API and the structured JSON output.

This is the public contract of the pipeline: ``POST /extract`` returns an
:class:`ExtractResponse`, whose :class:`DocumentResult` is the JSON document
described in PROJECT_SPEC.md::

    {pages: [{regions: [{type, bbox, text, confidence}]}],
     questions: [...], full_text: "..."}
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

#: Languages accepted by the recognition stage.
Language = Literal["en", "mar", "hin"]


class RegionType(str, Enum):
    """Layout classes produced by the (rule-based) layout parser."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    QUESTION_BLOCK = "question_block"
    OTHER = "other"


class BBox(BaseModel):
    """Axis-aligned bounding box, pixel coordinates in the deskewed image."""

    x1: float = Field(ge=0, description="Left edge.")
    y1: float = Field(ge=0, description="Top edge.")
    x2: float = Field(ge=0, description="Right edge (>= x1).")
    y2: float = Field(ge=0, description="Bottom edge (>= y1).")


class Region(BaseModel):
    """One classified layout region with its recognized text."""

    type: RegionType
    bbox: BBox
    polygon: list[tuple[float, float]] | None = Field(
        default=None,
        description="Optional detector polygon (vertices clockwise from top-left).",
    )
    text: str = ""
    confidence: float = Field(
        ge=0.0, le=1.0, description="Mean recognition confidence over the region."
    )


class Question(BaseModel):
    """One parsed exam-paper question with its options."""

    question_number: str = Field(description='As printed, e.g. "1", "2a", "१".')
    question: str
    options: list[str] = Field(
        default_factory=list, description="Option texts in printed order, e.g. (a)-(d)."
    )


class Page(BaseModel):
    """A single processed page."""

    page_number: int = Field(default=1, ge=1)
    width: int = Field(gt=0, description="Deskewed page width, px.")
    height: int = Field(gt=0, description="Deskewed page height, px.")
    regions: list[Region] = Field(default_factory=list)


class DocumentResult(BaseModel):
    """Structured output for one input image (the spec's output JSON)."""

    pages: list[Page] = Field(default_factory=list)
    questions: list[Question] = Field(default_factory=list)
    full_text: str = ""
    language: Language = "en"
    warnings: list[str] = Field(
        default_factory=list,
        description='Degradation notes, e.g. "boundary detection failed; used full frame".',
    )
    retake_photo: bool = Field(
        default=False,
        description="True when the blur score is below preprocess.blur.laplacian_threshold.",
    )
    blur_score: float | None = Field(
        default=None, description="Variance of the Laplacian of the input image."
    )
    timings_ms: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Wall-clock milliseconds per stage (docint.pipeline.STAGES keys, plus "
            "namespaced sub-steps like 'preprocess.deskew')."
        ),
    )


class ExtractResponse(BaseModel):
    """Response body of ``POST /extract``."""

    result: DocumentResult
    deskewed_image_png_b64: str | None = Field(
        default=None,
        description="Deskewed page as base64 PNG (present when return_debug_images=true).",
    )
    debug_images_png_b64: dict[str, str] | None = Field(
        default=None,
        description=(
            "Per-stage debug images as base64 PNGs, keyed like "
            "PipelineResult.intermediates (present when return_debug_images=true)."
        ),
    )


class ExtractError(BaseModel):
    """Error body for ``POST /extract`` rejections.

    ``error`` values: ``invalid_image`` (400), ``upload_too_large`` (413),
    ``blurry_image`` (422, with ``blur_score``).
    """

    error: str
    detail: str | None = None
    blur_score: float | None = None
