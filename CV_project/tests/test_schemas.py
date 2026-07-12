"""Contract tests for schemas + config — these run green on the skeleton."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import docint
from docint.schemas import BBox, DocumentResult, Page, Question, Region, RegionType


def test_package_imports_and_has_version() -> None:
    assert docint.__version__ == "0.1.0"


def test_document_result_json_round_trip() -> None:
    doc = DocumentResult(
        pages=[
            Page(
                width=800,
                height=1100,
                regions=[
                    Region(
                        type=RegionType.HEADING,
                        bbox=BBox(x1=10, y1=10, x2=400, y2=60),
                        text="PHYSICS",
                        confidence=0.98,
                    )
                ],
            )
        ],
        questions=[
            Question(
                question_number="1",
                question="What is the SI unit of force?",
                options=["joule", "newton", "watt", "pascal"],
            )
        ],
        full_text="PHYSICS ...",
        language="en",
        timings_ms={"preprocess": 42.0},
    )
    restored = DocumentResult.model_validate_json(doc.model_dump_json())
    assert restored == doc


def test_region_confidence_is_bounded() -> None:
    with pytest.raises(ValidationError):
        Region(
            type=RegionType.PARAGRAPH,
            bbox=BBox(x1=0, y1=0, x2=1, y2=1),
            text="x",
            confidence=1.5,
        )


def test_language_is_restricted_to_supported_codes() -> None:
    with pytest.raises(ValidationError):
        DocumentResult(language="fr")


def test_default_config_covers_every_stage(default_cfg: dict) -> None:
    for section in ("preprocess", "detect", "recognize", "layout", "pipeline", "api", "eval"):
        assert section in default_cfg, f"configs/default.yaml missing section {section!r}"

    assert isinstance(default_cfg["preprocess"]["blur"]["laplacian_threshold"], (int, float))
    assert default_cfg["detect"]["backend"] == "paddle"
    for lang in ("en", "mar", "hin"):
        assert lang in default_cfg["recognize"]["paddle"]["lang_map"]
    parser_cfg = default_cfg["layout"]["question_parser"]
    assert parser_cfg["question_start"] and parser_cfg["option_marker"]
