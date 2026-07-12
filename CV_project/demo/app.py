"""Gradio demo — every intermediate pipeline stage, with per-stage latency.

Run (after installing requirements)::

    python demo/app.py

Upload a phone photo, pick a language, press Extract: the gallery shows
original -> boundary overlay -> warped -> deskewed -> shadow-free ->
enhanced -> detection boxes, each captioned with its latency. Tabs show the
searchable full text, the parsed questions, the complete DocumentResult
JSON and the raw per-stage timings; a banner flags blurry photos (retake).
"""

from __future__ import annotations

import cv2
import gradio as gr
import numpy as np

from docint.pipeline import Pipeline

#: (intermediates key, caption, timings_ms key) in pipeline order.
STAGE_VIEWS: tuple[tuple[str, str, str], ...] = (
    ("boundary_overlay", "1 · Boundary detection", "preprocess.boundary"),
    ("warped", "2 · Perspective correction", "preprocess.warp"),
    ("deskewed", "3 · Deskew", "preprocess.deskew"),
    ("shadow_free", "4 · Shadow removal", "preprocess.shadow"),
    ("enhanced", "5 · CLAHE / threshold", "preprocess.enhance"),
    ("detections_overlay", "6 · Text detection", "detect"),
)

_pipeline: Pipeline | None = None


def _get_pipeline() -> Pipeline:
    """Build the pipeline once per process; OCR models load on first run."""
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline()
    return _pipeline


def _to_rgb(image: np.ndarray) -> np.ndarray:
    """Gradio displays RGB; the pipeline works in BGR/grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB if image.ndim == 2 else cv2.COLOR_BGR2RGB)


def run_pipeline_ui(image: np.ndarray | None, lang: str):
    """Gradio callback: run the pipeline and fan results out to components."""
    if image is None:
        raise gr.Error("Upload a document photo first.")

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # gr.Image(type="numpy") is RGB
    result = _get_pipeline().run(bgr, lang=lang)
    document = result.document
    timings = document.timings_ms

    gallery: list[tuple[np.ndarray, str]] = [(image, "0 · Original")]
    for key, caption, timing_key in STAGE_VIEWS:
        stage_image = result.intermediates.get(key)
        if stage_image is None:
            continue
        latency = timings.get(timing_key)
        label = f"{caption} — {latency:.0f} ms" if latency is not None else caption
        gallery.append((_to_rgb(stage_image), label))

    if document.retake_photo:
        banner = f"⚠️ **Retake photo** — blur score {document.blur_score:.0f} is below threshold."
    else:
        banner = f"✅ Sharpness OK (blur score {document.blur_score:.0f})."
    if document.warnings:
        banner += "\n\n" + "\n".join(f"- {warning}" for warning in document.warnings)

    questions = [question.model_dump() for question in document.questions]
    return (
        gallery,
        document.model_dump(mode="json"),
        questions,
        document.full_text,
        timings,
        banner,
    )


def build_demo() -> gr.Blocks:
    """Assemble the Blocks UI (declarative — no models load here)."""
    with gr.Blocks(title="docint — Document Intelligence") as demo:
        gr.Markdown("# docint — phone photo → structured text")
        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(label="Document photo", type="numpy")
                lang_in = gr.Radio(choices=["en", "mar", "hin"], value="en", label="Language")
                run_btn = gr.Button("Extract", variant="primary")
                banner_out = gr.Markdown()
            with gr.Column(scale=2):
                gallery_out = gr.Gallery(
                    label="Pipeline stages (captioned with latency)",
                    columns=4,
                    height=380,
                    object_fit="contain",
                )
                with gr.Tab("Full text"):
                    text_out = gr.Textbox(label="full_text", lines=12, show_copy_button=True)
                with gr.Tab("Questions"):
                    questions_out = gr.JSON(label="parsed questions")
                with gr.Tab("Structured JSON"):
                    json_out = gr.JSON(label="DocumentResult")
                with gr.Tab("Timings"):
                    timings_out = gr.JSON(label="ms per stage")
        run_btn.click(
            run_pipeline_ui,
            inputs=[image_in, lang_in],
            outputs=[gallery_out, json_out, questions_out, text_out, timings_out, banner_out],
        )
    return demo


if __name__ == "__main__":
    build_demo().launch()
