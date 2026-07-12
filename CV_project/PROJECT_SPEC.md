# PROJECT SPEC — Document Intelligence Pipeline (`docint`)

A production-quality system that takes a phone photo of a document (skewed,
shadowed, low-light, printed or handwritten, English or Marathi/Devanagari)
and returns clean structured output (JSON + searchable text).

---

## Pipeline stages

### 1. Preprocessing (OpenCV)

- Document boundary detection: Canny + contours, with a Hough-line fallback.
- 4-point perspective correction.
- Adaptive thresholding / CLAHE enhancement.
- Deskewing (minAreaRect / Hough).
- Optional shadow removal (dilate + median-blur background estimation).
- Blur detection via Laplacian variance — return a **"retake photo"** flag if
  below threshold.

### 2. Text detection

- PaddleOCR **DBNet** detector as default.
- Design the module with a common interface so **CRAFT** or **EAST** can be
  swapped in later.
- Output: polygons / bounding boxes sorted into **reading order**
  (top-to-bottom, left-to-right).

### 3. Text recognition

- PaddleOCR recognizer as default; pluggable interface for **Tesseract** and
  **TrOCR**.
- Must support language parameter: `"en"`, `"mar"`, `"hin"`.

### 4. Layout parsing

- Classify regions (heading / paragraph / table / question block) — start
  rule-based (position + relative text size).
- Include a **question-paper parser** that segments patterns like
  `Q1 ... (a)...(b)...(c)...(d)` into
  `{question_number, question, options[]}` JSON.

### 5. Output

Structured JSON:

```json
{
  "pages": [{"regions": [{"type": "...", "bbox": "...", "text": "...", "confidence": 0.0}]}],
  "questions": ["..."],
  "full_text": "..."
}
```

plus the deskewed image.

---

## Tech stack

- Python 3.11, OpenCV, PaddleOCR, pytesseract (optional), FastAPI, Gradio,
  pytest, jiwer (CER/WER metrics), Docker.
- **Pin all dependencies** in `requirements.txt`.

---

## Repo structure

```
docint/                  # repo root (here: D:\CV_project)
  src/docint/
    preprocess.py        # each step = separate pure function
    detect.py            # TextDetector interface + PaddleDetector impl
    recognize.py         # TextRecognizer interface + PaddleRecognizer impl
    layout.py            # region classification + question parser
    pipeline.py          # orchestrates stages, returns PipelineResult dataclass
    schemas.py           # pydantic models for API + JSON output
  api/main.py            # FastAPI: POST /extract (multipart image upload)
  demo/app.py            # Gradio UI showing EVERY intermediate stage
  eval/benchmark.py      # CER/WER per condition, latency per stage
  tests/                 # pytest unit tests
  configs/default.yaml   # all thresholds/params — nothing hardcoded
  data/samples/          # test images (gitignored except 2-3 small ones)
  Dockerfile
  README.md
```

---

## Engineering rules

1. Every pipeline stage is **independently callable and testable**.
2. **Type hints everywhere**; docstrings on public functions.
3. **No magic numbers** — all thresholds live in `configs/default.yaml`.
4. **Graceful degradation**: if boundary detection fails, fall back to the
   full frame and set a warning flag in the result; **never crash on a bad
   image**.
5. **Log timing (ms) for each stage** into the `PipelineResult`.
6. Write **unit tests alongside each module** — corner ordering, blur
   detector, and the question-regex parser must have tests.
