# docint — Document Intelligence Pipeline

Phone photo of a document in → clean structured JSON + searchable text out.
Handles skew, shadows, low light, printed and handwritten text, in English,
Marathi and Hindi (Devanagari).

**Status: core pipeline complete.** All five stages are implemented and
tested — preprocessing, detection, recognition, rule-based layout
(heading/paragraph/table/question_block) and the question-paper parser.
Still stubs: Tesseract/TrOCR recognition backends and the evaluation
harness. The full specification lives in [PROJECT_SPEC.md](PROJECT_SPEC.md).

## Architecture

```
photo ──▶ 1·preprocess ──▶ 2·detect ──▶ 3·recognize ──▶ 4·layout ──▶ 5·output
          boundary, warp,   text boxes   text + conf     regions,      JSON +
          deskew, enhance,  in reading   per box,        question      deskewed
          shadow, blur ✓    order        en/mar/hin      parsing       image
```

| # | Stage | Module | Default backend | Swappable via |
|---|-------|--------|-----------------|---------------|
| 1 | Preprocess | `src/docint/preprocess.py` | OpenCV (pure functions) | config flags per step |
| 2 | Detection | `src/docint/detect.py` | PaddleOCR DBNet | `TextDetector` registry (CRAFT/EAST later) |
| 3 | Recognition | `src/docint/recognize.py` | PaddleOCR | `TextRecognizer` registry (Tesseract, TrOCR) |
| 4 | Layout | `src/docint/layout.py` | rule-based + question regex | patterns in config |
| 5 | Output | `src/docint/pipeline.py` + `schemas.py` | pydantic `DocumentResult` | — |

Design rules (from the spec):

- Every stage is a pure function or small class — independently callable and testable.
- **No magic numbers**: every threshold lives in [configs/default.yaml](configs/default.yaml).
- **Graceful degradation**: boundary-detection failure falls back to the full
  frame with a warning; a blurry photo returns `retake_photo: true`; a bad
  image never crashes the pipeline.
- Per-stage wall-clock timings (ms) are recorded in `DocumentResult.timings_ms`.
- Heavy deps (paddle, tesseract, torch) import lazily — the package imports
  with numpy + pydantic + PyYAML alone.

## Repo layout

```
CV_project/                  (repo root; package name is docint)
├── src/docint/
│   ├── preprocess.py        # stage 1 — boundary, warp, deskew, enhance, shadow, blur
│   ├── detect.py            # stage 2 — TextDetector interface + PaddleDetector
│   ├── recognize.py         # stage 3 — TextRecognizer + Paddle/Tesseract/TrOCR
│   ├── layout.py            # stage 4 — region classifier + question parser
│   ├── pipeline.py          # orchestration, PipelineResult, stage timings
│   ├── schemas.py           # pydantic contracts (DocumentResult, ExtractResponse…)
│   └── config.py            # YAML loader + deep-merge overrides
├── api/main.py              # FastAPI: POST /extract, GET /health
├── demo/app.py              # Gradio UI — every intermediate stage
├── eval/                    # benchmark.py (CER/WER, pre ON vs OFF) · make_gt.py · RESULTS.md
├── scripts/
│   └── visualize_preprocess.py  # dev tool: side-by-side grid of all stage-1 steps
├── tests/                   # pytest — preprocess suite green, later-stage tests skip-marked
├── configs/default.yaml     # all thresholds/params
├── data/samples/            # gitignored except sample1–3.jpeg
├── Dockerfile               # python:3.11-slim + tesseract eng/mar/hin
└── requirements.txt         # pinned deps
```

## Quickstart

Python 3.11 is the target runtime (the Dockerfile is canonical).

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install --no-deps -e .

pytest                          # Paddle models are mocked — no weights needed
python -m docint.pipeline data/samples/sample1.jpeg --lang en --out out.json
python eval/benchmark.py        # CER/WER + latency report (see Benchmark section)
python scripts/visualize_preprocess.py data/samples/sample1.jpeg   # stage-1 grid
```

## API (FastAPI)

```bash
uvicorn api.main:app --reload   # http://localhost:8000 — OpenAPI docs at /docs
```

```bash
# extract structured JSON from a photo
curl -s -X POST "http://localhost:8000/extract?lang=en" \
     -F "file=@data/samples/sample1.jpeg"

# include the deskewed page + per-stage debug images (base64 PNGs)
curl -s -X POST "http://localhost:8000/extract?lang=en&return_debug_images=true" \
     -F "file=@data/samples/sample1.jpeg"

curl -s http://localhost:8000/health    # {"status":"ok"}
```

Error semantics: a non-decodable upload returns **400**
`{"error": "invalid_image"}`, an oversize upload (over `api.max_upload_mb`)
returns **413**, and a photo whose Laplacian blur score falls below
`preprocess.blur.laplacian_threshold` returns **422**
`{"error": "blurry_image", "blur_score": ...}`. The `Pipeline` is built once
at startup; OCR models load lazily on the first request and stay resident.

## Demo (Gradio)

```bash
python demo/app.py              # http://localhost:7860
```

Upload a photo, pick a language, press Extract — the gallery shows every
stage (original → boundary overlay → warp → deskew → shadow removal →
enhancement → detection boxes), each captioned with its latency, plus tabs
for full text, parsed questions, the DocumentResult JSON and raw timings.

## Docker

```bash
docker build -t docint .
docker run -p 8000:8000 -v paddle-models:/home/appuser/.paddleocr docint
```

The image runs as a non-root `appuser`; the named volume keeps PaddleOCR's
downloaded weights across restarts.

## Benchmark

Current numbers: **[eval/RESULTS.md](eval/RESULTS.md)**.

`eval/benchmark.py` measures OCR quality and speed on your own labelled
photos, which live outside version control:

```
data/benchmark/              # gitignored — your data stays local
  clean_scan/     page1.jpg  page1.gt.txt
  angled_photo/   ...
  low_light/      ...
  handwritten/    ...
```

Workflow:

1. Drop images into the condition folders above.
2. `python eval/make_gt.py` creates a `<image>.gt.txt` template per new
   image — or `python eval/make_gt.py --ocr` pre-fills each template with
   the pipeline's own output so you only correct its mistakes. Non-empty
   files are never overwritten.
3. Type/fix the exact visible text in each `.gt.txt`.
4. `python eval/benchmark.py` prints the report and writes `eval/RESULTS.md`.

Methodology: every image is OCR'd **twice** — once through the full
pipeline (*preprocess ON*) and once feeding the raw image straight to the
detector and recognizer with identical confidence filtering and line
joining (*preprocess OFF*), so the delta isolates exactly what the
preprocessing stage buys per condition. CER and WER are corpus-level
(`jiwer`) over whitespace-normalized, case-sensitive text; unlabelled or
empty-GT images are skipped with a warning. Latency is reported as mean and
p95 per stage across the ON runs, with extra rows timing the ablation
path. The committed RESULTS.md comes from a small synthetic seed set
(rendered pages with exactly known text) — swap in real photos for numbers
that mean something.

## Languages

`lang` parameter: `en` | `mar` | `hin`. PaddleOCR maps `mar`/`hin` to its
`devanagari` recognition model; the Tesseract backend uses the `mar`/`hin`
traineddata packages (already installed in the Docker image). Mappings live
in `configs/default.yaml → recognize.*.lang_map`.
