"""CER / WER benchmark across capture conditions, with and without preprocessing.

Dataset layout (gitignored — your data stays local)::

    data/benchmark/
      clean_scan/     page1.jpg  page1.gt.txt
      angled_photo/   ...
      low_light/      ...
      handwritten/    ...

Each image needs a UTF-8 ``<stem>.gt.txt`` transcription next to it — create
templates with ``python eval/make_gt.py`` (``--ocr`` pre-fills them).
Images with a missing or empty ground-truth file are skipped with a warning.

Run::

    python eval/benchmark.py                 # writes + prints eval/RESULTS.md
    python eval/benchmark.py --lang en --data-dir data/benchmark --out eval/RESULTS.md

Every sample is OCR'd twice:

* **preprocess ON** — the production path (:meth:`docint.pipeline.Pipeline.run`).
* **preprocess OFF** — the raw image goes straight to the detector and
  recognizer, with the same confidence filter and line joining, isolating
  what the preprocessing stage buys.

Metrics: corpus-level CER and WER per condition (``jiwer``, whitespace-
normalized, case-sensitive) plus per-stage latency mean and p95.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Allow running from a checkout without `pip install -e .`.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from docint.config import load_config
from docint.pipeline import STAGES, Pipeline, stage_timer
from docint.preprocess import Image
from docint.recognize import join_spans
from docint.schemas import Language

#: Image files considered benchmark samples.
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")

#: Latency rows for the ablation path (kept distinct from the ON stages).
OFF_DETECT = "detect (no preprocess)"
OFF_RECOGNIZE = "recognize (no preprocess)"


@dataclass(frozen=True)
class Sample:
    """One labelled benchmark image."""

    image_path: Path
    gt_path: Path
    condition: str


@dataclass
class ConditionStats:
    """Corpus-level accuracy for one condition (or 'overall')."""

    condition: str
    n: int
    cer_on: float
    wer_on: float
    cer_off: float
    wer_off: float


@dataclass
class BenchmarkReport:
    """Everything the markdown report is rendered from."""

    per_condition: list[ConditionStats]
    overall: ConditionStats
    latency_ms: dict[str, tuple[float, float]]  # stage -> (mean, p95)
    meta: dict[str, str]


def discover_samples(data_dir: Path, conditions: list[str]) -> tuple[list[Sample], list[str]]:
    """Find image + ground-truth pairs under ``data_dir``.

    Configured conditions come first (in config order), then any extra
    directories found on disk (flagged with a warning).

    Returns:
        ``(samples, warnings)`` — warnings cover missing/empty ground truth
        and unknown condition directories; such images are skipped.
    """
    warnings: list[str] = []
    if not data_dir.is_dir():
        return [], [f"benchmark directory {data_dir} does not exist"]

    found = {p.name: p for p in data_dir.iterdir() if p.is_dir()}
    ordered = [found[c] for c in conditions if c in found]
    ordered += [d for name, d in sorted(found.items()) if name not in conditions]

    samples: list[Sample] = []
    for condition_dir in ordered:
        if condition_dir.name not in conditions:
            warnings.append(
                f"condition '{condition_dir.name}' is not in eval.conditions — including anyway"
            )
        for image_path in sorted(condition_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            gt_path = image_path.with_name(image_path.stem + ".gt.txt")
            if not gt_path.exists():
                warnings.append(
                    f"skipping {image_path.name} ({condition_dir.name}): no {gt_path.name} "
                    "— run eval/make_gt.py"
                )
                continue
            if not gt_path.read_text(encoding="utf-8").strip():
                warnings.append(
                    f"skipping {image_path.name} ({condition_dir.name}): ground truth is empty"
                )
                continue
            samples.append(Sample(image_path, gt_path, condition_dir.name))
    return samples, warnings


def _normalize(text: str) -> str:
    """Whitespace-normalize (case-sensitive) for CER/WER comparison."""
    return " ".join(text.split())


def run_without_preprocess(
    pipeline: Pipeline, image: Image, lang: Language, timings_ms: dict[str, float]
) -> str:
    """The ablation path: raw image -> detector -> recognizer -> text.

    Applies the same ``recognize.min_confidence`` filter and line joining as
    the production path, so the comparison isolates preprocessing itself.
    """
    with stage_timer(timings_ms, "detect"):
        boxes = pipeline.detector.detect(image)
    with stage_timer(timings_ms, "recognize"):
        spans = pipeline.recognizer.recognize(image, boxes, lang) if boxes else []
    min_confidence = float(pipeline.cfg["recognize"]["min_confidence"])
    spans = [span for span in spans if span.confidence >= min_confidence]
    return join_spans(spans, float(pipeline.cfg["detect"]["reading_order"]["line_overlap_frac"]))


def _stats(condition: str, refs: list[str], on: list[str], off: list[str]) -> ConditionStats:
    import jiwer  # heavy-ish; keep module import light

    return ConditionStats(
        condition=condition,
        n=len(refs),
        cer_on=float(jiwer.cer(refs, on)),
        wer_on=float(jiwer.wer(refs, on)),
        cer_off=float(jiwer.cer(refs, off)),
        wer_off=float(jiwer.wer(refs, off)),
    )


def evaluate(samples: list[Sample], pipeline: Pipeline, lang: Language) -> BenchmarkReport:
    """Run both pipeline variants on every sample and aggregate metrics.

    One untimed warmup run precedes the measured loop so lazy model loading
    doesn't contaminate the latency stats.
    """
    import cv2

    refs: dict[str, list[str]] = defaultdict(list)
    hyps_on: dict[str, list[str]] = defaultdict(list)
    hyps_off: dict[str, list[str]] = defaultdict(list)
    latencies: dict[str, list[float]] = defaultdict(list)

    warmup = cv2.imread(str(samples[0].image_path))
    if warmup is not None:
        print("warmup run (model load excluded from latency stats)...", file=sys.stderr)
        pipeline.run(warmup, lang=lang)

    for index, sample in enumerate(samples, start=1):
        image = cv2.imread(str(sample.image_path))
        if image is None:
            print(f"warning: could not read {sample.image_path}; skipped", file=sys.stderr)
            continue
        print(
            f"[{index}/{len(samples)}] {sample.condition}/{sample.image_path.name}",
            file=sys.stderr,
        )
        reference = _normalize(sample.gt_path.read_text(encoding="utf-8"))

        result = pipeline.run(image, lang=lang)
        for stage in STAGES:
            latencies[stage].append(result.document.timings_ms.get(stage, 0.0))

        off_timings: dict[str, float] = {}
        hyp_off = run_without_preprocess(pipeline, image, lang, off_timings)
        latencies[OFF_DETECT].append(off_timings["detect"])
        latencies[OFF_RECOGNIZE].append(off_timings["recognize"])

        refs[sample.condition].append(reference)
        hyps_on[sample.condition].append(_normalize(result.document.full_text))
        hyps_off[sample.condition].append(_normalize(hyp_off))

    per_condition = [
        _stats(condition, refs[condition], hyps_on[condition], hyps_off[condition])
        for condition in refs
    ]
    all_refs = [r for values in refs.values() for r in values]
    all_on = [h for values in hyps_on.values() for h in values]
    all_off = [h for values in hyps_off.values() for h in values]
    overall = _stats("overall", all_refs, all_on, all_off)

    latency_ms = {
        stage: (float(np.mean(values)), float(np.percentile(values, 95)))
        for stage, values in latencies.items()
    }
    meta = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "images": str(overall.n),
        "lang": lang,
        "detector": pipeline.detector.name,
        "recognizer": pipeline.recognizer.name,
    }
    return BenchmarkReport(per_condition, overall, latency_ms, meta)


def render_markdown(report: BenchmarkReport, data_dir: Path) -> str:
    """Render the report as the RESULTS.md document."""

    def pct(value: float) -> str:
        return f"{value * 100:.1f}%"

    lines = [
        "# docint benchmark results",
        "",
        f"_Generated {report.meta['generated']} - {report.meta['images']} images - "
        f"lang={report.meta['lang']} - detector/recognizer: {report.meta['detector']}/"
        f"{report.meta['recognizer']} - data: `{data_dir.as_posix()}`_",
        "",
        "## Accuracy - CER / WER per condition",
        "",
        "Lower is better. \"pre ON\" is the full pipeline; \"pre OFF\" feeds the raw",
        "image straight to the detector (same confidence filter and line joining).",
        "A positive CER delta means preprocessing helped.",
        "",
        "| Condition | n | CER (pre ON) | WER (pre ON) | CER (pre OFF) | WER (pre OFF) | CER delta (OFF-ON) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in [*report.per_condition, report.overall]:
        bold = "**" if row.condition == "overall" else ""
        delta_pp = (row.cer_off - row.cer_on) * 100
        lines.append(
            f"| {bold}{row.condition}{bold} | {row.n} | {pct(row.cer_on)} | {pct(row.wer_on)} "
            f"| {pct(row.cer_off)} | {pct(row.wer_off)} | {delta_pp:+.1f} pp |"
        )

    lines += [
        "",
        "## Latency per stage",
        "",
        f"Across the {report.meta['images']} preprocess-ON runs, after one untimed",
        "warmup run (model load excluded); the \"no preprocess\" rows time the",
        "ablation path on the raw image.",
        "",
        "| Stage | mean ms | p95 ms |",
        "|---|---:|---:|",
    ]
    for stage, (mean_ms, p95_ms) in report.latency_ms.items():
        lines.append(f"| {stage} | {mean_ms:.0f} | {p95_ms:.0f} |")

    lines += [
        "",
        "## Method",
        "",
        "Corpus-level CER/WER via jiwer over whitespace-normalized, case-sensitive",
        "text. Dataset layout, labelling workflow and caveats: see the Benchmark",
        "section of the [README](../README.md).",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="Benchmark dataset root")
    parser.add_argument("--out", type=Path, default=None, help="Markdown report path")
    parser.add_argument("--lang", choices=("en", "mar", "hin"), default="en")
    parser.add_argument("--config", type=Path, default=None, help="Alternative config YAML")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    data_dir = args.data_dir or REPO_ROOT / cfg["eval"]["benchmark_dir"]
    out_path = args.out or REPO_ROOT / "eval" / "RESULTS.md"

    samples, warnings = discover_samples(data_dir, list(cfg["eval"]["conditions"]))
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if not samples:
        print(
            f"no labelled samples under {data_dir}.\n"
            "Add images to data/benchmark/<condition>/ and create ground truth with:\n"
            "    python eval/make_gt.py",
            file=sys.stderr,
        )
        return 1

    report = evaluate(samples, Pipeline(config=cfg), args.lang)
    markdown = render_markdown(report, data_dir)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")
    try:  # Windows consoles may default to a non-UTF-8 code page
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    print(markdown)
    print(f"report written to {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
