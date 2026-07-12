"""Create ground-truth template files for benchmark images.

Scans ``data/benchmark/<condition>/`` for images without a matching
``<stem>.gt.txt`` and creates one. By default the template is empty — open
it and type the exact text visible in the image. With ``--ocr`` the
template is pre-filled with the pipeline's own output, so you only fix its
mistakes (much faster for long pages). Existing non-empty files are never
touched; ``--ocr`` also fills files that exist but are still empty.

Run::

    python eval/make_gt.py            # empty templates
    python eval/make_gt.py --ocr      # pre-filled with OCR output (slower)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from a checkout without `pip install -e .`.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from docint.config import load_config

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="Benchmark dataset root")
    parser.add_argument(
        "--ocr", action="store_true", help="Pre-fill templates with the pipeline's OCR output"
    )
    parser.add_argument("--lang", choices=("en", "mar", "hin"), default="en")
    parser.add_argument("--config", type=Path, default=None, help="Alternative config YAML")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    data_dir = args.data_dir or REPO_ROOT / cfg["eval"]["benchmark_dir"]
    if not data_dir.is_dir():
        print(f"benchmark directory {data_dir} does not exist — create it first", file=sys.stderr)
        return 1

    pipeline = None
    if args.ocr:
        from docint.pipeline import Pipeline

        pipeline = Pipeline(config=cfg)

    created: list[Path] = []
    for condition_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for image_path in sorted(condition_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            gt_path = image_path.with_name(image_path.stem + ".gt.txt")
            if gt_path.exists() and gt_path.read_text(encoding="utf-8").strip():
                continue  # already transcribed — never overwrite
            if pipeline is not None:
                print(f"OCR: {condition_dir.name}/{image_path.name}", file=sys.stderr)
                text = pipeline.run_path(image_path, lang=args.lang).document.full_text
            else:
                text = ""
            gt_path.write_text(text, encoding="utf-8")
            created.append(gt_path)

    if created:
        for path in created:
            print(f"created {path.relative_to(data_dir)}")
        action = "review and fix the OCR draft" if args.ocr else "type the exact visible text"
        print(f"{len(created)} template(s) created — {action} in each file.")
    else:
        print("all images already have ground-truth files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
