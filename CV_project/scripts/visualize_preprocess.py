"""Save a side-by-side grid of every preprocessing stage for one image.

Usage::

    python scripts/visualize_preprocess.py data/samples/sample1.jpeg
    python scripts/visualize_preprocess.py photo.jpg -o grid.png --tile-height 420

Prints corners / skew / blur / warnings / per-step timings to stdout and
writes the annotated grid PNG (default: ``<input stem>_preprocess.png`` in
the current directory).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# Allow running from a checkout without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np

from docint.config import load_config
from docint.preprocess import PreprocessResult, preprocess

#: images-dict key -> timings-dict key, for tile captions.
_TIMING_KEY = {
    "boundary_overlay": "boundary",
    "warped": "warp",
    "deskewed": "deskew",
    "shadow_free": "shadow",
    "enhanced": "enhance",
}

# Presentation-only constants (not pipeline thresholds).
LABEL_BAR_PX = 28
GRID_COLUMNS = 3
BACKGROUND_BGR = (24, 24, 24)


def _as_tile(image: np.ndarray, height: int) -> np.ndarray:
    """BGR copy resized to a common tile height (aspect preserved)."""
    bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    scale = height / bgr.shape[0]
    width = max(1, round(bgr.shape[1] * scale))
    return cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)


def build_grid(original: np.ndarray, result: PreprocessResult, tile_height: int) -> np.ndarray:
    """Compose original + every intermediate into one labeled grid image."""
    tiles: list[tuple[str, np.ndarray]] = [("original", _as_tile(original, tile_height))]
    for key, image in result.images.items():
        timing = result.timings_ms.get(_TIMING_KEY.get(key, ""))
        label = f"{key} ({timing:.0f} ms)" if timing is not None else key
        tiles.append((label, _as_tile(image, tile_height)))

    cell_w = max(tile.shape[1] for _, tile in tiles)
    cell_h = tile_height + LABEL_BAR_PX
    rows = math.ceil(len(tiles) / GRID_COLUMNS)
    canvas = np.full((rows * cell_h, GRID_COLUMNS * cell_w, 3), BACKGROUND_BGR, dtype=np.uint8)

    for index, (label, tile) in enumerate(tiles):
        row, col = divmod(index, GRID_COLUMNS)
        y0 = row * cell_h
        x0 = col * cell_w + (cell_w - tile.shape[1]) // 2
        cv2.putText(
            canvas,
            label,
            (col * cell_w + 8, y0 + LABEL_BAR_PX - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        canvas[y0 + LABEL_BAR_PX : y0 + LABEL_BAR_PX + tile.shape[0], x0 : x0 + tile.shape[1]] = tile
    return canvas


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("image", type=Path, help="Input photo (JPEG/PNG/WebP)")
    parser.add_argument("-o", "--out", type=Path, default=None, help="Output PNG path")
    parser.add_argument("--config", type=Path, default=None, help="Alternative config YAML")
    parser.add_argument("--tile-height", type=int, default=360, help="Tile height in px")
    args = parser.parse_args(argv)

    original = cv2.imread(str(args.image))
    if original is None:
        print(f"error: could not read image {args.image}", file=sys.stderr)
        return 2

    result = preprocess(original, load_config(args.config))

    if result.corners is not None:
        corner_text = ", ".join(f"({x:.0f},{y:.0f})" for x, y in result.corners)
        print(f"corners:      found  [{corner_text}]")
    else:
        print("corners:      NOT FOUND (full frame fallback)")
    print(f"skew angle:   {result.skew_angle_deg:+.2f} deg")
    print(f"blur score:   {result.blur_score:.1f}  ->  retake_photo={result.retake_photo}")
    for warning in result.warnings:
        print(f"warning:      {warning}")
    print("timings (ms): " + ", ".join(f"{k}={v:.1f}" for k, v in result.timings_ms.items()))

    out = args.out or Path(f"{args.image.stem}_preprocess.png")
    cv2.imwrite(str(out), build_grid(original, result, args.tile_height))
    print(f"grid saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
