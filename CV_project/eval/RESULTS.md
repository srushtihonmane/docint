# docint benchmark results

_Generated 2026-07-11 04:42 - 8 images - lang=en - detector/recognizer: paddle/paddle - data: `D:/CV_project/data/benchmark`_

## Accuracy - CER / WER per condition

Lower is better. "pre ON" is the full pipeline; "pre OFF" feeds the raw
image straight to the detector (same confidence filter and line joining).
A positive CER delta means preprocessing helped.

| Condition | n | CER (pre ON) | WER (pre ON) | CER (pre OFF) | WER (pre OFF) | CER delta (OFF-ON) |
|---|---:|---:|---:|---:|---:|---:|
| clean_scan | 2 | 0.0% | 0.0% | 0.0% | 0.0% | +0.0 pp |
| angled_photo | 2 | 0.0% | 0.0% | 0.0% | 0.0% | +0.0 pp |
| low_light | 2 | 0.0% | 0.0% | 0.0% | 0.0% | +0.0 pp |
| handwritten | 2 | 1.6% | 8.0% | 1.9% | 10.0% | +0.3 pp |
| **overall** | 8 | 0.4% | 2.0% | 0.5% | 2.5% | +0.1 pp |

## Latency per stage

Across the 8 preprocess-ON runs, after one untimed
warmup run (model load excluded); the "no preprocess" rows time the
ablation path on the raw image.

| Stage | mean ms | p95 ms |
|---|---:|---:|
| preprocess | 87 | 96 |
| detect | 224 | 471 |
| recognize | 1248 | 1524 |
| layout | 2 | 2 |
| output | 1 | 1 |
| detect (no preprocess) | 224 | 521 |
| recognize (no preprocess) | 1006 | 1266 |

## Method

Corpus-level CER/WER via jiwer over whitespace-normalized, case-sensitive
text. Dataset layout, labelling workflow and caveats: see the Benchmark
section of the [README](../README.md).
