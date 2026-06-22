"""Dataset splitting for LAnoBERT (chronological, train-on-normal-only).

LAnoBERT pretrains the MLM on *normal* logs and evaluates on a held-out test
portion. Unlike retrieval-based follow-ups (e.g. RAPID) that randomly shuffle
the normal pool, LAnoBERT keeps the **temporal order** of the log stream:

  line method (BGL / Thunderbird):
    - label each line by its leading marker (first token == `label_marker`
      -> normal, else anomaly);
    - take the first `train_ratio` of the *whole* stream as the train region
      and the remainder as the test region (no shuffle);
    - train = normal lines inside the train region only;
    - test  = every line in the test region, with its label.

  block method (HDFS):
    - group lines by block id (`blk_-?\\d+`), label each block via
      `anomaly_label.csv`;
    - normal blocks are split by `train_ratio` (order preserved); train = the
      train-side normal blocks; test = remaining normal blocks + all anomaly
      blocks.

Outputs (raw, still un-normalized) are written to the config's
`paths.train_raw`, `paths.test_raw`, `paths.test_label`; run `preprocess`
afterwards to normalize them.

CLI:
    python -m lanobert.split --config configs/bgl.yaml --train_ratio 0.8
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import OrderedDict
from typing import List, Optional, Tuple

from tqdm import tqdm

from .utils import ensure_dir, load_config


def _write_lines(lines, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln if ln.endswith("\n") else ln + "\n")


def split_line_stream(
    raw_path: str, train_ratio: float, label_marker: str
) -> Tuple[List[str], List[str], List[int]]:
    """Chronological line split. Returns (train_normal, test_lines, test_labels)."""
    with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f]

    n = len(lines)
    cut = int(n * train_ratio)
    train_region, test_region = lines[:cut], lines[cut:]

    def is_normal(ln: str) -> bool:
        return ln[:1] == label_marker

    # Paper split: normal logs are divided ~80/20 (chronological); train uses the
    # train-region normals only. ALL anomalies (from the entire stream) go to the
    # test set -- anomalies in the train region must NOT be discarded, otherwise
    # the test anomaly count is far below the paper's (e.g. BGL 46k vs 348k).
    train_normal = [ln for ln in train_region if is_normal(ln)]
    test_normal = [ln for ln in test_region if is_normal(ln)]
    abnormal_all = [ln for ln in lines if not is_normal(ln)]
    test_lines = test_normal + abnormal_all
    test_labels = [0] * len(test_normal) + [1] * len(abnormal_all)
    return train_normal, test_lines, test_labels


def split_blocks(
    raw_path: str, label_csv: str, train_ratio: float, block_id_regex: str
) -> Tuple[List[str], List[str], List[int]]:
    """Block-grouped split for HDFS. Returns (train_lines, test_lines, test_labels).

    Each emitted "line" is a whole block's logs concatenated (one block = one
    example). Train = train-side normal blocks; test = remaining normal blocks
    + all anomaly blocks.
    """
    import csv

    blk_re = re.compile(block_id_regex)

    # label map: BlockId -> 1 (Anomaly) / 0 (Normal)
    label_map = {}
    with open(label_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_map[row["BlockId"]] = 1 if row["Label"].strip().lower() == "anomaly" else 0

    # group logs by block id, preserving first-seen order
    blocks: "OrderedDict[str, List[str]]" = OrderedDict()
    with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in tqdm(f, desc="group hdfs blocks"):
            ids = set(blk_re.findall(line))
            for blk in ids:
                blocks.setdefault(blk, []).append(line.strip())

    normal_blocks, anomaly_blocks = [], []
    for blk, logs in blocks.items():
        joined = " ".join(logs)
        if label_map.get(blk, 0) == 1:
            anomaly_blocks.append(joined)
        else:
            normal_blocks.append(joined)

    cut = int(len(normal_blocks) * train_ratio)
    train_lines = normal_blocks[:cut]
    test_normal = normal_blocks[cut:]
    test_lines = test_normal + anomaly_blocks
    test_labels = [0] * len(test_normal) + [1] * len(anomaly_blocks)
    return train_lines, test_lines, test_labels


def run(cfg, train_ratio: Optional[float] = None) -> dict:
    scfg = cfg.get("split", {}) or {}
    method = str(scfg.get("method", "line"))
    ratio = train_ratio if train_ratio is not None else float(scfg.get("train_ratio", 0.8))

    raw_log = cfg.get_path("paths.raw_log")
    train_raw = cfg.get_path("paths.train_raw")
    test_raw = cfg.get_path("paths.test_raw")
    test_label = cfg.get_path("paths.test_label")

    print(f"[split] {cfg.get('dataset')}  method={method}  train_ratio={ratio}")
    if method == "block":
        train, test, labels = split_blocks(
            raw_log,
            label_csv=cfg.get_path("paths.test_label_csv") or scfg.get("label_csv"),
            train_ratio=ratio,
            block_id_regex=str(scfg.get("block_id_regex", r"blk_-?\d+")),
        )
    else:
        train, test, labels = split_line_stream(
            raw_log,
            train_ratio=ratio,
            label_marker=str(scfg.get("label_marker", "-")),
        )

    _write_lines(train, train_raw)
    _write_lines(test, test_raw)
    _write_lines([str(x) for x in labels], test_label)

    stats = {
        "train_normal": len(train),
        "test_total": len(test),
        "test_anomaly": int(sum(labels)),
        "test_normal": int(len(labels) - sum(labels)),
    }
    stats_path = os.path.join(os.path.dirname(train_raw) or ".", "split_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[split] {stats}")
    print(f"[split] wrote: {train_raw} | {test_raw} | {test_label}")
    return stats


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="LAnoBERT chronological data split")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train_ratio", type=float, default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    run(cfg, train_ratio=args.train_ratio)


if __name__ == "__main__":
    main()
