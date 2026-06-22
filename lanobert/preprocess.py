"""Unified log preprocessing for HDFS, BGL and Thunderbird.

Replaces the dataset-specific scripts (preprocess.py / preprocess_thunder.py /
*_regex_parsing.ipynb) with one config-driven parser.

The parser normalizes each raw log line into a token sequence by masking
volatile fields (block ids, IPs, numbers) so the language model learns log
*templates* rather than memorizing identifiers.

CLI:
    python -m lanobert.preprocess --config configs/bgl.yaml --split train
    python -m lanobert.preprocess --config configs/thunderbird.yaml --split test
"""
from __future__ import annotations

import argparse
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from tqdm import tqdm

from .utils import ensure_dir, load_config

# Pre-compiled regexes (compiled once, reused per line).
_BLOCK_ID_RE = re.compile(r"blk_-?\d+")
_IP_RE = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d{1,5})?")
_NUMBER_RE = re.compile(r"\d+")
_NON_ALPHA_RE = re.compile(r"[^a-zA-Z.!?]+")
_PUNCT_RE = re.compile(r"([.!?])")

# --- Faithful ports of the original LAnoBERT preprocessing (reference code) ---
# BGL: preprocess/preprocess_bgl.py  (placeholders applied to the RAW line, in
# this exact order, BEFORE lower-casing; then lower + strip-non-alpha; drop[3:]).
_BGL_DATETIME_RE = re.compile(r"\d{1,4}\-\d{1,2}\-\d{1,2}-\d{1,2}.\d{1,2}.\d{1,2}.\d+")
_BGL_DATE_RE = re.compile(r"\d{1,4}\.\d{1,2}\.\d{1,2}")
_BGL_IP_RE = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d{1,5})?")
_BGL_SERVER_RE = re.compile(r"\S+(?=.*[0-9])(?=.*[a-zA-Z])(?=[:]+)\S+")
_BGL_SERVER2_RE = re.compile(r"\S+(?=.*[0-9])(?=.*[a-zA-Z])(?=[-])\S+")
_BGL_ECID_RE = re.compile(r"[A-Z0-9]{28}")
_BGL_SERIAL_RE = re.compile(r"[a-zA-Z0-9]{48}")
_BGL_MEMORY_RE = re.compile(r"0[xX][0-9a-fA-F]\S+")
_BGL_PATH_RE = re.compile(r".\S+(?=.[0-9a-zA-Z])(?=[/]).\S+")
_BGL_IAR_RE = re.compile(r"[0-9a-fA-F]{8}")
_BGL_NUM_RE = re.compile(r"(\d+)")
_BGL_NONALPHA_RE = re.compile(r"[^a-zA-Z<>]+")

# HDFS: preprocess/preprocess.py  (lower first, then blk/ip/num placeholders,
# then collapse [.!?] and strip non-alpha; keeps '.!?').
_HDFS_ID_RE = re.compile(r"blk_.\d+")
_HDFS_IP_RE = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d{1,5})?")
_HDFS_NUM_RE = re.compile(r"\d*\d")
_HDFS_NONALPHA_RE = re.compile(r"[^a-zA-Z.!?]+")

# Thunderbird: preprocess/pre_thunder_multi.py  (placeholders + punctuation
# stripping on line[2:] -- the leading "- " label is dropped; then lower +
# strip-non-alpha keeping '<>').
_TB_DATE_RE = re.compile(r"\d{1,4}\.\d{1,2}\.\d{1,2}")
_TB_DATE2_RE = re.compile(
    r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?"
    r"|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})\s+"
)
_TB_TIME_RE = re.compile(r"\d{1,2}\:\d{1,2}\:\d{1,2}")
_TB_ACCOUNT_RE = re.compile(r"(\w+[\w\.]*)@(\w+[\w\.]*)\-(\w+[\w\.]*)")
_TB_ACCOUNT2_RE = re.compile(r"(\w+[\w\.]*)@(\w+[\w\.]*)")
_TB_IAR_RE = re.compile(r"[0-9a-fA-F]{10}")
_TB_ID_RE = re.compile(r"(\w+[\w\.]*-\w+[\w\.]*)")
_TB_ID2_RE = re.compile(r"DATE\s\w+[\w\.]*")
_TB_NUM_RE = re.compile(r"(\[\d+\])")
_TB_EXP_RE = re.compile(r"\s[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]`'…》]")
_TB_EXP2_RE = re.compile(r"[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]`'…》]\s")
_TB_EXP3_RE = re.compile(r"[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]`'…》]")
_TB_NONALPHA_RE = re.compile(r"[^a-zA-Z<>]+")


@dataclass
class ParseOptions:
    """Normalization switches resolved from a config's `preprocess` block."""

    mask_block_id: bool = True
    mask_ip: bool = True
    mask_number: bool = True
    drop_header_fields: int = 0
    lowercase: bool = True
    regex_profile: str = "generic"


def _unicode_to_ascii(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def _bgl_parse(line: str, drop_header_fields: int = 3) -> str:
    """Exact port of reference preprocess_bgl.py (log_parser + normalizeString)."""
    tmp = _BGL_IP_RE.sub(" IP ", line)
    tmp = _BGL_DATETIME_RE.sub(" TIME ", tmp)
    tmp = _BGL_DATE_RE.sub(" TIME ", tmp)
    tmp = _BGL_PATH_RE.sub(" PATH ", tmp)
    tmp = _BGL_SERVER_RE.sub(" SERVER ", tmp)
    tmp = _BGL_SERVER2_RE.sub(" SERVER ", tmp)
    tmp = _BGL_ECID_RE.sub(" ECID ", tmp)
    tmp = _BGL_SERIAL_RE.sub(" SERIAL ", tmp)
    tmp = _BGL_MEMORY_RE.sub(" MEMORY ", tmp)
    tmp = _BGL_IAR_RE.sub(" IAR ", tmp)
    tmp = _BGL_NUM_RE.sub(" NUM ", tmp)
    # normalizeString: lower + ascii + strip everything but letters and <>
    s = _unicode_to_ascii(tmp.lower().strip())
    s = _BGL_NONALPHA_RE.sub(" ", s)
    return " ".join(s.split()[drop_header_fields:])


def _hdfs_parse(line: str, drop_header_fields: int = 0) -> str:
    """Exact port of reference preprocess.py (HDFS log_parser + normalize_string)."""
    log = _unicode_to_ascii(line.lower().strip())
    tmp = _HDFS_ID_RE.sub("BLK", log)
    tmp = _HDFS_IP_RE.sub("IP", tmp)
    tmp = _HDFS_NUM_RE.sub("NUM", tmp)
    # normalize_string: collapse [.!?] to space, then strip non-alpha (keep .!?)
    s = _PUNCT_RE.sub(" ", tmp)
    s = _HDFS_NONALPHA_RE.sub(" ", s).strip()
    if drop_header_fields > 0:
        s = " ".join(s.split()[drop_header_fields:])
    return s


def _tb_parse(line: str) -> str:
    """Exact port of reference pre_thunder_multi.py (log_parser + normalizeString).

    The reference drops the first two characters (`line[2:]`) to strip the
    leading "- "/"+ " label marker before parsing.
    """
    log = line[2:]
    tmp = _TB_DATE_RE.sub("DATE", log)
    tmp = _TB_DATE2_RE.sub("DATE ", tmp)
    tmp = _TB_TIME_RE.sub("TIME", tmp)
    tmp = _TB_ACCOUNT_RE.sub("ACCOUNT", tmp)
    tmp = _TB_ACCOUNT2_RE.sub("ACCOUNT", tmp)
    tmp = _TB_ID_RE.sub("ID", tmp)
    tmp = _TB_ID2_RE.sub("DATE ID", tmp)
    tmp = _TB_IAR_RE.sub("IAR", tmp)
    tmp = _TB_NUM_RE.sub(" NUM ", tmp)
    tmp = _TB_EXP_RE.sub("", tmp)
    tmp = _TB_EXP2_RE.sub(" ", tmp)
    tmp = _TB_EXP3_RE.sub(" ", tmp)
    # normalizeString: lower + ascii + strip everything but letters and <>
    s = _unicode_to_ascii(tmp.lower().strip())
    s = _TB_NONALPHA_RE.sub(" ", s)
    return " ".join(s.split())


def normalize_line(line: str, opt: ParseOptions) -> str:
    """Normalize a single raw log line into a clean, template-like token string."""
    if opt.regex_profile == "bgl":
        return _bgl_parse(line, drop_header_fields=opt.drop_header_fields)
    if opt.regex_profile == "hdfs":
        return _hdfs_parse(line, drop_header_fields=opt.drop_header_fields)
    if opt.regex_profile == "thunderbird":
        return _tb_parse(line)

    s = line.strip()
    if opt.lowercase:
        s = s.lower()
    s = _unicode_to_ascii(s)

    if opt.mask_block_id:
        s = _BLOCK_ID_RE.sub("BLK", s)
    if opt.mask_ip:
        s = _IP_RE.sub("IP", s)
    if opt.mask_number:
        s = _NUMBER_RE.sub("NUM", s)

    # collapse punctuation and non-alpha noise into single spaces
    s = _PUNCT_RE.sub(" ", s)
    s = _NON_ALPHA_RE.sub(" ", s)
    s = s.strip()

    if opt.drop_header_fields > 0:
        s = " ".join(s.split()[opt.drop_header_fields:])
    return s


def get_block_id(line: str) -> Optional[str]:
    """Return the HDFS block id in a line, if present (used to group HDFS logs)."""
    m = _BLOCK_ID_RE.search(line)
    return m.group() if m else None


def iter_lines(path: str, errors: str = "ignore") -> Iterable[str]:
    """Stream a (possibly very large) log file line by line."""
    with open(path, "r", encoding="utf-8", errors=errors) as f:
        for line in f:
            yield line


def split_thunderbird(raw_path: str, label_marker: str = "-") -> Tuple[List[str], List[str]]:
    """Split a raw Thunderbird log into (normal, abnormal) lines.

    Convention from the original loghub release: lines whose first character is
    `label_marker` ('-') are normal, everything else is anomalous.
    """
    normal, abnormal = [], []
    for line in tqdm(iter_lines(raw_path), desc="thunderbird split"):
        if line[:1] == label_marker:
            normal.append(line)
        else:
            abnormal.append(line)
    return normal, abnormal


def _options_from_config(cfg) -> ParseOptions:
    pre = cfg.get("preprocess", {}) or {}
    return ParseOptions(
        mask_block_id=bool(pre.get("mask_block_id", True)),
        mask_ip=bool(pre.get("mask_ip", True)),
        mask_number=bool(pre.get("mask_number", True)),
        drop_header_fields=int(pre.get("drop_header_fields", 0)),
        lowercase=bool(pre.get("lowercase", True)),
        regex_profile=str(pre.get("regex_profile", "generic")),
    )


def preprocess_file(in_path: str, out_path: str, opt: ParseOptions) -> int:
    """Normalize every line of `in_path` and write to `out_path`. Returns line count."""
    ensure_dir(os.path.dirname(out_path) or ".")
    n = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for line in tqdm(iter_lines(in_path), desc=f"parse {os.path.basename(in_path)}"):
            norm = normalize_line(line, opt)
            if norm:
                out.write(norm + "\n")
                n += 1
    return n


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="LAnoBERT log preprocessing")
    parser.add_argument("--config", required=True, help="path to a dataset yaml config")
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="train",
        help="which file in the config's `paths` to parse",
    )
    parser.add_argument("--in_path", default=None, help="override input log path")
    parser.add_argument("--out_path", default=None, help="override output path")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    opt = _options_from_config(cfg)

    # inputs are the raw split files produced by `lanobert.split`
    if args.split == "train":
        in_path = args.in_path or cfg.get_path("paths.train_raw")
        out_path = args.out_path or cfg.get_path("paths.train_normal")
    else:
        in_path = args.in_path or cfg.get_path("paths.test_raw")
        out_path = args.out_path or cfg.get_path("paths.test_log")

    print(f"[preprocess] {cfg.get('dataset')} / {args.split}")
    print(f"[preprocess] in : {in_path}")
    print(f"[preprocess] out: {out_path}")
    count = preprocess_file(in_path, out_path, opt)
    print(f"[preprocess] wrote {count} lines")


if __name__ == "__main__":
    main()
