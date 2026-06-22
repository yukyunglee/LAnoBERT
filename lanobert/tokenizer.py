"""Train and load a WordPiece tokenizer over normalized log corpora.

CLI:
    python -m lanobert.tokenizer --config configs/bgl.yaml
"""
from __future__ import annotations

import argparse
import os
from typing import List, Optional

from .utils import ensure_dir, load_config

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def train_tokenizer(
    train_files,
    out_dir: str,
    vocab_size: int = 10000,
    min_frequency: int = 2,
    lowercase: bool = False,
    name: str = "LogBERT",
) -> str:
    """Train a BertWordPieceTokenizer and save `{out_dir}/{name}-vocab.txt`."""
    from tokenizers import BertWordPieceTokenizer

    if isinstance(train_files, str):
        train_files = [train_files]

    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=False,
        lowercase=lowercase,
        wordpieces_prefix="##",
    )
    tokenizer.train(
        files=train_files,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=True,
        special_tokens=SPECIAL_TOKENS,
        wordpieces_prefix="##",
    )
    ensure_dir(out_dir)
    tokenizer.save_model(out_dir, name)
    vocab_path = os.path.join(out_dir, f"{name}-vocab.txt")
    print(f"[tokenizer] saved vocab -> {vocab_path}")
    return vocab_path


def load_tokenizer(vocab_file: str, max_len: int = 512):
    """Load a fast BERT tokenizer from a vocab file (case-sensitive by default)."""
    from transformers import BertTokenizerFast

    return BertTokenizerFast(
        vocab_file=vocab_file,
        max_len=max_len,
        do_lower_case=False,
    )


def vocab_path_for(cfg) -> str:
    """Resolve the standard vocab path for a config's dataset."""
    return os.path.join(
        cfg.get_path("paths.tokenizer_dir"),
        f"{cfg.get('dataset')}_LogBERT-vocab.txt",
    )


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="LAnoBERT tokenizer training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train_file", default=None, help="override training corpus path")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    tok = cfg.get("tokenizer", {})
    train_file = args.train_file or cfg.get_path("paths.train_normal")

    print(f"[tokenizer] training on {train_file}")
    train_tokenizer(
        train_files=train_file,
        out_dir=cfg.get_path("paths.tokenizer_dir"),
        vocab_size=int(tok.get("vocab_size", 10000)),
        min_frequency=int(tok.get("min_frequency", 2)),
        lowercase=bool(tok.get("lowercase", False)),
        name=f"{cfg.get('dataset')}_LogBERT",
    )


if __name__ == "__main__":
    main()
