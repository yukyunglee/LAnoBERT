"""MLM-only dataset for LAnoBERT.

The original implementation reused HuggingFace's
`TextDatasetForNextSentencePrediction`, which builds NSP pairs. LAnoBERT does
**not** use next-sentence prediction -- each log line is an independent example
and the only objective is masked-language-modeling. This module therefore
treats one normalized log line as one example, which is simpler, faster, and
matches the paper.
"""
from __future__ import annotations

import os
from typing import List

import torch
from torch.utils.data import Dataset


class LogLineDataset(Dataset):
    """One normalized log line -> one tokenized example (no NSP).

    Args:
        tokenizer: a fast BERT tokenizer.
        file_path: path to a newline-delimited normalized corpus.
        max_len: max sequence length (longer lines are truncated).
        skip_empty: drop blank lines.
    """

    def __init__(self, tokenizer, file_path: str, max_len: int = 512, skip_empty: bool = True):
        assert os.path.isfile(file_path), f"Input file not found: {file_path}"
        self.max_len = max_len

        with open(file_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f]
        if skip_empty:
            lines = [ln for ln in lines if ln]

        # Pre-tokenize all lines once so __getitem__ is just an index lookup.
        print(f"[dataset] pre-tokenizing {len(lines):,} lines...")
        batch_enc = tokenizer(
            lines,
            truncation=True,
            max_length=max_len,
            return_special_tokens_mask=True,
        )
        self.input_ids: List[List[int]] = batch_enc["input_ids"]
        self.attention_mask: List[List[int]] = batch_enc["attention_mask"]
        self.special_tokens_mask: List[List[int]] = batch_enc["special_tokens_mask"]
        print(f"[dataset] pre-tokenization done.")

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int):
        # Return plain lists so DataCollatorForLanguageModeling can
        # dynamically pad each batch to its longest sequence instead of
        # always padding to max_len.
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "special_tokens_mask": self.special_tokens_mask[idx],
        }


def read_lines(file_path: str, limit: int | None = None) -> List[str]:
    """Utility: read normalized lines from a corpus, optionally capped at `limit`."""
    out: List[str] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                out.append(ln)
            if limit is not None and len(out) >= limit:
                break
    return out
