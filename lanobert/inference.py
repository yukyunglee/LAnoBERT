"""LAnoBERT inference: masked-token anomaly scoring.

Core idea (from the LAnoBERT paper):
  1. Pretrain a BERT MLM on *normal* logs only.
  2. For a test log line, mask each content token one at a time and ask the
     model for the predictive probability of the *true* token.
  3. A normal log is well predicted (high probability); an anomalous log
     contains tokens the model finds surprising (low probability).
  4. Turn per-token surprise (1 - p) into one anomaly score per line by
     aggregating (mean / max / top-k mean).

`top_k` keeps the model's top-k predictions per masked position -- this is the
"retriever" step in the original code (`pipeline("fill-mask", top_k=5)`),
generalized here to full-corpus scoring.

CLI:
    python -m lanobert.inference --config configs/bgl.yaml
"""
from __future__ import annotations

import argparse
import bisect
import os
from typing import List, Optional, Sequence

import numpy as np
from tqdm import tqdm

from .metrics import evaluate
from .tokenizer import load_tokenizer
from .utils import ensure_dir, get_device, load_config, set_seed


def _aggregate(surprise: np.ndarray, mode: str, top_k: int) -> float:
    """Reduce per-token surprise (1 - p) to a single line-level anomaly score."""
    if surprise.size == 0:
        return 0.0
    if mode == "mean":
        return float(surprise.mean())
    if mode == "max":
        return float(surprise.max())
    if mode == "topk_mean":
        k = min(top_k, surprise.size)
        return float(np.sort(surprise)[-k:].mean())
    if mode == "bottomk_mean":
        # mean of the k SMALLEST values. For the prob score (surprise = 1-max_p)
        # this selects the words the model is *most confident* about, matching the
        # original LAnoBERT aggregation (mean of top-k largest predictive prob).
        k = min(top_k, surprise.size)
        return float(np.sort(surprise)[:k].mean())
    if mode == "pctl":
        # length-adaptive: q-th percentile (q passed via top_k). Robust to the
        # huge line-length variance in block-aggregated logs (e.g. HDFS).
        return float(np.percentile(surprise, top_k))
    raise ValueError(f"unknown score mode: {mode}")


class LAnoBERTScorer:
    """Wraps a trained MLM and produces per-line anomaly scores."""

    def __init__(self, model_dir: str, vocab_file: Optional[str] = None,
                 max_len: int = 512, device=None, pretrained_model: Optional[str] = None,
                 subfolder: Optional[str] = None):
        """Load the scoring model.

        If `pretrained_model` is set (e.g. "bert-base-uncased"), an off-the-shelf
        HuggingFace MLM and its own tokenizer are loaded — this is the
        *training-free* baseline (no from-scratch pretraining on logs).
        Otherwise the trained model at `model_dir` is used. `model_dir` may be a
        local path **or** a HuggingFace Hub repo id (e.g. "yukyung/LAnoBERT");
        `subfolder` selects a per-dataset checkpoint inside that repo
        (e.g. "bgl" / "hdfs" / "thunderbird").
        """
        import torch  # noqa: F401
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.device = device or get_device()
        # subfolder is only meaningful for Hub repos / nested local dirs.
        hub_kwargs = {"subfolder": subfolder} if subfolder else {}

        if pretrained_model:
            print(f"[infer] training-free mode: pretrained '{pretrained_model}'")
            self.model = AutoModelForMaskedLM.from_pretrained(pretrained_model).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        else:
            # AutoModel/AutoTokenizer so trained models of any architecture load
            # correctly (custom-vocab BERT, bert-base TAPT / random-init, ...),
            # from a local dir or the HuggingFace Hub.
            where = f"{model_dir}" + (f"/{subfolder}" if subfolder else "")
            print(f"[infer] loading trained model from '{where}'")
            self.model = AutoModelForMaskedLM.from_pretrained(model_dir, **hub_kwargs).to(self.device)
            # Prefer a tokenizer saved alongside the model; fall back to a vocab file.
            if vocab_file and os.path.isfile(vocab_file):
                self.tokenizer = load_tokenizer(vocab_file, max_len=max_len)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_dir, **hub_kwargs)

        self.model.eval()
        self.max_len = max_len
        self.mask_id = self.tokenizer.mask_token_id

    def score_line(self, line: str, score: str = "topk_mean", top_k: int = 5,
                   batch_size: int = 64, mask_unit: str = "word"):
        """Return two anomaly scores (abnormal_error, abnormal_prob) for a log line.

        Two quantities are measured at each masked position:
          * error_i = cross-entropy loss = -log p(true token)   (high => anomaly)
          * prob_i  = max predictive probability over the vocab  (low  => anomaly)
        Per-position series are reduced to a line-level score via top-k. Both
        follow "higher = more anomalous" (prob is stored as 1 - max_prob).

        `mask_unit="word"` (default, faithful to the reference LAnoBERT code)
        splits the line on whitespace and replaces each *whole word* with a
        single [MASK]. `mask_unit="subword"` masks each wordpiece token instead.
        """
        errs, probs = self.score_line_raw(line, batch_size=batch_size, mask_unit=mask_unit)
        return _aggregate(errs, score, top_k), _aggregate(probs, score, top_k)

    def score_line_raw(self, line: str, batch_size: int = 64, mask_unit: str = "word"):
        """Return raw per-position (error, prob_surprise) arrays (no aggregation).

        Aggregation (mean/max/top-k) is deferred to the caller so a single
        forward pass can be re-aggregated for many top_k values for free.
        """
        if mask_unit == "word":
            return self._score_line_word(line, batch_size)
        return self._score_line_subword(line, batch_size)

    def _score_line_word(self, line, batch_size):
        """Word-level masking: replace each whitespace-delimited word with [MASK].

        Long lines are scored with a **sliding window**: consecutive words are
        greedily packed into windows whose tokenized length stays within
        ``max_len`` (minus the 2 special tokens). Each word is masked and scored
        inside its own window, and the per-word scores from every window are
        concatenated in order. This guarantees *every* word is scored instead of
        silently dropping words past ``max_len`` (the previous behaviour, which
        truncated away ~6.3% of long HDFS anomaly blocks). When the whole line
        fits in one window this is identical to the original single-pass scoring.
        """
        import torch

        words = line.split()
        if not words:
            return 0.0, 0.0

        mask_tok = self.tokenizer.mask_token
        fast = getattr(self.tokenizer, "is_fast", False)

        def word_first_ids_and_lens(word_list):
            """For each whitespace word return (in-context first-subword id, token
            count). The CE target must be the token the model expects at the [MASK]
            slot. For leading-space BPE tokenizers a word's
            mid-line token (``Ġword``) differs from its context-free token
            (``word``); using the context-free id makes the CE target wrong and
            inverts the error score.

            We tokenize the joined string with char offsets and assign each token
            to the whitespace word containing its first non-space char. This is
            robust even when the tokenizer's own pre-tokenizer splits on
            punctuation (so ``word_ids()`` would not align with ``str.split()``).
            Slow tokenizers fall back to context-free tokenization."""
            n = len(word_list)
            if fast:
                s = " ".join(word_list)
                starts, c = [], 0
                for w in word_list:
                    starts.append(c)
                    c += len(w) + 1
                enc = self.tokenizer(s, add_special_tokens=False,
                                     return_offsets_mapping=True)
                first = [None] * n
                cnt = [0] * n
                for tid, (a, b) in zip(enc["input_ids"], enc["offset_mapping"]):
                    if a >= b:
                        continue  # special / empty span
                    while a < b and s[a] == " ":
                        a += 1  # skip leading space so 'Ġword' maps to its word
                    wi = bisect.bisect_right(starts, a) - 1
                    if 0 <= wi < n:
                        cnt[wi] += 1
                        if first[wi] is None:
                            first[wi] = tid
                return first, [max(1, x) for x in cnt]
            first, lens = [], []
            for w in word_list:
                ids = self.tokenizer(w, add_special_tokens=False)["input_ids"]
                first.append(ids[0] if ids else None)
                lens.append(max(1, len(ids)))
            return first, lens

        # token count per word (in-context) = window-packing budget
        _, wlens = word_first_ids_and_lens(words)

        # Greedily pack words into windows that fit the token budget.
        budget = max(1, self.max_len - 2)
        windows: List[tuple] = []
        s = 0
        while s < len(words):
            e, tot = s, 0
            while e < len(words) and (e == s or tot + wlens[e] <= budget):
                tot += wlens[e]
                e += 1
            windows.append((s, e))
            s = e

        errors: List[float] = []
        prob_surprises: List[float] = []
        with torch.no_grad():
            for (ws, we) in windows:
                win_words = words[ws:we]
                # CE targets computed from THIS window's tokenization so they match
                # exactly what the model sees at each [MASK] (incl. window-initial
                # word having no leading space).
                win_true, _ = word_first_ids_and_lens(win_words)
                masked_strs = []
                for i in range(len(win_words)):
                    w = list(win_words)
                    w[i] = mask_tok
                    masked_strs.append(" ".join(w))
                for start in range(0, len(masked_strs), batch_size):
                    chunk = masked_strs[start:start + batch_size]
                    enc = self.tokenizer(
                        chunk, return_tensors="pt", padding=True,
                        truncation=True, max_length=self.max_len,
                    ).to(self.device)
                    logits = self.model(**enc).logits
                    log_probs = torch.log_softmax(logits, dim=-1)
                    ids = enc["input_ids"]
                    for j in range(len(chunk)):
                        mask_pos = (ids[j] == self.mask_id).nonzero(as_tuple=True)[0]
                        if mask_pos.numel() == 0:
                            continue  # window mask was truncated away (rare)
                        pos = int(mask_pos[0].item())
                        lp = log_probs[j, pos]
                        max_log_p = torch.max(lp).item()
                        prob_surprises.append(1.0 - float(np.exp(max_log_p)))
                        tid = win_true[start + j]
                        if tid is not None:
                            errors.append(-lp[tid].item())

        return np.asarray(errors), np.asarray(prob_surprises)

    def _score_line_subword(self, line, batch_size):
        """Subword-level masking: mask each wordpiece token one at a time."""
        import torch

        enc = self.tokenizer(
            line, truncation=True, max_length=self.max_len,
            return_special_tokens_mask=True,
        )
        input_ids = enc["input_ids"]
        special = enc["special_tokens_mask"]
        content_pos = [i for i, s in enumerate(special) if s == 0]
        if not content_pos:
            return 0.0, 0.0

        base = torch.tensor(input_ids, dtype=torch.long)
        # one masked copy per content position
        batch = base.repeat(len(content_pos), 1)
        for row, pos in enumerate(content_pos):
            batch[row, pos] = self.mask_id

        errors: List[float] = []
        prob_surprises: List[float] = []
        with torch.no_grad():
            for start in range(0, batch.size(0), batch_size):
                chunk = batch[start:start + batch_size].to(self.device)
                logits = self.model(input_ids=chunk).logits
                log_probs = torch.log_softmax(logits, dim=-1)
                for j in range(chunk.size(0)):
                    pos = content_pos[start + j]
                    true_id = input_ids[pos]
                    # cross-entropy loss at the masked position
                    errors.append(-log_probs[j, pos, true_id].item())
                    # 1 - top-1 confidence (low max-prob => high surprise)
                    max_log_p = torch.max(log_probs[j, pos]).item()
                    prob_surprises.append(1.0 - float(np.exp(max_log_p)))

        return np.asarray(errors), np.asarray(prob_surprises)

    def score_corpus(self, lines: Sequence[str], score: str = "topk_mean", top_k: int = 5,
                     batch_size: int = 64, dedup: bool = True, mask_unit: str = "word"):
        """Score every line; returns (error_scores, prob_scores) as two arrays.

        Logs are extremely repetitive (the same template recurs thousands of
        times). With `dedup=True` we score each *unique* line only once, cache
        it in a {line: (err, prob)} table, then look up every line (duplicates
        are free). This is the key-value caching the original LAnoBERT used and
        is dramatically faster on BGL/HDFS/Thunderbird. Set `dedup=False` to
        score every line independently.
        """
        if not dedup:
            pairs = [
                self.score_line(ln, score=score, top_k=top_k, batch_size=batch_size, mask_unit=mask_unit)
                for ln in tqdm(lines, desc="scoring")
            ]
        else:
            uniques = list(dict.fromkeys(lines))  # preserve order, drop duplicates
            print(f"[infer] dedup: {len(lines)} lines -> {len(uniques)} unique "
                  f"({100 * (1 - len(uniques) / max(len(lines), 1)):.1f}% saved)")
            score_table = {
                ln: self.score_line(ln, score=score, top_k=top_k, batch_size=batch_size, mask_unit=mask_unit)
                for ln in tqdm(uniques, desc="scoring unique")
            }
            pairs = [score_table[ln] for ln in lines]

        errors = np.asarray([p[0] for p in pairs])
        probs = np.asarray([p[1] for p in pairs])
        return errors, probs

    def score_corpus_topk_sweep(self, lines: Sequence[str], top_ks: Sequence[int],
                                batch_size: int = 64, dedup: bool = True,
                                mask_unit: str = "word"):
        """Compute raw per-position arrays *once* (dedup-cached), then aggregate
        with top-k mean for every k in `top_ks`.

        The expensive part (BERT forward passes) is done a single time per unique
        line; re-aggregating for different k is essentially free. Returns
        ``{k: (error_scores, prob_scores, probtopk_scores)}`` aligned to `lines`:
          * error      = top-k LARGEST CE loss (most-surprising true tokens).
          * prob       = top-k LARGEST predictive prob (= k SMALLEST `1-max_p`,
                         bottom-k) -- the ORIGINAL LAnoBERT score.
          * probtopk   = top-k LARGEST `1-max_p` (least-confident words) -- the
                         alternative aggregation proposed during this project.
        """
        uniques = list(dict.fromkeys(lines))
        if dedup:
            print(f"[infer] dedup: {len(lines)} lines -> {len(uniques)} unique "
                  f"({100 * (1 - len(uniques) / max(len(lines), 1)):.1f}% saved)")
        raw = {
            ln: self.score_line_raw(ln, batch_size=batch_size, mask_unit=mask_unit)
            for ln in tqdm(uniques, desc="scoring unique")
        }
        out = {}
        for k in top_ks:
            errs = np.asarray([_aggregate(raw[ln][0], "topk_mean", k) for ln in lines])
            probs = np.asarray([_aggregate(raw[ln][1], "bottomk_mean", k) for ln in lines])
            probs_topk = np.asarray([_aggregate(raw[ln][1], "topk_mean", k) for ln in lines])
            out[int(k)] = (errs, probs, probs_topk)
        # length-adaptive (k-independent) error aggregations. mean_all is the most
        # BALANCED single method across BGL/HDFS/Thunderbird -- fixed top-k breaks
        # on block-aggregated logs (HDFS, avg 245 words/line) whereas the line mean
        # of CE is length-robust (HDFS AUROC 0.928 -> 0.997, others stay ~1.0).
        # error_* uses the CE signal raw[ln][0]; prob_* uses the (1-max_p) signal
        # raw[ln][1]. error_mean is the recommended balanced method -- prob_* are
        # kept for parity but are NOT balanced (great on HDFS, collapse on
        # BGL/Thunderbird under any aggregation).
        adaptive = {
            "error_mean": np.asarray([_aggregate(raw[ln][0], "mean", 0) for ln in lines]),
            "error_pctl99": np.asarray([_aggregate(raw[ln][0], "pctl", 99) for ln in lines]),
            "prob_mean": np.asarray([_aggregate(raw[ln][1], "mean", 0) for ln in lines]),
            "prob_pctl99": np.asarray([_aggregate(raw[ln][1], "pctl", 99) for ln in lines]),
        }
        return out, adaptive


def _load_test(cfg, limit: Optional[int]):
    """Load normalized test lines and integer labels (1=anomaly, 0=normal)."""
    from .dataset import read_lines

    test_path = cfg.get_path("paths.test_log")
    label_path = cfg.get_path("paths.test_label")
    lines = read_lines(test_path, limit=limit)

    labels: List[int] = []
    if label_path and os.path.isfile(label_path):
        with open(label_path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                # accept "1"/"0", "anomaly"/"normal", or Thunderbird "-" marker
                if ln in ("1", "anomaly", "Anomaly", "abnormal"):
                    labels.append(1)
                elif ln in ("0", "normal", "Normal"):
                    labels.append(0)
                else:
                    labels.append(0 if ln[:1] == "-" else 1)
                if limit is not None and len(labels) >= limit:
                    break
    labels = labels[:len(lines)]
    return lines, labels


def run(cfg) -> dict:
    icfg = cfg.get("inference", {})
    tcfg = cfg.get("train", {})
    set_seed(int(tcfg.get("seed", 42)))

    # `hf_model` loads the released checkpoint straight from the HuggingFace Hub
    # (e.g. "yukyung/LAnoBERT" with `hf_subfolder: bgl`), skipping local
    # training. Falls back to the locally trained model at paths.model_dir.
    hf_model = icfg.get("hf_model", None)
    subfolder = icfg.get("hf_subfolder", None)
    if hf_model:
        model_dir = hf_model
    else:
        model_dir = os.path.join(cfg.get_path("paths.model_dir"), "final")
        if not os.path.isdir(model_dir):
            model_dir = cfg.get_path("paths.model_dir")

    # `pretrained_model` (e.g. "bert-base-uncased") enables the training-free
    # baseline: skip from-scratch training and score with an off-the-shelf LM.
    pretrained = icfg.get("pretrained_model", None)

    scorer = LAnoBERTScorer(
        model_dir=model_dir,
        vocab_file=None,
        max_len=int(tcfg.get("max_len", 512)),
        pretrained_model=pretrained,
        subfolder=subfolder,
    )

    limit = icfg.get("max_eval_samples", None)
    lines, labels = _load_test(cfg, limit=limit)
    print(f"[infer] test lines: {len(lines)}  labels: {len(labels)}")

    # top_k sweep: a single forward pass per unique line, re-aggregated for
    # every k. Configurable via inference.top_ks; defaults to [1, 3, 5, 10].
    top_ks = icfg.get("top_ks", None)
    if top_ks is None:
        top_ks = [int(icfg.get("top_k", 5))]
    top_ks = [int(k) for k in top_ks]

    sweep = scorer.score_corpus_topk_sweep(
        lines,
        top_ks=top_ks,
        batch_size=int(icfg.get("batch_size", 16)),
        dedup=bool(icfg.get("dedup", True)),
        mask_unit=str(icfg.get("mask_unit", "word")),
    )
    sweep, adaptive = sweep

    result_dir = ensure_dir(cfg.get_path("paths.result_dir"))
    dataset = cfg.get("dataset")
    results: dict = {}
    for k in top_ks:
        err_scores, prob_scores, probtopk_scores = sweep[k]
        np.save(os.path.join(result_dir, f"scores_error_k{k}.npy"), err_scores)
        np.save(os.path.join(result_dir, f"scores_prob_k{k}.npy"), prob_scores)
        np.save(os.path.join(result_dir, f"scores_probtopk_k{k}.npy"), probtopk_scores)
        if not (labels and len(labels) == len(err_scores)):
            print(f"[infer] k={k}: no/mismatched labels -- saved scores only")
            results[k] = {"num_scored": int(len(err_scores))}
            continue
        print(f"[infer] ===== top_k={k} =====")
        print(f"[infer] --- abnormal_error (CE loss), k={k} ---")
        err_metrics = evaluate(err_scores, labels, result_dir=result_dir,
                               tag=f"{dataset}_error_k{k}")
        print(f"[infer] --- abnormal_prob (top-k largest max_p, original), k={k} ---")
        prob_metrics = evaluate(prob_scores, labels, result_dir=result_dir,
                                tag=f"{dataset}_prob_k{k}")
        print(f"[infer] --- abnormal_probtopk (top-k largest 1-max_p), k={k} ---")
        probtopk_metrics = evaluate(probtopk_scores, labels, result_dir=result_dir,
                                    tag=f"{dataset}_probtopk_k{k}")
        results[k] = {"error": err_metrics, "prob": prob_metrics, "probtopk": probtopk_metrics}

    # length-adaptive error scores (k-independent). error_mean is the recommended
    # BALANCED method across datasets; see score_corpus_topk_sweep docstring.
    for name, scores in adaptive.items():
        np.save(os.path.join(result_dir, f"scores_{name}.npy"), scores)
        if labels and len(labels) == len(scores):
            print(f"[infer] --- {name} (length-adaptive) ---")
            results[name] = evaluate(scores, labels, result_dir=result_dir,
                                     tag=f"{dataset}_{name}")
    return results


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="LAnoBERT inference")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
