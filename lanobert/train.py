"""Masked-language-model pretraining for LAnoBERT.

Trains a BERT encoder from scratch on *normal* logs only, with the MLM
objective (no NSP). The resulting model is used at inference time to score how
"surprising" each token in a test log is.

CLI:
    python -m lanobert.train --config configs/bgl.yaml
"""
from __future__ import annotations

import argparse
import os
from typing import List, Optional

from .dataset import LogLineDataset
from .tokenizer import load_tokenizer, vocab_path_for
from .utils import ensure_dir, load_config, set_seed


def build_model(vocab_size: int, max_len: int, attn_implementation: str = "sdpa"):
    """Create a fresh BertForMaskedLM with config sized for log vocab.

    `attn_implementation="sdpa"` uses PyTorch's scaled-dot-product-attention
    (Flash/memory-efficient kernels) -- mathematically identical to eager
    attention but faster and lighter, so it does not change reproducibility.
    """
    from transformers import BertConfig, BertForMaskedLM

    config = BertConfig(
        vocab_size=vocab_size,
        max_position_embeddings=max_len,
        # remaining hyper-params use BERT-base defaults; override here if needed
    )
    try:
        return BertForMaskedLM(config=config, attn_implementation=attn_implementation)
    except (TypeError, ValueError):
        # older transformers without attn_implementation kwarg
        return BertForMaskedLM(config=config)


def train(cfg, vocab_file: Optional[str] = None) -> str:
    """Run MLM training and return the saved model directory."""
    tcfg = cfg.get("train", {})
    set_seed(int(tcfg.get("seed", 42)))

    vocab_file = vocab_file or vocab_path_for(cfg)
    max_len = int(tcfg.get("max_len", 512))

    # Enable TF32 matmuls on Ampere+ GPUs (free speedup, no accuracy loss).
    import torch
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    from transformers import (
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    # Three ways to obtain the model + tokenizer:
    #   (a) base_model unset            -> custom log vocab + random-init BERT
    #       (the original LAnoBERT "init" setting).
    #   (b) base_model + warm_start=True -> load a pretrained HF checkpoint
    #       (e.g. bert-base-uncased) and continue MLM training on
    #       logs == task-adaptive pre-training (the paper's "pretrained").
    #   (c) base_model + warm_start=False -> take only the *architecture* of a
    #       pretrained model (its config) and random-init the weights.
    #   (d) base_model + warm_start=False + custom_vocab=True -> take the
    #       *architecture* of a pretrained model but resize it to the custom log
    #       vocab and train from scratch with the custom WordPiece tokenizer.
    # In (b)/(c) the pretrained model's OWN tokenizer is used (its embeddings
    # are tied to that vocab), so the custom log vocab is ignored. In (d) the
    # custom log vocab is used and config.vocab_size is overridden to match.
    base_model = tcfg.get("base_model", None)
    attn = str(tcfg.get("attn_implementation", "sdpa"))
    if base_model:
        from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

        warm_start = bool(tcfg.get("warm_start", False))
        custom_vocab = bool(tcfg.get("custom_vocab", False))
        if warm_start:
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            print(f"[train] base_model={base_model} warm_start=True "
                  f"(vocab={tokenizer.vocab_size})")
            try:
                model = AutoModelForMaskedLM.from_pretrained(base_model, attn_implementation=attn)
            except (TypeError, ValueError):
                model = AutoModelForMaskedLM.from_pretrained(base_model)
        elif custom_vocab:
            # (d) architecture-only shell, resized to the custom log vocab.
            tokenizer = load_tokenizer(vocab_file, max_len=max_len)
            config = AutoConfig.from_pretrained(base_model)
            config.vocab_size = tokenizer.vocab_size
            config.max_position_embeddings = max(getattr(config, "max_position_embeddings", max_len), max_len)
            for attr, tok_id in (("pad_token_id", tokenizer.pad_token_id),
                                 ("bos_token_id", tokenizer.cls_token_id),
                                 ("eos_token_id", tokenizer.sep_token_id)):
                if tok_id is not None and hasattr(config, attr):
                    setattr(config, attr, tok_id)
            print(f"[train] base_model={base_model} warm_start=False custom_vocab=True "
                  f"(arch shell + log vocab={tokenizer.vocab_size})")
            try:
                model = AutoModelForMaskedLM.from_config(config, attn_implementation=attn)
            except (TypeError, ValueError):
                model = AutoModelForMaskedLM.from_config(config)
        else:
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            print(f"[train] base_model={base_model} warm_start=False "
                  f"(arch shell + own vocab={tokenizer.vocab_size})")
            config = AutoConfig.from_pretrained(base_model)
            try:
                model = AutoModelForMaskedLM.from_config(config, attn_implementation=attn)
            except (TypeError, ValueError):
                model = AutoModelForMaskedLM.from_config(config)
    else:
        tokenizer = load_tokenizer(vocab_file, max_len=max_len)
        print(f"[train] loaded tokenizer (vocab={tokenizer.vocab_size})")
        model = build_model(
            vocab_size=tokenizer.vocab_size,
            max_len=max_len,
            attn_implementation=attn,
        )
    print(f"[train] model params: {model.num_parameters():,}")

    full_dataset = LogLineDataset(
        tokenizer=tokenizer,
        file_path=cfg.get_path("paths.train_normal"),
        max_len=max_len,
    )

    # Hold out a small validation split so we can track eval_loss and keep the
    # best (lowest-loss) checkpoint instead of the possibly-diverged final one.
    import torch
    from torch.utils.data import random_split

    eval_ratio = float(tcfg.get("eval_ratio", 0.01))
    eval_size = max(1, int(len(full_dataset) * eval_ratio))
    train_size = len(full_dataset) - eval_size
    train_dataset, eval_dataset = random_split(
        full_dataset, [train_size, eval_size],
        generator=torch.Generator().manual_seed(int(tcfg.get("seed", 42))),
    )
    print(f"[train] examples: train={train_size:,} eval={eval_size:,}")

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=float(tcfg.get("mlm_probability", 0.15)),
        pad_to_multiple_of=8,
    )

    model_dir = ensure_dir(cfg.get_path("paths.model_dir"))
    eval_steps = int(tcfg.get("eval_steps", tcfg.get("save_steps", 50000)))
    seed = int(tcfg.get("seed", 42))
    training_args = TrainingArguments(
        output_dir=model_dir,
        overwrite_output_dir=True,
        seed=seed,
        data_seed=seed,
        full_determinism=bool(tcfg.get("full_determinism", False)),
        num_train_epochs=float(tcfg.get("num_train_epochs", 10)),
        per_device_train_batch_size=int(tcfg.get("per_device_train_batch_size", 8)),
        per_device_eval_batch_size=int(tcfg.get("per_device_eval_batch_size", 64)),
        learning_rate=float(tcfg.get("learning_rate", 5e-5)),
        weight_decay=float(tcfg.get("weight_decay", 0.01)),
        warmup_ratio=float(tcfg.get("warmup_ratio", 0.1)),
        lr_scheduler_type=str(tcfg.get("lr_scheduler_type", "cosine")),
        adam_beta2=float(tcfg.get("adam_beta2", 0.98)),
        adam_epsilon=float(tcfg.get("adam_epsilon", 1e-6)),
        bf16=bool(tcfg.get("bf16", torch.cuda.is_available())),
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=int(tcfg.get("save_total_limit", 2)),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=int(tcfg.get("logging_steps", 1000)),
        logging_dir=os.path.join(model_dir, "logs"),
        dataloader_num_workers=4,
        report_to=["tensorboard"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("[train] start")
    trainer.train()

    final_dir = os.path.join(model_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    # Save a copy of the config for reproducibility
    import shutil
    config_path = cfg._source_path if hasattr(cfg, '_source_path') else None
    if config_path and os.path.isfile(config_path):
        shutil.copy2(config_path, os.path.join(model_dir, "config_used.yaml"))
    print(f"[train] saved final model -> {final_dir}")
    return final_dir


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="LAnoBERT MLM training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--vocab_file", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    train(cfg, vocab_file=args.vocab_file)


if __name__ == "__main__":
    main()
