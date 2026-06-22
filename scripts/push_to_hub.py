#!/usr/bin/env python
"""Push the trained LAnoBERT checkpoints to the HuggingFace Hub.

All three MAIN (batch-32, from-scratch) checkpoints go into a SINGLE repo, one
subfolder per dataset:

    yukyung/LAnoBERT
    ├── bgl/           (config.json, model.safetensors, tokenizer files)
    ├── hdfs/
    ├── thunderbird/
    └── README.md      (model card)

The repo is created automatically (`create_repo(exist_ok=True)`) -- you do NOT
need to make it on the website first. You DO need a write token:

    huggingface-cli login            # or: export HF_TOKEN=hf_xxx

Usage:
    python scripts/push_to_hub.py                       # push all 3 datasets
    python scripts/push_to_hub.py --repo me/lanobert    # custom repo id
    python scripts/push_to_hub.py --only bgl            # one dataset
    python scripts/push_to_hub.py --private             # create as private
    python scripts/push_to_hub.py --dry-run             # print actions only
"""
import argparse
import os
import sys

# dataset -> (main config, hub subfolder)
DATASETS = {
    "bgl": ("configs/bgl.yaml", "bgl"),
    "hdfs": ("configs/hdfs.yaml", "hdfs"),
    "thunderbird": ("configs/thunderbird.yaml", "thunderbird"),
}

# files in model/final we don't want to publish
IGNORE = ["training_args.bin", "optimizer.pt", "scheduler.pt", "*.out", "rng_state*"]

# Code repository URL shown in the model card. Replace with the public repo
# once it is online (e.g. https://github.com/yukyunglee/LAnoBERT).
CODE_URL = "https://github.com/yukyunglee/LAnoBERT"

MODEL_CARD = """---
license: cc-by-4.0
library_name: transformers
pipeline_tag: fill-mask
tags:
  - anomaly-detection
  - log-analysis
  - lanobert
  - bert
---

# LAnoBERT checkpoints (BGL / HDFS / Thunderbird)

From-scratch, custom-vocabulary BERT encoders trained with a **masked-language-
modeling objective on normal system logs only** (no next-sentence prediction),
following the **LAnoBERT** log anomaly detection method. One checkpoint per
dataset, stored as a subfolder of this repo.

- **Code:** {code_url}
- **Paper:** Yukyung Lee, Jina Kim, Pilsung Kang. *LAnoBERT: System log anomaly
  detection based on BERT masked language model.* Applied Soft Computing,
  Vol. 146, 2023, 110689. https://doi.org/10.1016/j.asoc.2023.110689

| subfolder      | dataset      | vocab | batch | AUROC / best-F1 (`error_mean`) |
| ---            | ---          | ---   | ---   | ---                            |
| `bgl`          | BGL          | 1000  | 32    | 1.000 / 1.000                  |
| `hdfs`         | HDFS         | 200   | 32    | 0.997 / 0.969                  |
| `thunderbird`  | Thunderbird  | 10000 | 32    | 1.000 / 1.000                  |

## Usage

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

sub = "bgl"  # or "hdfs" / "thunderbird"
tok = AutoTokenizer.from_pretrained("{repo}", subfolder=sub)
model = AutoModelForMaskedLM.from_pretrained("{repo}", subfolder=sub)
```

## Scoring

Anomaly score = **mean per-word cross-entropy** over a log line (`error_mean`),
which is length-adaptive and balanced across datasets. See the code repository
for the full inference pipeline.

## Citation

```bibtex
@article{lee2023lanobert,
  title   = {LAnoBERT: System log anomaly detection based on BERT masked language model},
  author  = {Lee, Yukyung and Kim, Jina and Kang, Pilsung},
  journal = {Applied Soft Computing},
  volume  = {146},
  pages   = {110689},
  year    = {2023},
  issn    = {1568-4946},
  doi     = {10.1016/j.asoc.2023.110689}
}
```
"""


def resolve_model_dir(config_path):
    """Return the trained-model directory for a config (prefers .../final)."""
    sys.path.insert(0, os.getcwd())
    from lanobert.utils import load_config

    cfg = load_config(config_path)
    base = cfg.get_path("paths.model_dir")
    final = os.path.join(base, "final")
    return final if os.path.isdir(final) else base


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="yukyung/LAnoBERT",
                    help="target Hub repo id (default: yukyung/LAnoBERT)")
    ap.add_argument("--only", choices=list(DATASETS), default=None,
                    help="push a single dataset instead of all three")
    ap.add_argument("--private", action="store_true", help="create the repo as private")
    ap.add_argument("--token", default=None, help="HF write token (else uses login / HF_TOKEN)")
    ap.add_argument("--dry-run", action="store_true", help="print what would be uploaded, do nothing")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    token = args.token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    targets = {args.only: DATASETS[args.only]} if args.only else DATASETS

    # resolve + validate all local model dirs first
    plan = []
    for name, (config_path, subfolder) in targets.items():
        model_dir = resolve_model_dir(config_path)
        if not os.path.isdir(model_dir):
            sys.exit(f"[push] ERROR: model dir not found for {name}: {model_dir}\n"
                     f"        train it first (run_pipeline.sh {config_path}).")
        plan.append((name, model_dir, subfolder))
        print(f"[push] {name:12s} {model_dir}  ->  {args.repo}/{subfolder}")

    if args.dry_run:
        print("[push] --dry-run: nothing uploaded.")
        return

    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    print(f"[push] repo ready: https://huggingface.co/{args.repo}")

    for name, model_dir, subfolder in plan:
        print(f"[push] uploading {name} -> {subfolder}/ ...")
        api.upload_folder(
            repo_id=args.repo,
            repo_type="model",
            folder_path=model_dir,
            path_in_repo=subfolder,
            ignore_patterns=IGNORE,
            commit_message=f"Add {name} (batch-32 from-scratch LAnoBERT) checkpoint",
        )

    # model card at repo root
    import io
    card = MODEL_CARD.replace("{repo}", args.repo).replace("{code_url}", CODE_URL).encode()
    api.upload_file(
        path_or_fileobj=io.BytesIO(card),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="model",
        commit_message="Add model card",
    )
    print(f"[push] done -> https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
