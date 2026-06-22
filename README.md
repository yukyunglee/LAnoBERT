<div align="center">

# LAnoBERT: System Log Anomaly Detection based on BERT Masked Language Model (ASC 2023)

[![Paper](https://img.shields.io/badge/Paper-Applied%20Soft%20Computing%202023-blue)](https://doi.org/10.1016/j.asoc.2023.110689) [![arXiv](https://img.shields.io/badge/arXiv-2111.09564-red)](https://arxiv.org/abs/2111.09564) [![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Models-yukyung%2FLAnoBERT-yellow)](https://huggingface.co/yukyung/LAnoBERT)

**Yukyung Lee**, **Jina Kim**, **Pilsung Kang**

Korea University

</div>

This is the official implementation of **LAnoBERT**, a **parser-free** approach to system log anomaly detection. Unlike methods that depend on a log parser, LAnoBERT trains a BERT encoder from scratch on *normal* logs using the masked language modeling (MLM) objective. At inference it masks each word of a log line and measures how surprised the model is by the true token; lines the model finds surprising are flagged as anomalies.

It is evaluated on the BGL, HDFS, and Thunderbird log datasets, with pretrained checkpoints available on the HuggingFace Hub ([`yukyung/LAnoBERT`](https://huggingface.co/yukyung/LAnoBERT)).

> **What's new in this release.** The code is updated to the latest `transformers` / `torch`, every model is re-trained from scratch, and the checkpoints are released on the Hub.

## How it works

1. **Preprocess.** A simple regular-expression-based step replaces variable fields with placeholder tokens.
2. **Train.** A WordPiece tokenizer and BERT are trained from scratch on the normal logs with MLM.
3. **Score.** Each word is masked and predicted; the model's reaction to the true word gives a per-word signal — its cross-entropy (`error`) or the top prediction probability (`prob`).

These per-word signals are pooled into a line score (by mean or top-k). We recommend **`error_mean`**, the line's mean cross-entropy, which was the most robust in the updated version; `inference.py` saves the rest for comparison.

**Main results (from-scratch BERT, batch 32), AUROC / best-F1:**

| dataset      | `error_mean` (recommended) | fixed top-k (k=5) |
| ---          | ---                        | ---               |
| BGL          | **1.000 / 1.000**          | 1.000 / 0.999     |
| HDFS         | **0.997 / 0.969**          | 0.928 / 0.919     |
| Thunderbird  | **1.000 / 1.000**          | 1.000 / 0.999     |

## Released checkpoints (HuggingFace Hub)

The three main (batch-32) checkpoints are released as **`yukyung/LAnoBERT`**, with one subfolder per dataset (`bgl/`, `hdfs/`, `thunderbird/`).

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer
tok   = AutoTokenizer.from_pretrained("yukyung/LAnoBERT", subfolder="bgl")
model = AutoModelForMaskedLM.from_pretrained("yukyung/LAnoBERT", subfolder="bgl")
```

To score a dataset with the released model instead of training, set `hf_model: yukyung/LAnoBERT` and `hf_subfolder: bgl` under `inference:` in the config, then run `python -m lanobert.inference --config configs/bgl.yaml`.

## Layout

```
LAnoBERT/
├── configs/                  # bgl / hdfs / thunderbird (main) + ablations/
├── lanobert/                 # split, preprocess, tokenizer, dataset, train, inference, metrics
└── scripts/                  # download_data.sh, run_pipeline.sh, push_to_hub.py
```

## Install

```bash
pip install -r requirements.txt
```

## Data

Raw logs are not committed. Download them from [loghub](https://github.com/logpai/loghub) and place each `*.log` at the `paths.raw_log` in its config.

```bash
bash scripts/download_data.sh bgl     # -> data/BGL/BGL.log
bash scripts/download_data.sh hdfs    # -> data/HDFS/HDFS.log + anomaly_label.csv
bash scripts/download_data.sh tbird   # -> data/Thunderbird/Thunderbird.log
```

## Usage

Run the full pipeline (split → preprocess → tokenizer → train → inference):

```bash
bash scripts/run_pipeline.sh configs/bgl.yaml
```

Or step by step:

```bash
python -m lanobert.split      --config configs/bgl.yaml
python -m lanobert.preprocess --config configs/bgl.yaml --split train
python -m lanobert.preprocess --config configs/bgl.yaml --split test
python -m lanobert.tokenizer  --config configs/bgl.yaml
python -m lanobert.train      --config configs/bgl.yaml
python -m lanobert.inference  --config configs/bgl.yaml
```

Results (AUROC/F1 report, ROC png, `scores_*.npy`) are written to `outputs/<dataset>/results/`.

## Ablations

Every variant uses the same BERT encoder; they differ only in the vocabulary and how the weights are trained. The released checkpoints are the main model (row 1). Each cell is **AUROC / best-F1**: top line `error_mean`, bottom line fixed top-k (k=5).

| # | Vocabulary | Training | BGL | HDFS | Thunderbird |
| --- | --- | --- | --- | --- | --- |
| 1 | log-specific (main) | from-scratch | **1.000 / 1.000**<br>1.000 / 0.999 | **0.997 / 0.969**<br>0.928 / 0.919 | **1.000 / 1.000**<br>1.000 / 0.999 |
| 2 | `bert-base` | from-scratch (rand-init) | 1.000 / 0.999<br>0.999 / 0.988 | 0.761 / 0.489<br>0.651 / 0.419 | 0.801 / 0.529<br>0.824 / 0.495 |
| 3 | `bert-base` | TAPT (warm-start) | 0.877 / 0.814<br>0.997 / 0.991 | 0.992 / 0.955<br>0.821 / 0.874 | 0.473 / 0.261<br>0.978 / 0.802 |
| 4 | `bert-base` | none (off-the-shelf) | 0.394 / 0.528<br>0.231 / 0.436 | 0.961 / 0.761<br>0.587 / 0.559 | 0.915 / 0.720<br>0.826 / 0.516 |

The main model (row 1) is the only one strong and stable across all three datasets: the log-specific vocabulary is the key factor (rows 1 vs. 2), and the off-the-shelf model (row 4) is weak. Row 1 uses the main config (`configs/<dataset>.yaml`); rows 2–4 use `configs/ablations/<dataset>_{bertbase_init,bertbase_tapt,pretrained}.yaml`.

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
