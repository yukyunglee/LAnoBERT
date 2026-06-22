#!/usr/bin/env bash
# End-to-end LAnoBERT pipeline for one dataset.
# Usage: bash scripts/run_pipeline.sh configs/bgl.yaml
#
# Assumes the raw log is already downloaded to the path in the config
# (see scripts/download_data.sh).
# Steps whose output files already exist are skipped automatically.
set -euo pipefail

CONFIG="${1:-configs/bgl.yaml}"
echo "==> config: $CONFIG"

# helper: read a yaml value by key (simple flat lookup via python)
yq() { python -c "import yaml,sys; c=yaml.safe_load(open('$CONFIG')); 
keys='$1'.split('.'); v=c
for k in keys: v=v[k]
print(v)"; }

TOK_DIR=$(yq paths.tokenizer_dir)
MODEL_DIR=$(yq paths.model_dir)

echo "==> [1-2/5] split + preprocess (lock-guarded, shared across parallel jobs)"
bash scripts/ensure_data.sh "$CONFIG"

echo "==> [3/5] train tokenizer"
if [ -d "$TOK_DIR" ] && [ "$(ls -A "$TOK_DIR" 2>/dev/null)" ]; then
    echo "    SKIP (already exists: $TOK_DIR)"
else
    python -m lanobert.tokenizer --config "$CONFIG"
fi

echo "==> [4/5] train MLM"
if [ -d "$MODEL_DIR/final" ]; then
    echo "    SKIP (already exists: $MODEL_DIR/final)"
else
    python -m lanobert.train --config "$CONFIG"
fi

echo "==> [5/5] inference + evaluate"
python -m lanobert.inference --config "$CONFIG"

echo "==> done. results under outputs/<dataset>/results"
