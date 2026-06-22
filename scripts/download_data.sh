#!/usr/bin/env bash
# Download raw log datasets from loghub (https://github.com/logpai/loghub).
#
# loghub hosts the full datasets on Zenodo (record 8196385). Direct download
# URLs occasionally change; if a link 404s, open the loghub README and grab the
# current "Full dataset" link for that dataset.
#
# Usage:
#   bash scripts/download_data.sh bgl
#   bash scripts/download_data.sh hdfs
#   bash scripts/download_data.sh tbird       # ~30GB, takes a while
set -euo pipefail

DATASET="${1:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

dl() {  # dl <url> <out>
  mkdir -p "$(dirname "$2")"
  echo "downloading -> $2"
  if command -v wget >/dev/null; then wget -c -O "$2" "$1"; else curl -L -o "$2" "$1"; fi
}

case "$DATASET" in
  bgl)
    dl "https://zenodo.org/records/8196385/files/BGL.zip" "$ROOT/data/BGL/BGL.zip"
    unzip -o "$ROOT/data/BGL/BGL.zip" -d "$ROOT/data/BGL/"
    echo "expected: data/BGL/BGL.log"
    ;;
  hdfs)
    dl "https://zenodo.org/records/8196385/files/HDFS_v1.zip" "$ROOT/data/HDFS/HDFS_v1.zip"
    unzip -o "$ROOT/data/HDFS/HDFS_v1.zip" -d "$ROOT/data/HDFS/"
    echo "expected: data/HDFS/HDFS.log  +  data/HDFS/anomaly_label.csv"
    ;;
  tbird|thunderbird)
    dl "https://zenodo.org/records/8196385/files/Thunderbird.tar.gz" "$ROOT/data/Thunderbird/Thunderbird.tar.gz"
    tar -xzf "$ROOT/data/Thunderbird/Thunderbird.tar.gz" -C "$ROOT/data/Thunderbird/"
    echo "expected: data/Thunderbird/Thunderbird.log"
    ;;
  *)
    echo "usage: bash scripts/download_data.sh [bgl|hdfs|tbird]"
    echo "datasets are listed at https://github.com/logpai/loghub (Zenodo record 8196385)"
    exit 1
    ;;
esac

echo "done. Verify the raw_log path in the matching config, then run scripts/run_pipeline.sh."
