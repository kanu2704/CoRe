#!/usr/bin/env bash
# Zip this repo for upload as a Kaggle Dataset (SoftGym, git history and caches left out).
#   bash kaggle/make_upload_zip.sh   ->  ../CoRe_psvm_kaggle.zip
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="$(cd .. && pwd)/CoRe_psvm_kaggle.zip"
rm -f "$OUT"
zip -qr "$OUT" . -x 'softgym/*' '.git/*' '*__pycache__*' '*.pyc' 'runs/*' 'results/*' '.venv/*'
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
