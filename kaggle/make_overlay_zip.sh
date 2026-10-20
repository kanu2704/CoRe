#!/usr/bin/env bash
# Zip only the files we changed or added on top of CoRe's official code (~100 KB instead of ~86 MB).
# On Kaggle, clone CoRe at the same commit from GitHub and unzip this on top:
#   git clone https://github.com/nihx9919/CoRe.git CoRe && cd CoRe && git checkout d663cabc38e6d6f39ac5e1e4327d94f17f84fa59
#   unzip -o CoRe_psvm_changes.zip -d .
#
#   bash kaggle/make_overlay_zip.sh   ->  ../CoRe_psvm_changes.zip
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="$(cd .. && pwd)/CoRe_psvm_changes.zip"
rm -f "$OUT"
{ git ls-files -m; git ls-files -o --exclude-standard; } \
  | grep -v -E '__pycache__|\.pyc$|^\.venv/|^runs|^results' | sort -u | zip -q "$OUT" -@
echo "wrote $OUT ($(du -h "$OUT" | cut -f1)); upstream commit $(git rev-parse HEAD)"
