#!/usr/bin/env bash
# Fallback only (if the browser path fails): push one notebook with the Kaggle CLI,
# wait for it, download only its summary files outside the repo, and delete the
# run log the CLI always saves next to them.
#
# Usage (from the repo root, on Ablay's Mac):
#   bash scripts/kpush.sh notebooks/n0
#
# Needs: pip install kaggle==2.2.5 ; kaggle auth login
set -euo pipefail

nb_dir="${1:?usage: bash scripts/kpush.sh notebooks/<name>}"
meta="$nb_dir/kernel-metadata.json"
[ -f "$meta" ] || { echo "missing $meta"; exit 1; }

slug="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['id'])" "$meta")"
timeout_s="$(cat "$nb_dir/timeout_s" 2>/dev/null || echo 21600)"
out="$HOME/kaggle_out/${slug#*/}"
mkdir -p "$out"

echo "push $slug (cap ${timeout_s}s)"
kaggle kernels push -p "$nb_dir" -t "$timeout_s"

echo "waiting for $slug"
while true; do
  status="$(kaggle kernels status "$slug" 2>/dev/null | tr '[:upper:]' '[:lower:]' || true)"
  case "$status" in
    *complete*) echo "complete"; break ;;
    *error*|*cancel*) echo "run ended with: $status"; break ;;
  esac
  sleep 60
done

kaggle kernels output "$slug" -p "$out" --file-pattern "^(agg_.*|run)\.json$"
rm -f "$out"/*.log
echo "summary files in $out (log deleted). Paste only agg_*.json or run.json to Claude."
