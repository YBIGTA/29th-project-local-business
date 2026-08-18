#!/usr/bin/env bash
set -euo pipefail

# Run from repository root after: conda activate ybigta-amazon
EXP_DIR="01_midterm/2-4_model_validation/2-4-03_Randomness_Correction"

python -m pip install --upgrade "python-igraph>=0.11" "leidenalg>=0.10" "pynndescent>=0.5" \
  "sentence-transformers>=3" "hdbscan>=0.8"

# First ablation: frozen MiniLM. This is the recommended run before the much
# slower fold-wise SimCSE adaptation run.
PHRASE_DOMAIN_ADAPT=0 \
PHRASE_MAX_CLUSTER_FIT=40000 \
PHRASE_MAX_RETAINED_CLUSTERS=40 \
PHRASE_BOOTSTRAPS=3 \
python "$EXP_DIR/run_phrase_dualview_burst_fusion.py" \
  2>&1 | tee "$EXP_DIR/phrase_dualview_burst/phrase_dualview_burst_run.log"

python -m json.tool "$EXP_DIR/phrase_dualview_burst/phrase_dualview_burst_summary.json"

python - <<'PY'
import pandas as pd

p = "01_midterm/2-4_model_validation/2-4-03_Randomness_Correction/phrase_dualview_burst/phrase_dualview_burst_metrics.csv"
df = pd.read_csv(p)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
print(df.to_string(index=False))
PY
