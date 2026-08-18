#!/usr/bin/env bash
set -euo pipefail
EXP='01_midterm/2-4_model_validation/2-4-03_Randomness_Correction'
python -m pip install -q 'sentence-transformers>=3' 'torch>=2.2'
MIL_EPOCHS=18 python "$EXP/run_true_bag_mil_fusion.py" 2>&1 | tee "$EXP/true_bag_mil/true_bag_mil_run.log"
python -m json.tool "$EXP/true_bag_mil/true_bag_mil_summary.json"
