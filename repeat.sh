#!/bin/bash

while true; do
    # python -m app.main --fname configs/train/vitl16/pretrain-256px-16f.yaml && break
    python -m evals.main --fname configs/eval/vitl/k400.yaml  2>&1 | tee -a logs/k400_eval_$(date +%Y%m%d_%H%M%S).log
    echo "Run failed (exit $?).  Retrying in 10 s …" >&2
    sleep 10
done
echo "✓  Finished successfully."
