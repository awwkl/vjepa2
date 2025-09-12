#!/bin/bash

while true; do
    python -m app.main --fname configs/train/vitl16/pretrain-256px-16f.yaml && break
    # python -m evals.main --fname configs/eval/vitl/k400_finish.yaml
    # python -m evals.main --fname configs/eval/vitl/k400.yaml
    # python -m evals.main --fname configs/eval/vitl/in1k.yaml
    # python -m app.main --fname configs/train/vitl16/cooldown-256px-64f.yaml && break
    echo "Run failed (exit $?).  Retrying in 10 s …" >&2
    sleep 10
done
echo "✓  Finished successfully."
