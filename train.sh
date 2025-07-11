#!/usr/bin/env bash
while true; do
    python -m app.main --fname configs/train/vitl16/pretrain-256px-16f_babyview.yaml && break
    echo "Run failed (exit $?).  Retrying in 10 s …" >&2
    sleep 10
done
echo "✓  Finished successfully."
