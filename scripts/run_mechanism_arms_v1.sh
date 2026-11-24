#!/usr/bin/env bash
# Matched retraining queue for the mechanism-control arms.
#
#   run_mechanism_arms_v1.sh <gpu> <arm>:<seed> [<arm>:<seed> ...]
#
# Each job trains the pair stage for 30 epochs from the common original-model
# initialization built from the same upstream binding checkpoint. Interrupted runs are
# resumed from pair/last.pt rather than restarted.
set -euo pipefail
GPU="$1"; shift
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK=/media/xd/code1/dyGF
for job in "$@"; do
  ARM="${job%%:*}"; SEED="${job##*:}"
  OUT="$WORK/outputs/mechanism_arms_v1/${ARM}_seed${SEED}"
  mkdir -p "$(dirname "$OUT")"   # the trainer creates $OUT itself
  echo "[$(date -Is)] gpu${GPU} start ${ARM} seed${SEED}"
  if [ -f "$OUT/plan.json" ]; then RESUME="--resume"; else RESUME=""; fi
  PYTHONPATH="$PKG/code:$WORK" OMP_NUM_THREADS=1 \
    /home/xd/anaconda3/envs/openfold/bin/python \
    "$PKG/code/train/train_mechanism_arms_v1.py" \
    --seed "$SEED" --arm "$ARM" --device "cuda:${GPU}" --epochs 30 \
    --output "$OUT" $RESUME 2>&1 | tail -40
  echo "[$(date -Is)] gpu${GPU} done ${ARM} seed${SEED}"
done
