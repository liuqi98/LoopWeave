#!/usr/bin/env bash
# Matched retraining queue for the three sequence-prior conditions.
#   run_sequence_prior_arms_v1.sh <gpu> <condition>:<seed> ...
set -euo pipefail
GPU="$1"; shift
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK=/media/xd/code1/dyGF
for job in "$@"; do
  CONDITION="${job%%:*}"; SEED="${job##*:}"
  OUT="$WORK/outputs/sequence_prior_arms_v1/${CONDITION}_seed${SEED}"
  mkdir -p "$(dirname "$OUT")"
  echo "[$(date -Is)] gpu${GPU} start ${CONDITION} seed${SEED}"
  if [ -f "$OUT/plan.json" ]; then RESUME="--resume"; else RESUME=""; fi
  PYTHONPATH="$PKG/code:$WORK" OMP_NUM_THREADS=1 \
    /home/xd/anaconda3/envs/openfold/bin/python \
    "$PKG/code/train/train_sequence_prior_arms_v1.py" \
    --prior-root "$PKG/results/sequence_prior_v1" \
    --seed "$SEED" --condition "$CONDITION" --device "cuda:${GPU}" --epochs 30 \
    --output "$OUT" $RESUME 2>&1 | tail -40
  echo "[$(date -Is)] gpu${GPU} done ${CONDITION} seed${SEED}"
done
