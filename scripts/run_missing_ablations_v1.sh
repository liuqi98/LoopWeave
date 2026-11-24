#!/usr/bin/env bash
# GPU runbook for the experiments that could not be trained in the CPU-only sandbox.
#
# Every step is matched to the archived 2x2 protocol: same seed, same split, same
# frozen generator caches and binding features, same optimizer/loss/guards, and
# identical epoch budgets. Nothing here selects a checkpoint on test results.
#
# Usage:
#   bash code/scripts/run_missing_ablations_v1.sh train-base
#   bash code/scripts/run_missing_ablations_v1.sh train-interface
#   bash code/scripts/run_missing_ablations_v1.sh score-2x2
#   bash code/scripts/run_missing_ablations_v1.sh score-interface
#   bash code/scripts/run_missing_ablations_v1.sh factorial
set -euo pipefail

WORKSPACE="${WORKSPACE:-/media/xd/code1/dyGF}"
DEVICE="${DEVICE:-cuda:0}"
SEEDS="${SEEDS:-2026 2027 2028}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/../.." && pwd)"
export PYTHONPATH="$PKG/code:${PYTHONPATH:-}"

train () {  # variant
  local variant="$1"
  for seed in $SEEDS; do
    python "$PKG/code/train/train_matched_ablation_v3.py" \
      --workspace "$WORKSPACE" --seed "$seed" --variant "$variant" \
      --output "$WORKSPACE/outputs/matched_ablation_v3_${variant}_seed${seed}" \
      --device "$DEVICE"
  done
}

case "${1:-}" in
  train-base)
    # The missing 2x2 cell: the binding branch is bypassed for the 60-epoch
    # binding stage, and both branches are bypassed for the 30-epoch pair stage.
    train no_pair_no_binding
    ;;
  train-interface)
    train features_surface_geometry
    train features_electrostatics_chemistry
    train features_none
    train static_neighborhood
    ;;
  score-2x2)
    # All four cells in one directory; refuses to run unless every cell exists.
    # Writes scores.csv with H3-cluster dependence groups for the factorial step.
    python "$PKG/code/eval/test_matched_ablations_v3.py" \
      --workspace "$WORKSPACE" --device "$DEVICE" \
      --output "$WORKSPACE/outputs/matched_ablations_test_v3_2x2" \
      --variants full only_pair only_binding base --seeds $SEEDS
    ;;
  score-interface)
    # Kept in a separate directory: interface levels are not factorial arms.
    python "$PKG/code/eval/test_matched_ablations_v3.py" \
      --workspace "$WORKSPACE" --device "$DEVICE" \
      --output "$WORKSPACE/outputs/matched_ablations_test_v3_interface" \
      --variants full features_surface_geometry features_electrostatics_chemistry \
                 features_none static_neighborhood --seeds $SEEDS
    ;;
  factorial)
    # Reverse the sign of error metrics, average paired seeds, resample whole
    # dependence groups, and report the interaction plus conditional/main effects.
    python -m research_audit factorial \
      --input "$WORKSPACE/outputs/matched_ablations_test_v3_2x2/scores.csv" \
      --output "$WORKSPACE/outputs/matched_ablations_test_v3_2x2/interaction.json" \
      --metric aligned_rmsd:min --metric interface_fape:min --metric contact_f1:max \
      --metric cn_mae:min --metric clash:min \
      --seeds $SEEDS
    ;;
  *)
    echo "usage: $0 {train-base|train-interface|score-2x2|score-interface|factorial}" >&2
    exit 2
    ;;
esac
