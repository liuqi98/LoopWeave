#!/usr/bin/env bash
# One-command GPU pipeline for the two GPU-blocked priorities.
#
#   1. three matched seed runs of the missing 2x2 cell (no_pair_no_binding)
#   2. score all four cells together, re-audit the protocol, run the factorial
#   3. DiffAb on all 20 external complexes, then the baseline comparison
#
# Preconditions verified before anything starts:
#   * torch reports a CUDA device (i.e. /dev/nvidia* is passed through)
#   * all three parent replicas are complete
# Steps are resumable: a completed training run is skipped, never overwritten.
#
# Usage:
#   bash code/scripts/run_gpu_priority_v1.sh            # everything
#   bash code/scripts/run_gpu_priority_v1.sh base       # only the 2x2 training
#   bash code/scripts/run_gpu_priority_v1.sh score      # scoring + factorial
#   bash code/scripts/run_gpu_priority_v1.sh diffab     # DiffAb + comparison
set -euo pipefail

WORKSPACE="${WORKSPACE:-/media/xd/code1/dyGF}"
DEVICE="${DEVICE:-cuda:0}"
SEEDS="${SEEDS:-2026 2027 2028}"
PY="${PY:-python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/../.." && pwd)"
STAGE="${1:-all}"
export PYTHONPATH="$PKG/code:${PYTHONPATH:-}"

fail () { echo "ABORT: $*" >&2; exit 1; }

preflight () {
  "$PY" - <<'EOF' || fail "torch cannot see a CUDA device; pass /dev/nvidia* into the container"
import torch, sys
if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
    print('cuda_available:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())
    sys.exit(2)
print('cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count(),
      'name', torch.cuda.get_device_name(0), 'torch', torch.__version__)
EOF
  for seed in $SEEDS; do
    [ -f "$WORKSPACE/outputs/full_pipeline_seed${seed}_v1/status.json" ] \
      || fail "missing parent replica for seed $seed"
  done
}

train_base () {
  for seed in $SEEDS; do
    local run="$WORKSPACE/outputs/matched_ablation_v3_no_pair_no_binding_seed${seed}"
    if [ -f "$run/status.json" ] && grep -q '"complete"' "$run/status.json"; then
      echo "skip seed $seed (already complete)"
      continue
    fi
    [ -e "$run" ] && fail "$run exists but is not complete; inspect it, do not overwrite"
    echo "== training base seed $seed on $DEVICE"
    "$PY" "$PKG/code/train/train_matched_ablation_v3.py" \
      --workspace "$WORKSPACE" --seed "$seed" --variant no_pair_no_binding \
      --device "$DEVICE" --output "$run"
  done
}

score_and_test () {
  local out="$WORKSPACE/outputs/matched_ablations_test_v3_2x2"
  [ -e "$out" ] && fail "$out already exists; refusing to overwrite a scored 2x2"
  "$PY" "$PKG/code/eval/test_matched_ablations_v3.py" \
    --workspace "$WORKSPACE" --device "$DEVICE" --output "$out" \
    --variants full only_pair only_binding base --seeds $SEEDS
  # Audit the four cells before interpreting anything (rule R7).
  "$PY" -m research_audit audit-history --workspace "$WORKSPACE" --verify-artifacts \
    --seeds $SEEDS --output "$out/protocol_audit.json"
  "$PY" - <<EOF
import json, sys
doc = json.load(open('$out/protocol_audit.json'))
if not doc.get('complete'):
    print('audit is not complete; the interaction must NOT be reported', file=sys.stderr)
    sys.exit(3)
EOF
  "$PY" -m research_audit factorial --input "$out/scores.csv" \
    --output "$out/interaction.json" --seeds $SEEDS \
    --metric aligned_rmsd:min --metric interface_fape:min --metric contact_f1:max \
    --metric cn_mae:min --metric clash:min --metric rmsd:min \
    --protocol-audit "$out/protocol_audit.json"
  echo "== interaction written to $out/interaction.json (apply rules R1-R7 in REPORT section 1d)"
}

run_diffab () {
  local out="$PKG/results/diffab_external_locked_v1"
  [ -e "$out" ] && fail "$out already exists; move it aside before regenerating"
  "$PY" "$PKG/code/baselines/generate_diffab_external_v1.py" \
    --orig "$WORKSPACE" --output "$out" --threads 4
  "$PY" - <<EOF
import json, sys
doc = json.load(open('$out/generation_audit.json'))
if not doc.get('complete'):
    print('DiffAb cohort incomplete; NOT scoring it', file=sys.stderr)
    sys.exit(3)
EOF
  "$PY" "$PKG/code/eval/compare_external_baselines_v2.py" \
    --output "$PKG/results/external_baselines_v2_diffab"
  for method in diffab; do
    "$PY" -m research_audit baselines \
      --input "$PKG/results/external_baselines_v2_diffab/ours_vs_${method}.csv" \
      --output "$PKG/results/external_baselines_v2_diffab/stats_ours_vs_${method}.json" \
      --reference ours --comparator "$method" \
      --metric aligned_rmsd:min --metric rmsd:min --metric contact_f1:max \
      --metric cn_mae:min --metric interface_fape:min --metric clash:min --metric aar:max
  done
  echo "== DiffAb added; reproduce the merged table with compare_external_baselines_v2.py"
}

preflight
case "$STAGE" in
  base)    train_base ;;
  score)   score_and_test ;;
  diffab)  run_diffab ;;
  all)     train_base; score_and_test; run_diffab ;;
  *)       fail "unknown stage: $STAGE (use base|score|diffab|all)" ;;
esac
echo "GPU pipeline stage '$STAGE' finished"
