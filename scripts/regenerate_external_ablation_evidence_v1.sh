#!/usr/bin/env bash
# Regenerate the CPU-only external-ablation evidence chain from committed checkpoints.
#
# Inputs are the archived validation-selected checkpoints, the locked external
# cohort and its group definition. Nothing is trained and nothing is selected on
# evaluation results. Output directories must not exist; every report is written
# once and refuses to overwrite.
#
# Usage:
#   bash code/scripts/regenerate_external_ablation_evidence_v1.sh
# Optional: GROUP_SOURCE=/path/to/evaluation_lock.json to point at another locked cohort definition.
set -euo pipefail

GROUP_SOURCE="${GROUP_SOURCE:-/media/xd/code1/dyGF/outputs/external_temporal_evaluation_v1/evaluation_lock.json}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/../.." && pwd)"
PY="${PY:-/home/xd/anaconda3/envs/openfold/bin/python}"
export PYTHONPATH="$PKG/code:${PYTHONPATH:-}"
cd "$PKG"

"$PY" code/eval/evaluate_external_ablations_v1.py \
  --output results/external_interface_ablations_v1 --groups "$GROUP_SOURCE"

"$PY" code/scripts/build_paired_subsets_v1.py \
  --input results/external_interface_ablations_v1/scores.csv --reference full

for arm in mask_surface_geometry mask_electrostatics_chemistry mask_none static_neighborhood; do
  "$PY" -m research_audit paired \
    --input "results/external_interface_ablations_v1/paired_full_vs_${arm}.csv" \
    --output "results/external_interface_ablations_v1/stats_full_vs_${arm}.json" \
    --reference full --comparator "$arm" --long-min 16 \
    --metric aligned_rmsd:min --metric rmsd:min --metric contact_f1:max \
    --metric cn_mae:min --metric interface_fape:min --metric clash:min
done

"$PY" -m research_audit drift \
  --input results/external_interface_ablations_v1/drift_records.json \
  --output results/external_interface_ablations_v1/numerical_drift.json \
  --metric aligned_rmsd --metric contact_f1 --metric rmsd --metric cn_mae

"$PY" -m research_audit costs \
  --input results/external_interface_ablations_v1/costs.csv \
  --output results/external_interface_ablations_v1/costs.json

"$PY" code/scripts/consolidate_geometry_and_cost_v1.py \
  --ablations results/external_interface_ablations_v1 --output results/geometry_and_cost_v1

"$PY" -m research_audit costs \
  --input results/geometry_and_cost_v1/costs.csv \
  --output results/geometry_and_cost_v1/costs.json

"$PY" code/eval/audit_fape_and_cn_claims_v1.py \
  --output results/fape_cn_audit_v2 --groups "$GROUP_SOURCE"

echo "external ablation evidence and FAPE/C-N audit regenerated"
