"""Supplement the mechanism-arm comparison with the trade-off readouts the record asks for.

The scorer returns the primary readouts; the record also asks for precision, recall,
TP/FP/FN, clashes, sequence recovery and geometry trade-offs. Those come from
``contact_budget`` (contact accounting at the 8 A CA threshold) and ``geometry``
(peptide-bond and torsion terms), both reused unchanged from the archived code.

Inputs are the committed per-arm predictions from the locked evaluation; nothing is
regenerated, so the arm-to-arm comparison stays identical to the primary table.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train.train_abgaze_reconstructed import dump
from eval.audit_fape_and_cn_claims_v1 import contact_budget
from eval.evaluate_external_temporal_v1 import geometry
from research_audit.statistics import interval

WORKSPACE = Path('/media/xd/code1/dyGF')
ARMS = ('full', 'reset_z', 'pair_mlp', 'feature_only')
SEEDS = (2026, 2027, 2028)
LOWER_IS_BETTER = {'fp', 'n_predicted', 'predicted_to_native_ratio', 'clash',
                   'cn_error_gt_0_1_fraction', 'internal_cn_error_A',
                   'left_anchor_cn_error_A', 'right_anchor_cn_error_A',
                   'omega_planarity_deviation_deg', 'phi_native_abs_error_deg',
                   'psi_native_abs_error_deg', 'interface_distance_mae',
                   'all_pair_distance_mae'}
HIGHER_IS_BETTER = {'tp', 'n_native', 'precision', 'recall', 'aar', 'oracle_aar',
                    'mean_sample_aar'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replicates', type=int, default=10000)
    args = parser.parse_args()
    evaluation = args.evaluation.resolve()
    out = args.output.resolve()

    rows = torch.load(WORKSPACE / 'data/abgaze_reconstructed_v1/tensors/test.pt',
                      map_location='cpu', weights_only=False)
    ids = [row[0] for row in rows]
    clusters = {json.loads(line)['pdb']: f"cluster_{json.loads(line)['cluster']}"
                for line in (WORKSPACE / 'data/abgaze_reconstructed_v1/splits/test.json')
                .read_text().splitlines() if line.strip()}

    per_arm = {}
    for arm in ARMS:
        for seed in SEEDS:
            label = f'{arm}_{seed}'
            generated = json.loads((evaluation / label / 'generated.json').read_text())
            budget, geom = [], []
            for (sid, x, y), entry in zip(rows, generated):
                if sid != entry['sample_id']:
                    raise ValueError('prediction order mismatch')
                prediction = torch.tensor(entry['backbone'])
                counts = contact_budget(prediction, x, y)
                counts['sample_id'] = sid
                budget.append(counts)
                geom.append(dict(sample_id=sid, **{k: v for k, v in
                                                   _geometry_row(x, y, prediction).items()}))
            per_arm[label] = dict(budget=budget, geometry=geom)
            print(json.dumps(dict(label=label, precision=round(
                statistics.mean([b['precision'] for b in budget]), 4))), flush=True)

    metrics = ['tp', 'fp', 'fn', 'n_native', 'n_predicted', 'precision', 'recall',
               'predicted_to_native_ratio', 'interface_distance_mae',
               'all_pair_distance_mae', 'cn_error_gt_0_1_fraction',
               'internal_cn_error_A', 'left_anchor_cn_error_A', 'right_anchor_cn_error_A',
               'omega_planarity_deviation_deg', 'phi_native_abs_error_deg',
               'psi_native_abs_error_deg']
    summary = dict(metrics=metrics, arms={}, contrasts={},
                   orientation='improvement > 0 means the full arm was better, so the '
                               'intervention harmed that readout',
                   source='contact_budget (8 A CA contacts) and geometry() reuse from the '
                          'archived evaluation code; predictions are the locked ones')
    for arm in ARMS:
        summary['arms'][arm] = {}
        for key in metrics:
            values = []
            for seed in SEEDS:
                table = per_arm[f'{arm}_{seed}']
                source = table['budget'] if key in table['budget'][0] else table['geometry']
                clean = [row[key] for row in source if row[key] is not None]
                values.append(statistics.mean(clean) if clean else float('nan'))
            summary['arms'][arm][key] = dict(mean=statistics.mean(values),
                                             per_seed=values,
                                             sample_std=statistics.stdev(values))
    for arm in ARMS:
        if arm == 'full':
            continue
        summary['contrasts'][arm] = {}
        for key in metrics:
            # Some readouts are undefined for a complex with no near contacts; drop
            # those complexes for this metric only, keeping every seed aligned.
            used = [sid for sid in ids
                    if all(per_arm[f'{arm}_{seed}']['budget' if key in
                                   per_arm[f'{arm}_{seed}']['budget'][0] else 'geometry']
                           and {row['sample_id']: row[key] for row in
                                per_arm[f'{arm}_{seed}']['budget' if key in
                                        per_arm[f'{arm}_{seed}']['budget'][0] else 'geometry']}[sid]
                           is not None
                           for seed in SEEDS)]
            if not used:
                continue
            matrix = []
            for seed in SEEDS:
                arm_table = per_arm[f'{arm}_{seed}']
                full_table = per_arm[f'full_{seed}']
                source = 'budget' if key in arm_table['budget'][0] else 'geometry'
                arm_rows = {row['sample_id']: row[key] for row in arm_table[source]}
                full_rows = {row['sample_id']: row[key] for row in full_table[source]}
                matrix.append(np.array([arm_rows[sid] - full_rows[sid] for sid in used]))
            matrix = np.stack(matrix)
            sign = -1. if key in LOWER_IS_BETTER else 1.
            stats = interval(sign * matrix, [clusters[sid] for sid in used], args.replicates)
            summary['contrasts'][arm][key] = dict(
                arm_minus_full=float(matrix.mean()), improvement=stats['mean'],
                ci95=stats['conditional_group_ci95'],
                per_seed_mean=stats['per_seed_mean'],
                significant=bool(stats['conditional_group_ci95'][0] *
                                 stats['conditional_group_ci95'][1] > 0))
    out.mkdir(parents=True, exist_ok=False)
    dump(out / 'tradeoff_summary.json', summary)

    for arm in summary['contrasts']:
        significant = {key: round(value['improvement'], 4)
                       for key, value in summary['contrasts'][arm].items()
                       if value['significant']}
        print(json.dumps({arm: significant}), flush=True)


def _geometry_row(x, y, prediction):
    """Per-complex torsion and peptide-bond terms from the archived geometry()."""
    fake = dict(sample_id='x', backbone=prediction.tolist(), sequence_logits=[[0.] * 20])
    result = geometry([('x', x, y)], [fake])
    row = dict(result['per_complex'][0])
    row.pop('sample_id', None)
    return row


if __name__ == '__main__':
    main()
