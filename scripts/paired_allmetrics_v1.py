"""Paired all-metric contrasts against RAAD, and a per-complex wide table.

``research_audit baselines`` needs a complete metric grid, but a few metrics are
undefined on a native-defined subset (interface-restricted FAPE and interface
distance MAE require at least one antigen residue within 8 A of the native loop).
This script therefore reuses the audited ``research_audit.statistics.interval``
group-bootstrap directly on the subset of complexes where a metric is defined for
every seed of both arms, and records that subset size next to each interval.

Pairing follows the same convention as ``research_audit baselines``: average each
method's seeds within a complex first, then pair complexes, then resample whole
dependence groups. Seeds never become extra samples.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT.parent
sys.path.insert(0, str(ROOT))
from research_audit.common import digest, finite
from research_audit.statistics import interval

HIGHER = ('aar', 'first_sample_aar', 'mean_sample_aar', 'oracle_aar', 'contact_f1',
          'precision', 'recall', 'tp', 'n_native', 'n_predicted')
LOWER = ('aligned_rmsd', 'rmsd', 'cn_mae', 'cn_p90', 'cn_frac_gt_0_1', 'cn_frac_gt_0_2',
         'cn_left_anchor', 'cn_right_anchor', 'cn_internal',
         'peptide_angle_at_C_dev_deg', 'peptide_angle_at_N_dev_deg',
         'omega_planarity_deviation_deg', 'omega_native_abs_error_deg',
         'phi_native_abs_error_deg', 'psi_native_abs_error_deg', 'clash',
         'interface_fape', 'fape_raw_all', 'fape_raw_interface',
         'fape_clamp5_all', 'fape_clamp5_interface',
         'fape_clamp10_all', 'fape_clamp10_interface',
         'fape_clamp20_all', 'fape_clamp20_interface',
         'interface_distance_mae', 'all_pair_distance_mae')
REFERENCE = 'raad'


def load(path):
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        return list(reader), reader.fieldnames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True,
                        help='dev60_all_metrics.csv produced by compare_dev60_allmetrics_v1.py')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replicates', type=int, default=10000)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows, fields = load(args.input)
    arms = [a for a in dict.fromkeys(r['arm'] for r in rows) if a != REFERENCE]
    metrics = [m for m in fields if m not in ('sample_id', 'group', 'length', 'arm', 'seed')]
    directions = {m: 1 for m in HIGHER if m in metrics}
    directions.update({m: -1 for m in LOWER if m in metrics})
    descriptive = [m for m in metrics if m not in directions]

    # arm -> seed -> sample -> values
    table = {}
    for row in rows:
        table.setdefault(row['arm'], {}).setdefault(row['seed'], {})[row['sample_id']] = row
    ids = sorted({r['sample_id'] for r in rows})
    groups = {r['sample_id']: r['group'] for r in rows}
    lengths = {r['sample_id']: r['length'] for r in rows}

    def value(arm, sid, metric):
        """Mean over that arm's seeds, or None if any seed lacks the metric."""
        vals = []
        for seed_rows in table[arm].values():
            row = seed_rows.get(sid)
            if row is None:
                return None
            raw = row.get(metric, '')
            if raw in ('', None):
                return None
            vals.append(finite(raw, f'{arm}/{sid}/{metric}'))
        return float(np.mean(vals))

    contrasts = {}
    for arm in arms:
        contrasts[arm] = {}
        for metric, direction in directions.items():
            deltas, used = [], []
            for sid in ids:
                a, b = value(arm, sid, metric), value(REFERENCE, sid, metric)
                if a is None or b is None:
                    continue
                deltas.append(direction * (a - b))
                used.append(sid)
            if len(used) < 3:
                contrasts[arm][metric] = dict(status='not_estimable', n=len(used))
                continue
            stats = interval(np.asarray([deltas], dtype=float),
                             [groups[s] for s in used], args.replicates)
            contrasts[arm][metric] = dict(
                sign='positive means the arm beats RAAD', mean=stats['mean'],
                ci95=stats['conditional_group_ci95'], n=len(used),
                n_groups=len({groups[s] for s in used}),
                arm_mean=float(np.mean([value(arm, s, metric) for s in used])),
                raad_mean=float(np.mean([value(REFERENCE, s, metric) for s in used])),
                significant=bool(stats['conditional_group_ci95'][0] *
                                 stats['conditional_group_ci95'][1] > 0))

    # Per-complex wide table for the headline metrics, one column per arm.
    headline = ['aligned_rmsd', 'rmsd', 'contact_f1', 'precision', 'recall',
                'interface_fape', 'cn_mae', 'clash', 'aar']
    wide_fields = ['sample_id', 'group', 'length']
    for arm in [REFERENCE] + arms:
        for metric in headline:
            wide_fields.append(f'{arm}.{metric}')
    wide = []
    for sid in ids:
        row = dict(sample_id=sid, group=groups[sid], length=lengths[sid])
        for arm in [REFERENCE] + arms:
            for metric in headline:
                value_ = value(arm, sid, metric)
                row[f'{arm}.{metric}'] = '' if value_ is None else round(value_, 6)
        wide.append(row)
    with (out / 'per_complex_wide.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=wide_fields)
        writer.writeheader()
        writer.writerows(wide)

    (out / 'contrasts_vs_raad.json').write_text(json.dumps(dict(
        reference=REFERENCE, arms=arms, n_complexes=len(ids),
        n_groups=len(set(groups.values())),
        input=str(args.input), input_sha256=digest(args.input),
        pairing='average each arm\'s seeds within a complex, then pair complexes, then '
                'resample whole dependence groups; seeds are not extra samples',
        subset_note='a metric undefined for a complex (genuine native-defined subset such as '
                    'interface FAPE without an 8 A native contact) is excluded from that '
                    'contrast and n is reported; no imputation',
        descriptive_metrics=descriptive,
        contrasts=contrasts), indent=2) + '\n')

    order = ['aligned_rmsd', 'rmsd', 'interface_fape', 'contact_f1', 'precision', 'recall',
             'cn_mae', 'cn_frac_gt_0_2', 'clash', 'fape_raw_interface',
             'fape_clamp10_interface', 'interface_distance_mae', 'aar', 'oracle_aar']
    for arm in arms:
        print(f'== {arm} vs {REFERENCE} (positive = {arm} better)')
        for metric in order:
            entry = contrasts[arm].get(metric)
            if not entry or entry.get('status') == 'not_estimable':
                continue
            star = '*' if entry['significant'] else ' '
            print('   %-26s d=%+.4f [%+.4f,%+.4f] %s n=%d' % (
                metric, entry['mean'], entry['ci95'][0], entry['ci95'][1], star, entry['n']))
    print(f'wrote {out / "contrasts_vs_raad.json"} and {out / "per_complex_wide.csv"}')


if __name__ == '__main__':
    main()
