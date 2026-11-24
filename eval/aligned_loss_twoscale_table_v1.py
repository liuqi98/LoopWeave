"""Aligned-loss weight comparison on the two-scale architecture: w=0 versus w=0.3.

Reads the per-complex scores written by ``eval/test_twoscale_ablations_v1.py`` (arms 'w0' and
'full'), reports mean +- sample SD across the three seeds, and the paired w=0 minus w=0.3
difference with a conditional group bootstrap.
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

PACKAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE / 'code'))
sys.path.insert(0, '/media/xd/code1/dyGF')
from research_audit.statistics import interval

METRICS = {'aligned_rmsd': 1, 'rmsd': 1, 'contact_f1': -1, 'interface_fape': 1,
           'cn_mae': 1, 'aar': -1, 'clash': 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scores', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replicates', type=int, default=10000)
    args = parser.parse_args()
    rows = [r for r in csv.DictReader(args.scores.open()) if r['arm'] in ('w0', 'full')]
    ids = sorted({r['sample_id'] for r in rows})
    groups = {r['sample_id']: r['group'] for r in rows}
    seeds = sorted({r['seed'] for r in rows})
    lookup = {(r['arm'], r['seed'], r['sample_id']): r for r in rows}
    table = {}
    for metric, sign in METRICS.items():
        seed_means = {}
        for arm in ('w0', 'full'):
            per_seed = [np.mean([float(lookup[arm, s, i][metric]) for i in ids]) for s in seeds]
            seed_means[arm] = dict(mean=float(np.mean(per_seed)), sd=float(np.std(per_seed, ddof=1))
                                   if len(per_seed) > 1 else 0.0, per_seed=per_seed)
        delta = np.array([sign * (np.mean([float(lookup['w0', s, i][metric]) for s in seeds])
                                  - np.mean([float(lookup['full', s, i][metric]) for s in seeds]))
                          for i in ids])
        stats = interval(np.array([delta]), [groups[i] for i in ids], args.replicates)
        lo, hi = stats['conditional_group_ci95']
        table[metric] = dict(w0=seed_means['w0'], w03=seed_means['full'],
                             w0_minus_w03=dict(mean=float(stats['mean']), ci_low=float(lo),
                                               ci_high=float(hi), significant=bool(lo * hi > 0)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(table, indent=2) + '\n')
    for metric, block in table.items():
        print('%-16s w=0 %.4f +- %.4f | w=0.3 %.4f +- %.4f | diff %+0.4f [%+0.4f,%+0.4f]' % (
            metric, block['w0']['mean'], block['w0']['sd'], block['w03']['mean'],
            block['w03']['sd'], block['w0_minus_w03']['mean'], block['w0_minus_w03']['ci_low'],
            block['w0_minus_w03']['ci_high']))


if __name__ == '__main__':
    main()
