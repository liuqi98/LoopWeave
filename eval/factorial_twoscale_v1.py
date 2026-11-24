"""2x2 factorial summary for the two-scale architecture.

Reads the scores produced by ``eval/test_twoscale_ablations_v1.py`` and reports arm means,
the full-minus-base contrast, the two single-branch contrasts, and the interaction of the
environment and pair branches, each with a conditional group bootstrap over H3 clusters.
"""
import argparse, csv, json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

PACKAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE / 'code'))
sys.path.insert(0, '/media/xd/code1/dyGF')
from research_audit.statistics import interval

ARMS = ('base', 'only_binding', 'only_pair', 'full')
METRICS = {'aligned_rmsd': 1, 'rmsd': 1, 'interface_fape': 1, 'contact_f1': -1, 'cn_mae': 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scores', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replicates', type=int, default=10000)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.scores.open()))
    ids = sorted({r['sample_id'] for r in rows})
    groups = {r['sample_id']: r['group'] for r in rows}
    lookup = {(r['arm'], r['seed'], r['sample_id']): r for r in rows}
    seeds = sorted({r['seed'] for r in rows})
    # seed-average within target, then treat the target as the unit
    values = {arm: {m: np.array([np.mean([float(lookup[arm, s, i][m]) for s in seeds]) for i in ids])
                    for m in METRICS} for arm in ARMS}
    order = [groups[i] for i in ids]
    out = dict(n=len(ids), n_groups=len(set(order)), seeds=seeds, metrics={})
    for metric, sign in METRICS.items():
        v = values[metric] if False else values
        arm_means = {arm: float(values[arm][metric].mean()) for arm in ARMS}
        contrasts = {}
        pairs = {'full_minus_base': ('full', 'base'),
                 'only_pair_minus_base': ('only_pair', 'base'),
                 'only_binding_minus_base': ('only_binding', 'base'),
                 'full_minus_only_binding': ('full', 'only_binding'),
                 'full_minus_only_pair': ('full', 'only_pair')}
        for name, (a, b) in pairs.items():
            delta = sign * (values[b][metric] - values[a][metric])  # positive = a better
            stats = interval(np.array([delta]), order, args.replicates)
            lo, hi = stats['conditional_group_ci95']
            contrasts[name] = dict(mean=float(stats['mean']), ci_low=float(lo), ci_high=float(hi),
                                   significant=bool(lo * hi > 0))
        # Benefit-oriented interaction = benefit(full) - benefit(binding) - benefit(pair),
        # with benefit(x) = sign * (base - x) so that a positive value means the two branches
        # together do better than the sum of their individual contributions.
        interaction = sign * ((values['base'][metric] - values['full'][metric])
                              - (values['base'][metric] - values['only_binding'][metric])
                              - (values['base'][metric] - values['only_pair'][metric]))
        stats = interval(np.array([interaction]), order, args.replicates)
        lo, hi = stats['conditional_group_ci95']
        out['metrics'][metric] = dict(raw_arm_means=arm_means, contrasts=contrasts,
                                      interaction=dict(mean=float(stats['mean']),
                                                       ci_low=float(lo), ci_high=float(hi),
                                                       significant=bool(lo * hi > 0)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({m: {k: round(v['raw_arm_means'][k], 4) for k in ARMS}
                      for m, v in out['metrics'].items()}, indent=2))


if __name__ == '__main__':
    main()
