"""Estimable part of the 2x2 interaction design when the base arm is missing.

The four cells are base (no pair, no binding), only_pair (binding removed),
only_binding (pair removed) and full. The interaction

    I = full - only_pair - only_binding + base

and the two "when the other branch is absent" conditional effects require the
base arm. When base is absent this script reports those quantities as
``not_estimable`` instead of substituting a proxy, and reports only the two
conditional effects the observed cells identify. It never labels a partial
design as synergy.

Pairing, seed handling and group resampling reuse
``research_audit.statistics.rectangular``/``interval`` so numbers match the
audited ``factorial`` command on the cells that exist.
"""
import argparse
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_audit.common import read_csv, write_json, digest, metric_spec
from research_audit.statistics import ARMS, CONTRASTS, rectangular, interval

ESTIMABLE_WITHOUT_BASE = ('pair_when_binding_present', 'binding_when_pair_present')
NEEDS_BASE = ('interaction', 'pair_when_binding_absent', 'binding_when_pair_absent',
              'pair_main_effect', 'binding_main_effect', 'full_minus_base')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--metric', action='append', required=True,
                        help='metric:min or metric:max; may repeat')
    parser.add_argument('--seeds', type=int, nargs='+', default=[2026, 2027, 2028])
    parser.add_argument('--replicates', type=int, default=10000)
    args = parser.parse_args()
    specs = [metric_spec(s) for s in args.metric]
    rows = read_csv(args.input)
    present = [a for a in ARMS if a in {r['arm'] for r in rows}]
    unknown = {r['arm'] for r in rows} - set(ARMS)
    if unknown:
        raise ValueError(f'Unknown arms: {sorted(unknown)}')
    if 'full' not in present:
        raise ValueError('The full arm is required as the reference cell')
    missing = [a for a in ARMS if a not in present]

    report = dict(present_arms=present, missing_arms=missing,
                  formula='I = full - only_pair - only_binding + base (direction-corrected)',
                  estimability={}, metrics={},
                  inference='Exploratory, conditional on fitted seeds; unadjusted intervals; '
                            'a partial design is not a synergy test.',
                  protocol_verified=False)
    if 'base' in missing:
        report['estimability']['interaction'] = dict(
            status='not_estimable',
            reason='The base (no_pair_no_binding) arm is absent, so '
                   'full - only_pair - only_binding + base cannot be formed. '
                   'A partial design must not be reported as synergy.')
        for name in NEEDS_BASE:
            report['estimability'][name] = dict(
                status='not_estimable', reason='Requires the absent base arm.')
    for name in ESTIMABLE_WITHOUT_BASE:
        report['estimability'][name] = dict(
            status='estimable' if {'only_pair', 'only_binding'}.issubset(present) else 'not_estimable',
            reason='Both cells named by this contrast are present.'
                   if {'only_pair', 'only_binding'}.issubset(present)
                   else 'One of the two cells named by this contrast is absent.')

    if {'only_pair', 'only_binding'}.issubset(present):
        metrics = [m for m, _ in specs]
        values, seeds, ids, groups, _ = rectangular(rows, present, metrics, args.seeds)
        index = {arm: i for i, arm in enumerate(present)}
        for metric, direction in specs:
            entry = dict(direction='higher' if direction == 1 else 'lower',
                         raw_arm_means={a: float(values[metric][:, :, i].mean())
                                        for a, i in index.items()},
                         n=len(ids), n_groups=len(set(groups)))
            for name in ESTIMABLE_WITHOUT_BASE:
                weights = np.zeros(len(present))
                for arm, weight in zip(ARMS, CONTRASTS[name]):
                    if arm in index:
                        weights[index[arm]] = weight
                delta = direction * (values[metric] @ weights)
                entry[name] = interval(delta, groups, args.replicates)
            entry['interaction'] = dict(status='not_estimable', value=None)
            report['metrics'][metric] = entry
    else:
        report['note'] = ('Conditional effects need both only_pair and only_binding; '
                          'only raw arm means are reportable.')
    report['input_sha256'] = digest(args.input)
    report['analysis_code_sha256'] = {Path(__file__).name: digest(Path(__file__))}
    write_json(args.output, report)
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
