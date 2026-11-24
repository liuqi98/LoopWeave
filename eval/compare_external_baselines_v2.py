"""Assemble the shared external-cohort score table for method comparison.

Sources are already committed score files on the identical locked 20-sample,
13-dependence-group cohort:
  ours   results/external_temporal_evaluation_v1/metrics_seed{2026,2027,2028}.json
  raad   results/external_raad_comparison_v1/metrics_seed{2026,2027,2028}.json
  mean   results/mean_external_locked_v1/metrics.json   (single training seed)

Only the score table and per-contrast subsets are produced here; the audited
``research_audit baselines`` command performs the actual paired comparison and
never pairs training seeds across methods.
"""
import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path
import torch

PACKAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE / 'code'))
from eval.test_loop_pair_replicates_v1 import score
GROUP_SOURCE = Path('/media/xd/code1/dyGF/outputs/external_temporal_evaluation_v1/evaluation_lock.json')
METRICS = ('aar', 'first_sample_aar', 'mean_sample_aar', 'oracle_aar', 'rmsd', 'aligned_rmsd',
           'contact_f1', 'cn_mae', 'clash', 'interface_fape')
SOURCES = {
    'ours': [(2026, 'external_temporal_evaluation_v1', 'metrics_seed2026.json'),
             (2027, 'external_temporal_evaluation_v1', 'metrics_seed2027.json'),
             (2028, 'external_temporal_evaluation_v1', 'metrics_seed2028.json')],
    'raad': [(2026, 'external_raad_comparison_v1', 'metrics_seed2026.json'),
             (2027, 'external_raad_comparison_v1', 'metrics_seed2027.json'),
             (2028, 'external_raad_comparison_v1', 'metrics_seed2028.json')],
    'mean': [(2026, 'mean_external_locked_v1', 'metrics.json')],
    # DiffAb commits joint samples; the shared scorer is applied here so every
    # method is scored by exactly the same code path.
    'diffab': [(2026, 'diffab_external_locked_v1', 'generated.json')],
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    group_doc = json.loads(GROUP_SOURCE.read_text())
    ids, groups = group_doc['ids'], group_doc['groups']
    membership = {s: g for g, ss in groups.items() for s in ss}

    rows, provenance, per_complex = [], {}, {}
    for arm, runs in SOURCES.items():
        provenance[arm] = {}
        for seed, directory, name in runs:
            path = PACKAGE / 'results' / directory / name
            document = json.loads(path.read_text())
            if 'per_complex' in document:
                cases = document['per_complex']
            else:
                cohort_rows = torch.load(PACKAGE / 'data/external_test/inputs.pt',
                                         map_location='cpu', weights_only=False)
                if [r[0] for r in cohort_rows] != ids or [g['sample_id'] for g in document] != ids:
                    raise ValueError(f'cohort/order mismatch while scoring {path}')
                scored = score(cohort_rows, document)
                cases = scored['per_complex']
                scored['test_sha256'] = sha(PACKAGE / 'data/external_test/inputs.pt')
                scored['scored_from'] = str(path.relative_to(PACKAGE))
                scored['scorer'] = 'eval/test_loop_pair_replicates_v1.score'
                target = path.parent / 'metrics.json'
                if not target.exists():
                    target.write_text(json.dumps(scored, ensure_ascii=False, indent=2) + '\n')
            if [c['sample_id'] for c in cases] != ids:
                raise ValueError(f'Cohort/order mismatch in {path}')
            provenance[arm][str(seed)] = dict(path=str(path.relative_to(PACKAGE)), sha256=sha(path))
            per_complex[(arm, seed)] = {c['sample_id']: c for c in cases}
            for case in cases:
                row = dict(sample_id=case['sample_id'], group=membership[case['sample_id']],
                           arm=arm, seed=seed)
                row.update({k: case[k] for k in METRICS})
                rows.append(row)
    fields = ['sample_id', 'group', 'arm', 'seed'] + list(METRICS)
    with (out / 'external_scores.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for comparator in ('mean', 'raad', 'diffab'):
        keep = [r for r in rows if r['arm'] in ('ours', comparator)]
        with (out / f'ours_vs_{comparator}.csv').open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(keep)

    summary = {}
    for arm, seeds in SOURCES.items():
        summary[arm] = {}
        for metric in METRICS:
            per_seed = [statistics.mean(per_complex[(arm, seed)][s][metric] for s in ids)
                        for seed, _, _ in seeds]
            summary[arm][metric] = dict(
                mean=statistics.mean(per_seed), per_seed=per_seed,
                seed_sd=statistics.stdev(per_seed) if len(per_seed) > 1 else None)
    dump(out / 'summary.json', dict(
        cohort='locked external nonpathogen cohort', n=len(ids), n_groups=len(groups),
        group_source=str(GROUP_SOURCE), group_source_sha256=sha(GROUP_SOURCE),
        provenance=provenance, summary=summary,
        comparison='Run: python -m research_audit baselines --input ours_vs_<method>.csv '
                   '--reference ours --comparator <method> --metric aligned_rmsd:min ...',
        limitations=['ours and raad have three training seeds; mean has one, so no training-seed '
                     'spread is claimed for mean.',
                     'Different methods still differ in input representation and compute budget.',
                     'The cohort was inspected during method development; this is not an untouched test.',
                     'greedy AAR, first_sample_aar and oracle_aar are distinct and are never merged.']))
    print(json.dumps({arm: {m: round(v['mean'], 4) for m, v in s.items()}
                      for arm, s in summary.items()}, indent=1))


if __name__ == '__main__':
    main()
