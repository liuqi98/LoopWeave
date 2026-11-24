"""Development-60 comparison: original, aligned-retrained, RAAD, MEAN, DiffAb.

All sources were scored on the same evaluation tensor (``test_sha256`` is asserted
equal) and the same 60 complexes, so the table is directly comparable. Only the
aggregate metric definitions are shared; per-method sampling differs and is
labelled.

Sources
  ours          results/aligned_shape_v1/test_w0_{original,seed2027,seed2028}/metrics.json
  ours_aligned  results/aligned_shape_v1/test_w0.3_seed{2026,2027,2028}/metrics.json
  raad          outputs/raad_abgaze_reconstructed_test_v1 / raad_reconstructed_seed{2027,2028}_test_v2
  mean          outputs/mean_reconstructed_test_v1/metrics.json
  diffab        results/diffab_reconstructed_seed2026_v1/test_metrics.json  (draw 0 primary)

Writes a combined score table, two-arm subsets for ``research_audit baselines``,
and a descriptive summary.
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT.parent
sys.path.insert(0, str(ROOT))
from research_audit.common import digest

METRICS = ('aligned_rmsd', 'rmsd', 'contact_f1', 'interface_fape', 'cn_mae', 'clash',
           'aar', 'oracle_aar')
SOURCES = {
    # Same locked harness as ours_aligned, so the two of our arms are scored
    # identically. These checkpoints reproduce the archived metrics except for
    # single 8 A threshold flips caused by float nondeterminism.
    'ours': [('results/aligned_shape_v1/test_w0_original/metrics.json', 2026),
             ('results/aligned_shape_v1/test_w0_seed2027/metrics.json', 2027),
             ('results/aligned_shape_v1/test_w0_seed2028/metrics.json', 2028)],
    'ours_aligned': [('results/aligned_shape_v1/test_w0.3_seed2026/metrics.json', 2026),
                     ('results/aligned_shape_v1/test_w0.3_seed2027/metrics.json', 2027),
                     ('results/aligned_shape_v1/test_w0.3_seed2028/metrics.json', 2028)],
    'raad': [('outputs/raad_abgaze_reconstructed_test_v1/metrics.json', 2026),
             ('outputs/raad_reconstructed_seed2027_test_v2/metrics.json', 2027),
             ('outputs/raad_reconstructed_seed2028_test_v2/metrics.json', 2028)],
    'mean': [('outputs/mean_reconstructed_test_v1/metrics.json', 2026)],
    'diffab': [('results/diffab_reconstructed_seed2026_v1/test_metrics.json', 2026)],
}


def resolve(rel):
    for base in (PACKAGE, Path('/media/xd/code1/dyGF')):
        p = base / rel
        if p.is_file():
            return p
    raise FileNotFoundError(rel)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    records = [json.loads(l) for l in
               Path('/media/xd/code1/dyGF/data/abgaze_reconstructed_v1/splits/test.json')
               .read_text().splitlines() if l.strip()]
    ids = [r['pdb'] for r in records]
    groups = {r['pdb']: f"cluster_{r['cluster']}" for r in records}
    lengths = {r['pdb']: (len(r['cdrh3_seq']) if r.get('cdrh3_seq') else None) for r in records}

    rows, provenance, test_hashes = [], {}, set()
    per_arm = {}
    for arm, entries in SOURCES.items():
        provenance[arm] = {}
        per_arm[arm] = {}
        for rel, seed in entries:
            path = resolve(rel)
            document = json.loads(path.read_text())
            provenance[arm][str(seed)] = dict(path=str(path), sha256=digest(path))
            if document.get('test_sha256'):
                test_hashes.add(document['test_sha256'])
            cases = document['per_complex']
            if [c['sample_id'] for c in cases] != ids:
                raise ValueError(f'cohort/order mismatch: {path}')
            per_arm[arm][str(seed)] = []
            for case in cases:
                flat = case.get('first', case)
                if arm == 'diffab':
                    flat = dict(flat, oracle_aar=case['oracle_aar_selected']['aar'])
                missing = [m for m in METRICS if m not in flat]
                if missing:
                    raise ValueError(f'{path} lacks {missing}')
                per_arm[arm][str(seed)].append(flat)
                rows.append(dict(sample_id=case['sample_id'], group=groups[case['sample_id']],
                                 seed=seed, arm=arm,
                                 length=case.get('length', lengths[case['sample_id']]),
                                 **{m: flat[m] for m in METRICS}))
    if len(test_hashes) != 1:
        raise ValueError(f'evaluation tensors differ across sources: {test_hashes}')

    fields = ['sample_id', 'group', 'seed', 'arm', 'length'] + list(METRICS)
    with (out / 'dev60_scores.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for comparator in ('raad', 'mean', 'diffab'):
        for reference in ('ours_aligned', 'ours'):
            keep = [r for r in rows if r['arm'] in (reference, comparator)]
            with (out / f'{reference}_vs_{comparator}.csv').open('x', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(keep)

    summary = {}
    for arm, by_seed in per_arm.items():
        summary[arm] = {}
        for metric in METRICS:
            per_seed = [statistics.mean(c[metric] for c in cases) for cases in by_seed.values()]
            summary[arm][metric] = dict(mean=statistics.mean(per_seed), per_seed=per_seed,
                                        seed_sd=statistics.stdev(per_seed) if len(per_seed) > 1 else None,
                                        n_seeds=len(per_seed))
    (out / 'summary.json').write_text(json.dumps(dict(
        cohort='development-60 benchmark', n=len(ids), n_groups=len(set(groups.values())),
        test_sha256=sorted(test_hashes)[0], provenance=provenance, per_arm=summary,
        sampling_note=('ours / ours_aligned / raad: greedy sequence on one deterministic backbone; '
                       'mean: five sequences sharing one backbone; diffab: draw 0 of five joint '
                       'samples, so its aar is not a greedy decode.'),
        limitations=['Repeatedly inspected development benchmark; not a confirmatory holdout.',
                     'ours, ours_aligned and raad have three training seeds; mean and diffab have one, '
                     'so no training-seed spread is claimed for them.']), indent=2) + '\n')
    print(json.dumps({arm: {m: round(v['mean'], 4) for m, v in s.items()} for arm, s in summary.items()},
                     indent=1))


if __name__ == '__main__':
    main()
