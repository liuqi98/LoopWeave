"""Collect native and generated CDRH3 sequences for the figure reproduction.

Sources (all committed; nothing is re-trained):
  native dev-60 / external-20 / train-380  from the repository records
  generated per arm                        from committed generated.json files
  FDA-approved reference                  from the archived reference table

Every generated file is hashed and its cohort order asserted, so a figure can never
mix two different sample orders.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT.parent
sys.path.insert(0, str(ROOT))

ORIG = Path('/media/xd/code1/dyGF')
GENCDRH3 = Path('/media/xd/code2/GenCDRH3_final')

# arm -> cohort -> list of committed generated.json paths (one per training seed)
GENERATED = {
    'Ours (full)': {
        'dev60': ['results/aligned_shape_v1/test_w0_original/generated.json',
                  'results/aligned_shape_v1/test_w0_seed2027/generated.json',
                  'results/aligned_shape_v1/test_w0_seed2028/generated.json'],
        'external20': ['results/external_interface_ablations_v1/generated_full_seed2026.json',
                       'results/external_interface_ablations_v1/generated_full_seed2027.json',
                       'results/external_interface_ablations_v1/generated_full_seed2028.json'],
    },
    'Ours (aligned, converged)': {
        'dev60': ['results/aligned_shape_v1/test_w0.3_converged_seed2026/generated.json',
                  'results/aligned_shape_v1/test_w0.3_converged_seed2027/generated.json',
                  'results/aligned_shape_v1/test_w0.3_converged_seed2028/generated.json'],
    },
    'RAAD': {
        'dev60': ['outputs/raad_abgaze_reconstructed_test_v1/generated.json',
                  'outputs/raad_reconstructed_seed2027_test_v2/generated.json',
                  'outputs/raad_reconstructed_seed2028_test_v2/generated.json'],
    },
    'MEAN': {
        'dev60': ['outputs/mean_reconstructed_test_v1/generated.json'],
    },
    'DiffAb': {
        'dev60': ['outputs/diffab_reconstructed_seed2026_v1/generated.json'],
    },
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def resolve(rel):
    for base in (PACKAGE, ORIG):
        path = base / rel
        if path.is_file():
            return path
    raise FileNotFoundError(rel)


def native(cohort):
    """Native CDRH3 strings, in the cohort order used by the generated files."""
    if cohort == 'dev60':
        records = [json.loads(l) for l in
                   (ORIG / 'data/abgaze_reconstructed_v1/splits/test.json')
                   .read_text().splitlines() if l.strip()]
        return [(r['pdb'], r['cdrh3_seq']) for r in records]
    if cohort == 'external20':
        records = json.loads((PACKAGE / 'data/external_test/records.json').read_text())
        ids = json.loads((PACKAGE / 'data/dataset_index.json').read_text())['external_test']
        lookup = {r['pdb']: r['cdrh3_seq'] for r in records}
        return [(sid, lookup[sid]) for sid in ids]
    if cohort == 'train380':
        records = json.loads((PACKAGE / 'data/train/records.json').read_text())
        return [(r['pdb'], r['cdrh3_seq']) for r in records]
    raise ValueError(cohort)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    payload = {'native': {}, 'generated': {}, 'provenance': {}}

    for cohort, splits in (('dev60', None), ('external20', None), ('train380', None)):
        payload['native'][cohort] = native(cohort)

    for arm, cohorts in GENERATED.items():
        payload['generated'][arm] = {}
        payload['provenance'][arm] = {}
        for cohort, paths in cohorts.items():
            ids = [sid for sid, _ in native(cohort)]
            sequences, per_seed = [], {}
            payload['provenance'][arm][cohort] = []
            for index, rel in enumerate(paths):
                path = resolve(rel)
                document = json.loads(path.read_text())
                if [r['sample_id'] for r in document] != ids:
                    raise ValueError(f'cohort/order mismatch: {path}')
                payload['provenance'][arm][cohort].append(
                    dict(path=str(path), sha256=sha(path), seed=2026 + index))
                seeded = []
                for record in document:
                    candidates = record.get('candidates')
                    if candidates is None:                      # DiffAb joint draws
                        candidates = [d['sequence'] for d in record['draws']]
                    seeded.append((record['sample_id'], list(candidates)))
                per_seed[2026 + index] = seeded
                for sid, candidates in seeded:
                    sequences.extend((sid, seq) for seq in candidates)
            payload['generated'][arm][cohort] = sequences
            payload['generated'][arm][f'{cohort}_per_seed'] = {
                str(seed): value for seed, value in per_seed.items()}

    reference = GENCDRH3 / 'cal_AAR/fda_approved_cdrh3_REAL.csv'
    payload['fda_approved'] = []
    if reference.is_file():
        import csv
        with reference.open(newline='') as handle:
            for row in csv.DictReader(handle):
                if row.get('cdrh3'):
                    payload['fda_approved'].append((row.get('name', ''), row['cdrh3']))
    payload['reference_sources'] = dict(
        fda_approved=dict(path=str(reference), sha256=sha(reference))
        if reference.is_file() else None,
        thera_sabdab_clinical=None,
        note='Thera-SAbDab clinical reference (n=350) is not present in this workspace; the '
             'archived figure used it, so that comparison cannot be reproduced here.')

    (out / 'sequences.json').write_text(json.dumps(payload, indent=2) + '\n')
    counts = {arm: {c: len(v) for c, v in per.items() if not c.endswith('_per_seed')}
              for arm, per in payload['generated'].items()}
    print(json.dumps(dict(native={k: len(v) for k, v in payload['native'].items()},
                          generated=counts,
                          fda_approved=len(payload['fda_approved'])), indent=1))


if __name__ == '__main__':
    main()
