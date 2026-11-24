"""All-metric comparison of every model on RAAD's 60-complex development benchmark.

Arms (each scored on the identical evaluation tensor, asserted by hash):
  ours30        original full model, 3 training seeds, 30-epoch final stage
  align30       aligned-backbone retraining (w=0.3), 3 seeds, 30-epoch final stage
  alignconv     aligned-backbone retraining, 3 seeds, continued to convergence
  raad          RAAD, 3 training seeds
  mean          MEAN, 1 seed
  diffab        adapted DiffAb, 1 seed, draw 0 of five joint samples

Metric families (all recomputed here from committed backbones, never re-trained):
  * geometry/accuracy: aligned_rmsd, rmsd, interface_fape, contact_f1/precision/recall
  * continuity/quality: cn_mae, cn_p90, tail fractions, anchors, internal bonds,
    peptide angle deviations, omega planarity, phi/psi native error, clash
  * interface placement: FAPE at several clamps and scopes, interface distance MAE
  * contact budget: TP/FP/FN, native vs predicted contact counts, over-prediction ratio
  * sequence: greedy AAR, first/mean over five samples, native-label oracle

Per-complex rows are written so any metric can be re-paired later; aggregates are
sample-weighted means with the seed spread shown. Paired contrasts are computed by
the audited ``research_audit baselines`` command from the emitted CSV.
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT.parent
sys.path.insert(0, str(ROOT))
from eval.test_loop_pair_replicates_v1 import score
from eval.evaluate_external_temporal_v1 import geometry
from eval.audit_fape_and_cn_claims_v1 import fape, cn_details, contact_budget, CLAMPS
from research_audit.common import digest

ORIG = Path('/media/xd/code1/dyGF')
ARMS = {
    'ours30': ['results/aligned_shape_v1/test_w0_original/generated.json',
               'results/aligned_shape_v1/test_w0_seed2027/generated.json',
               'results/aligned_shape_v1/test_w0_seed2028/generated.json'],
    'align30': ['results/aligned_shape_v1/test_w0.3_seed2026/generated.json',
                'results/aligned_shape_v1/test_w0.3_seed2027/generated.json',
                'results/aligned_shape_v1/test_w0.3_seed2028/generated.json'],
    'alignconv': ['results/aligned_shape_v1/test_w0.3_converged_seed2026/generated.json',
                  'results/aligned_shape_v1/test_w0.3_converged_seed2027/generated.json',
                  'results/aligned_shape_v1/test_w0.3_converged_seed2028/generated.json'],
    'no_chem': ['results/mechanism_arms_v1_no_chem/no_chem_2026/generated.json',
                'results/mechanism_arms_v1_no_chem/no_chem_2027/generated.json',
                'results/mechanism_arms_v1_no_chem/no_chem_2028/generated.json'],
    'raad': ['outputs/raad_abgaze_reconstructed_test_v1/generated.json',
             'outputs/raad_reconstructed_seed2027_test_v2/generated.json',
             'outputs/raad_reconstructed_seed2028_test_v2/generated.json'],
    'geoab': ['results/geoab_design_v1/generated_run1.json',
              'results/geoab_design_v1/generated_run2.json',
              'results/geoab_design_v1/generated_run3.json'],
    'mean': ['outputs/mean_reconstructed_test_v1/generated.json',
             'outputs/mean_reconstructed_seed2027_test/generated.json',
             'outputs/mean_reconstructed_seed2028_test/generated.json'],
    'diffab': ['outputs/diffab_reconstructed_seed2026_v1/generated.json',
               'outputs/diffab_reconstructed_seed2027_v1/generated.json',
               'outputs/diffab_reconstructed_seed2028_v1/generated.json'],
}
SEED_OF = {'ours30': [2026, 2027, 2028], 'align30': [2026, 2027, 2028],
           'alignconv': [2026, 2027, 2028], 'raad': [2026, 2027, 2028],
           'no_chem': [2026, 2027, 2028],
           'mean': [2026, 2027, 2028], 'diffab': [2026, 2027, 2028],
           'geoab': [2026, 2027, 2028]}
SCORE_KEYS = ('aar', 'first_sample_aar', 'mean_sample_aar', 'oracle_aar', 'rmsd',
              'aligned_rmsd', 'contact_f1', 'cn_mae', 'clash', 'interface_fape')
CN_KEYS = ('cn_p90', 'cn_frac_gt_0_1', 'cn_frac_gt_0_2', 'cn_left_anchor',
           'cn_right_anchor', 'cn_internal', 'peptide_angle_at_C_dev_deg',
           'peptide_angle_at_N_dev_deg')
GEOM_KEYS = ('omega_planarity_deviation_deg', 'omega_native_abs_error_deg',
             'phi_native_abs_error_deg', 'psi_native_abs_error_deg')
BUDGET_KEYS = ('tp', 'fp', 'fn', 'n_native', 'n_predicted', 'precision', 'recall',
               'predicted_to_native_ratio', 'interface_distance_mae',
               'all_pair_distance_mae')


def resolve(rel):
    for base in (PACKAGE, ORIG):
        path = base / rel
        if path.is_file():
            return path
    raise FileNotFoundError(rel)


def normalise(document, arm):
    """DiffAb commits joint draws; every other arm commits a standard record."""
    if 'draws' not in document[0] and 'sequence_logits' not in document[0]:
        # Arms stored as plain backbones plus a decoded sequence: build a one-hot logit stand-in.
        alphabet = 'ACDEFGHIKLMNPQRSTVWY'
        out = []
        for row in document:
            sequence = row.get('greedy') or row.get('sequence') or row['candidates'][0]
            onehot = torch.nn.functional.one_hot(
                torch.tensor([alphabet.index(a) for a in sequence]), 20).float() * 10
            record = dict(row, sequence_logits=onehot.tolist(), greedy=sequence)
            record.setdefault('candidates', [sequence] * 5)
            out.append(record)
        return out
    if 'draws' in document[0]:
        out = []
        for row in document:
            primary = row['draws'][0]
            alphabet = 'ACDEFGHIKLMNPQRSTVWY'
            onehot = torch.nn.functional.one_hot(
                torch.tensor([alphabet.index(a) for a in primary['sequence']]), 20).float() * 10
            out.append(dict(sample_id=row['sample_id'], backbone=primary['backbone'],
                            sequence_logits=onehot.tolist(), greedy=primary['sequence'],
                            candidates=[d['sequence'] for d in row['draws']]))
        return out
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replicates', type=int, default=10000)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    dataset = ORIG / 'data/abgaze_reconstructed_v1'
    test_path = dataset / 'tensors/test.pt'
    rows = torch.load(test_path, weights_only=False)
    records = [json.loads(l) for l in (dataset / 'splits/test.json').read_text().splitlines()
               if l.strip()]
    ids = [r[0] for r in rows]
    if ids != [r['pdb'] for r in records]:
        raise ValueError('test rows and split records disagree')
    groups = {r['pdb']: f"cluster_{r['cluster']}" for r in records}
    lengths = {r['pdb']: len(r['cdrh3_seq']) for r in records}

    per_complex, provenance = {}, {}
    for arm, paths in ARMS.items():
        per_complex[arm] = {}
        provenance[arm] = {}
        for path_rel, seed in zip(paths, SEED_OF[arm]):
            path = resolve(path_rel)
            document = normalise(json.loads(path.read_text()), arm)
            if [r['sample_id'] for r in document] != ids:
                raise ValueError(f'cohort/order mismatch: {path}')
            provenance[arm][str(seed)] = dict(path=str(path), sha256=digest(path))
            scored = {c['sample_id']: c for c in score(rows, document)['per_complex']}
            geom = {c['sample_id']: c for c in geometry(rows, document)['per_complex']}
            cases = {}
            for (sid, x, y), record in zip(rows, document):
                prediction = torch.tensor(record['backbone'])
                entry = {}
                entry.update({k: scored[sid][k] for k in SCORE_KEYS})
                entry.update({k: geom[sid][k] for k in GEOM_KEYS})
                entry.update(cn_details(prediction, x, y))
                entry.update(contact_budget(prediction, x, y))
                for clamp in CLAMPS:
                    label = 'raw' if clamp is None else f'clamp{int(clamp)}'
                    for scope in ('all', 'interface'):
                        value = fape(prediction, x, y, clamp, scope == 'interface')
                        if value is not None:
                            entry[f'fape_{label}_{scope}'] = value['mean']
                            entry[f'fape_{label}_{scope}_saturated'] = value['saturated_fraction']
                cases[sid] = entry
            per_complex[arm][str(seed)] = cases

    metrics = sorted({k for by_seed in per_complex.values()
                      for cases in by_seed.values() for case in cases.values() for k in case})
    # A metric missing for some complex (native-defined subsets) is averaged over
    # the complexes where it is defined; the counts are reported alongside.
    rows_out = []
    summary = {}
    for arm, by_seed in per_complex.items():
        summary[arm] = {}
        for metric in metrics:
            per_seed, n = [], 0
            for seed, cases in by_seed.items():
                values = [c[metric] for c in cases.values()
                          if c.get(metric) is not None]
                if values:
                    per_seed.append(statistics.mean(values))
                    n = max(n, len(values))
            if not per_seed:
                continue
            summary[arm][metric] = dict(
                mean=statistics.mean(per_seed), per_seed=per_seed, n=n,
                seed_sd=statistics.stdev(per_seed) if len(per_seed) > 1 else None,
                n_seeds=len(per_seed))
    for arm, by_seed in per_complex.items():
        for seed, cases in by_seed.items():
            for sid, case in cases.items():
                rows_out.append(dict(sample_id=sid, group=groups[sid], length=lengths[sid],
                                     arm=arm, seed=seed, **case))
    fields = ['sample_id', 'group', 'length', 'arm', 'seed'] + metrics
    scores_csv = out / 'dev60_all_metrics.csv'
    with scores_csv.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows_out)

    directions = {m: -1 for m in metrics}
    for m in ('aar', 'first_sample_aar', 'mean_sample_aar', 'oracle_aar', 'contact_f1',
              'contact_precision', 'contact_recall', 'precision', 'recall', 'tp',
              'n_native', 'n_predicted'):
        directions[m] = 1
    descriptive = {'predicted_to_native_ratio'} | {m for m in metrics if 'saturated' in m}
    for m in descriptive:
        directions.pop(m, None)

    contrasts = {}
    for arm in ARMS:
        if arm == 'raad':
            continue
        subset = [r for r in rows_out if r['arm'] in (arm, 'raad')]
        path = out / f'paired_{arm}_vs_raad.csv'
        with path.open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(subset)
        contrasts[arm] = str(path)

    (out / 'summary.json').write_text(json.dumps(dict(
        cohort='RAAD development-60 benchmark', n=len(ids),
        n_groups=len(set(groups.values())), test_sha256=digest(test_path),
        arms={a: list(v) for a, v in ARMS.items()},
        provenance=provenance, per_arm=summary, directions=directions,
        descriptive=sorted(descriptive),
        paired_csv=contrasts,
        sampling_note=('ours30 / align30 / alignconv / raad report a greedy sequence on one '
                       'deterministic backbone; mean reports five sequences sharing one backbone; '
                       'diffab is draw 0 of five joint samples, so its aar is not a greedy decode '
                       'and its AAR columns must not be ranked against the others.'),
        limitations=['Repeatedly inspected development benchmark; not a confirmatory holdout.',
                     'ours/raad have three training seeds, mean and diffab one, so no seed spread is '
                     'claimed for the latter.',
                     'Seeds are not extra samples; intervals are conditional on the fitted seeds.']),
        indent=2) + '\n')
    print(json.dumps({arm: {m: round(v['mean'], 4) for m, v in s.items()
                           if m in ('aligned_rmsd', 'contact_f1', 'cn_mae', 'interface_fape')}
                      for arm, s in summary.items()}, indent=1))
    print(f'{len(metrics)} metrics x {len(rows_out)} rows -> {scores_csv}')


if __name__ == '__main__':
    main()
