"""Consolidate geometry quality and cost/provenance evidence from committed runs.

No model is executed. Geometry is recomputed from already committed backbones and
already committed native coordinates; cost rows are read from archived epoch logs
and from the measured inference timings of the ablation evaluation.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT.parent
sys.path.insert(0, str(ROOT))
import torch
from eval.evaluate_external_temporal_v1 import geometry
from eval.evaluate_external_ablations_v1 import ARMS

ARCHIVED_FULL = Path('/media/xd/code1/dyGF/outputs/external_temporal_evaluation_v1')
ARCHIVED_ABLATIONS = Path('/media/xd/code1/dyGF')


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


def stage_seconds(run, stage):
    """Logged epoch seconds include training, validation and checkpoint saving."""
    total, epochs = 0.0, []
    for path in sorted((run / stage).glob('epoch_*.json')):
        record = json.loads(path.read_text())
        if record.get('epoch', 0) > 0:
            total += float(record['seconds'])
            epochs.append(record['epoch'])
    return total, epochs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ablations', type=Path, required=True,
                        help='Results dir produced by evaluate_external_ablations_v1.py')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    ablation = args.ablations.resolve()

    rows = torch.load(PACKAGE / 'data/external_test/inputs.pt', map_location='cpu', weights_only=False)
    seeds = json.loads((ablation / 'lock.json').read_text())['seeds']
    arms = [a for a, _, _ in ARMS]
    per_arm, reproduction = {}, {}
    for arm in arms:
        per_arm[arm] = {}
        for seed in seeds:
            generated = json.loads((ablation / f'generated_{arm}_seed{seed}.json').read_text())
            result = geometry(rows, generated)
            per_arm[arm][str(seed)] = result
        if arm == 'full':
            for seed in seeds:
                archived = json.loads((ARCHIVED_FULL / f'geometry_seed{seed}.json').read_text())
                diffs = {k: abs(per_arm[arm][str(seed)]['summary'][k] - archived['summary'][k])
                         for k in archived['summary']}
                reproduction[str(seed)] = dict(
                    max_abs_summary_difference=max(diffs.values()),
                    per_key_abs_difference=diffs,
                    archived_sha256=sha(ARCHIVED_FULL / f'geometry_seed{seed}.json'))
    dump(out / 'geometry.json', dict(
        cohort='locked external nonpathogen cohort', seeds=seeds, arms=arms,
        per_arm_seed=per_arm,
        full_arm_reproduction_vs_archived_gpu=reproduction,
        interpretation='Geometry quality recomputed from committed backbones; planarity and clash '
                       'are proxies, not all-atom MolProbity validation.'))

    cost_rows = []
    measured = {r['run']: r for r in csv.DictReader((ablation / 'costs.csv').open())
                if int(r['seed']) == seeds[0]}
    archived = {
        'full': (ARCHIVED_ABLATIONS / f'outputs/full_pipeline_seed{seeds[0]}_v1', ('binding', 'pair')),
        'only_pair': (ARCHIVED_ABLATIONS / f'outputs/matched_ablation_v2_no_binding_seed{seeds[0]}',
                      ('binding', 'pair')),
        'only_binding': (ARCHIVED_ABLATIONS / f'outputs/matched_ablation_v2_no_pair_seed{seeds[0]}',
                         ('pair',)),
    }
    archived_training = {}
    for arm, (run, stages) in archived.items():
        detail = {stage_sec: dict(epochs=stage_seconds(run, stage_sec)[1])
                  for stage_sec in stages}
        total = sum(stage_seconds(run, stage_sec)[0] for stage_sec in stages)
        archived_training[arm] = dict(run=str(run), stages=detail, training_seconds=total,
                                      definition='Sum of logged epoch seconds; includes validation and saving.')
    for arm in arms:
        source = dict(measured[arm])
        source.update(run=arm, seed=seeds[0],
                      timing_scope='measured single forward, batch 1, CPU, float32, threads=1',
                      training_seconds=archived_training.get(arm, {}).get('training_seconds'),
                      peak_memory_bytes='')
        cost_rows.append(source)
    with (out / 'costs.csv').open('x', newline='') as handle:
        fields = ['run', 'seed', 'device', 'precision', 'timing_scope', 'parameter_count',
                  'trainable_parameter_count', 'inference_seconds', 'n_inference_examples',
                  'training_seconds', 'peak_memory_bytes']
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(cost_rows)
    dump(out / 'training_cost.json', dict(
        seed=seeds[0], arms=archived_training,
        limitations=['Logged seconds include validation and checkpoint serialization; they are not pure forward time.',
                     'The three trained arms do not share FLOPs or parameter counts.',
                     'Peak GPU memory was never recorded and is reported as missing, not as checkpoint size.',
                     'Inference timings are CPU-only and are not comparable with the archived GPU training cost.']))

    drift = json.loads((ablation / 'numerical_drift.json').read_text())
    spread = max((c['metrics'][m]['spread'] for c in drift['per_complex'] for m in c['metrics']), default=None)
    backbone = max((c.get('backbone', {}).get('max_abs_pairwise', 0.) for c in drift['per_complex']), default=None)
    metrics_repro = json.loads((ablation / 'comparison.json').read_text())
    dump(out / 'reproduction.json', dict(
        repeat_drift=dict(scope='identical checkpoint, inputs, RNG seed, device, precision and runtime id',
                          max_metric_spread=spread, max_coordinate_pairwise_difference_A=backbone,
                          cases=len(drift['per_complex']),
                          note='Exactly zero on CPU here. This bounds this machine only; CUDA nondeterminism '
                               'is not measured and must not be inferred from this number.'),
        cross_device=dict(scope='CPU rerun of the committed full model versus the archived GPU run',
                          full_arm_aligned_rmsd=metrics_repro['mean_over_seeds']['full']['aligned_rmsd'],
                          note='Cross-device agreement is a reproduction check, not repeat drift.'),
        seeds=seeds))
    print(json.dumps(dict(geometry=str(out / 'geometry.json'), costs=str(out / 'costs.csv'),
                          cost_report=str(out / 'costs.json'),
                          reproduction=str(out / 'reproduction.json')), indent=1))


if __name__ == '__main__':
    main()
