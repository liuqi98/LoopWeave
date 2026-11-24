"""Locked external-cohort evaluation of inference-time interface interventions.

Runs the *existing* validation-selected full-pipeline checkpoints on the locked
20-sample, 13-dependence-group external cohort and changes only what the binding
branch can see, or when its interface members are chosen:

  full                           joint features, dynamic members (control)
  mask_surface_geometry          surface-geometry scalar/vector columns zeroed
  mask_electrostatics_chemistry  electrostatics/chemistry columns zeroed
  mask_none                      every scalar/vector column zeroed
  static_neighborhood            members frozen from the initial loop conformation

These measure DEPENDENCE of the trained model on an information block and the
value of re-selecting members at inference. They are not retrained ablations and
cannot establish a block's learnable value; the matched training entry is
``train/train_matched_ablation_v3.py``.

Nothing here trains, selects, or reads labels into a model. Outputs:
  lock.json, protocol.json, scores.csv, comparison.json, generated_*.json,
  stages.json, distances.json, drift_records.json, costs.csv, status.json
"""
import argparse
import csv
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT.parent
sys.path.insert(0, str(ROOT))
from frame_relation_refiner import FrameRelationGraph, FrameRelationRefiner
from interface_ablation_refiner import AblatedPersistentLoopPairRefiner
from eval.test_loop_pair_replicates_v1 import score

ARMS = (
    ('full', 'joint', 'dynamic'),
    ('mask_surface_geometry', 'surface_geometry', 'dynamic'),
    ('mask_electrostatics_chemistry', 'electrostatics_chemistry', 'dynamic'),
    ('mask_none', 'none', 'dynamic'),
    ('static_neighborhood', 'joint', 'static'),
)
ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
METRIC_KEYS = ('aar', 'first_sample_aar', 'mean_sample_aar', 'oracle_aar', 'rmsd',
               'aligned_rmsd', 'contact_f1', 'cn_mae', 'clash', 'interface_fape', 'length')
CONTRAST_METRICS = (('aligned_rmsd', -1), ('rmsd', -1), ('contact_f1', 1), ('cn_mae', -1),
                    ('clash', -1), ('interface_fape', -1), ('aar', 1), ('oracle_aar', 1))
REPEATS = 10000
STAGE_ARMS = ('full', 'static_neighborhood')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def tensor_digest(mapping):
    h = hashlib.sha256()
    for key in sorted(mapping):
        value = mapping[key]
        if torch.is_tensor(value):
            h.update(key.encode())
            h.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def dump(path, value):
    """Refuse to overwrite: every report is written once."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def ca_distances(native, prediction, x, y):
    observed = np.asarray(y['mask'], dtype=bool)[:, 1]
    context = np.asarray(x['context_bb'])[:, 1]
    valid_context = (np.asarray(x['antigen_mask'], dtype=bool)
                     & np.asarray(x['context_mask'], dtype=bool)[:, 1])
    ref_ca = np.asarray(native)[observed, 1]
    pred_ca = np.asarray(prediction)[observed, 1]
    ag_ca = context[valid_context]
    if not len(ref_ca) or not len(ag_ca):
        raise ValueError('empty CA support')
    return (np.linalg.norm(ref_ca[:, None] - ag_ca[None], axis=-1).tolist(),
            np.linalg.norm(pred_ca[:, None] - ag_ca[None], axis=-1).tolist())


def rounded(tensor, ndigits=3):
    """Three decimals are enough to recompute 8 A contacts and keep the archive small."""
    return np.round(tensor.detach().cpu().numpy(), ndigits).tolist()


def generate(model, rows, base, seed_stream=2026, trace=False):
    generated, stages, seconds = [], [], 0.
    rng = torch.Generator().manual_seed(seed_stream)
    for sid, _, _ in rows:
        inputs, _ = base[sid]
        model.stage_trace = [] if trace else None
        start = time.monotonic()
        with torch.no_grad():
            prediction = model(dict(inputs))
        seconds += time.monotonic() - start
        layers = model.stage_trace
        model.stage_trace = None
        backbone, logits = prediction['backbone'], prediction['sequence_logits']
        if not (torch.isfinite(backbone).all() and torch.isfinite(logits).all()):
            raise ValueError(f'Nonfinite output: {sid}')
        samples = torch.multinomial((logits / .5).softmax(-1), 5,
                                    replacement=True, generator=rng).T
        generated.append(dict(
            sample_id=sid, backbone=backbone.tolist(), sequence_logits=logits.tolist(),
            greedy=''.join(ALPHABET[i] for i in logits.argmax(-1).tolist()),
            candidates=[''.join(ALPHABET[i] for i in s.tolist()) for s in samples]))
        if trace:
            stages.append(dict(sample_id=sid, stages=[rounded(base[sid][1])]
                               + [rounded(t) for t in layers]))
    return generated, stages, seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--groups', type=Path, required=True,
                        help='Locked group source, e.g. '
                             'outputs/external_temporal_evaluation_v1/evaluation_lock.json')
    parser.add_argument('--seeds', type=int, nargs='+', default=[2026, 2027, 2028])
    parser.add_argument('--repeat-seed', type=int, default=2026,
                        help='Re-run this seed to measure numerical repeat drift')
    args = parser.parse_args()
    torch.set_num_threads(1)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    group_doc = json.loads(args.groups.read_text())
    ids, groups = group_doc['ids'], group_doc['groups']
    membership = {s: g for g, ss in groups.items() for s in ss}
    if set(membership) != set(ids):
        raise ValueError('Group membership does not match the locked cohort')

    rows = torch.load(PACKAGE / 'data/external_test/inputs.pt', map_location='cpu', weights_only=False)
    graphs = {m['sample_id']: g for m, g in torch.load(
        PACKAGE / 'data/external_test/graphs.pt', map_location='cpu', weights_only=False)}
    if [r[0] for r in rows] != ids:
        raise ValueError('Input order differs from the locked cohort order')
    features = {sid: torch.load(PACKAGE / f'data/external_test/features/{sid}/features.pt',
                                map_location='cpu', weights_only=False)['features'] for sid in ids}
    checkpoints = {}
    for seed in args.seeds:
        gen_path = PACKAGE / f'checkpoints/original_seed{seed}/generator.pt'
        pair_path = PACKAGE / f'checkpoints/original_seed{seed}/pair.pt'
        checkpoints[str(seed)] = dict(
            generator_sha256=sha(gen_path), pair_sha256=sha(pair_path),
            pair_epoch=torch.load(pair_path, map_location='cpu', weights_only=False)['epoch'])
    lock = dict(cohort='locked external nonpathogen cohort', ids=ids, groups=groups,
                n=len(ids), n_groups=len(groups), seeds=list(args.seeds),
                group_source=str(args.groups), group_source_sha256=sha(args.groups),
                inputs_sha256=sha(PACKAGE / 'data/external_test/inputs.pt'),
                graphs_sha256=sha(PACKAGE / 'data/external_test/graphs.pt'),
                code_sha256={str(p.relative_to(PACKAGE)): sha(p) for p in (
                    Path(__file__).resolve(), ROOT / 'interface_ablation_refiner.py',
                    ROOT / 'eval/test_loop_pair_replicates_v1.py',
                    ROOT / 'research_audit/diagnostics.py')},
                checkpoints=checkpoints,
                arm_definitions={a: dict(feature_groups=f, neighborhood=n) for a, f, n in ARMS},
                K=5, temperature=.5, sampling_seed=2026,
                selection='Pre-existing validation-selected checkpoints; no test-time selection.',
                failure_policy='Abort on any nonfinite output; no silent sample removal.',
                limitations=['Inference-time interventions only; no parameters are updated.',
                             'Dependence on an input block is not evidence of its learnable value.',
                             'Neighbor positions/distances are the shared geometric scaffold and are never '
                             'removed by feature_groups, so mask_none is chemistry-blind but not position-blind.',
                             'Static membership freezes the member set from the initial loop conformation while '
                             'feature values are still recomputed, isolating member re-selection only.',
                             'A repeatedly inspected cohort; not untouched confirmatory evidence.',
                             'CPU execution: repeat drift bounds this machine, not CUDA nondeterminism.'])
    dump(out / 'lock.json', lock)
    dump(out / 'protocol.json', dict(lock_sha256=sha(out / 'lock.json'), **lock))

    metric_rows, stage_records, cost_rows, drift_records = [], [], [], []
    final_distances, stage_distances = [], []
    summaries = {str(s): {} for s in args.seeds}
    for seed in args.seeds:
        gen_state = torch.load(PACKAGE / f'checkpoints/original_seed{seed}/generator.pt',
                               map_location='cpu', weights_only=False)
        pair_state = torch.load(PACKAGE / f'checkpoints/original_seed{seed}/pair.pt',
                                map_location='cpu', weights_only=False)
        generator = FrameRelationRefiner(**gen_state['model_config'])
        generator.load_state_dict(gen_state['model'])
        generator.eval()
        models = {}
        for arm, feature_groups, neighborhood in ARMS:
            model = AblatedPersistentLoopPairRefiner(**pair_state['model_config'],
                                                     feature_groups=feature_groups,
                                                     neighborhood=neighborhood)
            model.load_state_dict(pair_state['model'])
            model.eval()
            models[arm] = model
        base = {}
        with torch.no_grad():
            for sid, x, y in rows:
                raw = generator(FrameRelationGraph(**graphs[sid]))
                inputs = {**x, **features[sid]}
                inputs.update(base_bb=raw['h3_backbone'],
                              base_logits=torch.zeros_like(inputs['base_logits']))
                base[sid] = (inputs, raw['h3_backbone'].clone())
        for arm, _, _ in ARMS:
            model = models[arm]
            generated, stages, seconds = generate(model, rows, base, trace=True)
            dump(out / f'generated_{arm}_seed{seed}.json', generated)
            result = score(rows, generated)
            summaries[str(seed)][arm] = result['summary']
            dump(out / f'metrics_{arm}_seed{seed}.json', result)
            n_parameters = sum(p.numel() for p in model.parameters())
            cost_rows.append(dict(run=arm, seed=seed, device='cpu', precision='float32',
                                  timing_scope='single forward, batch 1, torch.set_num_threads(1)',
                                  parameter_count=n_parameters,
                                  trainable_parameter_count=n_parameters,
                                  inference_seconds=round(seconds, 6),
                                  n_inference_examples=len(ids),
                                  training_seconds=None, peak_memory_bytes=None))
            for case in result['per_complex']:
                metric_rows.append(dict(sample_id=case['sample_id'], group=membership[case['sample_id']],
                                        seed=seed, arm=arm,
                                        **{k: case[k] for k in METRIC_KEYS if k in case}))
            for record in stages:
                sid = record['sample_id']
                x, y = next((r[1], r[2]) for r in rows if r[0] == sid)
                chain = [('generator', np.asarray(record['stages'][0]))]
                chain += [(f'layer_{i}', np.asarray(t)) for i, t in enumerate(record['stages'][1:])]
                for stage_index, (stage_name, coordinates) in enumerate(chain):
                    ref, pred = ca_distances(y['bb'], coordinates, x, y)
                    entry = dict(arm=arm, seed=seed, sample_id=sid,
                                 reference_distances=ref, predicted_distances=pred)
                    if stage_index == len(chain) - 1:
                        final_distances.append(dict(entry, stage_index=0, stage_name='final'))
                    if seed == args.repeat_seed and arm in STAGE_ARMS:
                        stage_distances.append(dict(entry, stage_index=stage_index,
                                                    stage_name=stage_name))
            if arm in STAGE_ARMS:
                stage_records.append(dict(arm=arm, seed=seed,
                                          samples=[dict(sample_id=r['sample_id'], stages=r['stages'])
                                                   for r in stages]))
        # Numerical repeat: identical checkpoint, inputs, RNG and execution config.
        model = models['full']
        if seed == args.repeat_seed:
            first = {c['sample_id']: c for c in json.loads(
                (out / f'metrics_full_seed{seed}.json').read_text())['per_complex']}
            generated, _, _ = generate(model, rows, base)
            result = score(rows, generated)
            dump(out / f'metrics_full_seed{seed}_repeat2.json', result)
            second = {c['sample_id']: c for c in result['per_complex']}
            runtime = dict(device='cpu', precision='float32', inference_seed=2026,
                           runtime_id=f'torch{torch.__version__}|cpu|threads1|float32',
                           checkpoint_sha256=checkpoints[str(seed)]['pair_sha256'])
            g1 = {r['sample_id']: r['backbone'] for r in json.loads(
                (out / f'generated_full_seed{seed}.json').read_text())}
            g2 = {r['sample_id']: r['backbone'] for r in generated}
            for sid, _, _ in rows:
                inputs, _ = base[sid]
                for repeat_id, source, backbone in (('r1', first[sid], g1[sid]),
                                                    ('r2', second[sid], g2[sid])):
                    drift_records.append(dict(
                        arm='full', sample_id=sid, repeat_id=repeat_id,
                        input_sha256=tensor_digest(inputs), **runtime,
                        aligned_rmsd=source['aligned_rmsd'], contact_f1=source['contact_f1'],
                        rmsd=source['rmsd'], cn_mae=source['cn_mae'], backbone=backbone))
    dump(out / 'stages.json', dict(scope=f'per-layer H3 backbones (3 decimals), arms {list(STAGE_ARMS)}',
                                   all_seeds=True, records=stage_records))
    # Contact diagnostics reuse the audited offline implementation; raw distance
    # matrices are recomputable from stages.json + data/external_test and are not
    # archived, keeping the package small.
    from research_audit.diagnostics import contacts
    dump(out / 'contact_thresholds.json', dict(
        scope='final prediction, all arms and seeds',
        thresholds=[6., 7., 8., 9., 10.],
        result=contacts(final_distances, [6., 7., 8., 9., 10.])))
    dump(out / 'contact_stages.json', dict(
        scope=f'per-layer turnover, seed {args.repeat_seed}, arms {list(STAGE_ARMS)}',
        thresholds=[8.],
        result=contacts(stage_distances, [8.])))
    dump(out / 'drift_records.json', drift_records)
    with (out / 'scores.csv').open('x', newline='') as handle:
        fields = ['sample_id', 'group', 'seed', 'arm'] + list(METRIC_KEYS)
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(metric_rows)
    with (out / 'costs.csv').open('x', newline='') as handle:
        fields = ['run', 'seed', 'device', 'precision', 'timing_scope', 'parameter_count',
                  'trainable_parameter_count', 'inference_seconds', 'n_inference_examples',
                  'training_seconds', 'peak_memory_bytes']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(cost_rows)

    rng = np.random.default_rng(2026)
    sample_groups = [membership[s] for s in ids]
    unique_groups = sorted(groups)
    blocks = [np.flatnonzero(np.asarray(sample_groups) == g) for g in unique_groups]
    sizes = np.array([len(b) for b in blocks])
    draw = rng.integers(len(blocks), size=(REPEATS, len(blocks)))
    contrasts = {}
    for arm, _, _ in ARMS[1:]:
        contrasts[arm] = {}
        for metric, direction in CONTRAST_METRICS:
            per_seed = {}
            for seed in args.seeds:
                a = {r['sample_id']: r for r in json.loads(
                    (out / f'metrics_{arm}_seed{seed}.json').read_text())['per_complex']}
                b = {r['sample_id']: r for r in json.loads(
                    (out / f'metrics_full_seed{seed}.json').read_text())['per_complex']}
                per_seed[seed] = np.array([direction * (a[s][metric] - b[s][metric]) for s in ids])
            stacked = np.stack([per_seed[s] for s in args.seeds])
            sums = np.stack([stacked[:, b].sum(axis=1) for b in blocks], axis=1)
            estimates = sums.mean(axis=0)[draw].sum(axis=1) / sizes[draw].sum(axis=1)
            contrasts[arm][metric] = dict(
                sign='positive means the arm beats full',
                per_seed_mean={str(s): float(per_seed[s].mean()) for s in args.seeds},
                mean=float(stacked.mean()),
                group_balanced_mean=float(np.mean([stacked[:, b].mean() for b in blocks])),
                ci95=np.quantile(estimates, [.025, .975]).tolist(),
                n_groups=len(blocks))
    mean_over_seeds = {arm: {k: statistics.mean(summaries[str(s)][arm][k] for s in args.seeds)
                             for k in summaries[str(args.seeds[0])][arm]} for arm, _, _ in ARMS}
    dump(out / 'comparison.json', dict(
        n=len(ids), n_groups=len(groups), seeds=list(args.seeds),
        per_seed_summaries=summaries, mean_over_seeds=mean_over_seeds,
        contrast_vs_full=contrasts,
        inference='Exploratory group-resampled intervals; unadjusted for multiplicity; conditional on '
                  'the three fitted seeds. Seeds are not extra independent samples.',
        interpretation='Inference-time dependence and member-selection sensitivity, not trained ablations.'))
    dump(out / 'status.json', dict(phase='complete', n=len(ids), arms=[a for a, _, _ in ARMS],
                                   seeds=list(args.seeds), failures=0))
    print(json.dumps(dict(arms=[a for a, _, _ in ARMS], seeds=list(args.seeds),
                          mean_over_seeds=mean_over_seeds), indent=1))


if __name__ == '__main__':
    main()
