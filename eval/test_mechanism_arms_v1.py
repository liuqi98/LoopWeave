"""Score the four mechanism-control arms and compare them with the full arm.

Protocol follows ``manuscript/nature_communications_v1/mechanism_ablation_implementation.md``:

* all twelve checkpoints (four arms by three prespecified seeds) are **locked before**
  any test tensor is read; the lock records the checkpoint hash and the training-plan
  hash of every run;
* each run must be complete, non-smoke, and must carry the same parent plan, caches
  and feature plan as the original model;
* generation uses the frozen generator and the same sampling budget as the original
  model (K = 5, temperature 0.5, seed 2026), so only the pair stage differs;
* primary readouts are C-N MAE, interface FAPE and contact F1; precision, recall,
  TP/FP/FN, RMSD, aligned RMSD, clashes and sequence recovery are reported as
  trade-offs;
* contrasts are paired within seed (arm minus full) and then grouped by H3 cluster,
  with seeds averaged inside each complex before resampling.

Positive ``improvement`` always means the **full** arm was better, i.e. the
intervention was harmful; negative means the intervention helped.
"""
import argparse
import json
import shutil
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from frame_relation_refiner import FrameRelationRefiner, FrameRelationGraph
from persistent_loop_pair_refiner import PersistentLoopPairRefiner
from mechanism_pair_ablation import MechanismPairRefiner
from train.train_abgaze_reconstructed import sha, dump, move
from eval.test_loop_pair_replicates_v1 import score
from research_audit.statistics import interval

ARMS = ('full', 'reset_z', 'pair_mlp', 'feature_only', 'no_chem')

# Identical channel mask to the training script.
CHEMICAL_CHANNELS = dict(binding_surface_scalars=(4, 5, 6, 7, 8),
                         binding_surface_vectors=(1, 2, 3),
                         binding_atom_scalars=(1, 2, 3, 4, 7, 8),
                         binding_atom_vectors=(0, 1))
SEEDS = (2026, 2027, 2028)
WORKSPACE = Path('/media/xd/code1/dyGF')
LOWER_IS_BETTER = {'cn_mae', 'cn_frac_gt_0_1', 'cn_frac_gt_0_2', 'cn_p90', 'rmsd', 'aligned_rmsd',
                   'interface_fape', 'fape_raw_interface', 'fape_clamp10_interface', 'clash',
                   'interface_distance_mae', 'all_pair_distance_mae', 'omega_planarity_deviation_deg',
                   'phi_native_abs_error_deg', 'psi_native_abs_error_deg'}
REPORTED = ('cn_mae', 'interface_fape', 'contact_f1', 'precision', 'recall', 'tp', 'fp', 'fn',
            'rmsd', 'aligned_rmsd', 'clash', 'cn_frac_gt_0_2', 'aar', 'oracle_aar',
            'mean_sample_aar', 'omega_planarity_deviation_deg')


def zero_chemical_channels(module, arguments):
    """Same channel mask as the training script; kept here so evaluation is identical."""
    batch = arguments[0]
    for key, channels in CHEMICAL_CHANNELS.items():
        if key not in batch:
            continue
        value = batch[key]
        for channel in channels:
            value[:, channel] = 0
    return arguments


def build_model(arm, config, state):
    if arm in ('full', 'no_chem'):
        model = PersistentLoopPairRefiner(**config)
    else:
        model = MechanismPairRefiner(mechanism=arm, **config)
    model.load_state_dict(state['model'])
    if arm == 'no_chem':
        model.register_forward_pre_hook(zero_chemical_channels)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runs', type=Path,
                        default=WORKSPACE / 'outputs/mechanism_arms_v1')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--replicates', type=int, default=10000)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    # --- lock every checkpoint before any test access -------------------------
    parent = WORKSPACE / 'outputs/full_pipeline_seed2026_v1'
    source_plan = json.loads((parent / 'plan.json').read_text())
    locks = {}
    for seed in SEEDS:
        parent_seed = WORKSPACE / f'outputs/full_pipeline_seed{seed}_v1'
        for arm in ARMS:
            run = args.runs / f'{arm}_seed{seed}'
            plan_path = run / 'plan.json'
            status_path = run / 'status.json'
            if not plan_path.is_file() or not status_path.is_file():
                raise ValueError(f'incomplete run: {run}')
            plan = json.loads(plan_path.read_text())
            status = json.loads(status_path.read_text())
            if status['phase'] != 'complete' or plan.get('smoke'):
                raise ValueError(f'run not finished: {run} ({status["phase"]})')
            if plan['arm'] != arm or plan['seed'] != seed:
                raise ValueError(f'plan mismatch in {run}')
            if plan['parent_plan_sha256'] != sha(parent_seed / 'plan.json'):
                raise ValueError(f'parent plan mismatch in {run}')
            if plan['feature_plan_sha256'] != source_plan['feature_plan_sha256']:
                raise ValueError(f'feature plan mismatch in {run}')
            best = run / 'pair/best.pt'
            state = torch.load(best, map_location='cpu', weights_only=False)
            if state['plan_sha256'] != sha(plan_path):
                raise ValueError(f'checkpoint does not belong to its plan: {run}')
            target = out / f'{arm}_{seed}'
            target.mkdir()
            shutil.copyfile(best, target / 'locked.pt')
            locks[f'{arm}_{seed}'] = dict(
                arm=arm, seed=seed, epoch=int(state['epoch']),
                checkpoint_sha256=sha(target / 'locked.pt'),
                training_plan_sha256=sha(plan_path),
                selected_epoch=int(state['best_epoch']),
                parameters=plan['parameters'])
    dump(out / 'evaluation_lock.json', dict(
        protocol='all four arms by three seeds locked before test access',
        locks=locks,
        selection='validation-selected best.pt of each run; no test-time sweep',
        note='the test split is repeatedly inspected development data, not an untouched holdout'))
    print(json.dumps(dict(locked=len(locks))), flush=True)

    # --- shared inputs -------------------------------------------------------
    dataset = WORKSPACE / 'data/abgaze_reconstructed_v1'
    reference = WORKSPACE / 'outputs/persistent_loop_pair_test_v1'
    refplan = json.loads((reference / 'plan.json').read_text())
    graphs = {item['sample_id']: graph for item, graph in
              torch.load(WORKSPACE / 'data/raad_abgaze_reconstructed_v1/test.pt',
                         map_location='cpu', weights_only=False)}
    rows = torch.load(dataset / 'tensors/test.pt', map_location='cpu', weights_only=False)
    if args.limit:
        rows = rows[:args.limit]
    generator_path = WORKSPACE / 'outputs/reconstructed_two_stage_v1/generator/best.pt'
    if sha(generator_path) != refplan['checkpoints']['generator']:
        raise ValueError('generator hash mismatch')
    generator_state = torch.load(generator_path, map_location='cpu', weights_only=False)
    generator = FrameRelationRefiner(**generator_state['model_config']).to(device)
    generator.load_state_dict(generator_state['model'])
    generator.eval()
    inputs_by_id = {}
    with torch.no_grad():
        for sid, x, _ in rows:
            feature = torch.load(WORKSPACE / 'outputs/binding_interface_test_v1/features' / sid
                                 / 'features.pt', map_location='cpu', weights_only=False)
            if not feature['metadata']['antigen_only'] or feature['metadata']['native_h3_used']:
                raise ValueError(f'{sid}: binding features are not antigen-only')
            raw = generator(FrameRelationGraph(**graphs[sid]).to(device))
            inputs = move({**x, **feature['features']}, device)
            inputs.update(base_bb=raw['h3_backbone'],
                          base_logits=torch.zeros_like(inputs['base_logits']))
            inputs_by_id[sid] = inputs
    del generator

    clusters = {json.loads(line)['pdb']: f"cluster_{json.loads(line)['cluster']}"
                for line in (dataset / 'splits/test.json').read_text().splitlines() if line.strip()}
    ids = [row[0] for row in rows]
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'

    per_arm = {}
    for label, lock in locks.items():
        state = torch.load(out / label / 'locked.pt', map_location='cpu', weights_only=False)
        model = build_model(lock['arm'], state['model_config'], state).to(device)
        model.eval()
        rng = torch.Generator().manual_seed(2026)
        generated = []
        with torch.no_grad():
            for sid, _, _ in rows:
                result = model(dict(inputs_by_id[sid]))
                backbone = result['backbone'].cpu()
                logits = result['sequence_logits'].cpu()
                if not (torch.isfinite(backbone).all() and torch.isfinite(logits).all()):
                    raise ValueError(f'{label}/{sid}: nonfinite output')
                samples = torch.multinomial((logits / .5).softmax(-1), 5, replacement=True,
                                            generator=rng).T
                generated.append(dict(
                    sample_id=sid, backbone=backbone.tolist(),
                    sequence_logits=logits.tolist(),
                    greedy=''.join(alphabet[i] for i in logits.argmax(-1).tolist()),
                    candidates=[''.join(alphabet[i] for i in seq) for seq in samples.tolist()]))
        scored = score(rows, generated)
        dump(out / label / 'generated.json', generated)
        dump(out / label / 'metrics.json', scored)
        per_arm[label] = scored
        print(json.dumps(dict(label=label, epoch=lock['epoch'],
                              cn_mae=round(scored['summary']['cn_mae'], 4),
                              interface_fape=round(scored['summary']['interface_fape'], 4),
                              contact_f1=round(scored['summary']['contact_f1'], 4))), flush=True)

    # --- aggregate across seeds and contrast arms against full ---------------
    metrics = [key for key in REPORTED
               if all(key in per_arm[label]['per_complex'][0] for label in locks)]
    summary = dict(arms={}, contrasts={}, metrics=metrics,
                   orientation='improvement > 0 means the full arm was better, i.e. the '
                               'intervention harmed the metric')
    for arm in ARMS:
        for key in metrics:
            values = [per_arm[f'{arm}_{seed}']['summary'][key] for seed in SEEDS]
            summary['arms'].setdefault(arm, {})[key] = dict(
                mean=statistics.mean(values), per_seed=values,
                sample_std=statistics.stdev(values) if len(values) > 1 else 0.)
    for arm in ARMS:
        if arm == 'full':
            continue
        summary['contrasts'][arm] = {}
        for key in metrics:
            deltas = []
            for seed in SEEDS:
                arm_rows = {r['sample_id']: r for r in per_arm[f'{arm}_{seed}']['per_complex']}
                full_rows = {r['sample_id']: r for r in per_arm[f'full_{seed}']['per_complex']}
                deltas.append(np.array([arm_rows[sid][key] - full_rows[sid][key] for sid in ids]))
            matrix = np.stack(deltas)                       # seeds x samples
            value = float(matrix.mean())
            sign = -1. if key in LOWER_IS_BETTER else 1.
            stats = interval(sign * matrix, [clusters[sid] for sid in ids],
                             args.replicates)
            summary['contrasts'][arm][key] = dict(
                arm_minus_full=value, improvement=stats['mean'],
                ci95=stats['conditional_group_ci95'],
                per_seed_mean=stats['per_seed_mean'],
                significant=bool(stats['conditional_group_ci95'][0] *
                                 stats['conditional_group_ci95'][1] > 0))
    dump(out / 'mechanism_summary.json', summary)

    print(json.dumps({arm: {key: round(summary['contrasts'][arm][key]['improvement'], 4)
                            for key in ('cn_mae', 'interface_fape', 'contact_f1')}
                      for arm in summary['contrasts']}, indent=1))


if __name__ == '__main__':
    main()
