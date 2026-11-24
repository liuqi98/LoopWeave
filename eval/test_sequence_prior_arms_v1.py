"""Evaluate the three sequence-prior conditions.

For every condition (baseline / probability_init / full) and seed this generates H3
predictions with the frozen stage-1 backbone and the condition's own prior wiring, then
reports:

* **AAR** (primary, since the intervention targets sequence decoding);
* **interface FAPE** and **contact F1** (must not be traded away silently);
* **post-reconstruction C-N MAE** (the model's returned backbone, i.e. after the
  CA-preserving ``residue_geometry`` projection) and the **pre-reconstruction** value for
  reference;
* oracle AAR over five sampled sequences.

All checkpoints are locked in ``evaluation_lock.json`` before any test tensor is read.
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
from frame_relation_refiner import FrameRelationGraph, FrameRelationRefiner
from sequence_prior_refiner import SequencePriorRefiner
from structure_refinement_ablation import residue_geometry
from eval.test_loop_pair_replicates_v1 import score
from eval.audit_fape_and_cn_claims_v1 import cn_details
from train.train_abgaze_reconstructed import sha, dump, move
from research_audit.statistics import interval

WORKSPACE = Path('/media/xd/code1/dyGF')
PACKAGE = ROOT.parent
CONDITIONS = ('baseline', 'probability_init', 'full')
SEEDS = (2026, 2027, 2028)
ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prior-root', type=Path, required=True)
    parser.add_argument('--runs', type=Path,
                        default=WORKSPACE / 'outputs/sequence_prior_arms_v1')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--replicates', type=int, default=10000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    # --- lock every checkpoint first ----------------------------------------
    locks = {}
    for seed in SEEDS:
        for condition in CONDITIONS:
            run = args.runs / f'{condition}_seed{seed}'
            plan_path, status_path = run / 'plan.json', run / 'status.json'
            if not (plan_path.is_file() and status_path.is_file()):
                raise ValueError(f'incomplete run: {run}')
            plan = json.loads(plan_path.read_text())
            status = json.loads(status_path.read_text())
            if status['phase'] != 'complete' or plan.get('smoke'):
                raise ValueError(f'run not finished: {run} ({status["phase"]})')
            if plan['condition'] != condition or plan['seed'] != seed:
                raise ValueError(f'plan mismatch in {run}')
            best = run / 'pair/best.pt'
            state = torch.load(best, map_location='cpu', weights_only=False)
            if state['plan_sha256'] != sha(plan_path):
                raise ValueError(f'checkpoint does not match its plan: {run}')
            target = out / f'{condition}_{seed}'
            target.mkdir()
            shutil.copyfile(best, target / 'locked.pt')
            locks[f'{condition}_{seed}'] = dict(
                condition=condition, seed=seed, epoch=int(state['epoch']),
                selected_epoch=int(state['best_epoch']),
                checkpoint_sha256=sha(target / 'locked.pt'),
                training_plan_sha256=sha(plan_path),
                alpha=plan['alpha'], temperature=plan['temperature'],
                prior_sha256=plan['prior']['sha256'])
    dump(out / 'evaluation_lock.json', dict(
        protocol='three conditions by three seeds locked before test access',
        locks=locks, selection='validation-selected best.pt of each run',
        note='the RAbD test split is repeatedly inspected development data'))
    print(json.dumps(dict(locked=len(locks))), flush=True)

    # --- shared stage-1 geometry and inputs ---------------------------------
    dataset = WORKSPACE / 'data/abgaze_reconstructed_v1'
    reference = WORKSPACE / 'outputs/persistent_loop_pair_test_v1'
    refplan = json.loads((reference / 'plan.json').read_text())
    graphs = {item['sample_id']: graph for item, graph in
              torch.load(WORKSPACE / 'data/raad_abgaze_reconstructed_v1/test.pt',
                         map_location='cpu', weights_only=False)}
    rows = torch.load(dataset / 'tensors/test.pt', map_location='cpu', weights_only=False)
    ids = [row[0] for row in rows]
    clusters = {json.loads(line)['pdb']: f"cluster_{json.loads(line)['cluster']}"
                for line in (dataset / 'splits/test.json').read_text().splitlines()
                if line.strip()}

    per_run = {}
    for seed in SEEDS:
        generator_path = PACKAGE / f'checkpoints/original_seed{seed}/generator.pt'
        if sha(generator_path) != refplan['checkpoints']['generator']:
            raise ValueError('generator hash mismatch')
        state = torch.load(generator_path, map_location='cpu', weights_only=False)
        generator = FrameRelationRefiner(**state['model_config']).to(device)
        generator.load_state_dict(state['model'])
        generator.eval()
        prior = torch.load(args.prior_root / f'stage1_head_seed{seed}.pt',
                           map_location='cpu', weights_only=False)
        base_inputs = {}
        with torch.no_grad():
            for sid, x, _ in rows:
                feature = torch.load(WORKSPACE / 'outputs/binding_interface_test_v1/features'
                                     / sid / 'features.pt', map_location='cpu',
                                     weights_only=False)
                if not feature['metadata']['antigen_only'] or \
                        feature['metadata']['native_h3_used']:
                    raise ValueError(f'{sid}: binding features are not antigen-only')
                raw = generator(FrameRelationGraph(**graphs[sid]).to(device))
                inputs = move({**x, **feature['features']}, device)
                inputs.update(base_bb=raw['h3_backbone'],
                              base_logits=torch.zeros_like(inputs['base_logits']))
                base_inputs[sid] = inputs
        del generator
        prior_logits = {sid: prior['logits'][sid]['logits'].to(device) for sid in ids}

        for condition in CONDITIONS:
            label = f'{condition}_{seed}'
            state = torch.load(out / label / 'locked.pt', map_location='cpu',
                               weights_only=False)
            model = SequencePriorRefiner(**state['model_config']).to(device)
            model.load_state_dict(state['model'])
            model.eval()
            captured = {}
            handle = model.loop_pair_layers[-1].register_forward_hook(
                lambda module, inputs, output: captured.__setitem__(
                    'raw', output[1].detach().cpu().clone()))
            rng = torch.Generator().manual_seed(2026)
            generated, pre = [], []
            with torch.no_grad():
                for sid, _, _ in rows:
                    inputs = dict(base_inputs[sid])
                    if condition != 'baseline':
                        scale = state.get('alpha', 1.0) if 'alpha' in state else 1.0
                        prior_tensor = prior_logits[sid] * scale
                        inputs['base_logits'] = prior_tensor
                        inputs['base_logits_output'] = (
                            prior_tensor if condition == 'full'
                            else torch.zeros_like(prior_tensor))
                    captured.clear()
                    result = model(inputs)
                    backbone = result['backbone'].cpu()
                    logits = result['sequence_logits'].cpu()
                    if not (torch.isfinite(backbone).all() and torch.isfinite(logits).all()):
                        raise ValueError(f'{label}/{sid}: nonfinite output')
                    samples = torch.multinomial((logits / .5).softmax(-1), 5,
                                                replacement=True, generator=rng).T
                    generated.append(dict(
                        sample_id=sid, backbone=backbone.tolist(),
                        sequence_logits=logits.tolist(),
                        greedy=''.join(ALPHABET[i] for i in logits.argmax(-1).tolist()),
                        candidates=[''.join(ALPHABET[i] for i in seq)
                                    for seq in samples.tolist()]))
                    pre.append(captured.get('raw'))
            handle.remove()
            scored = score(rows, generated)
            # Pre-projection C-N for reference.
            pre_cn = []
            for (sid, x, y), raw_backbone in zip(rows, pre):
                if raw_backbone is None:
                    raise ValueError(f'{label}/{sid}: failed to capture pre-projection state')
                if not torch.allclose(residue_geometry(raw_backbone), torch.tensor(
                        generated[ids.index(sid)]['backbone']), atol=1e-5):
                    raise ValueError(f'{label}/{sid}: final is not residue_geometry(raw)')
                pre_cn.append(cn_details(raw_backbone, x, y)['cn_mae'])
            dump(out / label / 'generated.json', generated)
            dump(out / label / 'metrics.json', scored)
            per_run[label] = dict(scored=scored, pre_cn=pre_cn)
            print(json.dumps(dict(label=label, epoch=locks[label]['epoch'],
                                  aar=round(scored['summary']['aar'], 4),
                                  fape=round(scored['summary']['interface_fape'], 4),
                                  f1=round(scored['summary']['contact_f1'], 4),
                                  cn_post=round(scored['summary']['cn_mae'], 4),
                                  cn_pre=round(statistics.mean(pre_cn), 4))), flush=True)
            del model
            torch.cuda.empty_cache()

    # --- aggregate and contrast --------------------------------------------
    keys = ('aar', 'oracle_aar', 'mean_sample_aar', 'interface_fape', 'contact_f1',
            'cn_mae', 'rmsd', 'aligned_rmsd', 'clash')
    summary = dict(conditions=list(CONDITIONS), seeds=list(SEEDS), metrics=list(keys),
                   orientation='improvement > 0 means the condition is better than the baseline',
                   means={}, contrasts={},
                   notes=['cn_mae is the post-reconstruction value from the model output',
                          'cn_pre is the pre-projection C-N for reference',
                          'AAR is the greedy sequence accuracy; oracle AAR selects among five samples'])
    for condition in CONDITIONS:
        summary['means'][condition] = {}
        for key in keys:
            values = [per_run[f'{condition}_{seed}']['scored']['summary'][key]
                      for seed in SEEDS]
            summary['means'][condition][key] = dict(
                mean=statistics.mean(values), per_seed=values,
                sample_std=statistics.stdev(values))
        for key in ('cn_mae',):
            values = [statistics.mean(per_run[f'{condition}_{seed}']['pre_cn'])
                      for seed in SEEDS]
            summary['means'][condition]['cn_pre'] = dict(
                mean=statistics.mean(values), per_seed=values,
                sample_std=statistics.stdev(values))

    def contrast(first, second, key, pre=False):
        rows_matrix = []
        for seed in SEEDS:
            if pre:
                a = np.array(per_run[f'{first}_{seed}']['pre_cn'])
                b = np.array(per_run[f'{second}_{seed}']['pre_cn'])
            else:
                a = np.array([r[key] for r in per_run[f'{first}_{seed}']['scored']['per_complex']])
                b = np.array([r[key] for r in per_run[f'{second}_{seed}']['scored']['per_complex']])
            rows_matrix.append(a - b)
        matrix = np.stack(rows_matrix)
        sign = 1. if key in ('aar', 'oracle_aar', 'mean_sample_aar', 'contact_f1') else -1.
        stats = interval(sign * matrix, [clusters[sid] for sid in ids], args.replicates)
        return dict(difference=stats['mean'], ci95=stats['conditional_group_ci95'],
                    per_seed_mean=stats['per_seed_mean'],
                    significant=bool(stats['conditional_group_ci95'][0] *
                                     stats['conditional_group_ci95'][1] > 0))

    for first, second, name in (('probability_init', 'baseline', 'prob_init_vs_baseline'),
                                ('full', 'baseline', 'full_vs_baseline'),
                                ('full', 'probability_init', 'full_vs_prob_init')):
        summary['contrasts'][name] = {}
        for key in keys:
            summary['contrasts'][name][key] = contrast(first, second, key)
        summary['contrasts'][name]['cn_pre'] = contrast(first, second, 'cn_mae', pre=True)
    dump(out / 'sequence_prior_summary.json', summary)
    for name, values in summary['contrasts'].items():
        print(json.dumps({name: {key: round(value['difference'], 4)
                                 for key, value in values.items()}}), flush=True)


if __name__ == '__main__':
    main()
