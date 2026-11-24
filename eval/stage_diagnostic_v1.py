"""Stage-by-stage geometry diagnostic: initial prediction -> four pair layers -> final.

For the original model's three seeds this records, at every stage of the final
refinement, three readouts on the dev-60 cohort:

    C-N MAE                (split into in-loop bonds and the two framework flanks)
    interface FAPE
    contact F1             (hard 8 A CA contacts, same definition as the objective)

Stages captured
---------------
``initial``    the generator's ``base_bb``, i.e. the model's starting geometry
``before_pair``the backbone entering the first pair layer (after the earlier layers)
``layer1..4``  the backbone after each of the four loop-pair layers
``final``      the model's returned backbone

The metric definitions are imported from the archived code rather than re-derived:
``cn_details`` gives ``cn_internal`` / ``cn_left_anchor`` / ``cn_right_anchor``,
``interface_fape`` is the objective's implementation, and the contact F1 mirrors the
objective's hard 8 A CA definition exactly.

Nothing is trained and no test-set prediction is changed; this is a read-only probe of
intermediate states.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from frame_relation_refiner import FrameRelationRefiner, FrameRelationGraph
from persistent_loop_pair_refiner import PersistentLoopPairRefiner
from orientation_interface_refiner import interface_fape
from eval.audit_fape_and_cn_claims_v1 import cn_details
from train.train_abgaze_reconstructed import sha, dump, move
from research_audit.statistics import interval

WORKSPACE = Path('/media/xd/code1/dyGF')
PACKAGE = ROOT.parent
STAGES = ('initial', 'before_pair', 'layer1', 'layer2', 'layer3', 'layer4', 'final')


def contact_f1(prediction, x, y, threshold=8.):
    """Hard contact F1, identical to the objective's reported contact_f1."""
    valid = y['mask'][:, 1].bool()
    context = x['context_bb'][x['antigen_mask'] & x['context_mask'][:, 1], 1]
    predicted_ca, native_ca = prediction[valid, 1], y['bb'][valid, 1]
    if len(predicted_ca) < 1 or len(context) < 1:
        return float('nan')
    distances = torch.cdist(predicted_ca, context)
    truth = torch.cdist(native_ca, context)
    contacts = truth <= threshold
    predicted = distances <= threshold
    true_positive = (predicted & contacts).sum()
    denominator = (predicted.sum() + contacts.sum()).clamp_min(1)
    return float(2 * true_positive / denominator)


def metrics_at(prediction, x, y):
    row = dict(cn_details(prediction, x, y))
    row['interface_fape'] = float(interface_fape(prediction, x, y))
    row['contact_f1'] = contact_f1(prediction, x, y)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2026, 2027, 2028])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--replicates', type=int, default=10000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    dataset = WORKSPACE / 'data/abgaze_reconstructed_v1'
    graphs = {item['sample_id']: graph for item, graph in
              torch.load(WORKSPACE / 'data/raad_abgaze_reconstructed_v1/test.pt',
                         map_location='cpu', weights_only=False)}
    rows = torch.load(dataset / 'tensors/test.pt', map_location='cpu', weights_only=False)
    if args.limit:
        rows = rows[:args.limit]
    ids = [row[0] for row in rows]
    clusters = {json.loads(line)['pdb']: f"cluster_{json.loads(line)['cluster']}"
                for line in (dataset / 'splits/test.json').read_text().splitlines()
                if line.strip()}

    per_seed = {}
    for seed in args.seeds:
        generator_path = PACKAGE / f'checkpoints/original_seed{seed}/generator.pt'
        pair_path = PACKAGE / f'checkpoints/original_seed{seed}/pair.pt'
        generator_state = torch.load(generator_path, map_location='cpu', weights_only=False)
        generator = FrameRelationRefiner(**generator_state['model_config']).to(device)
        generator.load_state_dict(generator_state['model'])
        generator.eval()
        pair_state = torch.load(pair_path, map_location='cpu', weights_only=False)
        model = PersistentLoopPairRefiner(**pair_state['model_config']).to(device)
        model.load_state_dict(pair_state['model'])
        model.eval()

        captured = {}

        def record(name):
            def hook(module, inputs, output):
                captured[name] = output[1].detach().cpu().clone()
            return hook

        handles = [layer.register_forward_hook(record(f'layer{index + 1}'))
                   for index, layer in enumerate(model.loop_pair_layers)]
        handles.append(model.loop_pair_layers[0].register_forward_pre_hook(
            lambda module, inputs: captured.__setitem__('before_pair',
                                                        inputs[1].detach().cpu().clone())))

        records = []
        with torch.no_grad():
            for sid, x, y in rows:
                feature = torch.load(WORKSPACE / 'outputs/binding_interface_test_v1/features'
                                     / sid / 'features.pt', map_location='cpu',
                                     weights_only=False)
                if not feature['metadata']['antigen_only'] or \
                        feature['metadata']['native_h3_used']:
                    raise ValueError(f'{sid}: binding features are not antigen-only')
                raw = generator(FrameRelationGraph(**graphs[sid]).to(device))
                base_bb = raw['h3_backbone']
                inputs = move({**x, **feature['features']}, device)
                inputs.update(base_bb=base_bb,
                              base_logits=torch.zeros_like(inputs['base_logits']))
                captured.clear()
                result = model(dict(inputs))
                if not torch.isfinite(result['backbone']).all():
                    raise ValueError(f'{sid}: nonfinite final backbone')
                states = {'initial': base_bb.detach().cpu().clone(),
                          **captured,
                          'final': result['backbone'].detach().cpu()}
                entry = dict(sample_id=sid, seed=seed)
                for stage in STAGES:
                    if stage not in states:
                        continue
                    # Metrics are device-agnostic; keep everything on CPU so the
                    # captured state and the labels share a device.
                    for key, value in metrics_at(states[stage], x, y).items():
                        entry[f'{stage}_{key}'] = value
                records.append(entry)
                print(json.dumps(dict(seed=seed, sample_id=sid,
                                     initial_cn=round(entry['initial_cn_mae'], 4),
                                     final_cn=round(entry['final_cn_mae'], 4))), flush=True)
        for handle in handles:
            handle.remove()
        del generator, model
        torch.cuda.empty_cache()
        per_seed[seed] = records

    keys = ('cn_mae', 'cn_internal', 'cn_left_anchor', 'cn_right_anchor',
            'cn_frac_gt_0_2', 'interface_fape', 'contact_f1')
    summary = dict(
        cohort='dev-60 reconstructed benchmark', seeds=args.seeds, n=len(ids),
        stages=list(STAGES),
        metrics={k: {stage: dict(
            mean=statistics.mean(rec[f'{stage}_{k}'] for rec in per_seed[seed]),
            seed_std=0.) for stage in STAGES} for k in ()},   # filled below
        notes=['initial is the generator backbone (base_bb), so the gap to final isolates the '
               'refinement chain',
               'cn_internal covers loop-internal i,i+1 bonds; cn_left_anchor and '
               'cn_right_anchor are the two flanking framework connections',
               'contact F1 uses hard 8 A CA contacts, matching the objective definition',
               'read-only probe: no checkpoint or prediction is modified'])
    # per-seed means, then mean and spread over seeds
    summary['metrics'] = {}
    for key in keys:
        summary['metrics'][key] = {}
        for stage in STAGES:
            values = [statistics.mean(rec[f'{stage}_{key}'] for rec in per_seed[seed])
                      for seed in args.seeds]
            summary['metrics'][key][stage] = dict(
                mean=statistics.mean(values), per_seed=values,
                sample_std=statistics.stdev(values) if len(values) > 1 else 0.)
    # paired deltas against the initial state, grouped by H3 cluster
    summary['deltas'] = {}
    for key in keys:
        summary['deltas'][key] = {}
        for stage in STAGES:
            if stage == 'initial':
                continue
            matrix = []
            for seed in args.seeds:
                seed_rows = {rec['sample_id']: rec for rec in per_seed[seed]}
                matrix.append(np.array([seed_rows[sid][f'{stage}_{key}'] -
                                        seed_rows[sid][f'initial_{key}'] for sid in ids]))
            matrix = np.stack(matrix)
            stats = interval(matrix, [clusters[sid] for sid in ids], args.replicates)
            summary['deltas'][key][stage] = dict(
                delta=stats['mean'], ci95=stats['conditional_group_ci95'],
                per_seed_mean=stats['per_seed_mean'],
                significant=bool(stats['conditional_group_ci95'][0] *
                                 stats['conditional_group_ci95'][1] > 0))
    dump(out / 'stage_diagnostic.json', summary)
    import csv
    flat = [rec for seed in args.seeds for rec in per_seed[seed]]
    with (out / 'per_sample.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    for key in ('cn_mae', 'interface_fape', 'contact_f1'):
        print(json.dumps({stage: round(summary['metrics'][key][stage]['mean'], 4)
                          for stage in STAGES}), flush=True)
    print(json.dumps({'cn_internal': {stage: round(
        summary['metrics']['cn_internal'][stage]['mean'], 4) for stage in STAGES},
        'cn_left_anchor': {stage: round(
            summary['metrics']['cn_left_anchor'][stage]['mean'], 4) for stage in STAGES},
        'cn_right_anchor': {stage: round(
            summary['metrics']['cn_right_anchor'][stage]['mean'], 4)
            for stage in STAGES}}, indent=1))


if __name__ == '__main__':
    main()
