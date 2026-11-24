"""Extend the four-cell post-processing pilot from three cases to the full RAbD test set.

Four conditions per complex, all scored with the same definitions:

    ours: before reconstruction   the pair stack's raw backbone (before the final projection)
    ours: after reconstruction    the model output (after ``residue_geometry``)
    RAAD: raw output              RAAD's committed backbone
    RAAD: after the same reconstruction   ``residue_geometry`` applied to RAAD's backbone

The "reconstruction" is the CA-preserving geometric projection the model applies at the
end of ``StructureRefiner.forward`` (``residue_geometry``): it keeps CA and the CA-C
frame and rebuilds N, C and O with idealised intra-residue geometry. Because contact F1
and interface FAPE are defined on CA/backbone frames, they must be invariant under this
operator; the invariance is measured and reported as a consistency check.

Read-only: no checkpoint or prediction is modified. Metrics are the archived
``cn_details`` (C-N MAE), ``interface_fape`` and a hard 8 A CA contact F1 that matches the
objective's definition.
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
from structure_refinement_ablation import residue_geometry
from eval.stage_diagnostic_v1 import metrics_at
from orientation_interface_refiner import interface_fape as archived_fape
from train.train_abgaze_reconstructed import sha, dump, move
from research_audit.statistics import interval

WORKSPACE = Path('/media/xd/code1/dyGF')
PACKAGE = ROOT.parent
RAAD = WORKSPACE / 'outputs/raad_abgaze_reconstructed_test_v1/generated.json'
KEYS = ('cn_mae', 'interface_fape', 'contact_f1', 'contact_soft_f1')
PILOT = ('5mes', '5b8c', '4qci')


def soft_contact_f1(prediction, x, y, threshold=8., width=.75):
    """The objective's smooth contact F1, reported alongside the hard one so the
    pilot's convention can be identified."""
    valid = y['mask'][:, 1].bool()
    context = x['context_bb'][x['antigen_mask'] & x['context_mask'][:, 1], 1]
    if len(context) < 1:
        return float('nan')
    distances = torch.cdist(prediction[valid, 1], context)
    truth = torch.cdist(y['bb'][valid, 1], context)
    contacts = truth <= threshold
    soft = torch.sigmoid((threshold - distances) / width)
    return float((2 * (soft * contacts).sum() + 1) / (soft.sum() + contacts.sum() + 1))


def metrics_with_soft(prediction, x, y):
    row = metrics_at(prediction, x, y)
    row['contact_soft_f1'] = soft_contact_f1(prediction, x, y)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2026, 2027, 2028])
    parser.add_argument('--ids', nargs='+', default=None,
                        help='restrict to these complex ids (used to replicate the three-case pilot)')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--replicates', type=int, default=10000)
    parser.add_argument('--refiner-template',
                        default='checkpoints/original_seed{seed}/pair.pt')
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
    if args.ids:
        wanted = {value.lower() for value in args.ids}
        rows = [row for row in rows if row[0].lower() in wanted]
        if len(rows) != len(wanted):
            raise ValueError(f'requested {len(wanted)} ids, found {len(rows)}')
    ids = [row[0] for row in rows]
    clusters = {json.loads(line)['pdb']: f"cluster_{json.loads(line)['cluster']}"
                for line in (dataset / 'splits/test.json').read_text().splitlines()
                if line.strip()}
    raad = {row['sample_id']: row for row in json.loads(RAAD.read_text())}

    records = {name: {} for name in ('ours_before', 'ours_after', 'raad_raw', 'raad_after')}
    for seed in args.seeds:
        generator_state = torch.load(PACKAGE / f'checkpoints/original_seed{seed}/generator.pt',
                                     map_location='cpu', weights_only=False)
        generator = FrameRelationRefiner(**generator_state['model_config']).to(device)
        generator.load_state_dict(generator_state['model'])
        generator.eval()
        candidates = [PACKAGE / args.refiner_template.format(seed=seed),
                      WORKSPACE / args.refiner_template.format(seed=seed)]
        pair_path = next((c for c in candidates if c.is_file()), None)
        if pair_path is None:
            raise FileNotFoundError(args.refiner_template.format(seed=seed))
        pair_state = torch.load(pair_path, map_location='cpu', weights_only=False)
        model = PersistentLoopPairRefiner(**pair_state['model_config']).to(device)
        model.load_state_dict(pair_state['model'])
        model.eval()
        captured = {}
        handle = model.loop_pair_layers[-1].register_forward_hook(
            lambda module, inputs, output: captured.__setitem__(
                'raw', output[1].detach().cpu().clone()))

        for sid, x, y in rows:
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
            captured.clear()
            with torch.no_grad():
                result = model(dict(inputs))
            final = result['backbone'].detach().cpu()
            before = captured.get('raw')
            if before is None:
                raise ValueError(f'{sid}: failed to capture the pre-projection backbone')
            # The projection must map the captured state onto the returned backbone.
            if not torch.allclose(residue_geometry(before), final, atol=1e-5):
                raise ValueError(f'{sid}: final output is not residue_geometry(captured state)')
            for name, backbone in (('ours_before', before), ('ours_after', final)):
                records[name].setdefault(sid, {})[seed] = metrics_with_soft(backbone, x, y)
            if 'raad_raw' not in records or sid not in records['raad_raw']:
                raad_backbone = torch.tensor(raad[sid]['backbone'], dtype=torch.float32)
                records['raad_raw'][sid] = {0: metrics_with_soft(raad_backbone, x, y)}
                records['raad_after'][sid] = {0: metrics_with_soft(residue_geometry(raad_backbone),
                                                              x, y)}
        handle.remove()
        del generator, model
        torch.cuda.empty_cache()

    # invariance check: FAPE and contact F1 must not move under the projection
    invariance = {}
    for key in ('interface_fape', 'contact_f1', 'contact_soft_f1', 'cn_mae'):
        ours = [abs(records['ours_after'][sid][seed][key] - records['ours_before'][sid][seed][key])
                for sid in ids for seed in records['ours_before'][sid]]
        raad_values = [abs(records['raad_after'][sid][0][key] - records['raad_raw'][sid][0][key])
                       for sid in ids]
        invariance[key] = dict(ours_max=max(ours), ours_mean=statistics.mean(ours),
                               raad_max=max(raad_values), raad_mean=statistics.mean(raad_values))

    conditions = ('ours_before', 'ours_after', 'raad_raw', 'raad_after')
    summary = dict(
        cohort='RAbD test split (dev-60) of the reconstructed benchmark',
        n=len(ids), ids=ids, seeds=args.seeds, metrics=list(KEYS),
        reconstruction=('residue_geometry: keeps CA and the CA-C frame, rebuilds N, C and O with '
                        'idealised intra-residue geometry; the same operator the model applies '
                        'internally'),
        invariance=invariance,
        means={}, per_seed={}, deltas={},
        notes=['ours values are first averaged over the three seeds per complex, then over '
               'complexes',
               'RAAD is a single deterministic output, so it has no seed dimension',
               'contact F1 uses hard 8 A CA contacts; interface FAPE is the archived function'])
    for condition in conditions:
        summary['means'][condition] = {}
        summary['per_seed'][condition] = {}
        for key in KEYS:
            if condition.startswith('ours'):
                per_seed = []
                for seed in args.seeds:
                    per_seed.append(statistics.mean(
                        records[condition][sid][seed][key] for sid in ids))
                per_complex = [statistics.mean(records[condition][sid][seed][key]
                                               for seed in records[condition][sid])
                               for sid in ids]
                summary['per_seed'][condition][key] = per_seed
                summary['means'][condition][key] = dict(
                    mean=statistics.mean(per_complex),
                    sample_std=statistics.stdev(per_seed) if len(per_seed) > 1 else 0.,
                    per_seed_mean=per_seed, per_complex=per_complex)
            else:
                per_complex = [records[condition][sid][0][key] for sid in ids]
                summary['means'][condition][key] = dict(
                    mean=statistics.mean(per_complex), sample_std=0., per_complex=per_complex)

    def value(condition, sid, key, seed=None):
        """Per-complex value; ours are averaged over seeds unless a seed is given."""
        table = records[condition][sid]
        if seed is not None and seed in table:
            return table[seed][key]
        return statistics.mean(entry[key] for entry in table.values())

    def contrast(first, second, key, label, per_seed=False):
        """Paired difference first - second, grouped by H3 cluster.

        With ``per_seed`` the interval sees one row per seed (both sides are ours);
        otherwise ours are averaged over seeds inside each complex and the interval
        sees a single row, which is the correct unit for a cross-method comparison.
        """
        rows = []
        seeds = args.seeds if per_seed else [None]
        for seed in seeds:
            rows.append(np.array([value(first, sid, key, seed) - value(second, sid, key, seed)
                                  for sid in ids]))
        matrix = np.stack(rows)
        stats = interval(matrix, [clusters[sid] for sid in ids], args.replicates)
        summary['deltas'][label] = dict(
            difference=stats['mean'], ci95=stats['conditional_group_ci95'],
            per_seed_mean=stats['per_seed_mean'],
            significant=bool(stats['conditional_group_ci95'][0] *
                             stats['conditional_group_ci95'][1] > 0))
        return summary['deltas'][label]

    for key in KEYS:
        for label, first, second in (
                ('projection_effect_ours', 'ours_after', 'ours_before'),
                ('projection_effect_raad', 'raad_after', 'raad_raw'),
                ('ours_minus_raad_after', 'ours_after', 'raad_after'),
                ('ours_minus_raad_raw', 'ours_before', 'raad_raw')):
            contrast(first, second, key, f'{key}:{label}',
                     per_seed=(first.startswith('ours') and second.startswith('ours')))

    dump(out / 'postprocess_fullcohort.json', summary)
    import csv
    flat = []
    for condition in conditions:
        for sid in ids:
            row = dict(condition=condition, sample_id=sid)
            for key in KEYS:
                if condition.startswith('ours'):
                    row[key] = statistics.mean(records[condition][sid][seed][key]
                                               for seed in records[condition][sid])
                else:
                    row[key] = records[condition][sid][0][key]
            flat.append(row)
    with (out / 'per_sample.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)

    print('%-16s %10s %10s %10s' % ('condition', 'C-N MAE', 'FAPE', 'contact F1'))
    for condition in conditions:
        print('%-16s %10.4f %10.4f %10.4f' % (
            condition, summary['means'][condition]['cn_mae']['mean'],
            summary['means'][condition]['interface_fape']['mean'],
            summary['means'][condition]['contact_f1']['mean']))
    print(json.dumps({key: {label: round(value['difference'], 4)
                            for label, value in summary['deltas'].items()
                            if label.startswith(key)}
                      for key in KEYS}, indent=1))


if __name__ == '__main__':
    main()
