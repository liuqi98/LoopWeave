"""Six-cell comparison: what does the four-layer refinement add on top of
"initial prediction + the same reconstruction"?

Cells (rows = prediction source, columns = before / after the deterministic intra-residue
geometry reconstruction):

    stage-1 initial X^0        before: generator backbone      after: residue_geometry(X^0)
    full learned refinement    before: last pair-layer state   after: model output
    RAAD                       before: committed backbone      after: residue_geometry(RAAD)

``residue_geometry`` is a deterministic, parameter-free operator that keeps CA and the
CA-C frame and rebuilds N, C and O with idealised intra-residue geometry. It performs
**no inter-residue closure**, which is why contact F1 and interface FAPE cannot move
under it while C-N can.

Every cell is computed here from the same 60 test complexes and the same metric
functions (``cn_details`` for the C-N decomposition, ``interface_fape``, hard 8 A CA
contact F1), so the six numbers are directly comparable. Ours are averaged over the three
pretrained seeds; RAAD is a single deterministic output.
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
from frame_relation_refiner import FrameRelationGraph, FrameRelationRefiner
from persistent_loop_pair_refiner import PersistentLoopPairRefiner
from structure_refinement_ablation import residue_geometry
from eval.stage_diagnostic_v1 import metrics_at
from train.train_abgaze_reconstructed import move
from research_audit.statistics import interval

WORKSPACE = Path('/media/xd/code1/dyGF')
PACKAGE = ROOT.parent
SEEDS = (2026, 2027, 2028)
KEYS = ('cn_mae', 'cn_internal', 'cn_left_anchor', 'cn_right_anchor',
        'interface_fape', 'contact_f1')
LOWER_IS_BETTER = {'cn_mae', 'cn_internal', 'cn_left_anchor', 'cn_right_anchor',
                   'interface_fape'}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    parser.add_argument('--replicates', type=int, default=10000)
    parser.add_argument('--generator-template', default='checkpoints/original_seed{seed}/generator.pt',
                        help='path to the frozen generator, with {seed}; relative to the package or workspace')
    parser.add_argument('--refiner-template',
                        default='checkpoints/original_seed{seed}/pair.pt',
                        help='path to the refiner checkpoint, with {seed}')
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    def resolve(template, seed):
        # Checkpoints may live in the snapshot or in the source workspace.
        for base in (PACKAGE, WORKSPACE):
            candidate = base / template.format(seed=seed)
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(template.format(seed=seed))

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    dataset = WORKSPACE / 'data/abgaze_reconstructed_v1'
    graphs = {item['sample_id']: graph for item, graph in
              torch.load(WORKSPACE / 'data/raad_abgaze_reconstructed_v1/test.pt',
                         map_location='cpu', weights_only=False)}
    rows = torch.load(dataset / 'tensors/test.pt', map_location='cpu', weights_only=False)
    ids = [row[0] for row in rows]
    clusters = {json.loads(line)['pdb']: f"cluster_{json.loads(line)['cluster']}"
                for line in (dataset / 'splits/test.json').read_text().splitlines()
                if line.strip()}
    raad = {row['sample_id']: row for row in
            json.loads((WORKSPACE / 'outputs/raad_abgaze_reconstructed_test_v1'
                        '/generated.json').read_text())}

    # cells[source][when][seed] = {sid: metrics}
    cells = {source: {when: {} for when in ('before', 'after')}
             for source in ('initial', 'refinement', 'raad')}
    for seed in args.seeds:
        generator_state = torch.load(resolve(args.generator_template, seed),
                                     map_location='cpu', weights_only=False)
        generator = FrameRelationRefiner(**generator_state['model_config']).to(device)
        generator.load_state_dict(generator_state['model'])
        generator.eval()
        pair_state = torch.load(resolve(args.refiner_template, seed),
                                map_location='cpu', weights_only=False)
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
            base_bb = raw['h3_backbone'].detach().cpu()
            inputs = move({**x, **feature['features']}, device)
            inputs.update(base_bb=raw['h3_backbone'],
                          base_logits=torch.zeros_like(inputs['base_logits']))
            captured.clear()
            with torch.no_grad():
                result = model(dict(inputs))
            before = captured.get('raw')
            if before is None:
                raise ValueError(f'{sid}: failed to capture the pre-reconstruction state')
            after = result['backbone'].detach().cpu()
            if not torch.allclose(residue_geometry(before), after, atol=1e-5):
                raise ValueError(f'{sid}: the model output is not the reconstruction of the '
                                 f'captured pre-reconstruction state')
            for candidate, backbone in (
                    ('initial', (base_bb, residue_geometry(base_bb))),
                    ('refinement', (before, after))):
                cells[candidate]['before'].setdefault(seed, {})[sid] = metrics_at(
                    backbone[0], x, y)
                cells[candidate]['after'].setdefault(seed, {})[sid] = metrics_at(
                    backbone[1], x, y)
        handle.remove()
        del generator, model
        torch.cuda.empty_cache()

    # RAAD is seed-free.
    for sid, x, y in rows:
        backbone = torch.tensor(raad[sid]['backbone'], dtype=torch.float32)
        cells['raad']['before'][0] = cells['raad']['before'].get(0, {})
        cells['raad']['after'][0] = cells['raad']['after'].get(0, {})
        cells['raad']['before'][0][sid] = metrics_at(backbone, x, y)
        cells['raad']['after'][0][sid] = metrics_at(residue_geometry(backbone), x, y)

    def cell_values(source, when, key, seed=None):
        table = cells[source][when]
        if seed is None:
            return [statistics.mean(table[s][sid][key] for s in table) for sid in ids]
        return [table[seed][sid][key] for sid in ids]

    summary = dict(
        cohort='dev-60 RAbD test split', n=len(ids), seeds=args.seeds,
        checkpoint_sets=dict(generator=args.generator_template,
                             refiner=args.refiner_template),
        reconstruction=('residue_geometry: a deterministic, parameter-free reconstruction of '
                        'intra-residue geometry that keeps CA and the CA-C frame and rebuilds '
                        'N, C and O. It performs no inter-residue closure.'),
        cells={}, per_seed={}, contrasts={},
        keys=list(KEYS),
        notes=['ours are averaged over the three pretrained seeds; RAAD is one deterministic output',
               'C-N is split into in-loop bonds (cn_internal) and the two framework flanks',
               'contact F1 uses hard 8 A CA contacts; interface FAPE is the archived function',
               'FAPE and contact F1 cannot move under the reconstruction because CA is preserved; '
               'their equality across the two columns is a consistency check'])
    for source in ('initial', 'refinement', 'raad'):
        summary['cells'][source] = {}
        for when in ('before', 'after'):
            summary['cells'][source][when] = {
                key: dict(mean=float(np.mean(cell_values(source, when, key))))
                for key in KEYS}
            if source != 'raad':
                summary['per_seed'][f'{source}_{when}'] = {
                    key: [float(np.mean(cell_values(source, when, key, seed)))
                          for seed in args.seeds] for key in KEYS}

    def contrast(name, source_a, when_a, source_b, when_b, key):
        if source_a == 'raad' or source_b == 'raad':
            matrix = np.array([[cell_values(source_a, when_a, key)[index] -
                                cell_values(source_b, when_b, key)[index]
                                for index in range(len(ids))]])
        else:
            matrix = np.stack([
                np.array([cell_values(source_a, when_a, key, seed)[index] -
                          cell_values(source_b, when_b, key, seed)[index]
                          for index in range(len(ids))]) for seed in args.seeds])
        sign = -1. if key in LOWER_IS_BETTER else 1.
        stats = interval(sign * matrix, [clusters[sid] for sid in ids], args.replicates)
        summary['contrasts'].setdefault(name, {})[key] = dict(
            difference=float(matrix.mean()), improvement=stats['mean'],
            ci95=stats['conditional_group_ci95'],
            per_seed_mean=stats['per_seed_mean'],
            significant=bool(stats['conditional_group_ci95'][0] *
                             stats['conditional_group_ci95'][1] > 0))

    # improvement-oriented: positive = the first cell is better than the second
    for key in KEYS:
        contrast('reconstruction_effect_initial', 'initial', 'after', 'initial', 'before', key)
        contrast('reconstruction_effect_refinement', 'refinement', 'after',
                 'refinement', 'before', key)
        contrast('reconstruction_effect_raad', 'raad', 'after', 'raad', 'before', key)
        contrast('refinement_minus_initial_before', 'refinement', 'before',
                 'initial', 'before', key)
        contrast('refinement_minus_initial_after', 'refinement', 'after',
                 'initial', 'after', key)
        contrast('refinement_after_minus_raad_after', 'refinement', 'after',
                 'raad', 'after', key)
        contrast('initial_after_minus_raad_after', 'initial', 'after', 'raad', 'after', key)

    with (out / 'sixcell.json').open('x') as handle:
        json.dump(summary, handle, indent=2)
        handle.write('\n')

    print('%-28s %8s %8s %8s %8s %8s %8s' % ('cell', 'C-N', 'in-loop', 'left', 'right',
                                             'FAPE', 'F1'))
    for source, label in (('initial', 'stage-1 initial X^0'),
                          ('refinement', 'full learned refinement'),
                          ('raad', 'RAAD')):
        for when in ('before', 'after'):
            values = summary['cells'][source][when]
            print('%-28s %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f' % (
                f'{label} [{when}]', values['cn_mae']['mean'],
                values['cn_internal']['mean'], values['cn_left_anchor']['mean'],
                values['cn_right_anchor']['mean'], values['interface_fape']['mean'],
                values['contact_f1']['mean']))
    print()
    print('%-40s %8s %8s %8s' % ('contrast (positive = first better)', 'C-N', 'FAPE', 'F1'))
    for name in ('reconstruction_effect_initial', 'reconstruction_effect_refinement',
                 'reconstruction_effect_raad',
                 'refinement_minus_initial_before', 'refinement_minus_initial_after',
                 'refinement_after_minus_raad_after', 'initial_after_minus_raad_after'):
        values = summary['contrasts'][name]
        marks = ''.join('*' if values[key]['significant'] else ' '
                        for key in ('cn_mae', 'interface_fape', 'contact_f1'))
        print('%-40s %+8.4f%s %+8.4f%s %+8.4f%s' % (
            name, values['cn_mae']['improvement'], marks[0],
            values['interface_fape']['improvement'], marks[1],
            values['contact_f1']['improvement'], marks[2]))


if __name__ == '__main__':
    main()
