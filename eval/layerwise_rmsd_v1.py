"""Per-layer raw and aligned H3 C-alpha RMSD trajectory for the current checkpoints.

Motivation: the manuscript reports only the delivered backbone, so it is unknown whether the
four refinement layers keep improving loop geometry or start undoing good early predictions. This
probe records, for every layer, the backbone *before* the terminal intra-residue reconstruction:

    X^0            the frozen generator's initial backbone
    layer1..layer4 the backbone after each loop-pair layer
    final          the returned backbone (= reconstruction of layer4)

Two pose-dependent readouts are reported for every stage:

    raw      C-alpha RMSD in the supplied complex frame, no superposition
    aligned  C-alpha RMSD after superposing the loop on itself (Kabsch)

Raw RMSD is the primary readout because it also penalises displacement relative to the antigen;
aligned RMSD isolates shape. Nothing is trained and no prediction is modified.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

PACKAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE / 'code'))
sys.path.insert(0, '/media/xd/code1/dyGF')
from frame_relation_refiner import FrameRelationGraph, FrameRelationRefiner
from persistent_loop_pair_refiner import PersistentLoopPairRefiner
from train.train_abgaze_reconstructed import move

WORKSPACE = Path('/media/xd/code1/dyGF')
SEEDS = (2026, 2027, 2028)
THREE = {'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE', 'G': 'GLY',
         'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU', 'M': 'MET', 'N': 'ASN',
         'P': 'PRO', 'Q': 'GLN', 'R': 'ARG', 'S': 'SER', 'T': 'THR', 'V': 'VAL',
         'W': 'TRP', 'Y': 'TYR'}


def kabsch(prediction, reference):
    prediction, reference = prediction.double(), reference.double()
    p_centre, r_centre = prediction.mean(0), reference.mean(0)
    u, _, vt = torch.linalg.svd((prediction - p_centre).T @ (reference - r_centre))
    d = torch.sign(torch.linalg.det(vt.T @ u.T))
    rotation = vt.T @ torch.diag(torch.tensor([1., 1., d], dtype=torch.double)) @ u.T
    return rotation, r_centre - rotation @ p_centre


def rmsds(prediction, native):
    raw = float((prediction.double() - native.double()).square().sum(-1).mean().sqrt())
    rotation, translation = kabsch(prediction, native)
    moved = prediction.double() @ rotation.T + translation
    aligned = float((moved - native.double()).square().sum(-1).mean().sqrt())
    return raw, aligned


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--pair-template', default=None,
                        help='per-seed checkpoint template, e.g. {workspace}/outputs/arm_seed{seed}/pair/best.pt')
    parser.add_argument('--generator-template', default=None,
                        help='per-seed generator template; defaults to the published checkpoints')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    parser.add_argument('--save-backbones', type=Path, default=None,
                        help='optional JSON path for the delivered backbones and sequences')
    parser.add_argument('--coordinated-bounds', default=None,
                        help='per-layer coordinated bounds used at training time')
    parser.add_argument('--local-bounds', default=None,
                        help='per-layer local bounds used at training time')
    parser.add_argument('--coordinated-bound', type=float, default=None,
                        help='must match the value used when a two_scale checkpoint was trained')
    parser.add_argument('--local-bound', type=float, default=None,
                        help='must match the value used when a two_scale checkpoint was trained')
    parser.add_argument('--variant', default='baseline',
                        choices=('baseline', 'init_state', 'anchor', 'two_scale'),
                        help='loop-pair layer variant of the checkpoint being evaluated')
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    dataset = WORKSPACE / 'data/abgaze_reconstructed_v1'
    pairs = torch.load(WORKSPACE / 'data/raad_abgaze_reconstructed_v1/test.pt',
                       map_location='cpu', weights_only=False)
    graphs = {item['sample_id']: graph for item, graph in pairs}
    rows = torch.load(dataset / 'tensors/test.pt', map_location='cpu', weights_only=False)
    ids = [row[0] for row in rows]
    clusters = {json.loads(line)['pdb']: f"cluster_{json.loads(line)['cluster']}"
                for line in (dataset / 'splits/test.json').read_text().splitlines() if line.strip()}

    stages = ['X0', 'layer1', 'layer2', 'layer3', 'layer4', 'final']
    per_seed, per_complex, saved = {}, {}, {}
    seeds = tuple(args.seeds)
    for seed in seeds:
        generator_path = (Path(args.generator_template.format(seed=seed, workspace=WORKSPACE))
                          if args.generator_template
                          else PACKAGE / f'checkpoints/original_seed{seed}/generator.pt')
        generator_state = torch.load(generator_path,
                                     map_location='cpu', weights_only=False)
        generator = FrameRelationRefiner(**generator_state['model_config']).to(device)
        generator.load_state_dict(generator_state['model'])
        generator.eval()
        pair_path = (Path(args.pair_template.format(seed=seed, workspace=WORKSPACE))
                     if args.pair_template
                     else WORKSPACE / f'outputs/aligned_shape_w0.3_seed{seed}/pair/best.pt')
        pair_state = torch.load(pair_path,
                                map_location='cpu', weights_only=False)
        if args.variant == 'init_state':
            from init_state_pair_refiner import InitStatePairRefiner
            model = InitStatePairRefiner(**pair_state['model_config']).to(device)
        elif args.variant == 'anchor':
            from anchor_pair_refiner import AnchorPairRefiner
            model = AnchorPairRefiner(**pair_state['model_config']).to(device)
        elif args.variant == 'two_scale':
            from two_scale_pair_refiner import TwoScaleLoopPairLayer, TwoScalePairRefiner
            # The bounds are part of the forward dynamics, not learned parameters, so the
            # checkpoint must be evaluated with exactly the values it was trained with.
            if args.coordinated_bound is not None:
                TwoScaleLoopPairLayer.COORDINATED_BOUND = args.coordinated_bound
            if args.local_bound is not None:
                TwoScaleLoopPairLayer.LOCAL_BOUND = args.local_bound
            schedule = {}
            if args.coordinated_bounds:
                schedule['coordinated_bounds'] = [float(x) for x in
                                                  args.coordinated_bounds.split(',')]
            if args.local_bounds:
                schedule['local_bounds'] = [float(x) for x in args.local_bounds.split(',')]
            model = TwoScalePairRefiner(**pair_state['model_config'], **schedule).to(device)
        else:
            model = PersistentLoopPairRefiner(**pair_state['model_config']).to(device)
        model.load_state_dict(pair_state['model'])
        model.eval()

        captured = {}
        handles = []
        for index, layer in enumerate(model.loop_pair_layers):
            def hook(module, inputs, output, index=index):
                # LoopPairLayer returns (h, bb, z); the backbone is the second element.
                captured[f'layer{index + 1}'] = output[1].detach().cpu()
            handles.append(layer.register_forward_hook(hook))

        values = {stage: {k: [] for k in ('raw', 'aligned')} for stage in stages}
        delivered = []
        with torch.no_grad():
            for sid, x, y in rows:
                feature = torch.load(WORKSPACE / 'outputs/binding_interface_test_v1/features'
                                     / sid / 'features.pt', map_location='cpu', weights_only=False)
                raw = generator(FrameRelationGraph(**graphs[sid]).to(device))
                inputs = move({**x, **feature['features']}, device)
                inputs.update(base_bb=raw['h3_backbone'],
                              base_logits=torch.zeros_like(inputs['base_logits']))
                captured.clear()
                result = model(dict(inputs))
                native = y['bb'][y['mask'][:, 1], 1].cpu()
                states = {**captured,
                          'X0': raw['h3_backbone'].detach().cpu(),
                          'final': result['backbone'].detach().cpu()}
                predicted = result.get('sequence_logits')
                sequence = (''.join('ACDEFGHIKLMNPQRSTVWY'[i] for i in
                                    predicted.argmax(-1).tolist())
                            if predicted is not None else '')
                delivered.append((sid, result['backbone'].detach().cpu(), sequence,
                                  states['layer4']))
                for stage in stages:
                    backbone = states[stage]
                    valid = y['mask'][:, 1]
                    raw_rmsd, aligned_rmsd = rmsds(backbone[valid, 1], native)
                    values[stage]['raw'].append(raw_rmsd)
                    values[stage]['aligned'].append(aligned_rmsd)
        for handle in handles:
            handle.remove()
        per_seed[seed] = {stage: {k: float(np.mean(v)) for k, v in values[stage].items()}
                          for stage in stages}
        per_complex[seed] = {stage: {k: list(v) for k, v in values[stage].items()}
                             for stage in stages}
        for sid, backbone, sequence, pre in delivered:
            saved.setdefault(str(seed), []).append(
                dict(sample_id=sid, backbone=backbone.tolist(), greedy=sequence,
                     backbone_pre_reconstruction=pre.tolist()))
        del generator, model
        torch.cuda.empty_cache()
        print('seed', seed, json.dumps({s: {k: round(v, 4) for k, v in
                                            per_seed[seed][s].items()} for s in stages}))

    summary = dict(stages=stages, seeds=list(seeds),
                   note='X0 is the generator backbone; layer1-4 are pre-reconstruction; '
                        'final is the reconstruction of layer4 on the returned path',
                   means={stage: {k: float(np.mean([per_seed[s][stage][k] for s in seeds]))
                                  for k in ('raw', 'aligned')} for stage in stages},
                   per_seed={str(s): per_seed[s] for s in seeds},
                   per_complex={str(s): {stage: {k: v for k, v in d.items()}
                                         for stage, d in per_complex[s].items()}
                                for s in seeds})
    if args.save_backbones is not None:
        args.save_backbones.write_text(json.dumps(saved, indent=2) + '\n')
    (out / 'layerwise_rmsd.json').write_text(json.dumps(summary, indent=2) + '\n')
    print()
    print('%-8s %10s %10s   %s' % ('stage', 'raw', 'aligned', 'delta vs previous (raw)'))
    previous = None
    for stage in stages:
        block = summary['means'][stage]
        delta = '' if previous is None else '%+.4f' % (block['raw'] - previous['raw'])
        print('%-8s %10.4f %10.4f   %s' % (stage, block['raw'], block['aligned'], delta))
        previous = block


if __name__ == '__main__':
    main()
