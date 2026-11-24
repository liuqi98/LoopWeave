"""Stage-resolved endpoint values for the reported two-scale checkpoints.

For each fitted seed the frozen generator's backbone (X0), the output of each of the four
refinement layers, and the returned (reconstructed) backbone are scored on the same complexes with
the same definitions used in the main comparison: normalized interface FAPE, contact F1 on the
retained-antigen C-alpha contact set at 8 A, and C-N bond error overall and by region. The
deterministic reconstruction operator is also applied to the saved RAAD backbone so that the
before/after contrast uses one operator on both prediction sources.
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import torch

PACKAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE / 'code'))
sys.path.insert(0, '/media/xd/code1/dyGF')
from frame_relation_refiner import FrameRelationGraph, FrameRelationRefiner
from two_scale_pair_refiner import TwoScaleLoopPairLayer, TwoScalePairRefiner
from structure_refinement_ablation import residue_geometry
from eval.audit_fape_and_cn_claims_v1 import cn_details
from orientation_interface_refiner import interface_fape
from train.train_abgaze_reconstructed import dump, move

WS = Path('/media/xd/code1/dyGF')
ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
SEEDS = (2026, 2027, 2028)
CKPT = {2026: 'outputs/final_twoscale08_seed2026/pair/best.pt',
        2027: 'outputs/final_twoscale08_seed2027/pair/best.pt',
        2028: 'outputs/final_twoscale08_seed2028/pair/last.pt'}
RAAD = {2026: 'outputs/raad_abgaze_reconstructed_test_v1/generated.json',
        2027: 'outputs/raad_reconstructed_seed2027_test_v2/generated.json',
        2028: 'outputs/raad_reconstructed_seed2028_test_v2/generated.json'}
STAGES = ('X0', 'layer1', 'layer2', 'layer3', 'layer4', 'final')
CONTACT_CUT = 8.0
INTERFACE_RADIUS = 10.0


def contact_f1(prediction, x, y):
    """F1 between predicted and native H3-antigen C-alpha contacts on observed residues."""
    valid = y['mask'][:, 1].bool()
    antigen = x['context_bb'][:, 1]
    antigen_mask = x['context_mask'][:, 1] & x['antigen_mask']
    antigen = antigen[antigen_mask]
    if not len(antigen) or not valid.any():
        return None
    pred_ca = prediction[valid, 1]
    native_ca = y['bb'][valid, 1].to(prediction.device)
    pred_contacts = (torch.cdist(pred_ca, antigen) <= CONTACT_CUT)
    native_contacts = (torch.cdist(native_ca, antigen) <= CONTACT_CUT)
    tp = float((pred_contacts & native_contacts).sum())
    fp = float((pred_contacts & ~native_contacts).sum())
    fn = float((~pred_contacts & native_contacts).sum())
    if tp == 0:
        return dict(f1=0.0, precision=0.0, recall=0.0, tp=0, fp=int(fp), fn=int(fn))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return dict(f1=2 * precision * recall / (precision + recall), precision=precision,
                recall=recall, tp=int(tp), fp=int(fp), fn=int(fn))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workspace', type=Path, default=WS)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--coordinated-bound', type=float, default=0.8)
    parser.add_argument('--local-bound', type=float, default=0.2)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    dataset = WS / 'data/abgaze_reconstructed_v1'
    graph_path = WS / 'data/raad_abgaze_reconstructed_v1/test.pt'
    rows = torch.load(dataset / 'tensors/test.pt', weights_only=False)
    graphs = {item['sample_id']: g for item, g in torch.load(graph_path, weights_only=False)}
    clusters = {json.loads(line)['pdb']: f"cluster_{json.loads(line)['cluster']}"
                for line in (dataset / 'splits/test.json').read_text().splitlines() if line.strip()}
    features_dir = WS / 'outputs/binding_interface_test_v1/features'

    TwoScaleLoopPairLayer.COORDINATED_BOUND = args.coordinated_bound
    TwoScaleLoopPairLayer.LOCAL_BOUND = args.local_bound

    per_seed, per_complex = {}, {}
    raad_rows = {}
    for seed in args.seeds:
        state = torch.load(WS / CKPT[seed], map_location='cpu', weights_only=False)
        model = TwoScalePairRefiner(**state['model_config']).to(device)
        model.load_state_dict(state['model'], strict=True)
        model.eval()
        gs = torch.load(WS / f'outputs/full_pipeline_seed{seed}_v1/generator/best.pt',
                        map_location='cpu', weights_only=False)
        generator = FrameRelationRefiner(**gs['model_config']).to(device)
        generator.load_state_dict(gs['model'], strict=True)
        generator.eval()
        captured, handles = {}, []
        for index, layer in enumerate(model.loop_pair_layers):
            def hook(module, inputs, output, index=index):
                captured[f'layer{index + 1}'] = output[1].detach()
            handles.append(layer.register_forward_hook(hook))
        values = {stage: [] for stage in STAGES}
        records = {}
        with torch.no_grad():
            for sid, x, y in rows:
                feature = torch.load(features_dir / sid / 'features.pt', map_location='cpu',
                                     weights_only=False)
                raw = generator(FrameRelationGraph(**graphs[sid]).to(device))
                inputs = move({**x, **feature['features']}, device)
                inputs.update(base_bb=raw['h3_backbone'],
                              base_logits=torch.zeros_like(inputs['base_logits']))
                captured.clear()
                result = model(dict(inputs))
                states = {**captured, 'X0': raw['h3_backbone'].detach(),
                          'final': result['backbone'].detach()}
                row = {}
                xd, yd = move(x, device), move(y, device)
                for stage in STAGES:
                    backbone = states[stage]
                    fape_value = interface_fape(backbone, xd, yd)
                    cn = cn_details(backbone, xd, yd)
                    contacts = contact_f1(backbone, xd, yd)
                    row[stage] = dict(fape=float(fape_value.detach()) if fape_value is not None else None,
                                      **{k: float(v) for k, v in cn.items()},
                                      contact_f1=float(contacts['f1']) if contacts else None,
                                      precision=float(contacts['precision']) if contacts else None,
                                      recall=float(contacts['recall']) if contacts else None)
                    values[stage].append(row[stage])
                records[sid] = row
        for handle in handles:
            handle.remove()
        del generator, model
        torch.cuda.empty_cache()
        per_seed[str(seed)] = {stage: {key: float(np.mean([v[key] for v in values[stage]
                                                           if v[key] is not None]))
                                       for key in ('fape', 'contact_f1', 'cn_mae', 'cn_left_anchor',
                                                   'cn_right_anchor', 'cn_internal',
                                                   'cn_frac_gt_0_1', 'cn_frac_gt_0_2')}
                              for stage in STAGES}
        per_complex[str(seed)] = records
        print('seed', seed, json.dumps({s: {k: round(per_seed[str(seed)][s][k], 4)
                                            for k in ('fape', 'contact_f1', 'cn_mae')}
                                        for s in STAGES}))
        # RAAD with the same reconstruction operator applied to the saved backbone.
        saved = {r['sample_id']: r['backbone'] for r in json.loads((WS / RAAD[seed]).read_text())}
        raad_records = {}
        for sid, x, y in rows:
            bb = torch.tensor(saved[sid], dtype=torch.float32, device=device)
            before = bb
            after = residue_geometry(bb)
            xd, yd = move(x, device), move(y, device)
            for label, backbone in (('saved', before), ('reconstructed', after)):
                fape_value = interface_fape(backbone, xd, yd)
                cn = cn_details(backbone, xd, yd)
                contacts = contact_f1(backbone, xd, yd)
                raad_records.setdefault(sid, {})[label] = dict(
                    fape=float(fape_value.detach()) if fape_value is not None else None,
                    **{k: float(v) for k, v in cn.items()},
                    contact_f1=float(contacts['f1']) if contacts else None)
        raad_rows[str(seed)] = raad_records

    clusters_out = {sid: clusters[sid] for sid in per_complex[str(args.seeds[0])]}
    summary = dict(stages=STAGES, note=('X0 is the frozen generator backbone; layer1-4 are pre-'
                                        'reconstruction; final is the returned, reconstructed backbone. '
                                        'FAPE is the normalized H3-frame antigen error of the main comparison (clamp 10 A, no contact crop); contact F1 uses '
                                        'retained-antigen C-alpha contacts at 8 A; C-N error is in A.'),
                   seeds=list(args.seeds),
                   bounds=dict(coordinated=args.coordinated_bound, local=args.local_bound),
                   per_seed=per_seed, per_complex=per_complex,
                   raad_reconstruction={'per_seed': {
                       s: {label: {k: float(np.mean([v[label][k] for v in raad_rows[s].values()
                                                     if v[label][k] is not None]))
                                   for k in ('fape', 'contact_f1', 'cn_mae', 'cn_left_anchor',
                                             'cn_right_anchor', 'cn_internal')}
                           for label in ('saved', 'reconstructed')}
                       for s in raad_rows},
                       'per_complex': raad_rows})
    dump(out / 'stage_endpoints.json', summary)
    print('wrote', out / 'stage_endpoints.json')


if __name__ == '__main__':
    main()
