"""Five-candidate sequence sampling for the reported two-scale checkpoints.

Mirrors the archived sampling protocol exactly (K=5, temperature 0.5, fixed generator seed 2026, one
deterministic backbone per complex) so that the candidate-diversity figure is computed on the same
footing as the other arms, but for the two-scale model that the manuscript now reports.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

PACKAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE / 'code'))
sys.path.insert(0, '/media/xd/code1/dyGF')
from frame_relation_refiner import FrameRelationGraph, FrameRelationRefiner
from two_scale_pair_refiner import TwoScalePairRefiner
from train.train_abgaze_reconstructed import move

WORKSPACE = Path('/media/xd/code1/dyGF')
ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
CHECKS = {
    2026: 'outputs/final_twoscale08_seed2026/pair/best.pt',
    2027: 'outputs/final_twoscale08_seed2027/pair/best.pt',
    2028: 'outputs/final_twoscale08_seed2028/pair/last.pt',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed', type=int, required=True, choices=sorted(CHECKS))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    out = args.output.resolve()

    pair = torch.load(WORKSPACE / CHECKS[args.seed], map_location='cpu', weights_only=False)
    model = TwoScalePairRefiner(**pair['model_config'], coordinated_bounds=[0.80] * 4,
                                local_bounds=[0.20] * 4).to(device)
    model.load_state_dict(pair['model'])
    model.eval()
    generator_state = torch.load(PACKAGE / f'checkpoints/original_seed{args.seed}/generator.pt',
                                 map_location='cpu', weights_only=False)
    generator = FrameRelationRefiner(**generator_state['model_config']).to(device)
    generator.load_state_dict(generator_state['model'])
    generator.eval()

    dataset = WORKSPACE / 'data/abgaze_reconstructed_v1'
    graphs = {item['sample_id']: graph for item, graph in
              torch.load(WORKSPACE / 'data/raad_abgaze_reconstructed_v1/test.pt',
                         map_location='cpu', weights_only=False)}
    rows = torch.load(dataset / 'tensors/test.pt', map_location='cpu', weights_only=False)
    rng = torch.Generator().manual_seed(2026)
    generated = []
    with torch.no_grad():
        for sid, x, _ in rows:
            raw = generator(FrameRelationGraph(**graphs[sid]).to(device))
            feature = torch.load(WORKSPACE / 'outputs/binding_interface_test_v1/features' / sid
                                 / 'features.pt', map_location='cpu', weights_only=False)
            inputs = move({**x, **feature['features']}, device)
            inputs.update(base_bb=raw['h3_backbone'],
                          base_logits=torch.zeros_like(inputs['base_logits']))
            result = model(inputs)
            backbone = result['backbone'].cpu()
            logits = result['sequence_logits'].cpu()
            assert torch.isfinite(backbone).all() and torch.isfinite(logits).all(), sid
            samples = torch.multinomial((logits / .5).softmax(-1), 5, replacement=True,
                                        generator=rng).T
            generated.append(dict(
                sample_id=sid, backbone=backbone.tolist(), sequence_logits=logits.tolist(),
                greedy=''.join(ALPHABET[i] for i in logits.argmax(-1).tolist()),
                candidates=[''.join(ALPHABET[i] for i in seq) for seq in samples.tolist()]))
    out.mkdir(parents=True, exist_ok=False)
    (out / 'generated.json').write_text(json.dumps(generated) + '\n')
    plan = dict(K=5, temperature=.5, sampling_seed=2026, arm='twoscale_0.80_0.20',
                checkpoint=str(CHECKS[args.seed]), seed=args.seed,
                protocol='identical to the archived five-candidate sampling; one deterministic '
                         'backbone per complex, five categorical sequence draws')
    (out / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print('wrote', out, len(generated), 'complexes')


if __name__ == '__main__':
    main()
