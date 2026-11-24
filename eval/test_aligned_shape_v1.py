"""Locked single test pass for the aligned-backbone retraining.

The weight and checkpoint were chosen on validation (`train_aligned_shape_v1.py`
runs, selection = minimum validation aligned_rmsd subject to the original guards).
This script reads the locked checkpoint, generates on the development-60 benchmark
with the parent replica's own frozen generator, and scores with the unchanged
scorer. It performs one pass per run directory and refuses to overwrite.

This is the development benchmark that has been inspected repeatedly during method
development; it is not a confirmatory holdout.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train.train_abgaze_reconstructed import sha, dump
from persistent_loop_pair_refiner import PersistentLoopPairRefiner
from frame_relation_refiner import FrameRelationRefiner
from eval.test_matched_ablations_v3 import generate, SCORE_KEYS
from eval.test_loop_pair_replicates_v1 import score


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workspace', type=Path, default=Path('/media/xd/code1/dyGF'))
    parser.add_argument('--run', type=Path, required=True,
                        help='Training run directory containing plan.json and pair/best.pt')
    parser.add_argument('--label', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(1)
    workspace = args.workspace.resolve()
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    run = args.run.resolve()
    plan = read(run / 'plan.json')
    checkpoint = run / 'pair/best.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['plan_sha256'] != sha(run / 'plan.json'):
        raise ValueError('checkpoint does not belong to its plan')
    status = read(run / 'status.json')
    if status.get('phase') != 'complete':
        raise ValueError(f'run is not complete: {status}')
    if state['epoch'] != status.get('selected_epoch'):
        raise ValueError('checkpoint epoch differs from the validation-selected epoch')
    parent = Path(plan.get('parent', run))

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    dataset = workspace / 'data/abgaze_reconstructed_v1'
    graph_path = workspace / 'data/raad_abgaze_reconstructed_v1/test.pt'
    test_path = dataset / 'tensors/test.pt'
    features_dir = workspace / 'outputs/binding_interface_test_v1/features'
    rows = torch.load(test_path, weights_only=False)
    graphs = {item['sample_id']: g for item, g in torch.load(graph_path, weights_only=False)}
    metadata = {r['pdb']: r for r in
                map(json.loads, (dataset / 'splits/test.json').read_text().splitlines())}
    lock = dict(
        label=args.label, run=str(run), parent=str(parent),
        aligned_weight=plan.get('aligned_weight'), aligned_atom_set=plan.get('aligned_atom_set'),
        selected_epoch=state['epoch'], checkpoint_sha256=sha(checkpoint),
        plan_sha256=sha(run / 'plan.json'),
        generator_sha256=sha(parent / 'generator/best.pt'),
        test_sha256=sha(test_path), graph_sha256=sha(graph_path),
        K=5, temperature=.5, sampling_seed=2026,
        selection='weight and checkpoint chosen on validation before this test pass',
        code_sha256={Path(__file__).name: sha(Path(__file__))},
        caveat='Development-60 benchmark, repeatedly inspected; not a confirmatory holdout.')
    dump(out / 'lock.json', lock)

    model = PersistentLoopPairRefiner(**state['model_config']).to(device)
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    gs = torch.load(parent / 'generator/best.pt', map_location='cpu', weights_only=False)
    generator = FrameRelationRefiner(**gs['model_config']).to(device)
    generator.load_state_dict(gs['model'], strict=True)
    generator.eval()
    generated = generate(model, generator, rows, graphs, dataset, features_dir, device)
    result = score(rows, generated)
    result.update(test_sha256=sha(test_path), label=args.label,
                  aligned_weight=plan.get('aligned_weight'), selected_epoch=state['epoch'])
    dump(out / 'generated.json', generated)
    dump(out / 'metrics.json', result)
    with (out / 'scores.csv').open('x', newline='') as handle:
        fields = ['sample_id', 'arm', 'seed', 'length'] + list(SCORE_KEYS)
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for case in result['per_complex']:
            writer.writerow(dict(sample_id=case['sample_id'], arm=args.label,
                                 seed=plan['seed'], **{k: case[k] for k in SCORE_KEYS
                                                       if k in case}))
    dump(out / 'status.json', dict(phase='complete', label=args.label,
                                   selected_epoch=state['epoch'], n=result['n']))
    print(json.dumps(dict(label=args.label, epoch=state['epoch'],
                          summary={k: round(v, 4) for k, v in result['summary'].items()}),
                     indent=1))


if __name__ == '__main__':
    main()
