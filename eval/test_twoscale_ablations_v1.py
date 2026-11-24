"""Score the two-scale ablation arms on the development benchmark.

Cells of the 2x2: full = the reported two-scale model, only_pair = binding branch removed,
only_binding = pair branch removed, base = both removed. Also scores the aligned-loss w=0 arms.
Protocol is inherited from ``eval/test_matched_ablations_v3.py``: checkpoints are locked before
any test tensor is read, generation uses each parent's frozen generator, and scoring reuses the
unchanged replicate scorer.
"""
import argparse, csv, json, sys
from pathlib import Path
import torch

PACKAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE / 'code'))
sys.path.insert(0, '/media/xd/code1/dyGF')
from train.train_abgaze_reconstructed import sha, dump, move
from train.train_matched_ablation_twoscale_v1 import SPEC, configure
from frame_relation_refiner import FrameRelationGraph, FrameRelationRefiner
from two_scale_pair_refiner import TwoScalePairRefiner
from eval.test_loop_pair_replicates_v1 import score
from eval.test_matched_ablations_v3 import generate

TWO_BY_TWO = {'base': 'no_pair_no_binding', 'only_pair': 'no_binding', 'only_binding': 'no_pair'}
SEEDS = (2026, 2027, 2028)


def run_paths(workspace, arm, seed):
    if arm == 'full':
        run = workspace / f'outputs/final_twoscale08_seed{seed}'
        # 2028 selected epoch is the final one because the guard rejected later improvements
        checkpoint = run / 'pair/best.pt'
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if state['epoch'] == 0 or (seed == 2028):
            checkpoint = run / 'pair/last.pt'
        # the generator is frozen upstream, not part of these continuation runs
        return run, checkpoint, workspace / f'outputs/full_pipeline_seed{seed}_v1'
    if arm == 'w0':
        run = workspace / f'outputs/twoscale_w0_seed{seed}'
        return run, run / 'pair/best.pt', workspace / f'outputs/full_pipeline_seed{seed}_v1'
    variant = TWO_BY_TWO[arm]
    run = workspace / f'outputs/twoscale_abl_{variant}_seed{seed}'
    parent = Path(json.loads((run / 'plan.json').read_text())['parent'])
    return run, run / 'pair/best.pt', parent


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workspace', type=Path, default=Path('/media/xd/code1/dyGF'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arms', nargs='+',
                        default=['full', 'only_pair', 'only_binding', 'base'])
    parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(1)
    workspace = args.workspace.resolve()
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    dataset = workspace / 'data/abgaze_reconstructed_v1'
    graph_path = workspace / 'data/raad_abgaze_reconstructed_v1/test.pt'
    test_path = dataset / 'tensors/test.pt'
    features_dir = workspace / 'outputs/binding_interface_test_v1/features'
    rows = torch.load(test_path, weights_only=False)
    graphs = {item['sample_id']: g for item, g in torch.load(graph_path, weights_only=False)}
    metadata = {r['pdb']: r for r in map(json.loads,
                (dataset / 'splits/test.json').read_text().splitlines())}
    ids = [r[0] for r in rows]
    assert set(ids) == set(graphs) == set(metadata)

    locks = []
    for arm in args.arms:
        for seed in args.seeds:
            run, checkpoint, parent = run_paths(workspace, arm, seed)
            if not (run / 'plan.json').is_file():
                raise FileNotFoundError(run)
            status = json.loads((run / 'status.json').read_text())
            if status.get('phase') != 'complete':
                raise ValueError(f'Run is not complete: {run}')
            plan_path = run / 'plan.json'
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if state['plan_sha256'] != sha(plan_path):
                raise ValueError(f'Checkpoint does not belong to its plan: {run}')
            generator = parent / 'generator/best.pt'
            if not generator.is_file():
                generator = parent / 'generator/last.pt'
            locks.append(dict(arm=arm, seed=seed, run=str(run), parent=str(parent),
                              checkpoint=str(checkpoint), epoch=state['epoch'],
                              checkpoint_sha256=sha(checkpoint), plan_sha256=sha(plan_path),
                              generator_path=str(generator), generator_sha256=sha(generator)))
    dump(out / 'lock.json', dict(jobs=locks, test_sha256=sha(test_path),
                                 graph_sha256=sha(graph_path), K=5, temperature=.5,
                                 sampling_seed=2026,
                                 selection='Validation selections locked before test access',
                                 code_sha256={Path(__file__).name: sha(Path(__file__))}))

    results, metric_rows = {}, []
    for lock in locks:
        state = torch.load(lock['checkpoint'], map_location='cpu', weights_only=False)
        model = TwoScalePairRefiner(**state['model_config'], coordinated_bounds=[0.80] * 4,
                                    local_bounds=[0.20] * 4).to(device)
        if lock['arm'] in TWO_BY_TWO:
            model = configure(model, SPEC[TWO_BY_TWO[lock['arm']]]['disable'])
        model.load_state_dict(state['model'], strict=True)
        model.eval()
        gs = torch.load(lock['generator_path'], map_location='cpu', weights_only=False)
        generator = FrameRelationRefiner(**gs['model_config']).to(device)
        generator.load_state_dict(gs['model'], strict=True)
        generator.eval()
        generated = generate(model, generator, rows, graphs, dataset, features_dir, device)
        result = score(rows, generated)
        result.update(arm=lock['arm'], seed=lock['seed'], test_sha256=sha(test_path))
        dump(out / f'metrics_{lock["arm"]}_seed{lock["seed"]}.json', result)
        results[(lock['arm'], lock['seed'])] = result
        for case in result['per_complex']:
            metric_rows.append(dict(sample_id=case['sample_id'],
                                    group=f'cluster_{metadata[case["sample_id"]]["cluster"]}',
                                    arm=lock['arm'], seed=lock['seed'],
                                    **{k: case[k] for k in ('aar', 'rmsd', 'aligned_rmsd',
                                                            'contact_f1', 'cn_mae', 'clash',
                                                            'interface_fape')}))
    with (out / 'scores.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(metric_rows[0]))
        w.writeheader()
        w.writerows(metric_rows)
    summary = {}
    for arm in args.arms:
        keys = [k for k in results if k[0] == arm]
        summary[arm] = {metric: sum(results[k]['summary'][metric] for k in keys) / len(keys)
                        for metric in results[keys[0]]['summary']}
    dump(out / 'arm_means.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
