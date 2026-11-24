"""Score trained ablation runs (v3) on the 60-complex development benchmark.

Generalizes ``eval/test_matched_ablations_v2.py`` to every variant produced by
``train/train_matched_ablation_v3.py`` and to a configurable workspace, so the
missing double-removal cell can be scored by the same locked procedure:

  * all checkpoints are hashed and committed before any test tensor is read;
  * generation uses the parent replica's own frozen generator;
  * scoring reuses the unchanged ``eval.test_loop_pair_replicates_v1.score``;
  * a ``scores.csv`` with dependence groups is emitted so the audited
    ``research_audit factorial`` command can compute the interaction.

Use ``--variants no_pair no_binding no_pair_no_binding`` (with ``full`` always
included) to run the 2x2. Never select a checkpoint on test results.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train.train_abgaze_reconstructed import sha, dump, move
from train.train_matched_ablation_v3 import SPEC, build_pair_model, configure
from frame_relation_refiner import FrameRelationGraph, FrameRelationRefiner
from interface_ablation_refiner import AblatedPersistentLoopPairRefiner
from eval.test_loop_pair_replicates_v1 import score

ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
# 2x2 cells keep the arm names the factorial command requires. Interface
# ablations are one-factor levels and are scored in a separate output directory
# so that their label never enters a factorial CSV.
TWO_BY_TWO = {'base': 'no_pair_no_binding', 'only_pair': 'no_binding',
              'only_binding': 'no_pair', 'full': None}
DEFAULT_VARIANTS = ('full', 'only_pair', 'only_binding', 'base')
SCORE_KEYS = ('aar', 'first_sample_aar', 'mean_sample_aar', 'oracle_aar', 'rmsd',
              'aligned_rmsd', 'contact_f1', 'cn_mae', 'clash', 'interface_fape')


def read(path):
    return json.loads(Path(path).read_text())


def generate(model, generator, rows, graphs, dataset, features_dir, device):
    generated = []
    rng = torch.Generator().manual_seed(2026)
    with torch.no_grad():
        for sid, x, y in rows:
            directory = features_dir / sid
            path = directory / 'features.pt'
            digest = sha(path)
            assert digest == read(directory / 'ready.json')['sha256'], sid
            item = torch.load(path, map_location='cpu', weights_only=False)
            meta = item['metadata']
            assert meta['antigen_only'] and not meta['native_h3_used'], sid
            assert meta['pdb_sha256'] == sha(dataset / 'pdb' / f'{sid}.pdb'), sid
            assert not set(x) & set(item['features']), sid
            inputs = move({**x, **item['features']}, device)
            raw = generator(FrameRelationGraph(**graphs[sid]).to(device))
            inputs.update(base_bb=raw['h3_backbone'],
                          base_logits=torch.zeros_like(inputs['base_logits']))
            pred = model(inputs)
            bb, logits = pred['backbone'].cpu(), pred['sequence_logits'].cpu()
            if not (torch.isfinite(bb).all() and torch.isfinite(logits).all()):
                raise ValueError(f'Nonfinite output: {sid}')
            samples = torch.multinomial((logits / .5).softmax(-1), 5,
                                        replacement=True, generator=rng).T
            generated.append(dict(sample_id=sid, backbone=bb.tolist(), sequence_logits=logits.tolist(),
                                  greedy=''.join(ALPHABET[i] for i in logits.argmax(-1).tolist()),
                                  candidates=[''.join(ALPHABET[i] for i in s.tolist())
                                              for s in samples]))
    return generated


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workspace', type=Path, default=Path('/media/xd/code1/dyGF'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variants', nargs='+', default=None,
                        help='Labels to score: the 2x2 cells base/only_pair/only_binding/full, '
                             'or any train_matched_ablation_v3 variant name. Runs are expected at '
                             'outputs/matched_ablation_v3_<variant>_seed<seed>, with the legacy '
                             'outputs/matched_ablation_v2_<variant>_seed<seed> accepted as fallback. '
                             'Keep 2x2 cells and interface arms in separate output directories.')
    parser.add_argument('--seeds', type=int, nargs='+', default=[2026, 2027, 2028])
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(1)
    workspace = args.workspace.resolve()
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    arms = list(args.variants or DEFAULT_VARIANTS)
    resolved = {}
    for label in arms:
        if label in TWO_BY_TWO:
            resolved[label] = TWO_BY_TWO[label]
        elif label in SPEC:
            resolved[label] = label
        else:
            raise ValueError(f'Unknown arm or variant: {label}')
    cells = {'base', 'only_pair', 'only_binding'}
    two_by_two = cells | {'full'}
    if cells & set(resolved) and not two_by_two.issubset(resolved):
        raise ValueError('The 2x2 arms must be scored together: '
                         'request base, only_pair, only_binding and full.')
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
    ids = [r[0] for r in rows]
    if set(ids) != set(graphs) or set(ids) != set(metadata):
        raise ValueError('Test rows, graphs and metadata do not agree')

    locks = []
    for arm in arms:
        variant = resolved[arm]
        for seed in args.seeds:
            if variant is None:
                run = workspace / f'outputs/full_pipeline_seed{seed}_v1'
                checkpoint = run / 'pair/best.pt'
                parent = run
            else:
                run = workspace / f'outputs/matched_ablation_v3_{variant}_seed{seed}'
                if not (run / 'plan.json').is_file():
                    run = workspace / f'outputs/matched_ablation_v2_{variant}_seed{seed}'
                checkpoint = run / 'pair/best.pt'
                parent = Path(read(run / 'plan.json')['parent'])
            plan_path = run / 'plan.json'
            status = read(run / 'status.json')
            if status.get('phase') != 'complete':
                raise ValueError(f'Run is not complete: {run}')
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if state['plan_sha256'] != sha(plan_path):
                raise ValueError(f'Checkpoint does not belong to its plan: {run}')
            locks.append(dict(arm=arm, variant=variant, seed=seed, run=str(run),
                              parent=str(parent), epoch=state['epoch'],
                              checkpoint_sha256=sha(checkpoint), plan_sha256=sha(plan_path),
                              generator_path=str(parent / 'generator/best.pt'),
                              generator_sha256=sha(parent / 'generator/best.pt')))
    dump(out / 'lock.json', dict(jobs=locks, test_sha256=sha(test_path), graph_sha256=sha(graph_path),
                                 K=5, temperature=.5, sampling_seed=2026,
                                 selection='All validation selections locked before test access',
                                 code_sha256={Path(__file__).name: sha(Path(__file__))}))

    results, metric_rows = {}, []
    for lock in locks:
        key = (lock['arm'], lock['seed'])
        state = torch.load(Path(lock['run']) / 'pair/best.pt', map_location='cpu', weights_only=False)
        if lock['variant'] is None:
            model = AblatedPersistentLoopPairRefiner(**state['model_config']).to(device)
        else:
            model = build_pair_model(lock['variant'], state['model_config'], device)
            disable = set(SPEC[lock['variant']]['disable'])
            if 'binding' in disable:
                configure(model, {'binding'})
            if 'pair' in disable:
                configure(model, {'pair'})
        model.load_state_dict(state['model'], strict=True)
        model.eval()
        gs = torch.load(lock['generator_path'], map_location='cpu', weights_only=False)
        generator = FrameRelationRefiner(**gs['model_config']).to(device)
        generator.load_state_dict(gs['model'], strict=True)
        generator.eval()
        generated = generate(model, generator, rows, graphs, dataset, features_dir, device)
        result = score(rows, generated)
        result['test_sha256'] = sha(test_path)
        result['arm'] = lock['arm']
        result['seed'] = lock['seed']
        dump(out / f'metrics_{lock["arm"]}_seed{lock["seed"]}.json', result)
        results[key] = result
        for case in result['per_complex']:
            metric_rows.append(dict(sample_id=case['sample_id'],
                                    group=f'cluster_{metadata[case["sample_id"]]["cluster"]}',
                                    arm=lock['arm'], seed=lock['seed'],
                                    length=case['length'],
                                    **{k: case[k] for k in SCORE_KEYS}))
        print(json.dumps(dict(arm=lock['arm'], seed=lock['seed'], summary=result['summary'])), flush=True)
        del model, generator, state, gs
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    with (out / 'scores.csv').open('x', newline='') as handle:
        fields = ['sample_id', 'group', 'arm', 'seed', 'length'] + list(SCORE_KEYS)
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(metric_rows)
    summary = {}
    for arm in arms:
        runs = [results[(arm, s)] for s in args.seeds if (arm, s) in results]
        summary[arm] = {k: dict(mean=sum(r['summary'][k] for r in runs) / len(runs),
                                per_seed=[r['summary'][k] for r in runs]) for k in runs[0]['summary']}
    dump(out / 'comparison.json', dict(n=len(ids), seeds=list(args.seeds), arms=arms,
                                       per_arm_summary=summary,
                                       next_step=('python -m research_audit factorial --input scores.csv '
                                                  '--metric aligned_rmsd:min --metric contact_f1:max '
                                                  '--metric cn_mae:min --metric interface_fape:min'
                                                  if two_by_two.issubset(resolved) else
                                                  'Interface arms: compare with research_audit paired '
                                                  'against the full label in this same scores.csv.'),
                                       limitations=[
                                           'Repeatedly inspected development benchmark; not confirmatory.',
                                           'H3 clusters are not antigen families; intervals are conditional on fitted seeds.',
                                           'Matched epoch budgets do not imply matched FLOPs or parameter counts.',
                                           'An interaction is only estimable when all four cells are present.']))
    dump(out / 'status.json', dict(phase='complete', scored=len(results), total=len(locks)))
    print(json.dumps(dict(scored=len(results), output=str(out))))


if __name__ == '__main__':
    main()
