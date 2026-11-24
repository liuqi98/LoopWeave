"""Retrain the final stage of the original model with an aligned-backbone loss.

What is held fixed
------------------
Architecture (``PersistentLoopPairRefiner``), data split, feature tensors, the
frozen generator/refiner/binding stages, the optimizer and its per-group learning
rates, the per-epoch shuffle seeds, the guards and the validation-selection rule
are all exactly those of the original model's pair stage. The only difference is
the objective:

    loss = <unchanged base objective> + aligned_weight * <Kabsch-aligned backbone loss>

The aligned term is complementary to the existing absolute-coordinate term: it
superimposes the predicted backbone onto the native backbone and penalises the
residual conformation, so global placement error is factored out instead of being
paid for twice.

Why the pair stage
------------------
The original model's final weights are the pair stage's validation-selected
checkpoint, initialised from ``outputs/full_pipeline_seed<seed>_v1/binding/last.pt``.
Retraining that stage from the identical ancestor isolates the loss change; the
three upstream stages stay bit-identical to the original model.

Selection and locking
---------------------
Validation picks the checkpoint (minimum validation aligned_rmsd subject to the
original guards; epoch 0 is eligible, exactly as before). No test tensor is read
here. Scoring is a separate locked step, and the weight is chosen on validation
before any test evaluation.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from persistent_loop_pair_refiner import PersistentLoopPairRefiner
from train.compare_two_stage_binding import FeatureRows
from train.train_abgaze_reconstructed import sha, dump, save, move
from train.train_loop_pair_multiseed import eligible


def relaxed_guard(metric, baseline, cn_tolerance):
    """Same guard as ``eligible`` but with a configurable C-N tolerance."""
    if cn_tolerance is None:
        return eligible(metric, baseline)
    return (metric['rmsd'] <= baseline['rmsd'] + .03
            and metric['contact_f1'] >= baseline['contact_f1'] - .01
            and metric['cn_mae'] <= baseline['cn_mae'] + cn_tolerance
            and metric['clash'] <= baseline['clash'] * 1.1 + 1e-5
            and metric['aar'] >= baseline['aar'] - .015)
from train.train_reconstructed_two_stage import aligned_rmsd
from aligned_shape_objective import objective


def epoch(model, rows, device, optimizer=None, seed=0, status=None, aligned_weight=0.,
          aligned_atom_set='backbone'):
    """Same loop as the original (chunks of 16, grad clip 1., aligned metric added)."""
    model.train(optimizer is not None)
    order = list(range(len(rows)))
    if optimizer is not None:
        random.Random(seed).shuffle(order)
    totals = {}
    seen = 0
    with torch.set_grad_enabled(optimizer is not None):
        for start in range(0, len(order), 16):
            chunk = order[start:start + 16]
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            for index in chunk:
                sid, xcpu, ycpu, _ = rows[index]
                x, y = move(xcpu, device), move(ycpu, device)
                out = model(x)
                loss, metrics = objective(out, x, y, .5, 1., aligned_weight, aligned_atom_set)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Nonfinite loss {sid}')
                if optimizer is not None:
                    (loss / len(chunk)).backward()
                else:
                    mask = y['mask'][:, 1]
                    metrics['aligned_rmsd'] = aligned_rmsd(
                        out['backbone'][mask, 1].detach().cpu(), y['bb'][mask, 1].cpu())
                if not all(math.isfinite(v) for v in metrics.values()):
                    raise RuntimeError(f'Nonfinite metric {sid}')
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.) + value
                seen += 1
            if optimizer is not None:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(norm):
                    raise RuntimeError('Nonfinite gradient')
                optimizer.step()
            if status and (seen % 400 == 0 or seen == len(rows)):
                path, record = status
                dump(path, dict(**record, seen=seen, total=len(rows)))
                print(json.dumps(dict(**record, seen=seen,
                                      loss=totals['loss'] / seen)), flush=True)
    return {k: v / len(rows) for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workspace', type=Path, default=Path('/media/xd/code1/dyGF'))
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--aligned-weight', type=float, required=True)
    parser.add_argument('--aligned-atom-set', choices=('backbone', 'ca'), default='backbone')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--cn-tolerance', type=float, default=None,
                        help='relax the C-N part of the validation guard from 0.01 to this value; '
                             'used to test whether a small connectivity cost is worth a large '
                             'placement gain')
    parser.add_argument('--coordinated-bounds', default=None,
                        help='per-layer coordinated bounds, comma separated, e.g. 0.8,0.6,0.4,0.2')
    parser.add_argument('--local-bounds', default=None,
                        help='per-layer local bounds, comma separated')
    parser.add_argument('--coordinated-bound', type=float, default=None,
                        help='override the bounded coordinated step of the two-scale arm (A)')
    parser.add_argument('--local-bound', type=float, default=None,
                        help='override the bounded local per-atom step of the two-scale arm (A)')
    parser.add_argument('--two-scale', action='store_true',
                        help='split each layer translation into a bounded smooth component and a '
                             'bounded local per-atom component; the only change from baseline')
    parser.add_argument('--anchor-conditioned', action='store_true',
                        help='inject dual-anchor geometric features into every loop-pair layer; '
                             'the only change relative to the baseline arm')
    parser.add_argument('--init-state', action='store_true',
                        help='inject the initial-state displacement channels into every loop-pair '
                             'layer; the only change relative to the baseline arm')
    parser.add_argument('--patience', type=int, default=0,
                        help='Stop after this many epochs without a new guard-passing validation minimum '
                             '(0 disables; see CONVERGENCE_PREREGISTRATION_v1.md)')
    parser.add_argument('--hold-lr', action='store_true',
                        help='Hold the learning rate at the schedule floor (initial_lr * 0.1) for '
                             'continuation epochs instead of re-annealing')
    parser.add_argument('--resume', action='store_true',
                        help='Continue an existing run from pair/last.pt (or best.pt at epoch 0). '
                             'Used because a sandbox restart kills long GPU jobs.')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.aligned_weight < 0:
        raise ValueError('aligned weight must be non-negative')
    workspace = args.workspace.resolve()
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    parent = workspace / f'outputs/full_pipeline_seed{args.seed}_v1'
    pp = json.loads((parent / 'plan.json').read_text())
    assert json.loads((parent / 'status.json').read_text())['phase'] == 'complete'
    for name, digest in pp['code_sha256'].items():
        path = workspace / name
        assert path.is_file() and sha(path) == digest, f'parent code mismatch: {name}'
    features = workspace / 'data/abgaze_binding_features_finalized_v1'
    binding_plan = json.loads((workspace / 'outputs/two_stage_binding60_gpu_v1/plan.json').read_text())
    assert sha(features / 'plan.json') == pp['feature_plan_sha256']

    rows, hashes = {}, {}
    for split in ('train', 'valid'):
        path = parent / f'{split}_cache.pt'
        hashes[split] = sha(path)
        raw = torch.load(path, map_location='cpu', weights_only=False)
        assert [r[0] for r in raw] == pp['ids'][split]
        if args.smoke:
            raw = raw[:2]
        rows[split] = FeatureRows(raw, features, split, binding_plan['feature_sha256'])
    source_path = parent / 'binding/last.pt'
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    assert source['plan_sha256'] == sha(parent / 'plan.json')
    assert source['epoch'] == 60, source['epoch']
    config = source['model_config']
    out = args.output.resolve()
    guard = lambda metric, baseline: relaxed_guard(metric, baseline, args.cn_tolerance)
    resume_state = None
    if args.resume:
        if not (out / 'plan.json').is_file():
            raise ValueError('--resume needs an existing run directory with plan.json')
        candidates = [out / 'pair/last.pt', out / 'pair/best.pt']
        source_state = next((c for c in candidates if c.is_file()), None)
        if source_state is None:
            raise ValueError('--resume found no checkpoint to continue from')
        resume_state = torch.load(source_state, map_location='cpu', weights_only=False)
    else:
        out.mkdir(parents=True, exist_ok=False)
    epochs = 1 if args.smoke else args.epochs
    plan = dict(seed=args.seed, smoke=args.smoke, stage='pair',
                aligned_weight=args.aligned_weight, aligned_atom_set=args.aligned_atom_set,
                aligned_loss='Kabsch-aligned per-atom smooth-L1 on the observed backbone',
                objective='unchanged base geometry + 0.5 FAPE + CE, plus aligned_weight * aligned',
                epochs=dict(pair=epochs), parent=str(parent),
                parent_plan_sha256=sha(parent / 'plan.json'),
                source=str(source_path), source_sha256=sha(source_path), source_epoch=source['epoch'],
                cache_sha256=hashes, model_config=config,
                feature_plan_sha256=sha(features / 'plan.json'),
                upstream_frozen=True,
                selection='Unchanged validation aligned RMSD with original guards; epoch0 eligible; no early stopping',
                test_access=False,
                code_sha256={name: sha(workspace / name) for name in sorted(pp['code_sha256'])},
                new_code={str(p): sha(p) for p in (
                    Path(__file__).resolve(), ROOT / 'aligned_shape_objective.py')},
                caveats=['Only the final stage is retrained; the upstream three stages are inherited '
                         'bit-identically from the same replica, which is what isolates the loss change.',
                         'Aligned and absolute coordinate terms are complementary, not duplicates: the '
                         'aligned term is invariant to a rigid transform of the prediction.',
                         'Validation selects both the weight and the checkpoint; the test set is read '
                         'only after the weight and checkpoints are locked.',
                         'Weight sweep is small by design; large weights can trade interface placement '
                         'for loop shape.'])
    if resume_state is None:
        dump(out / 'plan.json', plan)
    else:
        existing = json.loads((out / 'plan.json').read_text())
        # A copied original pair stage has no ablation fields; missing means the
        # zero-weight, backbone-atom, same-seed configuration.
        if (float(existing.get('aligned_weight', 0.0)) != plan['aligned_weight']
                or existing.get('aligned_atom_set', 'backbone') != plan['aligned_atom_set']
                or (('seed' in existing) and existing['seed'] != plan['seed'])):
            raise ValueError('existing plan disagrees with the requested run')
    plan_sha = sha(out / 'plan.json')

    torch.manual_seed(args.seed)
    if args.two_scale:
        from two_scale_pair_refiner import TwoScaleLoopPairLayer, TwoScalePairRefiner
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
        model = TwoScalePairRefiner(**config, **schedule).to(device)
        shared = PersistentLoopPairRefiner(**config)
        shared.load_binding(source['model'])
        inherited = {k: v for k, v in shared.state_dict().items()
                     if not k.startswith('loop_pair_layers.')}
        model.load_state_dict(inherited, strict=False)
        model.load_binding(source['model'])
    elif args.anchor_conditioned:
        from anchor_pair_refiner import AnchorPairRefiner
        model = AnchorPairRefiner(**config).to(device)
        shared = PersistentLoopPairRefiner(**config)
        shared.load_binding(source['model'])
        inherited = {k: v for k, v in shared.state_dict().items()
                     if not k.startswith('loop_pair_layers.')}
        model.load_state_dict(inherited, strict=False)
        model.load_binding(source['model'])
    elif args.init_state:
        from init_state_pair_refiner import InitStatePairRefiner
        model = InitStatePairRefiner(**config).to(device)
        model.load_binding(source['model'])
        # Pair-layer tensors are new; only the shared upstream tensors are inherited.
        shared = PersistentLoopPairRefiner(**config)
        shared.load_binding(source['model'])
        inherited = {k: v for k, v in shared.state_dict().items()
                     if not k.startswith('loop_pair_layers.')}
        model.load_state_dict(inherited, strict=False)
        model.load_binding(source['model'])
    else:
        model = PersistentLoopPairRefiner(**config).to(device)
        model.load_binding(source['model'])
    directory = out / 'pair'
    if resume_state is None:
        directory.mkdir()
        initial = epoch(model, rows['valid'], device, aligned_weight=args.aligned_weight,
                        aligned_atom_set=args.aligned_atom_set)
        dump(directory / 'initial_validation.json', initial)
        best = initial['aligned_rmsd']
        best_epoch = 0
        start_epoch = 1
        since_best = 0
    else:
        initial = json.loads((directory / 'initial_validation.json').read_text())
        model.load_state_dict(resume_state['model'])
        best, best_epoch = resume_state['best_score'], resume_state['best_epoch']
        start_epoch = int(resume_state['epoch']) + 1
        since_best = max(0, int(resume_state['epoch']) - int(best_epoch))
        print(json.dumps(dict(resumed_from_epoch=resume_state['epoch'],
                              start_epoch=start_epoch, best_epoch=best_epoch)), flush=True)
    groups = [dict(params=[p for n, p in model.named_parameters()
                           if not n.startswith('loop_pair_layers.')], initial_lr=1e-5, lr=1e-5),
              dict(params=[p for n, p in model.named_parameters()
                           if n.startswith('loop_pair_layers.')], initial_lr=1e-4, lr=1e-4)]
    optimizer = torch.optim.Adam(groups)

    def state(epoch_index, validation):
        return dict(model=model.state_dict(), model_config=config, epoch=epoch_index, stage='pair',
                    validation=validation, best_epoch=best_epoch, best_score=best,
                    optimizer=optimizer.state_dict(), plan_sha256=plan_sha,
                    aligned_weight=args.aligned_weight,
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())

    if resume_state is None:
        save(directory / 'best.pt', state(0, initial))
    else:
        optimizer.load_state_dict(resume_state['optimizer'])
        # Provenance of the continuation, kept separate from the parent plan so a
        # copied original stage is never rewritten to look like a new run.
        if (out / 'extension.json').exists():
            previous = json.loads((out / 'extension.json').read_text())
        else:
            previous = dict(extensions=[])
        previous['extensions'].append(dict(
            aligned_weight=args.aligned_weight, aligned_atom_set=args.aligned_atom_set,
            hold_lr=args.hold_lr, patience=args.patience, max_epochs=epochs,
            start_epoch=start_epoch, resumed_from_epoch=resume_state['epoch'],
            resumed_checkpoint=str(source_state),
            parent_plan_sha256=plan_sha,
            note='continuation of an existing final stage; plan.json is untouched'))
        (out / 'extension.json').write_text(json.dumps(previous, indent=2) + '\n')
    stop_reason = 'max_epochs'
    converged = False
    last_epoch = start_epoch - 1
    for e in range(start_epoch, epochs + 1):
        last_epoch = e
        tick = time.monotonic()
        torch.manual_seed(args.seed + e)
        for group in optimizer.param_groups:
            if args.hold_lr and e > 30:
                group['lr'] = group['initial_lr'] * .1
            else:
                group['lr'] = group['initial_lr'] * .1 ** ((e - 1) / max(epochs - 1, 1))
        status = dict(phase='training', stage='pair', epoch=e, epochs=epochs, test_access=False)
        dump(out / 'status.json', status)
        train_metrics = epoch(model, rows['train'], device, optimizer, args.seed + e,
                              (out / 'status.json', status), args.aligned_weight,
                              args.aligned_atom_set)
        val = epoch(model, rows['valid'], device, aligned_weight=args.aligned_weight,
                    aligned_atom_set=args.aligned_atom_set)
        allowed = guard(val, initial)
        improved = allowed and val['aligned_rmsd'] < best
        if improved:
            best, best_epoch = val['aligned_rmsd'], e
            since_best = 0
        else:
            since_best += 1
        checkpoint = state(e, val)
        save(directory / 'last.pt', checkpoint)
        if improved:
            save(directory / 'best.pt', checkpoint)
        record = dict(epoch=e, train=train_metrics, validation=val, guard_passed=allowed,
                      best_epoch=best_epoch, seconds=time.monotonic() - tick,
                      aligned_weight=args.aligned_weight, since_best=since_best)
        dump(directory / f'epoch_{e:03d}.json', record)
        print(json.dumps(dict(epoch=e, val_aligned_rmsd=round(val['aligned_rmsd'], 4),
                              val_rmsd=round(val['rmsd'], 4), guard=allowed,
                              best_epoch=best_epoch, since_best=since_best)), flush=True)
        if args.patience and since_best >= args.patience:
            stop_reason = 'plateau'
            converged = best_epoch <= e - 3
            break
    dump(out / 'status.json', dict(phase='smoke_complete' if args.smoke else 'complete',
                                   stage='pair', selected_epoch=best_epoch,
                                   aligned_weight=args.aligned_weight, test_access=False,
                                   stop_reason=stop_reason, final_epoch=last_epoch,
                                   converged=converged, patience=args.patience,
                                   hold_lr=args.hold_lr, max_epochs=epochs))
    print(json.dumps(dict(seed=args.seed, aligned_weight=args.aligned_weight,
                          selected_epoch=best_epoch, output=str(out))))


if __name__ == '__main__':
    main()
