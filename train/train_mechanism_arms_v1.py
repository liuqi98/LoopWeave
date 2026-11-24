"""Matched retraining of the four mechanism-control arms.

Implements the protocol demanded by
``manuscript/nature_communications_v1/mechanism_ablation_implementation.md``:

* every arm starts from one **common original-model initialization**: a
  ``PersistentLoopPairRefiner`` loaded from the same upstream binding checkpoint
  (``outputs/full_pipeline_seed<seed>_v1/binding/last.pt``, epoch 60);
* shared tensors are copied with ``load_shared_initialization``; for the MLP arm only
  the replacement ``pair_mlp`` tensors are newly initialised;
* data split, feature tensors, objective, optimiser groups, learning rates, per-epoch
  shuffle seeds, guards, validation-selection rule and training budget are exactly
  those of the original pair stage;
* checkpoints are selected on validation only; the test split is never read here;
* the run records seed, input and checkpoint hashes, trainable parameter counts and
  the optimiser groups, plus the arm definition.

Arms (from ``mechanism_pair_ablation.py``):

    full          original pair computation
    reset_z       zero the incoming pair state before adding the current descriptors
    pair_mlp      third-residue triangle aggregation replaced by a pair-local MLP
    feature_only  direct pair rigid-coordinate readout disabled
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
sys.path.insert(0, '/media/xd/code1/dyGF')
from persistent_loop_pair_refiner import PersistentLoopPairRefiner
from mechanism_pair_ablation import MechanismPairRefiner
from train.compare_two_stage_binding import FeatureRows
from train.train_abgaze_reconstructed import sha, dump, save, move
from train.train_loop_pair_multiseed import eligible
from train.train_reconstructed_two_stage import aligned_rmsd
from aligned_shape_objective import objective

ARMS = ('full', 'reset_z', 'pair_mlp', 'feature_only', 'no_chem')

# Channels that carry explicit chemical description in the binding features. Geometry,
# exposure, atom radius, side-chain and added-atom flags are retained. Zeroing keeps the
# channel count and therefore the parameter count identical to the full arm.
CHEMICAL_CHANNELS = dict(binding_surface_scalars=(4, 5, 6, 7, 8),
                         binding_surface_vectors=(1, 2, 3),
                         binding_atom_scalars=(1, 2, 3, 4, 7, 8),
                         binding_atom_vectors=(0, 1))


def zero_chemical_channels(module, arguments):
    batch = arguments[0]
    for key, channels in CHEMICAL_CHANNELS.items():
        if key not in batch:
            continue
        value = batch[key]
        for channel in channels:
            value[:, channel] = 0
    return arguments
# Code lives either in the source workspace or in the snapshot's code directory.
CODE_PATHS = ('mechanism_pair_ablation.py', 'persistent_loop_pair_refiner.py',
              'aligned_shape_objective.py', 'v21_atomic_refiner.py',
              'orientation_interface_refiner.py', 'binding_interface_refiner.py')


def locate(workspace, name):
    for base in (workspace, ROOT):
        path = base / name
        if path.is_file():
            return path
    raise FileNotFoundError(f'{name} not found in the workspace or the snapshot code directory')


def trainable(model):
    return [(name, parameter) for name, parameter in model.named_parameters()
            if parameter.requires_grad]


def run_epoch(model, rows, device, optimizer=None, seed=0, status=None):
    """Identical loop to the original pair stage (chunks of 16, grad clip 1.)."""
    model.train(optimizer is not None)
    order = list(range(len(rows)))
    if optimizer is not None:
        random.Random(seed).shuffle(order)
    totals, seen = {}, 0
    with torch.set_grad_enabled(optimizer is not None):
        for start in range(0, len(order), 16):
            chunk = order[start:start + 16]
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            for index in chunk:
                sid, xcpu, ycpu, _ = rows[index]
                x, y = move(xcpu, device), move(ycpu, device)
                out = model(x)
                loss, metrics = objective(out, x, y, .5, 1., 0., 'backbone')
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
                norm = torch.nn.utils.clip_grad_norm_(
                    [p for _, p in trainable(model)], 1.)
                if not torch.isfinite(norm):
                    raise RuntimeError('Nonfinite gradient')
                optimizer.step()
            if status and (seen % 400 == 0 or seen == len(rows)):
                path, record = status
                dump(path, dict(**record, seen=seen, total=len(rows)))
    return {k: v / len(rows) for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workspace', type=Path, default=Path('/media/xd/code1/dyGF'))
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true',
                        help='Continue an interrupted run from pair/last.pt')
    args = parser.parse_args()
    torch.set_num_threads(1)
    workspace = args.workspace.resolve()
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    parent = workspace / f'outputs/full_pipeline_seed{args.seed}_v1'
    parent_plan = json.loads((parent / 'plan.json').read_text())
    if json.loads((parent / 'status.json').read_text())['phase'] != 'complete':
        raise ValueError('parent pipeline is not complete')
    for name, digest in parent_plan['code_sha256'].items():
        path = workspace / name
        if not (path.is_file() and sha(path) == digest):
            raise ValueError(f'parent code mismatch: {name}')
    features = workspace / 'data/abgaze_binding_features_finalized_v1'
    binding_plan = json.loads((workspace / 'outputs/two_stage_binding60_gpu_v1/plan.json').read_text())
    if sha(features / 'plan.json') != parent_plan['feature_plan_sha256']:
        raise ValueError('feature plan mismatch')

    rows, hashes = {}, {}
    for split in ('train', 'valid'):
        path = parent / f'{split}_cache.pt'
        hashes[split] = sha(path)
        raw = torch.load(path, map_location='cpu', weights_only=False)
        if [r[0] for r in raw] != parent_plan['ids'][split]:
            raise ValueError('cache id order mismatch')
        if args.smoke:
            raw = raw[:2]
        rows[split] = FeatureRows(raw, features, split, binding_plan['feature_sha256'])

    source_path = parent / 'binding/last.pt'
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    if source['plan_sha256'] != sha(parent / 'plan.json'):
        raise ValueError('binding checkpoint does not belong to this parent plan')
    if source['epoch'] != 60:
        raise ValueError(f"unexpected binding epoch {source['epoch']}")
    config = source['model_config']

    out = args.output.resolve()
    resume_state = None
    if args.resume:
        if not (out / 'plan.json').is_file():
            raise ValueError('--resume needs an existing run directory')
        existing = json.loads((out / 'plan.json').read_text())
        if existing['arm'] != args.arm or existing['seed'] != args.seed:
            raise ValueError('existing plan belongs to a different arm or seed')
        candidates = [out / 'pair/last.pt', out / 'pair/best.pt']
        chosen = next((c for c in candidates if c.is_file()), None)
        if chosen is None:
            raise ValueError('nothing to resume from')
        resume_state = torch.load(chosen, map_location='cpu', weights_only=False)
        out_plan = existing
    else:
        out.mkdir(parents=True, exist_ok=False)
        out_plan = None
    epochs = 1 if args.smoke else args.epochs

    # One common original-model initialization from the same upstream checkpoint.
    torch.manual_seed(args.seed)
    reference = PersistentLoopPairRefiner(**config).to(device)
    reference.load_binding(source['model'])
    # Guard against two-sided hashing mistakes: each arm must be built from this state.
    for name, parameter in reference.named_parameters():
        if name.endswith('native_head.0.weight') or name.endswith('native_head.0.bias'):
            pass                                  # zero-initialised by design
    shared_state = {k: v.detach().cpu().clone() for k, v in reference.state_dict().items()}

    torch.manual_seed(args.seed)
    if args.arm in ('full', 'no_chem'):
        model = PersistentLoopPairRefiner(**config).to(device)
        model.load_state_dict(shared_state, strict=True)
        if args.arm == 'no_chem':
            model.register_forward_pre_hook(zero_chemical_channels)
    else:
        model = MechanismPairRefiner(mechanism=args.arm, **config).to(device)
        model.load_shared_initialization(shared_state)
    model.load_binding(source['model'])           # non-pair tensors come from the parent
    if args.arm == 'pair_mlp':
        # load_binding would have touched ignored keys; re-assert the common part.
        model.load_shared_initialization(shared_state)

    trainable_names = [name for name, _ in trainable(model)]
    retained = [p for n, p in trainable(model) if not n.startswith('loop_pair_layers.')]
    pair = [p for n, p in trainable(model) if n.startswith('loop_pair_layers.')]
    groups = [dict(params=retained, initial_lr=1e-5, lr=1e-5),
              dict(params=pair, initial_lr=1e-4, lr=1e-4)]
    if sum(parameter.numel() for parameter in retained) + \
            sum(parameter.numel() for parameter in pair) == 0:
        raise ValueError('no trainable parameters')
    optimizer = torch.optim.Adam(groups)

    directory = out / 'pair'
    if resume_state is None:
        directory.mkdir()
        initial = run_epoch(model, rows['valid'], device)
        dump(directory / 'initial_validation.json', initial)
        best, best_epoch = initial['aligned_rmsd'], 0
        start_epoch = 1
    else:
        initial = json.loads((directory / 'initial_validation.json').read_text())
        model.load_state_dict(resume_state['model'])
        best, best_epoch = resume_state['best_score'], resume_state['best_epoch']
        start_epoch = int(resume_state['epoch']) + 1
        print(json.dumps(dict(resumed_from=resume_state['epoch'], start_epoch=start_epoch,
                              best_epoch=best_epoch)), flush=True)

    plan = dict(
        stage='pair', arm=args.arm, seed=args.seed, smoke=args.smoke,
        mechanism=dict(arms=list(ARMS), selected=args.arm,
                       description=dict(
                           full='original pair computation',
                           no_chem='explicit chemical binding channels zeroed, geometry retained',
                           reset_z='zero incoming pair state before adding current descriptors',
                           pair_mlp='third-residue triangle aggregation replaced by pair-local MLP',
                           feature_only='direct pair rigid-coordinate readout disabled')),
        parent=str(parent), parent_plan_sha256=sha(parent / 'plan.json'),
        source=str(source_path), source_sha256=sha(source_path), source_epoch=source['epoch'],
        cache_sha256=hashes, feature_plan_sha256=sha(features / 'plan.json'),
        model_config=config, epochs=dict(pair=epochs),
        objective='unchanged base geometry + 0.5 FAPE + CE (aligned weight 0)',
        validation_selection='minimum validation aligned_rmsd subject to the original guards; '
                             'epoch 0 eligible',
        optimizer=dict(type='Adam',
                       groups=[dict(purpose='retained', initial_lr=1e-5,
                                    parameters=sum(p.numel() for p in retained)),
                               dict(purpose='loop_pair_layers', initial_lr=1e-4,
                                    parameters=sum(p.numel() for p in pair))],
                       schedule='initial_lr * 0.1 ** ((epoch-1)/(epochs-1))',
                       accumulation=16, grad_clip=1.0),
        parameters=dict(total=sum(p.numel() for p in model.parameters()),
                        trainable=sum(p.numel() for _, p in trainable(model)),
                        non_trainable=sum(p.numel() for p in model.parameters()) -
                                      sum(p.numel() for _, p in trainable(model))),
        trainable_parameter_names=trainable_names,
        shared_initialization='PersistentLoopPairRefiner built from the same binding checkpoint, '
                              'then copied via load_shared_initialization',
        test_access=False,
        code_sha256={name: sha(locate(workspace, name)) for name in CODE_PATHS},
        new_code={str(Path(__file__).resolve()): sha(Path(__file__).resolve())})
    if out_plan is None:
        dump(out / 'plan.json', plan)
    plan_sha = sha(out / 'plan.json')

    def state(epoch_index, validation):
        return dict(model=model.state_dict(), model_config=config, epoch=epoch_index,
                    stage='pair', arm=args.arm, validation=validation,
                    best_epoch=best_epoch, best_score=best,
                    optimizer=optimizer.state_dict(), plan_sha256=plan_sha,
                    torch_rng=torch.get_rng_state())

    if resume_state is None:
        save(directory / 'best.pt', state(0, initial))
    else:
        optimizer.load_state_dict(resume_state['optimizer'])
    for e in range(start_epoch, epochs + 1):
        tick = time.monotonic()
        torch.manual_seed(args.seed + e)
        for group in optimizer.param_groups:
            group['lr'] = group['initial_lr'] * .1 ** ((e - 1) / max(epochs - 1, 1))
        status = dict(phase='training', stage='pair', arm=args.arm, seed=args.seed,
                      epoch=e, epochs=epochs, test_access=False)
        dump(out / 'status.json', status)
        train_metrics = run_epoch(model, rows['train'], device, optimizer,
                                  args.seed + e, (out / 'status.json', status))
        validation = run_epoch(model, rows['valid'], device)
        allowed = eligible(validation, initial)
        improved = allowed and validation['aligned_rmsd'] < best
        if improved:
            best, best_epoch = validation['aligned_rmsd'], e
        checkpoint = state(e, validation)
        save(directory / 'last.pt', checkpoint)
        if improved:
            save(directory / 'best.pt', checkpoint)
        dump(directory / f'epoch_{e:03d}.json',
             dict(epoch=e, arm=args.arm, seed=args.seed, train=train_metrics,
                  validation=validation, guard_passed=allowed, best_epoch=best_epoch,
                  seconds=time.monotonic() - tick))
        print(json.dumps(dict(arm=args.arm, seed=args.seed, epoch=e,
                              val_aligned_rmsd=round(validation['aligned_rmsd'], 4),
                              guard=allowed, best_epoch=best_epoch,
                              seconds=round(time.monotonic() - tick, 1))), flush=True)

    dump(out / 'status.json', dict(phase='smoke_complete' if args.smoke else 'complete',
                                   stage='pair', arm=args.arm, seed=args.seed,
                                   selected_epoch=best_epoch, test_access=False))
    print(json.dumps(dict(arm=args.arm, seed=args.seed, selected_epoch=best_epoch,
                          total=plan['parameters']['total'],
                          trainable=plan['parameters']['trainable'])))


if __name__ == '__main__':
    main()
