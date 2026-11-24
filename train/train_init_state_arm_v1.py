"""Matched continuation training for the initial-state-retention variant.

Protocol is copied from ``train_mechanism_arms_v1.py`` so the comparison is like for like: the same
common initialisation from the upstream binding checkpoint at epoch 60, the same 30-epoch budget,
the same guards, learning-rate schedule and validation-selection rule, and the same cached initial
backbones and feature tensors. Only the loop-pair layer input changes (initial-state channels).
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

PACKAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE / 'code'))
sys.path.insert(0, '/media/xd/code1/dyGF')
from init_state_pair_refiner import InitStatePairRefiner
from persistent_loop_pair_refiner import PersistentLoopPairRefiner
from train.compare_two_stage_binding import FeatureRows
from train.train_abgaze_reconstructed import dump, move, save, sha
from train.train_mechanism_arms_v1 import eligible, objective, run_epoch

WORKSPACE = Path('/media/xd/code1/dyGF')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    out = args.output.resolve()
    epochs = 1 if args.smoke else args.epochs

    parent = WORKSPACE / f'outputs/full_pipeline_seed{args.seed}_v1'
    source_path = parent / 'binding/last.pt'
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    if source['epoch'] != 60:
        raise ValueError(f"unexpected binding epoch {source['epoch']}")
    config = dict(source['model_config'])

    features = WORKSPACE / 'data/abgaze_binding_features_finalized_v1'
    plan_path = features / 'plan.json'
    raw = {split: torch.load(WORKSPACE / f'data/abgaze_reconstructed_v1/tensors/{split}.pt',
                             map_location='cpu', weights_only=False)
           for split in ('train', 'valid')}
    rows = {split: FeatureRows(raw[split], features, split, json.loads(plan_path.read_text()))
            for split in ('train', 'valid')}
    if args.smoke:
        rows = {split: rows[split][:8] for split in rows}

    resume_state = None
    if args.resume:
        resume_state = torch.load(out / 'pair/last.pt', map_location='cpu', weights_only=False)
        if resume_state.get('arm') != 'init_state' or resume_state.get('seed') != args.seed:
            raise ValueError('existing run belongs to a different arm or seed')

    if not args.resume:
        out.mkdir(parents=True, exist_ok=False)
        torch.manual_seed(args.seed)
        reference = PersistentLoopPairRefiner(**config)
        reference.load_binding(source['model'])
        shared_state = {k: v.detach().cpu().clone() for k, v in reference.state_dict().items()}

        torch.manual_seed(args.seed)
        model = InitStatePairRefiner(**config).to(device)
        result = model.load_state_dict(shared_state, strict=False)
        unexpected = [k for k in result.unexpected_keys if not k.startswith('loop_pair_layers.')]
        missing = [k for k in result.missing_keys if not k.startswith('loop_pair_layers.')]
        if unexpected or missing:
            raise ValueError(f'shared tensors do not match: {unexpected[:3]} {missing[:3]}')
        model.load_binding(source['model'])
        trainable_names = [n for n, _ in model.named_parameters() if n.startswith('loop_pair_layers.')]
        retained = [p for n, p in model.named_parameters()
                    if n.startswith('loop_pair_layers.') is False]
        pair = [p for n, p in model.named_parameters() if n.startswith('loop_pair_layers.')]
        groups = [dict(params=retained, initial_lr=1e-5, lr=1e-5),
                  dict(params=pair, initial_lr=1e-4, lr=1e-4)]
        optimizer = torch.optim.Adam(groups)
        directory = out / 'pair'
        directory.mkdir()
        initial = run_epoch(model, rows['valid'], device)
        dump(directory / 'initial_validation.json', initial)
        best, best_epoch = initial['aligned_rmsd'], 0
        parameters = dict(total=sum(p.numel() for p in model.parameters()),
                          pair_layers=sum(p.numel() for p in pair),
                          trainable=len(trainable_names))
        dump(out / 'plan.json', dict(arm='init_state', seed=args.seed, epochs=epochs,
                                     smoke=args.smoke, model_config=config,
                                     route='common initialisation from upstream binding epoch 60',
                                     change='initial-state displacement channels in every '
                                            'loop-pair layer; no pull-back penalty',
                                     parent_plan_sha256=sha(parent / 'plan.json'),
                                     feature_plan_sha256=sha(plan_path),
                                     parameters=parameters))
    else:
        optimizer = torch.optim.Adam(
            [dict(params=[p for n, p in model.named_parameters()
                          if not n.startswith('loop_pair_layers.')], initial_lr=1e-5),
             dict(params=[p for n, p in model.named_parameters()
                          if n.startswith('loop_pair_layers.')], initial_lr=1e-4)])
        model = InitStatePairRefiner(**config)
    plan_sha = sha(out / 'plan.json')
    if resume_state is not None:
        model.load_state_dict(resume_state['model'])
        model.to(device)
        optimizer.load_state_dict(resume_state['optimizer'])
        best, best_epoch = resume_state['best_score'], resume_state['best_epoch']
        start_epoch = int(resume_state['epoch']) + 1
    else:
        start_epoch = 1

    directory = out / 'pair'
    def state(epoch_index, validation):
        return dict(model=model.state_dict(), model_config=config, epoch=epoch_index,
                    stage='pair', arm='init_state', seed=args.seed, validation=validation,
                    best_epoch=best_epoch, best_score=best, optimizer=optimizer.state_dict(),
                    plan_sha256=plan_sha, torch_rng=torch.get_rng_state())

    if resume_state is None:
        save(directory / 'best.pt', state(0, initial))
    for e in range(start_epoch, epochs + 1):
        tick = time.monotonic()
        torch.manual_seed(args.seed + e)
        for group in optimizer.param_groups:
            group['lr'] = group['initial_lr'] * .1 ** ((e - 1) / max(epochs - 1, 1))
        dump(out / 'status.json', dict(phase='training', stage='pair', arm='init_state',
                                       seed=args.seed, epoch=e, epochs=epochs, test_access=False))
        train_metrics = run_epoch(model, rows['train'], device, optimizer, args.seed + e)
        validation = run_epoch(model, rows['valid'], device)
        allowed = eligible(validation, initial)
        improved = allowed and validation['aligned_rmsd'] < best
        if improved:
            best, best_epoch = validation['aligned_rmsd'], e
        save(directory / 'last.pt', state(e, validation))
        if improved:
            save(directory / 'best.pt', state(e, validation))
        dump(directory / f'epoch_{e:03d}.json',
             dict(epoch=e, arm='init_state', seed=args.seed, train=train_metrics,
                  validation=validation, guard_passed=allowed, best_epoch=best_epoch,
                  seconds=time.monotonic() - tick))
        print(json.dumps(dict(arm='init_state', seed=args.seed, epoch=e,
                              val_aligned_rmsd=round(validation['aligned_rmsd'], 4),
                              guard=allowed, best_epoch=best_epoch,
                              seconds=round(time.monotonic() - tick, 1))), flush=True)
    dump(out / 'status.json', dict(phase='smoke_complete' if args.smoke else 'complete',
                                   arm='init_state', seed=args.seed, best_epoch=best_epoch))


if __name__ == '__main__':
    main()
