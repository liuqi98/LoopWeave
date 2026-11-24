"""Matched retraining of the three sequence-prior conditions.

Conditions (from the pilot protocol):

    baseline            uniform input prior, zero residual decode
    probability_init    input prior B = alpha * S^0 / T, zero residual decode
    full                input prior B, and the residual decode also starts from B

All three use the **same stage-1 geometry** ``X^0``: the generator is frozen and its
backbone is used unchanged, so only the sequence-prior wiring differs. The refiner is
retrained from the same upstream binding checkpoint for every condition, because the
input distribution changes and inference-time switching alone would not be matched.

The prior tensor is ``S^0`` from ``train_stage1_sequence_head_v1.py`` scaled by
``alpha / T``; with ``--alpha 0`` this reproduces the current zero-logits baseline.
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
from sequence_prior_refiner import SequencePriorRefiner
from train.compare_two_stage_binding import FeatureRows
from train.train_abgaze_reconstructed import sha, dump, save, move
from train.train_loop_pair_multiseed import eligible
from train.train_reconstructed_two_stage import aligned_rmsd
from aligned_shape_objective import objective

WORKSPACE = Path('/media/xd/code1/dyGF')
CONDITIONS = ('baseline', 'probability_init', 'full')


def trainable(model):
    return [(name, parameter) for name, parameter in model.named_parameters()
            if parameter.requires_grad]


def load_prior(path, split, ids, alpha, temperature, device):
    """Scaled stage-1 logits B = alpha * S^0 / T for one split, in split order."""
    document = torch.load(path, map_location='cpu', weights_only=False)
    table = document['logits']
    missing = [sid for sid in ids if sid not in table]
    if missing:
        raise ValueError(f'stage-1 prior missing {len(missing)} ids, e.g. {missing[:3]}')
    return {sid: (table[sid]['logits'].to(device) * (alpha / temperature))
            for sid in ids}


def run_epoch(model, rows, device, priors, output_prior, optimizer=None, seed=0, status=None,
              condition='baseline'):
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
                inputs = dict(x)
                if condition != 'baseline':
                    prior = priors[sid].to(device)
                    inputs['base_logits'] = prior
                    inputs['base_logits_output'] = (prior if output_prior
                                                    else torch.zeros_like(prior))
                out = model(inputs)
                loss, metrics = objective(out, inputs, y, .5, 1., 0., 'backbone')
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
                norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable(model)], 1.)
                if not math.isfinite(float(norm)):
                    raise RuntimeError('Nonfinite gradient')
                optimizer.step()
            if status and (seen % 400 == 0 or seen == len(rows)):
                path, record = status
                dump(path, dict(**record, seen=seen, total=len(rows)))
    return {k: v / len(rows) for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workspace', type=Path, default=WORKSPACE)
    parser.add_argument('--prior-root', type=Path, required=True,
                        help='directory holding stage1_head_seed<seed>.pt')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--condition', choices=CONDITIONS, required=True)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    workspace = args.workspace.resolve()

    parent = workspace / f'outputs/full_pipeline_seed{args.seed}_v1'
    plan = json.loads((parent / 'plan.json').read_text())
    if json.loads((parent / 'status.json').read_text())['phase'] != 'complete':
        raise ValueError('parent pipeline is not complete')
    for name, digest in plan['code_sha256'].items():
        path = workspace / name
        if not (path.is_file() and sha(path) == digest):
            raise ValueError(f'parent code mismatch: {name}')
    features = workspace / 'data/abgaze_binding_features_finalized_v1'
    binding_plan = json.loads((workspace / 'outputs/two_stage_binding60_gpu_v1/plan.json')
                              .read_text())
    if sha(features / 'plan.json') != plan['feature_plan_sha256']:
        raise ValueError('feature plan mismatch')

    rows, hashes = {}, {}
    for split in ('train', 'valid'):
        path = parent / f'{split}_cache.pt'
        hashes[split] = sha(path)
        raw = torch.load(path, map_location='cpu', weights_only=False)
        if [r[0] for r in raw] != plan['ids'][split]:
            raise ValueError('cache id order mismatch')
        if args.smoke:
            raw = raw[:2]
        rows[split] = FeatureRows(raw, features, split, binding_plan['feature_sha256'])

    prior_path = args.prior_root / f'stage1_head_seed{args.seed}.pt'
    if not prior_path.is_file():
        raise ValueError(f'missing stage-1 prior: {prior_path}')
    priors = {split: load_prior(prior_path, split, plan['ids'][split], args.alpha,
                                args.temperature, device) for split in ('train', 'valid')}

    source_path = parent / 'binding/last.pt'
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    if source['epoch'] != 60:
        raise ValueError(f"unexpected binding epoch {source['epoch']}")
    config = source['model_config']

    out = args.output.resolve()
    resume_state = None
    if args.resume:
        if not (out / 'plan.json').is_file():
            raise ValueError('--resume needs an existing run directory')
        existing = json.loads((out / 'plan.json').read_text())
        if existing['condition'] != args.condition or existing['seed'] != args.seed:
            raise ValueError('existing plan belongs to a different condition or seed')
        candidates = [out / 'pair/last.pt', out / 'pair/best.pt']
        chosen = next((c for c in candidates if c.is_file()), None)
        if chosen is None:
            raise ValueError('nothing to resume from')
        resume_state = torch.load(chosen, map_location='cpu', weights_only=False)
    else:
        out.mkdir(parents=True, exist_ok=False)
    epochs = 1 if args.smoke else args.epochs

    torch.manual_seed(args.seed)
    model = SequencePriorRefiner(**config).to(device)
    model.load_binding(source['model'])
    directory = out / 'pair'
    if resume_state is None:
        directory.mkdir()
        initial = run_epoch(model, rows['valid'], device, priors['valid'], False,
                            condition=args.condition)
        dump(directory / 'initial_validation.json', initial)
        best, best_epoch, start_epoch = initial['aligned_rmsd'], 0, 1
    else:
        initial = json.loads((directory / 'initial_validation.json').read_text())
        model.load_state_dict(resume_state['model'])
        best, best_epoch = resume_state['best_score'], resume_state['best_epoch']
        start_epoch = int(resume_state['epoch']) + 1

    groups = [dict(params=[p for n, p in model.named_parameters()
                           if not n.startswith('loop_pair_layers.')], initial_lr=1e-5, lr=1e-5),
              dict(params=[p for n, p in model.named_parameters()
                           if n.startswith('loop_pair_layers.')], initial_lr=1e-4, lr=1e-4)]
    optimizer = torch.optim.Adam(groups)
    output_prior = args.condition == 'full'

    if resume_state is None:
        plan = dict(
            stage='pair', condition=args.condition, seed=args.seed, smoke=args.smoke,
            alpha=args.alpha, temperature=args.temperature,
            prior=dict(path=str(prior_path), sha256=sha(prior_path),
                       meaning='B = alpha * S^0 / T; S^0 is the stage-1 masked readout'),
            conditions=dict(baseline='zeros input, zeros residual',
                            probability_init='B input, zeros residual',
                            full='B input, B residual'),
            stage1_geometry='frozen; the same X^0 is used in every condition',
            parent=str(parent), parent_plan_sha256=sha(parent / 'plan.json'),
            source=str(source_path), source_sha256=sha(source_path),
            cache_sha256=hashes, feature_plan_sha256=sha(features / 'plan.json'),
            model_config=config, epochs=dict(pair=epochs),
            objective='unchanged base geometry + 0.5 FAPE + CE',
            selection='minimum validation aligned_rmsd subject to the original guards',
            optimizer=dict(type='Adam', accumulation=16, grad_clip=1.0,
                           schedule='initial_lr * 0.1 ** ((epoch-1)/(epochs-1))'),
            parameters=dict(total=sum(p.numel() for p in model.parameters()),
                            trainable=sum(p.numel() for _, p in trainable(model))),
            test_access=False,
            code_sha256={name: sha(Path(__file__).resolve())})
        dump(out / 'plan.json', plan)
    plan_sha = sha(out / 'plan.json')

    def state(epoch_index, validation):
        return dict(model=model.state_dict(), model_config=config, epoch=epoch_index,
                    stage='pair', condition=args.condition, validation=validation,
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
        status = dict(phase='training', stage='pair', condition=args.condition,
                      seed=args.seed, epoch=e, epochs=epochs, test_access=False)
        dump(out / 'status.json', status)
        train_metrics = run_epoch(model, rows['train'], device, priors['train'], output_prior,
                                  optimizer, args.seed + e, (out / 'status.json', status),
                                  condition=args.condition)
        validation = run_epoch(model, rows['valid'], device, priors['valid'], output_prior,
                               condition=args.condition)
        allowed = eligible(validation, initial)
        improved = allowed and validation['aligned_rmsd'] < best
        if improved:
            best, best_epoch = validation['aligned_rmsd'], e
        checkpoint = state(e, validation)
        save(directory / 'last.pt', checkpoint)
        if improved:
            save(directory / 'best.pt', checkpoint)
        dump(directory / f'epoch_{e:03d}.json',
             dict(epoch=e, condition=args.condition, seed=args.seed, train=train_metrics,
                  validation=validation, guard_passed=allowed, best_epoch=best_epoch,
                  seconds=time.monotonic() - tick))
        print(json.dumps(dict(condition=args.condition, seed=args.seed, epoch=e,
                              val_aligned_rmsd=round(validation['aligned_rmsd'], 4),
                              val_aar=round(validation['aar'], 4),
                              guard=allowed, best_epoch=best_epoch,
                              seconds=round(time.monotonic() - tick, 1))), flush=True)
    dump(out / 'status.json', dict(phase='smoke_complete' if args.smoke else 'complete',
                                   stage='pair', condition=args.condition, seed=args.seed,
                                   selected_epoch=best_epoch, test_access=False))
    print(json.dumps(dict(condition=args.condition, seed=args.seed,
                          selected_epoch=best_epoch)))


if __name__ == '__main__':
    main()
