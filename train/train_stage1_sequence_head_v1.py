"""Stage-1 sequence prior S^0: train the generator's existing sequence head only.

Background
----------
The generator was trained with the sequence weight set to zero, so its
``sequence_head`` (a single ``nn.Linear(hidden_dim, 20)``) is still at its random
initialisation. ``predict(stage='generator')`` even zeroes its output with the comment
"Geometry-only generator: its untrained sequence head is not a prior". This script makes
that head meaningful by fitting it with H3 cross-entropy while the geometry network
stays frozen.

Why the head can be fitted on cached features
---------------------------------------------
The masked H3 candidate one-hot in the graph is **all zeros** (the ``X`` placeholder from
``candidate()``), so the generator's sequence output is exactly ``sequence_head(h3_hidden)``
and the head does not feed back into the backbone. Two consequences:

* the backbone ``X^0`` is bit-identical before and after this script, which is the
  "fix stage-1 geometry" condition of the pilot;
* the head is linear in the cached H3 hidden state, so fitting on the cache is
  mathematically identical to training through the generator, but fast.

The native H3 sequence is used as a label only; it never enters the graph, so the
target-free property of the inputs is preserved.

Outputs per seed
----------------
``stage1_head_seed<seed>.pt``    head weights, S^0 logits for every split, selection record
``stage1_head_seed<seed>.json``  AAR before/after, geometry-invariance check
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from frame_relation_refiner import FrameRelationGraph, FrameRelationRefiner
from train.train_abgaze_reconstructed import sha, dump

WORKSPACE = Path('/media/xd/code1/dyGF')
PACKAGE = ROOT.parent
SPLITS = ('train', 'valid', 'test')


def cache(generator, device, split):
    """H3 hidden states (input of ``sequence_head``), native labels and backbone hashes."""
    graphs = torch.load(WORKSPACE / f'data/raad_abgaze_reconstructed_v1/{split}.pt',
                        map_location='cpu', weights_only=False)
    rows = torch.load(WORKSPACE / f'data/abgaze_reconstructed_v1/tensors/{split}.pt',
                      map_location='cpu', weights_only=False)
    labels = {sid: y['sequence'] for sid, _, y in rows}
    captured, hidden, target, ids = {}, [], [], []
    handle = generator.sequence_head.register_forward_hook(
        lambda module, inputs, output: captured.__setitem__('hidden', inputs[0]))
    backbone_checksums = []
    with torch.no_grad():
        for position, (item, graph) in enumerate(graphs):
            sample_id = item['sample_id']
            out = generator(FrameRelationGraph(**graph).to(device))
            hidden.append(captured['hidden'].detach().cpu())
            target.append(labels[sample_id].detach().cpu())
            ids.append(sample_id)
            if position < 8:
                backbone_checksums.append(float(out['h3_backbone'].double().sum()))
    handle.remove()
    return dict(hidden=hidden, target=target, ids=ids,
                backbone_checksums=backbone_checksums)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2026, 2027, 2028])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    out = args.output.resolve()
    # Existing seeds are skipped so additional seeds can be appended to the same directory.
    out.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        if (out / f'stage1_head_seed{seed}.pt').exists():
            print(json.dumps(dict(seed=seed, status='already present, skipped')), flush=True)
            continue
        generator_path = PACKAGE / f'checkpoints/original_seed{seed}/generator.pt'
        state = torch.load(generator_path, map_location='cpu', weights_only=False)
        generator = FrameRelationRefiner(**state['model_config']).to(device)
        generator.load_state_dict(state['model'])
        generator.eval()
        for parameter in generator.parameters():
            parameter.requires_grad_(False)

        data = {split: cache(generator, device, split) for split in SPLITS}
        width = int(state['model_config']['hidden_dim'])
        head = torch.nn.Linear(width, 20).to(device)
        head.weight.data = generator.sequence_head.weight.detach().clone()
        head.bias.data = generator.sequence_head.bias.detach().clone()

        def split_aar(weights, split='valid'):
            scores = []
            for hidden, target in zip(data[split]['hidden'], data[split]['target']):
                logits = hidden.to(device) @ weights[0].T + weights[1]
                scores.append((logits.argmax(-1).cpu() == target).float())
            return float(torch.cat(scores).mean())

        before = split_aar((head.weight.detach(), head.bias.detach()))
        optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
        best, best_epoch, best_state = -1.0, 0, None
        for epoch in range(1, args.epochs + 1):
            head.train()
            order = torch.randperm(len(data['train']['ids']))
            total, seen = 0.0, 0
            for index in order.tolist():
                hidden = data['train']['hidden'][index].to(device)
                target = data['train']['target'][index].to(device)
                loss = F.cross_entropy(head(hidden), target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
                total += float(loss.detach()) * len(target)
                seen += len(target)
            value = split_aar((head.weight.detach(), head.bias.detach()))
            if value > best:
                best, best_epoch = value, epoch
                best_state = (head.weight.detach().clone(), head.bias.detach().clone())
            if epoch % 10 == 0 or epoch == 1:
                print(json.dumps(dict(seed=seed, epoch=epoch,
                                      train_loss=round(total / seen, 4),
                                      valid_aar=round(value, 4), best=round(best, 4))),
                      flush=True)
        head.weight.data, head.bias.data = best_state

        # S^0 for every split, and a geometry-invariance statement.
        logits_out = {}
        for split in SPLITS:
            for hidden, target, sid in zip(data[split]['hidden'], data[split]['target'],
                                           data[split]['ids']):
                logits_out[sid] = dict(
                    logits=(hidden.to(device) @ head.weight.T + head.bias)
                    .detach().cpu(), target=target, split=split)
        aar_per_split = {split: split_aar((head.weight, head.bias), split) for split in SPLITS}
        geometry_note = ('the head does not feed the backbone: the masked candidate one-hot is '
                         'all zeros, so X^0 is unchanged by this training')
        torch.save(dict(seed=seed, width=width, weight=head.weight.detach().cpu(),
                        bias=head.bias.detach().cpu(), best_epoch=best_epoch,
                        best_valid_aar=best, aar_before_training=before,
                        aar_per_split=aar_per_split, logits=logits_out,
                        generator=str(generator_path),
                        generator_sha256=sha(generator_path),
                        backbone_checksums=data['test']['backbone_checksums'],
                        geometry_note=geometry_note),
                   out / f'stage1_head_seed{seed}.pt')
        dump(out / f'stage1_head_seed{seed}.json', dict(
            seed=seed, best_epoch=best_epoch, best_valid_aar=best,
            aar_before_training=before, aar_per_split=aar_per_split,
            epochs=args.epochs, lr=args.lr,
            candidate_one_hot='all zeros (masked X placeholder)',
            geometry_note=geometry_note,
            backbone_checksums_test8=data['test']['backbone_checksums']))
        print(json.dumps(dict(seed=seed, valid_aar_before=round(before, 4),
                              valid_aar_after=round(aar_per_split['valid'], 4),
                              test_aar=round(aar_per_split['test'], 4),
                              best_epoch=best_epoch)), flush=True)
        del generator
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
