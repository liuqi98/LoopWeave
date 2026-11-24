"""Generate the adapted-DiffAb baseline on the locked 20-complex external cohort.

All-or-nothing by design: with ``--limit`` below the cohort size the run is marked
incomplete and is NOT scored, because a partial cohort must never enter the main
table. Protocol is the archived one (``baselines/train_diffab_reconstructed.py``
``evaluate_test``): K=5 independent joint samples per complex, seed 2026, one
deterministic draw-0 primary.

Scoring is done by ``code/eval/compare_external_baselines_v2.py`` from
``generated.json`` once the run is complete, so the same scorer is used for every
method. The DiffAb-native first/oracle decomposition is recomputed and stored in
``metrics.json`` with the oracle explicitly labelled as native-label using.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'baselines'))
UPSTREAM = Path('/media/xd/code2/DiffCDRH3/external/diffab')
sys.path.insert(0, str(UPSTREAM))
from diffab_reconstructed import CONFIG, MaskedDiffAb, batch, convert, AA
from diffab.modules.common.geometry import reconstruct_backbone
from diffab.modules.common.so3 import so3vec_to_rotation

ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
K = 5
SEED = 2026


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_progress(path, value):
    """Progress files are updated in place; final artifacts use write-once dump."""
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def tensor_digest(mapping):
    h = hashlib.sha256()
    for key in sorted(mapping):
        value = mapping[key]
        if torch.is_tensor(value):
            h.update(key.encode())
            h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orig', type=Path, default=Path('/media/xd/code1/dyGF'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0, help='Probe only; >0 marks the run incomplete')
    parser.add_argument('--threads', type=int, default=20)
    parser.add_argument('--device', default='cpu', help='cuda:0 when a GPU is available')
    parser.add_argument('--checkpoint', type=Path,
                        default=PACKAGE / 'checkpoints/baselines/diffab/best.pt')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('requested CUDA but torch cannot see a device')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    ids = json.loads((PACKAGE / 'data/dataset_index.json').read_text())['external_test']
    items = {i['sample_id']: (i, g) for i, g in torch.load(
        args.orig / 'data/independent_nonpathogen_inputs_v1/graphs.pt',
        map_location='cpu', weights_only=False)}
    coords = {s: (s, x, y) for s, x, y in torch.load(
        args.orig / 'data/independent_nonpathogen_inputs_v1/test.pt',
        map_location='cpu', weights_only=False)}
    records = {r['pdb']: r for r in json.loads(
        (args.orig / 'data/independent_nonpathogen_inputs_v1/records.json').read_text())}
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    plan = json.loads((args.orig / 'outputs/diffab_reconstructed_seed2026_v1/plan.json').read_text())
    protocol = dict(
        method='DiffAb', implementation='adapted reconstructed backbone-context adapter',
        checkpoint=str(args.checkpoint), checkpoint_sha256=sha(args.checkpoint),
        checkpoint_epoch=state.get('epoch'), training_plan_sha256=plan.get('plan_sha256'),
        training_plan_sha256_recomputed=sha(
            args.orig / 'outputs/diffab_reconstructed_seed2026_v1/plan.json'),
        cohort='locked external nonpathogen cohort', ids=ids, n=len(ids),
        K=K, sampling_seed=SEED, device=str(device), precision='float32', threads=args.threads,
        torch_version=torch.__version__,
        selection='draw 0 is the primary deployable prediction; native-AAR oracle over the five '
                  'joint samples is auxiliary and uses labels',
        inputs_sha256=dict(
            graphs=sha(args.orig / 'data/independent_nonpathogen_inputs_v1/graphs.pt'),
            test=sha(args.orig / 'data/independent_nonpathogen_inputs_v1/test.pt'),
            records=sha(args.orig / 'data/independent_nonpathogen_inputs_v1/records.json')),
        limitations=['Adapted DiffAb, not the unchanged official protocol; missing target frames use '
                     'deterministic proxy geometry and there is no rotation loss.',
                     'Single training seed (2026); no training-seed spread.',
                     'The cohort was already inspected during method development.',
                     'Joint samples differ in both backbone and sequence; the native-AAR oracle is not '
                     'a deployable selection rule.'])
    dump(out / 'protocol.json', protocol)

    model = MaskedDiffAb().to(device)
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    rows, audits = [], []
    for sid in ids[: args.limit or len(ids)]:
        item, graph = items[sid]
        _, x, y = coords[sid]
        row = convert(item, graph, x, y, records[sid])
        # Native labels must not reach the conditioner: translating every native
        # H3 coordinate must leave the inference batch unchanged at generate
        # positions (which are replaced by the target-free proxy).
        altered = dict(y, bb=y['bb'] + 500.)
        alt = convert(item, graph, x, altered, records[sid])
        with torch.no_grad():
            a = batch([row], device, True)
            b = batch([alt], device, True)
        same = all(torch.equal(a[k], b[k]) for k in a)
        audits.append(dict(sample_id=sid, native_label_invariant=bool(same),
                           digest=tensor_digest(a), L=int(len(row['aa'])),
                           h3=int(row['generate_flag'].sum())))
        if not same:
            raise ValueError(f'native label leaks into conditioning: {sid}')
        rows.append((sid, row))

    inverse = {int(AA(a)): a for a in ALPHABET}
    torch.manual_seed(SEED)
    generated, timing = [], []
    for index, (sid, row) in enumerate(rows):
        start = time.monotonic()
        with torch.no_grad():
            b = batch([row] * K, device, True)
            v, p, s = model.sample(b)[0]
            bb = reconstruct_backbone(so3vec_to_rotation(v), p, s,
                                      b['chain_nb'], b['res_nb'], b['mask'])
        elapsed = time.monotonic() - start
        draws = []
        for j in range(K):
            g = b['generate_flag'][j]
            coordinates = bb[j, g].cpu() + row['origin']
            if not torch.isfinite(coordinates).all():
                raise ValueError(f'nonfinite coordinates: {sid}/{j}')
            sequence = ''.join(inverse[int(i)] for i in s[j, g])
            draws.append(dict(sequence=sequence, backbone=coordinates.tolist()))
        primary = draws[0]
        onehot = torch.nn.functional.one_hot(
            torch.tensor([ALPHABET.index(a) for a in primary['sequence']]), 20).float() * 10
        generated.append(dict(sample_id=sid, backbone=primary['backbone'],
                              sequence_logits=onehot.tolist(), greedy=primary['sequence'],
                              candidates=[d['sequence'] for d in draws],
                              draws=draws))
        timing.append(dict(sample_id=sid, seconds=round(elapsed, 2),
                           length=int(len(row['aa'])), h3=int(row['generate_flag'].sum())))
        write_progress(out / 'status.json',
                       dict(phase='generating', completed=len(generated),
                            total=len(rows), last_seconds=round(elapsed, 2)))
        print(json.dumps(dict(sample=sid, completed=len(generated), total=len(rows),
                              seconds=round(elapsed, 2))), flush=True)
    complete = len(generated) == len(ids)
    dump(out / 'generated.json', generated)
    dump(out / 'generation_audit.json', dict(
        sha256=sha(out / 'generated.json'), complete=complete,
        target_free_conditioning=audits, K=K, sampling_seed=SEED,
        timing=timing, total_seconds=round(sum(t['seconds'] for t in timing), 1),
        draw_order='draw 0 is primary and is what backbone/sequence_logits encode'))
    write_progress(out / 'status.json',
                   dict(phase='complete' if complete else 'probe_complete',
                        generated=len(generated), total=len(ids), scored=False))
    print(json.dumps(dict(complete=complete, n=len(generated))), flush=True)


if __name__ == '__main__':
    main()
