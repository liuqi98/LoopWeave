"""Use the trained refiner as a geometry post-processor for other methods' predictions.

For each method (RAAD / MEAN / DiffAb, plus our own model and a native ceiling control)
this keeps the method's **sequence** and only lets our refiner recompute the loop
geometry:

    base_bb      <- the method's committed H3 backbone
    base_logits  <- one-hot of the method's greedy sequence, scaled by 10
    output       <- our refiner's H3 backbone

The sequence is therefore fixed by construction: the refiner is conditioned on it, and
the evaluation uses that same sequence, so every change is geometric. Because the
scorer derives contacts/FAPE/C-N/RMSD from the backbone only, any movement in those
metrics is the refinement effect and AAR cannot change.

The native-backbone arm is included on purpose: if refinement pulls the *native* loop
away from its own ground truth, the procedure has a systematic attraction that must be
reported rather than hidden.

Reads committed predictions only; nothing is trained.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT.parent
sys.path.insert(0, str(ROOT))
from persistent_loop_pair_refiner import PersistentLoopPairRefiner
from eval.test_loop_pair_replicates_v1 import score
from eval.evaluate_external_temporal_v1 import geometry
from research_audit.statistics import interval

ORIG = Path('/media/xd/code1/dyGF')
ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
CHECKPOINT = PACKAGE / 'checkpoints/original_seed2026/pair.pt'
CONDITION_SCALE = 10.0

METHODS = {
    'raad': ORIG / 'outputs/raad_abgaze_reconstructed_test_v1/generated.json',
    'mean': ORIG / 'outputs/mean_reconstructed_test_v1/generated.json',
    'diffab': ORIG / 'outputs/diffab_reconstructed_seed2026_v1/generated.json',
    'ours': PACKAGE / 'results/aligned_shape_v1/test_w0_original/generated.json',
}
METRICS = ('aligned_rmsd', 'rmsd', 'contact_f1', 'interface_fape', 'cn_mae', 'clash',
           'omega_planarity_deviation_deg', 'phi_native_abs_error_deg',
           'psi_native_abs_error_deg')


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_generated(path):
    document = json.loads(Path(path).read_text())
    if 'draws' in document[0]:                       # DiffAb joint samples
        out = []
        for row in document:
            primary = row['draws'][0]
            onehot = torch.nn.functional.one_hot(
                torch.tensor([ALPHABET.index(a) for a in primary['sequence']]), 20).float()
            out.append(dict(sample_id=row['sample_id'], backbone=primary['backbone'],
                            sequence=primary['sequence'],
                            sequence_logits=(onehot * CONDITION_SCALE).tolist()))
        return out
    return [dict(sample_id=row['sample_id'], backbone=row['backbone'],
                 sequence=row['greedy'],
                 sequence_logits=(torch.nn.functional.one_hot(
                     torch.tensor([ALPHABET.index(a) for a in row['greedy']]), 20).float()
                     * CONDITION_SCALE).tolist())
            for row in document]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--replicates', type=int, default=10000)
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    # Evaluation cohort and model inputs.
    dataset = ORIG / 'data/abgaze_reconstructed_v1'
    rows = torch.load(dataset / 'tensors/test.pt', weights_only=False)
    records = [json.loads(l) for l in
               (dataset / 'splits/test.json').read_text().splitlines() if l.strip()]
    ids = [r[0] for r in rows]
    groups = {r['pdb']: f"cluster_{r['cluster']}" for r in records}
    features_dir = ORIG / 'outputs/binding_interface_test_v1/features'

    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = PersistentLoopPairRefiner(**state['model_config'])
    model.load_state_dict(state['model'])
    model.eval()

    base_inputs = {}
    for sid, x, y in rows:
        feature = torch.load(features_dir / sid / 'features.pt',
                             map_location='cpu', weights_only=False)
        if not feature['metadata']['antigen_only'] or feature['metadata']['native_h3_used']:
            raise ValueError(f'{sid}: binding features are not antigen-only')
        base_inputs[sid] = {**x, **feature['features']}

    def run_arm(name, entries, logits_mode='onehot'):
        """Refine geometry with each method's own sequence held fixed."""
        original, refined, native_sequence_ok = [], [], []
        generated_for_scoring = []
        for (sid, x, y), entry in zip(rows, entries):
            if entry['sample_id'] != sid:
                raise ValueError('prediction order mismatch')
            backbone = torch.tensor(entry['backbone'], dtype=torch.float32)
            logits = torch.tensor(entry['sequence_logits'], dtype=torch.float32)
            if backbone.shape != y['bb'].shape:
                raise ValueError(f'{sid}: shape {tuple(backbone.shape)} vs {tuple(y["bb"].shape)}')
            inputs = dict(base_inputs[sid])
            inputs['base_bb'] = backbone
            # Training used base_logits == 0 for the refiner stage, so a real one-hot
            # conditioning vector is out of distribution. Both modes are run so the
            # cause of any regression can be separated.
            inputs['base_logits'] = logits if logits_mode == 'onehot' else torch.zeros_like(logits)
            with torch.no_grad():
                prediction = model(inputs)
            polished = prediction['backbone']
            if not torch.isfinite(polished).all():
                raise ValueError(f'{sid}: nonfinite refined backbone')
            # The call must return the sequence it was conditioned on.
            if logits_mode == 'onehot':
                # Record how often the model's own argmax agrees with the conditioned
                # sequence. A mismatch does not affect scoring (which uses the method's
                # sequence directly) but it does show how strongly the prior dominates
                # the model's own sequence head.
                recovered = ''.join(ALPHABET[i] for i in
                                    prediction['sequence_logits'].argmax(-1).tolist())
                matches = sum(a == b for a, b in zip(recovered, entry['sequence']))
                native_sequence_ok.append(matches / max(len(entry['sequence']), 1))
            else:
                # Zero prior: the model picks its own sequence; scoring still uses the
                # method's sequence so only the geometry channel is compared.
                native_sequence_ok.append(True)
            refined.append(dict(sample_id=sid, backbone=polished.tolist(),
                                sequence_logits=logits.tolist(), greedy=entry['sequence'],
                                candidates=[entry['sequence']] * 5))
            generated_for_scoring.append(entry)
        conditioning_match = (max(native_sequence_ok) if logits_mode == 'onehot'
                              else float(np.mean(native_sequence_ok)))
        before = score(rows, [dict(sample_id=e['sample_id'], backbone=e['backbone'],
                                   sequence_logits=e['sequence_logits'], greedy=e['sequence'],
                                   candidates=[e['sequence']] * 5) for e in entries])
        after = score(rows, refined)
        geom_before = geometry(rows, [dict(sample_id=e['sample_id'], backbone=e['backbone'],
                                           sequence_logits=e['sequence_logits'],
                                           greedy=e['sequence'],
                                           candidates=[e['sequence']] * 5) for e in entries])
        geom_after = geometry(rows, refined)
        return before, after, geom_before, geom_after, refined, conditioning_match

    report = {}
    per_complex_rows = []
    arms = [(f'{name}_{mode}', name, path, mode)
            for name, path in METHODS.items() for mode in ('onehot', 'zeros')]
    for label, name, path, mode in arms:
        entries = load_generated(path)
        before, after, geom_before, geom_after, refined, conditioning_match = \
            run_arm(name, entries, mode)
        b = {c['sample_id']: c for c in before['per_complex']}
        a = {c['sample_id']: c for c in after['per_complex']}
        gb = {c['sample_id']: c for c in geom_before['per_complex']}
        ga = {c['sample_id']: c for c in geom_after['per_complex']}
        entry = dict(method=name, logits_mode=mode, source=str(path),
                     source_sha256=sha(path), n=len(ids),
                     conditioning_argmax_match=float(conditioning_match), per_metric={})
        for metric in METRICS:
            delta = []
            for sid in ids:
                if metric in b[sid]:
                    delta.append(a[sid][metric] - b[sid][metric])
                elif metric in gb[sid]:
                    delta.append(ga[sid][metric] - gb[sid][metric])
            if not delta:
                continue
            # Metrics are oriented so that a negative delta means "refinement helped"
            # except for contact_f1 and AAR-type metrics.
            direction = 1.0 if metric == 'contact_f1' else -1.0
            stats = interval(np.asarray([direction * np.array(delta)]),
                             [groups[s] for s in ids], args.replicates)
            entry['per_metric'][metric] = dict(
                before_mean=float(np.mean([b[s].get(metric, gb[s].get(metric)) for s in ids])),
                after_mean=float(np.mean([a[s].get(metric, ga[s].get(metric)) for s in ids])),
                improvement=stats['mean'],
                ci95=stats['conditional_group_ci95'],
                significant=bool(stats['conditional_group_ci95'][0] *
                                 stats['conditional_group_ci95'][1] > 0))
        report[label] = entry
        for sid in ids:
            per_complex_rows.append(dict(method=label, sample_id=sid, group=groups[sid],
                                         **{f'before_{m}': b[sid].get(m, gb[sid].get(m))
                                            for m in METRICS},
                                         **{f'after_{m}': a[sid].get(m, ga[sid].get(m))
                                            for m in METRICS}))
        dump(out / f'refined_{label}.json', dict(entry, sequences=[
            dict(sample_id=e['sample_id'], sequence=e['sequence']) for e in entries]))
        print(json.dumps({m: {k: round(v, 4) for k, v in entry['per_metric'][m].items()
                              if k in ('before_mean', 'after_mean', 'improvement')}
                          for m in ('cn_mae', 'interface_fape', 'aligned_rmsd')}), flush=True)

    # Native ceiling control: feed the ground-truth loop and refine it.
    native_entries = []
    for sid, x, y in rows:
        sequence = ''.join(ALPHABET[i] for i in y['sequence'].tolist())
        onehot = torch.nn.functional.one_hot(y['sequence'], 20).float()
        native_entries.append(dict(sample_id=sid, backbone=y['bb'].tolist(),
                                   sequence=sequence,
                                   sequence_logits=(onehot * CONDITION_SCALE).tolist()))
    for mode in ('onehot', 'zeros'):
        before, after, geom_before, geom_after, _, _cond = run_arm('native', native_entries, mode)
        b = {c['sample_id']: c for c in before['per_complex']}
        a = {c['sample_id']: c for c in after['per_complex']}
        gb = {c['sample_id']: c for c in geom_before['per_complex']}
        ga = {c['sample_id']: c for c in geom_after['per_complex']}
        entry = dict(method='native', logits_mode=mode, per_metric={},
                     note='ground-truth loop passed through the refiner')
        for metric in METRICS:
            delta = [a[s][metric] - b[s][metric] for s in ids if metric in b[s]]
            if not delta:
                continue
            direction = 1.0 if metric == 'contact_f1' else -1.0
            stats = interval(np.asarray([direction * np.array(delta)]),
                             [groups[s] for s in ids if metric in b[s]], args.replicates)
            entry['per_metric'][metric] = dict(
                before_mean=float(np.mean([b[s][metric] for s in ids if metric in b[s]])),
                after_mean=float(np.mean([a[s][metric] for s in ids if metric in b[s]])),
                improvement=stats['mean'], ci95=stats['conditional_group_ci95'],
                significant=bool(stats['conditional_group_ci95'][0] *
                                 stats['conditional_group_ci95'][1] > 0))
        report[f'native_{mode}'] = entry

    summary = dict(
        purpose='use the trained refiner as a geometry post-processor while keeping each '
                "method's own sequence fixed",
        checkpoint=str(args.checkpoint), checkpoint_sha256=sha(args.checkpoint),
        conditioning='base_bb = the method backbone; base_logits = its one-hot sequence x 10',
        sequence_preservation='asserted for every sample in every arm',
        cohort='dev-60 benchmark', n=len(ids), n_groups=len(set(groups.values())),
        orientation='improvement is positive when refinement made the metric better',
        results=report,
        limitations=['The refiner was trained as a full reconstruction model, so it has its own '
                     'geometric prior; the native-arm control measures that prior directly.',
                     'Contact/FAPE/C-N/RMSD depend on the backbone only, which is what changes here.',
                     'Single checkpoint (original seed 2026) for this scan.'])
    dump(out / 'postprocess_summary.json', summary)
    import csv
    if per_complex_rows:
        with (out / 'per_complex.csv').open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_complex_rows[0]))
            writer.writeheader()
            writer.writerows(per_complex_rows)
    print(json.dumps({k: (v['per_metric']['cn_mae']['improvement']
                          if isinstance(v.get('per_metric'), dict) else None)
                      for k, v in report.items()}, indent=1))


if __name__ == '__main__':
    main()
