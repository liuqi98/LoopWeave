"""Audit what the FAPE and C-N advantages of this model actually mean.

Three questions, each answered on the locked 20-complex external cohort with the
committed predictions of every method:

FAPE
    ``interface_fape`` places the fixed antigen backbone into the local frame of
    each H3 residue and averages the clamped frame error. The audit reports the
    same quantity with (a) the clamp removed, (b) different clamps, and (c) the
    antigen restricted to residues that are actually within 8 A of the native
    loop. A genuine interface-positioning gain must survive (a) and (c); a gain
    that only exists at 10 A on all antigen atoms is a truncated global term.

C-N continuity
    Beyond the mean |d - 1.329| error: anomaly fractions at 0.1 and 0.2 A, the
    two terminal anchor bonds, the internal bonds, the peptide bond angles at C
    and N, omega planarity and the clash proxy, so a continuity gain cannot be
    reported while silently paying elsewhere.

Contacts
    Precision, recall, F1 and the raw native/predicted pair counts alongside the
    turnover counts, so "more wrong additions than right additions" is explained
    by the contact budget rather than by a single turnover number.

AAR is decomposed into deployable fixed sampling (greedy / first / mean over five
samples) and the native-label oracle; the oracle never selects a sample whose
other metrics are then reported.

Reading only; no model is executed. Native labels are used solely to stratify the
evaluation (interface subset, oracle AAR), never as model input.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT.parent
sys.path.insert(0, str(ROOT))
from orientation_interface_refiner import local_frames
from eval.test_loop_pair_replicates_v1 import score
from eval.evaluate_external_temporal_v1 import geometry
from research_audit.statistics import interval
from research_audit.diagnostics import contact_counts

ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
SOURCES = {
    'ours': [('external_temporal_evaluation_v1', f'generated_seed{s}.json', s)
             for s in (2026, 2027, 2028)],
    'raad': [('external_raad_comparison_v1', f'generated_seed{s}.json', s)
             for s in (2026, 2027, 2028)],
    'mean': [('mean_external_locked_v1', 'generated.json', 2026)],
    'diffab': [('diffab_external_locked_v1', 'generated.json', 2026)],
}
CLAMPS = (None, 5., 10., 20.)
INTERFACE_RADIUS = 8.


def sha(path):
    import hashlib
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


def fape(prediction, x, y, clamp, interface_only):
    """Local-frame antigen error; clamp=None reports Angstrom without truncation."""
    po, pf, pv = local_frames(prediction, torch.ones_like(prediction[..., 0], dtype=torch.bool))
    to, tf, tv = local_frames(y['bb'], y['mask'])
    context = x['context_bb']
    mask = x['context_mask'] & x['antigen_mask'][:, None] & torch.isfinite(context).all(-1)
    if interface_only:
        observed = y['mask'][:, 1].bool()
        native_ca = y['bb'][observed, 1]
        antigen_ca = context[:, 1]
        if len(native_ca) and len(antigen_ca):
            near = torch.cdist(native_ca, antigen_ca).min(dim=0).values <= INTERFACE_RADIUS
            mask = mask & near[:, None] & x['antigen_mask'][:, None]
        else:
            return None
    if not bool(mask.any()):
        return None
    points = torch.where(mask[..., None], context, 0.).flatten(0, 1)
    plocal = torch.einsum('lpj,ljk->lpk', points[None] - po[:, None], pf)
    tlocal = torch.einsum('lpj,ljk->lpk', points[None] - to[:, None], tf)
    error = ((plocal - tlocal).square().sum(-1) + 1e-8).sqrt()
    if clamp is not None:
        saturated = (error >= clamp).float()
        error = error.clamp(max=clamp) / clamp
    else:
        saturated = torch.zeros_like(error)
    valid = (pv & tv)[:, None] & mask.flatten()[None]
    if not bool(valid.any()):
        return None
    return dict(mean=float(error[valid].mean()), saturated_fraction=float(saturated[valid].mean()),
                n_frames=int(valid.any(-1).sum()), n_pairs=int(valid.sum()))


def cn_details(prediction, x, y):
    chain = torch.cat([x['anchors'][:1], prediction, x['anchors'][1:]])
    native = torch.cat([x['anchors'][:1], y['bb'], x['anchors'][1:]])
    mask = torch.cat([x['anchor_mask'][:1], y['mask'], x['anchor_mask'][1:]])
    valid = mask[:-1, 2] & mask[1:, 0]
    errors = ((chain[:-1, 2] - chain[1:, 0]).norm(dim=-1) - 1.329).abs()
    errors = errors[valid]

    def angle(a, b, c, target, m):
        cosine = (torch.nn.functional.normalize(a - b, dim=-1)
                  * torch.nn.functional.normalize(c - b, dim=-1)).sum(-1).clamp(-1, 1)
        return (cosine.arccos() * 180. / np.pi - target).abs()[m]

    at_c = angle(chain[:-1, 1], chain[:-1, 2], chain[1:, 0], 116.2, valid & mask[:-1, 1])
    at_n = angle(chain[:-1, 2], chain[1:, 0], chain[1:, 1], 121.7, valid & mask[1:, 1])
    return dict(cn_mae=float(errors.mean()), cn_p90=float(torch.quantile(errors, .9)),
                cn_frac_gt_0_1=float((errors > .1).float().mean()),
                cn_frac_gt_0_2=float((errors > .2).float().mean()),
                cn_left_anchor=float(errors[0]), cn_right_anchor=float(errors[-1]),
                cn_internal=float(errors[1:-1].mean()),
                peptide_angle_at_C_dev_deg=float(at_c.mean()),
                peptide_angle_at_N_dev_deg=float(at_n.mean()),
                n_bonds=int(valid.sum()))


def contact_budget(prediction, x, y, threshold=8.):
    observed = y['mask'][:, 1].bool()
    context = x['context_bb'][:, 1]
    antigen = (x['antigen_mask'].bool() & x['context_mask'][:, 1].bool())
    reference = torch.cdist(y['bb'][observed, 1], context[antigen])
    predicted = torch.cdist(prediction[observed, 1], context[antigen])
    mask = torch.ones_like(reference, dtype=torch.bool)
    counts = contact_counts(reference.numpy(), predicted.numpy(), mask.numpy(), threshold)
    near = reference <= threshold
    counts['interface_distance_mae'] = (float((predicted - reference).abs()[near].mean())
                                        if bool(near.any()) else None)
    counts['all_pair_distance_mae'] = float((predicted - reference).abs().mean())
    counts['predicted_to_native_ratio'] = float(counts['n_predicted'] / max(counts['n_native'], 1))
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--groups', type=Path, required=True)
    parser.add_argument('--orig', type=Path, default=Path('/media/xd/code1/dyGF'),
                        help='Source workspace holding committed generated_*.json when absent from the package')
    parser.add_argument('--replicates', type=int, default=10000)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    group_doc = json.loads(args.groups.read_text())
    ids, groups = group_doc['ids'], group_doc['groups']
    membership = {s: g for g, ss in groups.items() for s in ss}
    rows = torch.load(PACKAGE / 'data/external_test/inputs.pt', map_location='cpu',
                      weights_only=False)
    if [r[0] for r in rows] != ids:
        raise ValueError('cohort order mismatch')

    per_method = {}
    provenance = {}
    for method, runs in SOURCES.items():
        per_method[method] = {}
        provenance[method] = {}
        for directory, name, seed in runs:
            candidates = [PACKAGE / 'results' / directory / name,
                          args.orig / 'outputs' / directory / name]
            path = next((c for c in candidates if c.is_file()), None)
            if path is None:
                raise FileNotFoundError(f'no committed predictions for {directory}/{name}')
            generated = json.loads(path.read_text())
            if [g['sample_id'] for g in generated] != ids:
                raise ValueError(f'sample order mismatch: {path}')
            provenance[method][str(seed)] = dict(path=str(path), sha256=sha(path), seed=seed)
            scored = score(rows, generated)
            geom = geometry(rows, generated)
            scored_by = {c['sample_id']: c for c in scored['per_complex']}
            geom_by = {c['sample_id']: c for c in geom['per_complex']}
            cases = {}
            for (sid, x, y), record in zip(rows, generated):
                prediction = torch.tensor(record['backbone'])
                entry = dict(length=int(y['mask'][:, 1].sum()))
                entry.update({k: scored_by[sid][k] for k in
                              ('aar', 'first_sample_aar', 'mean_sample_aar', 'oracle_aar')})
                entry['clash'] = scored_by[sid]['clash']
                entry['contact_f1_reported'] = scored_by[sid]['contact_f1']
                entry['cn_mae_reported'] = scored_by[sid]['cn_mae']
                entry.update({k: geom_by[sid][k] for k in
                              ('omega_planarity_deviation_deg', 'omega_native_abs_error_deg',
                               'phi_native_abs_error_deg', 'psi_native_abs_error_deg')})
                entry.update(cn_details(prediction, x, y))
                entry.update(contact_budget(prediction, x, y))
                for clamp in CLAMPS:
                    label = 'raw' if clamp is None else f'clamp{int(clamp)}'
                    for scope in ('all', 'interface'):
                        value = fape(prediction, x, y, clamp, scope == 'interface')
                        if value is not None:
                            entry[f'fape_{label}_{scope}'] = value['mean']
                            entry[f'fape_{label}_{scope}_saturated'] = value['saturated_fraction']
                cases[sid] = entry
            per_method[method][str(seed)] = cases

    metrics = sorted({k for seed_cases in per_method['ours'].values()
                      for k in next(iter(seed_cases.values()))})
    summary = {}
    for method, by_seed in per_method.items():
        summary[method] = {}
        for metric in metrics:
            values = [statistics.mean([c[metric] for c in cases.values() if c.get(metric) is not None])
                      for cases in by_seed.values()
                      if any(c.get(metric) is not None for c in cases.values())]
            if not values:
                continue
            summary[method][metric] = dict(
                mean=statistics.mean(values), per_seed=values,
                seed_sd=statistics.stdev(values) if len(values) > 1 else None,
                n_seeds=len(values))

    directions = {'aligned_rmsd': -1, 'rmsd': -1, 'interface_fape': -1}
    high_is_better = {'aar', 'first_sample_aar', 'mean_sample_aar', 'oracle_aar', 'contact_f1',
                      'contact_f1_reported'}
    low_is_better = {'clash', 'cn_mae', 'cn_mae_reported', 'cn_p90', 'cn_frac_gt_0_1',
                     'cn_frac_gt_0_2', 'cn_left_anchor', 'cn_right_anchor', 'cn_internal',
                     'peptide_angle_at_C_dev_deg', 'peptide_angle_at_N_dev_deg',
                     'omega_planarity_deviation_deg', 'omega_native_abs_error_deg',
                     'phi_native_abs_error_deg', 'psi_native_abs_error_deg',
                     'interface_distance_mae', 'all_pair_distance_mae'}
    directions.update({m: 1 for m in high_is_better})
    directions.update({m: -1 for m in low_is_better})
    for metric in metrics:
        if 'saturated' in metric or 'predicted_to_native_ratio' in metric:
            directions[metric] = 0
        elif any(k in metric for k in ('fape', 'clamp')):
            directions[metric] = -1

    sample_groups = [membership[s] for s in ids]
    contrasts = {}
    for comparator in ('raad', 'mean', 'diffab'):
        contrasts[comparator] = {}
        for metric, direction in directions.items():
            if direction == 0:
                continue
            # A metric may be undefined for a few complexes (e.g. an empty
            # native-interface subset). The subset is native-defined, so it is
            # identical for every method; pair on the common defined set only.
            usable = [sid for sid in ids
                      if all(all(cases[sid].get(metric) is not None
                                 for cases in per_method[m].values()) for m in ('ours', comparator))]
            if len(usable) < 2:
                continue
            a = [statistics.mean(cases[sid][metric] for cases in per_method['ours'].values())
                 for sid in usable]
            b = [statistics.mean(cases[sid][metric] for cases in per_method[comparator].values())
                 for sid in usable]
            delta = np.array([[direction * (a[i] - b[i]) for i in range(len(usable))]])
            stats = interval(delta, [membership[s] for s in usable], args.replicates)
            contrasts[comparator][metric] = dict(
                sign='positive means ours better', mean=stats['mean'],
                ci95=stats['conditional_group_ci95'], n_common=len(usable),
                ours=statistics.mean(a), comparator=statistics.mean(b))

    dump(out / 'audit.json', dict(
        cohort='locked external nonpathogen cohort', n=len(ids), n_groups=len(groups),
        group_source=str(args.groups), group_source_sha256=sha(args.groups),
        clamps=list(CLAMPS), interface_radius=INTERFACE_RADIUS,
        fape_definition=dict(
            frames='per H3 residue local frame: e1 = C-CA, e2 = N-CA orthogonalized, e3 = e1 x e2',
            target='fixed antigen backbone atoms (all atoms, or only antigen residues within 8 A of the native loop)',
            error='Euclidean error between predicted-frame and native-frame local coordinates',
            normalization='clamp10 divides the error by 10 after clamping; raw reports Angstrom with no truncation',
            mask='context_mask & antigen_mask & finite coordinates; H3 frames must be valid',
            native_use='native coordinates only define the interface subset and the oracle AAR; never model input'),
        provenance=provenance, summary=summary, contrasts_vs_ours=contrasts,
        seed_counts={m: len(v) for m, v in per_method.items()},
        limitations=['ours and RAAD have three training seeds; MEAN has one, so no training-seed spread is claimed for MEAN.',
                     'Interface subset and oracle AAR use native labels for evaluation only.',
                     'Five sampled sequences share one deterministic backbone for ours and MEAN, so sequence sampling is not independent evidence.',
                     'Clash is a backbone proxy, not an all-atom steric energy.',
                     'The cohort was inspected during method development.']))
    interesting = ['fape_raw_all', 'fape_raw_interface', 'fape_clamp10_all', 'fape_clamp10_interface',
                   'fape_clamp5_interface', 'fape_clamp20_interface',
                   'fape_clamp10_all_saturated', 'fape_clamp5_all_saturated',
                   'interface_distance_mae', 'contact_f1', 'precision', 'recall',
                   'predicted_to_native_ratio', 'tp', 'fp', 'fn', 'n_native', 'n_predicted',
                   'cn_mae', 'cn_frac_gt_0_1', 'cn_frac_gt_0_2',
                   'cn_left_anchor', 'cn_right_anchor', 'cn_internal',
                   'peptide_angle_at_C_dev_deg', 'peptide_angle_at_N_dev_deg',
                   'omega_planarity_deviation_deg', 'clash',
                   'aar', 'first_sample_aar', 'mean_sample_aar', 'oracle_aar']
    rowsout = []
    for metric in interesting:
        if metric not in summary['ours']:
            continue
        line = {'metric': metric}
        for method in summary:
            line[method] = round(summary[method][metric]['mean'], 4)
            line[method + '_sd'] = (round(summary[method][metric]['seed_sd'], 4)
                                    if summary[method][metric]['seed_sd'] is not None else None)
        for comparator in contrasts:
            if metric in contrasts[comparator]:
                c = contrasts[comparator][metric]
                line[f'vs_{comparator}'] = round(c['mean'], 4)
                line[f'vs_{comparator}_ci'] = [round(x, 4) for x in c['ci95']]
        rowsout.append(line)
    dump(out / 'table.json', dict(rows=rowsout))
    for line in rowsout:
        print(json.dumps(line, ensure_ascii=False))


if __name__ == '__main__':
    main()
