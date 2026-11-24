"""Descriptive paired statistics and masked peptide-link audit on the existing development benchmark."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train.train_abgaze_reconstructed import dump, sha


def distribution(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return dict(n=0)
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                p90=float(np.quantile(a, .9)), p95=float(np.quantile(a, .95)),
                maximum=float(a.max()), fraction_abs_error_gt_0_1=float((a > .1).mean()))


def paired(delta, groups, seed=2026, replicates=10000):
    """Resample groups, retain all member observations, report sample-weighted mean."""
    rng = np.random.default_rng(seed)
    unique = sorted(set(groups))
    blocks = [np.flatnonzero(np.asarray(groups) == g) for g in unique]
    sums = np.array([delta[b].sum() for b in blocks])
    sizes = np.array([len(b) for b in blocks])
    draw = rng.integers(0, len(blocks), size=(replicates, len(blocks)))
    estimates = sums[draw].sum(1) / sizes[draw].sum(1)
    return dict(mean_delta=float(delta.mean()), median_delta=float(np.median(delta)),
                ci95=np.quantile(estimates, [.025, .975]).tolist(), n_groups=len(blocks))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    runs = dict(new='persistent_loop_pair_test_v1', source='two_stage_binding_epoch60_gpu_test_v1',
                raad='raad_abgaze_reconstructed_test_v1')
    paths = {k: ROOT / 'outputs' / v for k, v in runs.items()}
    metrics = {k: json.loads((v / 'metrics.json').read_text()) for k, v in paths.items()}
    ids = [r['sample_id'] for r in metrics['new']['per_complex']]
    for k, m in metrics.items():
        assert [r['sample_id'] for r in m['per_complex']] == ids, k
        assert m['test_sha256'] == metrics['new']['test_sha256']
    split = ROOT / 'data/abgaze_reconstructed_v1/splits/test.json'
    metadata = {r['pdb']: r for r in map(json.loads, split.read_text().splitlines())}
    assert set(ids) == set(metadata)
    clusters = [metadata[s]['cluster'] for s in ids]
    keys = ['rmsd', 'aligned_rmsd', 'interface_fape', 'contact_f1', 'cn_mae', 'clash', 'aar', 'oracle_aar']
    report = {}
    for other in ('source', 'raad'):
        result = {}
        for key in keys:
            delta = np.array([a[key] - b[key] for a, b in zip(metrics['new']['per_complex'], metrics[other]['per_complex'])])
            higher = key in ('contact_f1', 'aar', 'oracle_aar')
            wins = delta > 0 if higher else delta < 0
            order = np.argsort(-np.abs(delta))
            result[key] = dict(
                direction='new minus comparator', higher_is_better=higher,
                paired_sample_bootstrap=paired(delta, ids),
                paired_h3_cluster_bootstrap=paired(delta, clusters),
                win_fraction=float(wins.mean()), tie_fraction=float((delta == 0).mean()),
                largest_absolute_deltas=[dict(sample_id=ids[i], delta=float(delta[i])) for i in order[:5]],
                per_complex_delta={sid: float(d) for sid, d in zip(ids, delta)})
        report[other] = result
    dump(out / 'paired_statistics.json', report)
    tensorpath = ROOT / 'data/abgaze_reconstructed_v1/tensors/test.pt'
    assert sha(tensorpath) == metrics['new']['test_sha256']
    rows = torch.load(tensorpath, map_location='cpu', weights_only=False)
    # Native and both in-house models share the exact same anchor coordinates and atom masks.
    generated = {k: {r['sample_id']: r for r in json.loads((paths[k] / 'generated.json').read_text())}
                 for k in ('new', 'source')}
    geometry = {}
    for name in ('native', 'source', 'new'):
        pooled = {k: [] for k in ('all', 'internal', 'left_anchor', 'right_anchor')}
        cases = []
        for sid, x, y in rows:
            bb = y['bb'] if name == 'native' else torch.tensor(generated[name][sid]['backbone'])
            chain = torch.cat([x['anchors'][:1], bb, x['anchors'][1:]])
            mask = torch.cat([x['anchor_mask'][:1], y['mask'], x['anchor_mask'][1:]])
            valid = mask[:-1, 2] & mask[1:, 0]
            distances = (chain[:-1, 2] - chain[1:, 0]).norm(dim=-1)
            errors = (distances - 1.329).abs()
            per = dict(sample_id=sid, links=[])
            for i in range(len(distances)):
                if not valid[i]:
                    continue
                kind = 'left_anchor' if i == 0 else 'right_anchor' if i == len(distances)-1 else 'internal'
                value = float(errors[i])
                pooled['all'].append(value)
                pooled[kind].append(value)
                per['links'].append(dict(index=i, kind=kind, distance=float(distances[i]), abs_error=value))
            per['mean_abs_error'] = float(errors[valid].mean()) if valid.any() else None
            cases.append(per)
        geometry[name] = dict(pooled={k: distribution(v) for k, v in pooled.items()},
                              macro_mean_abs_error=float(np.mean([c['mean_abs_error'] for c in cases if c['mean_abs_error'] is not None])),
                              per_complex=cases)
    dump(out / 'peptide_link_audit.json', geometry)
    dump(out / 'plan.json', dict(
        script_sha256=sha(Path(__file__)), metrics_sha256={k: sha(v/'metrics.json') for k,v in paths.items()},
        generated_sha256={k: sha(paths[k]/'generated.json') for k in generated},
        split_sha256=sha(split), test_sha256=sha(tensorpath), bootstrap_replicates=10000,
        seed=2026, n=len(ids), n_h3_clusters=len(set(clusters)),
        limitations=['Exploratory analysis of repeatedly inspected development test; not confirmatory.',
                     'H3 clusters are NOT antigen families; antigen-family bootstrap pending.',
                     'Single training seed; intervals measure sampling uncertainty, not training variability.',
                     'CN outlier threshold 0.1 A is descriptive, not a full stereochemical validation.',
                     'Native atom masks applied equally to all geometry audits; may differ from historical metric.']))
    dump(out / 'status.json', dict(phase='complete', n=len(ids)))
    print(json.dumps(dict(statistics={k: {m: v['paired_sample_bootstrap'] for m,v in r.items()} for k,r in report.items()},
                          geometry={k: v['pooled'] for k,v in geometry.items()})))


if __name__ == '__main__':
    main()
