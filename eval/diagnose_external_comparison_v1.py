"""Post-hoc, descriptive diagnostics only: no inference, fitting or selection."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parents[1]


def main():
    sources = {}

    def read(path):
        p = ROOT / path
        raw = p.read_bytes()
        sources[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    own = 'outputs/external_temporal_evaluation_v1'
    raad = 'outputs/external_raad_comparison_v1'
    lock = read(own + '/evaluation_lock.json')
    other_lock = read(raad + '/evaluation_lock.json')
    assert lock['ids'] == other_lock['ids']
    assert lock['groups'] == other_lock['groups']
    ids, seeds, groups = lock['ids'], lock['seeds'], lock['groups']
    membership = {sid: group for group, members in groups.items() for sid in members}
    comparison = read(raad + '/comparison.json')
    keys = list(comparison['ours_mean_std'])
    results, geometries = {}, {}
    for model, directory in [('ours', own), ('raad', raad)]:
        assert read(directory + '/status.json')['phase'] == 'complete'
        results[model], geometries[model] = {}, {}
        for seed in seeds:
            for target, prefix in [(results, 'metrics'), (geometries, 'geometry')]:
                doc = read(f'{directory}/{prefix}_seed{seed}.json')
                rows = doc['per_complex']
                assert [r['sample_id'] for r in rows] == ids
                target[model][seed] = {r['sample_id']: r for r in rows}
    geometry_keys = [k for k in geometries['ours'][seeds[0]][ids[0]] if k != 'sample_id']
    cases = []
    for sid in ids:
        length = results['ours'][seeds[0]][sid]['length']
        r = dict(sample_id=sid, group=membership[sid], length=length)
        for model in results:
            assert all(results[model][s][sid]['length'] == length for s in seeds)
            for source, names in [(results, keys), (geometries, geometry_keys)]:
                for k in names:
                    r[f'{model}_{k}'] = st.mean(source[model][s][sid][k] for s in seeds)
            # Descriptive alignment benefit, NOT a decomposition into pose error.
            r[f'{model}_alignment_benefit_A'] = r[f'{model}_rmsd'] - r[f'{model}_aligned_rmsd']
            r[f'{model}_zero_contact_all_seeds'] = all(results[model][s][sid]['contact_f1'] == 0 for s in seeds)
        for k in keys + geometry_keys:
            r[f'delta_{k}'] = r[f'ours_{k}'] - r[f'raad_{k}']
        assert all(math.isfinite(v) for v in r.values() if isinstance(v, (float, int)))
        cases.append(r)
    summary = {}
    for k in keys + geometry_keys:
        # Equal weight for dependence groups, then report training-seed SD.
        source = results if k in keys else geometries
        per_seed = {
            model: [st.mean(st.mean(source[model][s][sid][k] for sid in members)
                            for members in groups.values()) for s in seeds]
            for model in results}
        loo = {group: st.mean(r[f'delta_{k}'] for r in cases if r['group'] != group)
               for group in groups}
        summary[k] = dict(
            sample_mean_delta=st.mean(r[f'delta_{k}'] for r in cases),
            group_balanced={m: dict(mean=st.mean(v), sample_std=st.stdev(v)) for m, v in per_seed.items()},
            leave_one_group_out_delta=loo,
            leave_one_group_out_range=[min(loo.values()), max(loo.values())],
            n_ours_lower=sum(r[f'delta_{k}'] < 0 for r in cases),
            n_equal=sum(r[f'delta_{k}'] == 0 for r in cases),
            n_ours_higher=sum(r[f'delta_{k}'] > 0 for r in cases))
    report = dict(
        n=len(ids), n_groups=len(groups), seeds=seeds,
        analysis='Post-hoc descriptive diagnostics on all locked cases; no exclusions, new predictions or tuning.',
        metric_notes={
            'aar': 'Greedy argmax sequence recovery; oracle_aar is best native-AAR of five sampled sequences, auxiliary only.',
            'alignment_benefit_A': 'Raw minus loop-aligned CA RMSD. Not an independent docking error or additive pose/conformation decomposition.',
            'geometry': 'Backbone-only diagnostics; not all-atom validation or experimental binding evidence.',
            'leave_one_group_out': 'Sensitivity of sample-weighted mean delta to dropping each whole group in turn; not a confidence interval.'},
        metric_diagnostics=summary,
        worst_ours_aligned_rmsd=sorted(cases, key=lambda r: r['ours_aligned_rmsd'], reverse=True)[:5],
        largest_aligned_rmsd_disadvantages=sorted(cases, key=lambda r: r['delta_aligned_rmsd'], reverse=True)[:5],
        both_zero_contact_all_seeds=[r['sample_id'] for r in cases if r['ours_zero_contact_all_seeds'] and r['raad_zero_contact_all_seeds']],
        geometry_macro_mean={m: {k: st.mean(r[f'{m}_{k}'] for r in cases) for k in geometry_keys} for m in results},
        limitations=comparison['limitations'] + [
            'Post-hoc exploratory diagnostics, not pre-registered subgroup claims.',
            'No causal attribution to length, target identity or model components from this cohort.',
            'The evaluated cohort must not be reused for tuning and then presented as unseen confirmation.'])
    out = ROOT / 'outputs/external_failure_diagnostics_v1'
    out.mkdir(exist_ok=False)
    with (out / 'per_complex.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    sources[str(Path(__file__).relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for name, content in [('report.json', report), ('provenance.json', sources),
                          ('status.json', dict(phase='complete', n=len(ids), n_groups=len(groups), new_inference=False))]:
        with (out / name).open('x') as f:
            json.dump(content, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: summary[k] for k in ['aar', 'aligned_rmsd', 'contact_f1']}, indent=2))
    print(json.dumps(report['geometry_macro_mean'], indent=2))
    print('Both zero-contact:', report['both_zero_contact_all_seeds'])


if __name__ == '__main__':
    main()
