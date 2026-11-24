"""Repeat the fixed human-validation projection pilot on all three old seeds."""
import json
import math
from pathlib import Path
import statistics as st
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.validate_peptide_geometry_v1 import read, FeatureRows, objective, aligned_rmsd, geometry
from train.train_abgaze_reconstructed import sha, dump, move
from persistent_loop_pair_refiner import PersistentLoopPairRefiner
from peptide_geometry_refiner import PeptideConstrainedLoopPairRefiner


def main():
    torch.set_num_threads(1)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    first = ROOT/'outputs/peptide_geometry_human_valid_v1'
    assert read(first/'status.json')['phase'] == 'complete'
    plan = read(first/'plan.json')
    for name, digest in plan['code_sha256'].items():
        assert sha(ROOT/name) == digest, name
    assert sha(first/'scope_audit.json') == plan['scope_audit_sha256']
    ids = plan['ids']
    out = ROOT/'outputs/peptide_geometry_human_valid_replicates_v1'
    out.mkdir(exist_ok=False)
    sources = {}
    for seed in [2026, 2027, 2028]:
        parent = ROOT/f'outputs/full_pipeline_seed{seed}_v1'
        pp = read(parent/'plan.json')
        assert read(parent/'status.json')['phase'] == 'complete'
        for name, digest in pp['code_sha256'].items():
            assert sha(ROOT/name) == digest, name
        state = torch.load(parent/'pair/best.pt', map_location='cpu', weights_only=False)
        assert state['plan_sha256'] == sha(parent/'plan.json')
        assert state['model_config'] == plan['model_config']
        assert set(ids) <= set(pp['ids']['valid'])
        assert not set(ids) & set(pp['ids']['train'])
        sources[str(seed)] = dict(parent=str(parent), checkpoint_sha256=sha(parent/'pair/best.pt'),
            epoch=state['epoch'], plan_sha256=sha(parent/'plan.json'), cache_sha256=sha(parent/'valid_cache.pt'))
    assert sources['2026']['checkpoint_sha256'] == plan['checkpoint_sha256']
    dump(out/'plan.json', dict(ids=ids, seeds=[2026, 2027, 2028], sources=sources,
        projection_config=plan['projection_config'], source_pilot_plan_sha256=sha(first/'plan.json'),
        first_comparison_sha256=sha(first/'comparison.json'),
        script_sha256=sha(Path(__file__)), test_access=False, optimizer_steps=0,
        limitations=plan['limitations'] + ['Expanded to three pre-existing checkpoints after the first pilot; unchanged samples and settings, not independent confirmation.']))
    results = {'2026':read(first/'comparison.json')}
    features = ROOT/'data/abgaze_binding_features_finalized_v1'
    hashes = {f'valid/{sid}':digest for sid,digest in plan['feature_sha256'].items()}
    for seed in [2027, 2028]:
        source = sources[str(seed)]
        parent = Path(source['parent'])
        assert sha(parent/'valid_cache.pt') == source['cache_sha256']
        raw = torch.load(parent/'valid_cache.pt', map_location='cpu', weights_only=False)
        assert [r[0] for r in raw] == read(parent/'plan.json')['ids']['valid']
        lookup = {r[0]:r for r in raw}
        rows = FeatureRows([lookup[sid] for sid in ids], features, 'valid', hashes)
        state = torch.load(parent/'pair/best.pt', map_location='cpu', weights_only=False)
        baseline = PersistentLoopPairRefiner(**state['model_config']).to(device).eval()
        candidate = PeptideConstrainedLoopPairRefiner(**state['model_config'], **plan['projection_config']).to(device).eval()
        for model in [baseline, candidate]:
            model.load_state_dict(state['model'], strict=True)
        cases = []
        with torch.no_grad():
            for i in range(len(rows)):
                sid, xc, yc, _ = rows[i]
                x, y = move(xc, device), move(yc, device)
                predictions = dict(baseline=baseline(x), candidate=candidate(x))
                assert torch.equal(predictions['baseline']['sequence_logits'], predictions['candidate']['sequence_logits'])
                torch.testing.assert_close(predictions['baseline']['backbone'], predictions['candidate']['backbone_before_peptide_projection'])
                case = dict(sample_id=sid, length=len(y['bb']))
                for name, p in predictions.items():
                    _, m = objective(p, x, y)
                    mask = y['mask'][:, 1]
                    m['aligned_rmsd'] = aligned_rmsd(p['backbone'][mask, 1].cpu(), y['bb'][mask, 1].cpu())
                    m['projection_energy'] = float(candidate.peptide_projection.energy(p['backbone'], x['anchors'], x['anchor_mask']))
                    m.update(geometry([(sid, xc, yc)], [dict(backbone=p['backbone'].cpu().tolist())])['summary'])
                    assert all(math.isfinite(v) for v in m.values()), sid
                    case[name] = m
                assert case['candidate']['projection_energy'] <= case['baseline']['projection_energy'] + 1e-6
                case['delta'] = {k:case['candidate'][k]-case['baseline'][k] for k in case['baseline']}
                cases.append(case)
                dump(out/'status.json', dict(phase='validating', seed=seed, completed=len(cases), n=len(ids), test_access=False))
        summary = {name:{k:st.mean(r[name][k] for r in cases) for k in cases[0][name]} for name in ['baseline', 'candidate', 'delta']}
        results[str(seed)] = dict(n=len(ids), summary=summary, per_complex=cases)
        dump(out/f'seed{seed}.json', results[str(seed)])
        del baseline, candidate, state
        torch.cuda.empty_cache()
    for result in results.values():
        assert [r['sample_id'] for r in result['per_complex']] == ids
    aggregate = {name:{k:dict(mean=st.mean(r['summary'][name][k] for r in results.values()),
                           sample_std=st.stdev(r['summary'][name][k] for r in results.values()))
                      for k in results['2026']['summary'][name]} for name in ['baseline', 'candidate', 'delta']}
    counts = {k:dict(lower=sum(c['delta'][k] < -1e-8 for r in results.values() for c in r['per_complex']),
                     higher=sum(c['delta'][k] > 1e-8 for r in results.values() for c in r['per_complex']))
              for k in results['2026']['summary']['delta']}
    dump(out/'comparison.json', dict(n=len(ids), seeds=[2026, 2027, 2028], mean_std=aggregate,
        by_seed={s:r['summary'] for s,r in results.items()}, sample_seed_counts=counts,
        limitations=read(out/'plan.json')['limitations']))
    dump(out/'status.json', dict(phase='complete', n=len(ids), seeds=[2026, 2027, 2028], optimizer_steps=0, test_access=False))
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
