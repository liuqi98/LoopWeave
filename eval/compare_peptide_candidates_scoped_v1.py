"""Locked three-candidate comparison on an expanded benign validation subset.

No fitting, checkpoint sweep, or test-set access. Metadata scope is fixed before
inference. These are development validation samples, not a fresh holdout.
"""
import collections
import concurrent.futures
import json
import math
from pathlib import Path
import statistics as st
import sys

import numpy as np
import requests
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
from prepare_peptide_human_expanded_v1 import ALLOWED
from train.train_abgaze_reconstructed import sha, dump, move
from eval.validate_peptide_geometry_v1 import read, FeatureRows, geometry, aligned_rmsd
from eval.audit_loop_pair_paper import paired
from abgaze_reconstructed_objective import objective
from persistent_loop_pair_refiner import PersistentLoopPairRefiner
from peptide_geometry_refiner import PeptideConstrainedLoopPairRefiner


def close_contacts(bb, mask, x):
    """Counts under 1.5 A, backbone only, with terminal covalent neighbors omitted."""
    length = len(bb)
    atoms = bb.flatten(0, 1)
    valid = mask.flatten()
    positions = torch.arange(length).repeat_interleave(4)
    allowed = (positions[:, None]-positions[None, :]).abs() > 1
    allowed &= valid[:, None] & valid[None, :]
    allowed &= torch.triu(torch.ones_like(allowed), diagonal=1)
    internal = (torch.cdist(atoms, atoms) < 1.5) & allowed
    context = x['context_bb']
    allowed_context = valid[:, None] & x['context_mask'].flatten()[None]
    for loop_index, anchor in [(0, x['anchors'][0, 1]), (length-1, x['anchors'][1, 1])]:
        match = (context[:, 1]-anchor).norm(dim=-1) < 1e-3
        allowed_context[loop_index*4:(loop_index+1)*4, match.repeat_interleave(4)] = False
    external = (torch.cdist(atoms, context.flatten(0, 1)) < 1.5) & allowed_context
    antigen = external & x['antigen_mask'].repeat_interleave(4)[None]
    return dict(short_internal_pairs=int(internal.sum()), short_context_pairs=int(external.sum()),
                short_antigen_pairs=int(antigen.sum()), any_short_pair=float(bool(internal.any() or external.any())))


def main():
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    device = torch.device('cuda:0'); torch.cuda.set_device(device)
    out = ROOT/'outputs/peptide_candidate_scoped_comparison_v1'
    out.mkdir(exist_ok=False); (out/'metadata').mkdir()
    source = ROOT/'data/abgaze_reconstructed_v1/splits/valid.json'
    records = [json.loads(l) for l in source.read_text().splitlines()]
    parent = ROOT/'outputs/full_pipeline_seed2026_v1'
    training = ROOT/'outputs/peptide_geometry_human_train_pilot_v1'
    pp, tp = read(parent/'plan.json'), read(training/'plan.json')
    assert [r['pdb'] for r in records] == pp['ids']['valid']
    for name, digest in tp['code_sha256'].items():
        assert sha(ROOT/name) == digest, name
    assert read(training/'status.json')['phase'] == 'complete'
    old_ids = set(tp['ids']['valid'])
    batches = [records[i:i+30] for i in range(0, len(records), 30)]
    fields = ('rcsb_id polymer_entities{rcsb_id rcsb_polymer_entity{pdbx_description} '
              'rcsb_polymer_entity_container_identifiers{auth_asym_ids} '
              'rcsb_entity_source_organism{ncbi_taxonomy_id ncbi_scientific_name}}')

    def fetch(job):
        i, batch = job
        query = '{entries(entry_ids:'+json.dumps([r['pdb'].upper() for r in batch])+'){'+fields+'}}'
        response = requests.get('https://data.rcsb.org/graphql', params={'query':query}, timeout=30)
        response.raise_for_status(); result = response.json()
        if result.get('errors'): raise ValueError(result['errors'])
        dump(out/'metadata'/f'{i:03d}.json', result)
        return result

    entries = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(fetch, enumerate(batches)):
            for entry in result['data']['entries'] or []:
                if entry: entries[entry['rcsb_id'].lower()] = entry
    ids, decisions = [], []
    for r in records:
        chains = set(r['antigen_chains']); covered = set(); matched = []
        for e in entries.get(r['pdb'], {}).get('polymer_entities') or []:
            overlap = chains & set((e.get('rcsb_polymer_entity_container_identifiers') or {}).get('auth_asym_ids') or [])
            if overlap: covered.update(overlap); matched.append(e)
        ok = bool(chains) and covered == chains and all(
            (e.get('rcsb_polymer_entity') or {}).get('pdbx_description', '').lower() in ALLOWED
            and {s.get('ncbi_taxonomy_id') for s in e.get('rcsb_entity_source_organism') or []} == {9606}
            for e in matched)
        decisions.append(dict(sample_id=r['pdb'], retained=ok, entities=[e['rcsb_id'] for e in matched],
            reason='explicit_benign_human_target_scope' if ok else 'outside_scope_or_incomplete_metadata'))
        if ok: ids.append(r['pdb'])
    assert old_ids <= set(ids)
    assert not set(ids) & set(tp['ids']['train'])
    clusters = {r['pdb']:r['cluster'] for r in records}
    dump(out/'scope_audit.json', dict(ids=ids, decisions=decisions, allowed_descriptions=sorted(ALLOWED),
        allowed_taxonomy=[9606], source_sha256=sha(source), performance_filter=False,
        metadata_sha256={p.name:sha(p) for p in (out/'metadata').glob('*.json')}))
    checkpoint_paths = dict(baseline=parent/'pair/best.pt', projected=training/'best.pt', trained3=training/'last.pt')
    states = {name:torch.load(p, map_location='cpu', weights_only=False) for name,p in checkpoint_paths.items()}
    assert states['projected']['epoch'] == 0 and states['trained3']['epoch'] == 3
    assert sha(checkpoint_paths['baseline']) == tp['checkpoint_sha256']
    for name in ['projected', 'trained3']:
        assert states[name]['plan_sha256'] == sha(training/'plan.json')
    cache = parent/'valid_cache.pt'
    assert sha(cache) == tp['cache_sha256']['valid']
    raw = torch.load(cache, map_location='cpu', weights_only=False)
    assert [r[0] for r in raw] == pp['ids']['valid']
    lookup = {r[0]:r for r in raw}
    features = ROOT/'data/abgaze_binding_features_finalized_v1'
    fp = read(ROOT/'outputs/two_stage_binding60_gpu_v1/plan.json')
    assert sha(features/'plan.json') == pp['feature_plan_sha256']
    feature_hashes = {}
    for sid in ids:
        path = features/'valid'/sid/'features.pt'; digest = sha(path)
        assert digest == fp['feature_sha256'][f'valid/{sid}']
        meta = torch.load(path, map_location='cpu', weights_only=False)['metadata']
        assert meta['antigen_only'] and not meta['native_h3_used'] and not meta['native_contact_crop']
        assert meta['pdb_sha256'] == sha(ROOT/'data/abgaze_reconstructed_v1/pdb'/f'{sid}.pdb')
        feature_hashes[f'valid/{sid}'] = digest
    rows = FeatureRows([lookup[sid] for sid in ids], features, 'valid', feature_hashes)
    plan = dict(ids=ids, n=len(ids), old_pilot_ids=sorted(old_ids), new_to_pilot_ids=[s for s in ids if s not in old_ids],
        clusters={sid:clusters[sid] for sid in ids},
        checkpoints={k:dict(path=str(p), sha256=sha(p), epoch=states[k]['epoch']) for k,p in checkpoint_paths.items()},
        cache_sha256=sha(cache), feature_sha256=feature_hashes, scope_audit_sha256=sha(out/'scope_audit.json'),
        code_sha256={f:sha(ROOT/f) for f in sorted(set(tp['code_sha256']) | {
            'eval/compare_peptide_candidates_scoped_v1.py', 'scripts/prepare_peptide_human_expanded_v1.py',
            'eval/audit_loop_pair_paper.py'})},
        selection='Exactly original, projection-only, and fixed last epoch3; no sweep or fitting.',
        test_access=False, optimizer_steps=0,
        limitations=['One seed; validation used in historical model selection, not a fresh holdout.',
            'Expanded explicit human-target validation scope; does not represent full validation distribution.',
            'Last epoch3 has only 36 human-target training examples; no matched continued-training control.',
            'Bootstrap uses existing H3 clusters, not certified independent antigen families.',
            'Short-contact threshold 1.5 A is descriptive and backbone-only, not an all-atom clash score.',
            'New-to-pilot samples are not necessarily unseen by the original training/selection workflow.'])
    dump(out/'plan.json', plan)
    classes = dict(baseline=PersistentLoopPairRefiner, projected=PeptideConstrainedLoopPairRefiner, trained3=PeptideConstrainedLoopPairRefiner)
    models = {}
    for name, cls in classes.items():
        models[name] = cls(**states[name]['model_config']).to(device).eval()
        models[name].load_state_dict(states[name]['model'], strict=True)
    cases = []
    with torch.no_grad():
        for i in range(len(rows)):
            sid, xc, yc, _ = rows[i]; x, y = move(xc, device), move(yc, device)
            case = dict(sample_id=sid, cluster=clusters[sid], old_pilot=sid in old_ids,
                        native_close_contacts=close_contacts(yc['bb'], yc['mask'], xc))
            predictions = {name:model(x) for name,model in models.items()}
            assert torch.equal(predictions['baseline']['sequence_logits'], predictions['projected']['sequence_logits'])
            for name, pred in predictions.items():
                _, metrics = objective(pred, x, y)
                valid = y['mask'][:, 1]
                metrics['aligned_rmsd'] = aligned_rmsd(pred['backbone'][valid, 1].cpu(), y['bb'][valid, 1].cpu())
                bb = pred['backbone'].cpu()
                metrics.update(geometry([(sid, xc, yc)], [dict(backbone=bb.tolist())])['summary'])
                metrics.update(close_contacts(bb, yc['mask'], xc))
                assert all(math.isfinite(v) for v in metrics.values()), (sid,name)
                case[name] = metrics
            cases.append(case)
            dump(out/'status.json', dict(phase='evaluating', completed=len(cases), n=len(ids), test_access=False))
    keys = list(cases[0]['baseline'])
    subsets = dict(all=cases, old_pilot=[c for c in cases if c['old_pilot']], new_to_pilot=[c for c in cases if not c['old_pilot']])
    summaries = {}
    for label, subset in subsets.items():
        if not subset: continue
        metrics = {name:{k:st.mean(c[name][k] for c in subset) for k in keys} for name in classes}
        contrasts = {}
        for name, other in [('projected','baseline'),('trained3','baseline'),('trained3','projected')]:
            contrasts[name+'_minus_'+other] = {k:paired(np.array([c[name][k]-c[other][k] for c in subset]),
                [c['cluster'] for c in subset]) for k in ['aligned_rmsd','rmsd','contact_f1','interface_fape','cn_mae','clash','aar','omega_planarity_deviation_deg']}
        summaries[label] = dict(n=len(subset), n_h3_clusters=len({c['cluster'] for c in subset}), metrics=metrics, paired_contrasts=contrasts,
            native_close_contacts={k:st.mean(c['native_close_contacts'][k] for c in subset) for k in cases[0]['native_close_contacts']})
    dump(out/'comparison.json', dict(subsets=summaries, per_complex=cases, limitations=plan['limitations']))
    dump(out/'status.json', dict(phase='complete', n=len(ids), old_pilot_n=len(old_ids), new_to_pilot_n=len(ids)-len(old_ids), test_access=False))
    print(json.dumps({label:dict(n=s['n'],metrics={name:{k:v[k] for k in ['aligned_rmsd','rmsd','contact_f1','interface_fape','cn_mae','clash','aar','omega_planarity_deviation_deg','short_context_pairs','any_short_pair']} for name,v in s['metrics'].items()}) for label,s in summaries.items()},indent=2))


if __name__ == '__main__':
    main()
