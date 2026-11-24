"""Bounded human-target continuation pilot; no test access or baseline mutation."""
import concurrent.futures
import json
import math
from pathlib import Path
import random
import statistics as st
import sys
import time

import requests
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.validate_peptide_geometry_v1 import DESCRIPTIONS, read, FeatureRows, geometry, aligned_rmsd
from train.train_abgaze_reconstructed import sha, dump, save, move
from abgaze_reconstructed_objective import objective
from peptide_geometry_refiner import PeptideConstrainedLoopPairRefiner


def scope(out, parent_plan):
    path = ROOT/'data/abgaze_reconstructed_v1/splits/train.json'
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r['pdb'] for r in records] == parent_plan['ids']['train']
    entities, hashes = {}, {}
    for p in sorted((ROOT/'outputs/independent_domain_annotations_v1/batches').glob('*.json')):
        hashes[str(p.relative_to(ROOT))] = sha(p)
        for entry in read(p)['data']['entries']:
            if entry:
                entities[entry['rcsb_id'].lower()] = entry.get('polymer_entities') or []
    selected, decisions = {}, []
    for record in records:
        sid, chains = record['pdb'], set(record['antigen_chains'])
        covered, matched = set(), []
        for entity in entities.get(sid, []):
            chain_ids = set((entity.get('rcsb_polymer_entity_container_identifiers') or {}).get('auth_asym_ids') or [])
            if chains & chain_ids:
                covered.update(chains & chain_ids)
                matched.append(entity)
        eligible = bool(chains) and covered == chains and all(
            (e.get('rcsb_polymer_entity') or {}).get('pdbx_description', '').lower() in DESCRIPTIONS for e in matched)
        decisions.append(dict(sample_id=sid, preliminary_eligible=eligible,
                              reason='pending_taxonomy' if eligible else 'outside_scope_or_incomplete_mapping'))
        if eligible:
            selected[sid] = [e['rcsb_id'] for e in matched]
    (out/'metadata').mkdir()

    def fetch(eid):
        pdb, entity = eid.split('_')
        url = f'https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb}/{entity}'
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            value = response.json()
            dump(out/'metadata'/f'{eid}.json', value)
            assert value['rcsb_id'] == eid
            tax = {v['ncbi_taxonomy_id'] for v in value.get('rcsb_entity_source_organism') or []}
            description = value.get('rcsb_polymer_entity', {}).get('pdbx_description', '').lower()
            chains = value['rcsb_polymer_entity_container_identifiers']['auth_asym_ids']
            return eid, dict(passed=tax == {9606} and description in DESCRIPTIONS,
                             tax_ids=sorted(tax), description=description, chains=chains,
                             url=url, sha256=sha(out/'metadata'/f'{eid}.json'))
        except Exception as exc:
            return eid, dict(passed=False, error=repr(exc), url=url)

    targets = sorted({eid for values in selected.values() for eid in values})
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        taxonomy = dict(pool.map(fetch, targets))
    ids = []
    for record, decision in zip(records, decisions):
        sid = record['pdb']
        ok = sid in selected and all(taxonomy[e]['passed'] for e in selected[sid])
        if ok:
            fresh_chains = {c for e in selected[sid] for c in taxonomy[e]['chains']}
            ok = set(record['antigen_chains']) <= fresh_chains
        decision['retained'] = ok
        if sid in selected:
            decision['entities'] = selected[sid]
            decision['reason'] = 'verified_human_target' if ok else 'metadata_or_taxonomy_not_verified'
        if ok:
            ids.append(sid)
    assert ids
    dump(out/'scope_audit.json', dict(selected_ids=ids, decisions=decisions, taxonomy=taxonomy,
        train_split_sha256=sha(path), annotation_sha256=hashes,
        allowed_descriptions=sorted(DESCRIPTIONS), allowed_taxonomy=[9606], performance_selection=False))
    return ids


def evaluate(model, rows, device):
    model.eval()
    cases = []
    with torch.no_grad():
        for index in range(len(rows)):
            sid, xc, yc, _ = rows[index]
            x, y = move(xc, device), move(yc, device)
            pred = model(x)
            _, m = objective(pred, x, y)
            mask = y['mask'][:, 1]
            m['aligned_rmsd'] = aligned_rmsd(pred['backbone'][mask, 1].cpu(), y['bb'][mask, 1].cpu())
            m.update(geometry([(sid, xc, yc)], [dict(backbone=pred['backbone'].cpu().tolist())])['summary'])
            assert all(math.isfinite(v) for v in m.values()), sid
            cases.append(dict(sample_id=sid, **m))
    return dict(summary={k:st.mean(r[k] for r in cases) for k in cases[0] if k != 'sample_id'}, per_complex=cases)


def guards(m, baseline):
    return dict(
        aligned_rmsd=m['aligned_rmsd'] <= baseline['aligned_rmsd'] + .01,
        raw_rmsd=m['rmsd'] <= baseline['rmsd'] + .01,
        contact_f1=m['contact_f1'] >= baseline['contact_f1'] - .005,
        aar=m['aar'] >= baseline['aar'] - .01,
        clash=m['clash'] <= baseline['clash'] * 1.1 + 1e-5,
        cn=m['cn_mae'] <= baseline['cn_mae'],
        plane=m['omega_planarity_deviation_deg'] <= baseline['omega_planarity_deviation_deg'])


def main():
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    parent = ROOT/'outputs/full_pipeline_seed2026_v1'
    prior = ROOT/'outputs/peptide_geometry_human_valid_v1'
    pp, vp = read(parent/'plan.json'), read(prior/'plan.json')
    assert read(prior/'status.json')['phase'] == 'complete'
    for name, digest in vp['code_sha256'].items():
        assert sha(ROOT/name) == digest, name
    assert sha(prior/'scope_audit.json') == vp['scope_audit_sha256']
    assert sha(parent/'pair/best.pt') == vp['checkpoint_sha256']
    out = ROOT/'outputs/peptide_geometry_human_train_pilot_v1'
    out.mkdir(exist_ok=False)
    dump(out/'status.json', dict(phase='metadata_review', test_access=False))
    ids = dict(train=scope(out, pp), valid=vp['ids'])
    assert not set(ids['train']) & set(ids['valid'])
    features = ROOT/'data/abgaze_binding_features_finalized_v1'
    fp = read(ROOT/'outputs/two_stage_binding60_gpu_v1/plan.json')
    assert sha(features/'plan.json') == pp['feature_plan_sha256']
    for name, digest in read(features/'plan.json')['code_sha256'].items():
        assert sha(ROOT/name) == digest, name
    data, cache_hashes, feature_hashes = {}, {}, {}
    for split in ['train', 'valid']:
        cache = parent/f'{split}_cache.pt'
        cache_hashes[split] = sha(cache)
        raw = torch.load(cache, map_location='cpu', weights_only=False)
        assert [r[0] for r in raw] == pp['ids'][split]
        lookup = {r[0]:r for r in raw}
        for sid in ids[split]:
            path = features/split/sid/'features.pt'
            digest = sha(path)
            assert digest == fp['feature_sha256'][f'{split}/{sid}']
            feature_hashes[f'{split}/{sid}'] = digest
            meta = torch.load(path, map_location='cpu', weights_only=False)['metadata']
            assert meta['antigen_only'] and not meta['native_h3_used'] and not meta['native_contact_crop']
            assert meta['pdb_sha256'] == sha(ROOT/'data/abgaze_reconstructed_v1/pdb'/f'{sid}.pdb')
        data[split] = FeatureRows([lookup[sid] for sid in ids[split]], features, split, feature_hashes)
    state = torch.load(parent/'pair/best.pt', map_location='cpu', weights_only=False)
    assert state['plan_sha256'] == sha(parent/'plan.json')
    config = dict(**state['model_config'], **vp['projection_config'])
    plan = dict(seed=2026, epochs=3, lr=1e-5, accumulate=16, clip_norm=1., early_stopping=False,
        ids=ids, counts={s:len(v) for s,v in ids.items()}, model_class='PeptideConstrainedLoopPairRefiner',
        model_config=config, checkpoint_sha256=vp['checkpoint_sha256'], parent_plan_sha256=sha(parent/'plan.json'),
        scope_audit_sha256=sha(out/'scope_audit.json'), valid_scope_sha256=sha(prior/'scope_audit.json'),
        initial_pilot_plan_sha256=sha(prior/'plan.json'), initial_comparison_sha256=sha(prior/'comparison.json'),
        cache_sha256=cache_hashes, feature_sha256=feature_hashes,
        code_sha256={f:sha(ROOT/f) for f in sorted(set(vp['code_sha256']) | {'train/train_peptide_geometry_human_pilot_v1.py'})},
        trainable='All existing refiner parameters; generator frozen via prior caches; projection has no learned parameters.',
        objective='Unchanged geometry + 0.5 interface FAPE + sequence CE; gradients through final projection.',
        selection='Minimum equal-weight normalized CN MAE and omega planarity deviation under guards; epoch0 eligible; retain baseline if none eligible.',
        guards=dict(aligned_rmsd_max_increase=.01, raw_rmsd_max_increase=.01, contact_f1_max_drop=.005,
                    aar_max_drop=.01, clash_multiplier=1.1, clash_allowance=1e-5, cn_no_increase=True, plane_no_increase=True),
        limitations=['Small scoped human-target continuation, not full-distribution retraining or independent confirmation.',
                     'Single training seed, no matched additional-training control; cannot attribute changes solely to projection.',
                     'Validation already used for prior checkpoint selection; no test access.',
                     'Only three bounded epochs; no convergence claim.'], test_access=False)
    dump(out/'plan.json', plan)
    ph = sha(out/'plan.json')
    model = PeptideConstrainedLoopPairRefiner(**config).to(device)
    model.load_state_dict(state['model'], strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=plan['lr'])
    baseline = read(prior/'comparison.json')['summary']['baseline']
    initial = evaluate(model, data['valid'], device)
    for k, value in read(prior/'comparison.json')['summary']['candidate'].items():
        if k in initial['summary']:
            assert math.isclose(value, initial['summary'][k], abs_tol=1e-5), k
    dump(out/'initial_validation.json', initial)
    best, best_epoch = float('inf'), None
    evaluations = []

    def record(epoch, validation, train_metrics=None, seconds=0):
        nonlocal best, best_epoch
        m = validation['summary']
        checks = guards(m, baseline)
        score = .5*(m['cn_mae']/baseline['cn_mae'] + m['omega_planarity_deviation_deg']/baseline['omega_planarity_deviation_deg'])
        improved = all(checks.values()) and score < best
        if improved:
            best, best_epoch = score, epoch
        checkpoint = dict(model_class=plan['model_class'], model_config=config, model=model.state_dict(),
                          optimizer=optimizer.state_dict(), epoch=epoch, plan_sha256=ph, validation=m,
                          best_epoch=best_epoch, best_score=best if math.isfinite(best) else None)
        save(out/'last.pt', checkpoint)
        if improved:
            save(out/'best.pt', checkpoint)
        result = dict(epoch=epoch, validation=validation, train=train_metrics,
                      guards=checks, eligible=all(checks.values()), selection_score=score,
                      best_epoch=best_epoch, seconds=seconds)
        evaluations.append(result)
        dump(out/f'epoch_{epoch:03d}.json', result)
        print(json.dumps(dict(epoch=epoch, summary=m, guards=checks, best_epoch=best_epoch, seconds=seconds)), flush=True)

    record(0, initial)
    steps = 0
    for e in range(1, plan['epochs']+1):
        start = time.monotonic()
        model.train()
        torch.manual_seed(2026+e)
        order = list(range(len(data['train'])))
        random.Random(2026+e).shuffle(order)
        train_metrics = []
        for offset in range(0, len(order), plan['accumulate']):
            chunk = order[offset:offset+plan['accumulate']]
            optimizer.zero_grad(set_to_none=True)
            for index in chunk:
                sid, xc, yc, _ = data['train'][index]
                x, y = move(xc, device), move(yc, device)
                prediction = model(x)
                loss, metrics = objective(prediction, x, y)
                assert torch.isfinite(loss) and all(math.isfinite(v) for v in metrics.values()), sid
                (loss/len(chunk)).backward()
                train_metrics.append(metrics)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), plan['clip_norm'])
            assert torch.isfinite(norm), 'Nonfinite training gradient'
            optimizer.step()
            steps += 1
            dump(out/'status.json', dict(phase='training', epoch=e, epochs=plan['epochs'],
                seen=len(train_metrics), total=len(order), optimizer_steps=steps, test_access=False))
        val = evaluate(model, data['valid'], device)
        tr = {k:st.mean(m[k] for m in train_metrics) for k in train_metrics[0]}
        record(e, val, tr, time.monotonic()-start)
    dump(out/'comparison.json', dict(baseline=baseline, initial_projected=initial['summary'],
        epochs=evaluations, best_epoch=best_epoch,
        selected_validation=None if best_epoch is None else evaluations[best_epoch]['validation']['summary'],
        baseline_replaced=False, limitations=plan['limitations']))
    assert sha(parent/'pair/best.pt') == plan['checkpoint_sha256']
    dump(out/'status.json', dict(phase='complete', epochs=plan['epochs'], counts=plan['counts'],
        optimizer_steps=steps, best_epoch=best_epoch, baseline_replaced=False, test_access=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        out = ROOT/'outputs/peptide_geometry_human_train_pilot_v1'
        if out.exists():
            dump(out/'failure.json', dict(error=repr(exc), no_automatic_retry=True))
        raise
