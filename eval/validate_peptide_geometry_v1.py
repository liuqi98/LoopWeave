"""Fixed-setting geometry pilot on metadata-screened human validation targets.

Never reads a test tensor or fits weights. This restricted validation subset is
not an independent benchmark or representative of the full validation set.
"""
import concurrent.futures
import json
import math
from pathlib import Path
import statistics
import sys
import time

import requests
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train.train_abgaze_reconstructed import sha, dump, move
from train.compare_two_stage_binding import FeatureRows
from persistent_loop_pair_refiner import PersistentLoopPairRefiner
from peptide_geometry_refiner import PeptideConstrainedLoopPairRefiner
from abgaze_reconstructed_objective import objective
from eval.test_abgaze_reconstructed_ours import aligned_rmsd
from eval.evaluate_external_temporal_v1 import geometry


# Explicit benign protein-target scope, matched by deposited description, then
# confirmed by source taxonomy. No prediction metric participates in selection.
DESCRIPTIONS = {
    'programmed cell death 1 ligand 1', 'programmed cell death protein 1',
    'tumor necrosis factor receptor superfamily member 10b',
    'proprotein convertase subtilisin/kexin type 9',
    'interleukin-13 receptor subunit alpha-1',
    'tumor necrosis factor ligand superfamily member 13b',
    'integrin beta-8', 'integrin alpha-v', 'interleukin-6',
    'interleukin-23 subunit alpha', 'interleukin-12 subunit beta',
    'disintegrin and metalloproteinase domain-containing protein 10',
}


def read(p):
    return json.loads(p.read_text())


def main():
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    parent = ROOT/'outputs/full_pipeline_seed2026_v1'
    pp = read(parent/'plan.json')
    assert read(parent/'status.json')['phase'] == 'complete'
    for name, digest in pp['code_sha256'].items():
        assert sha(ROOT/name) == digest, name
    config_path = ROOT/'experiments/peptide_geometry_candidate_v1.json'
    config = read(config_path)['projection_config']
    out = ROOT/'outputs/peptide_geometry_human_valid_v1'
    out.mkdir(exist_ok=False)
    (out/'metadata').mkdir()
    dump(out/'status.json', dict(phase='metadata_review', test_access=False))
    source = ROOT/'data/abgaze_reconstructed_v1/splits/valid.json'
    records = [json.loads(line) for line in source.read_text().splitlines()]
    assert [r['pdb'] for r in records] == pp['ids']['valid']
    entities, annotation_hashes = {}, {}
    for path in sorted((ROOT/'outputs/independent_domain_annotations_v1/batches').glob('*.json')):
        annotation_hashes[str(path.relative_to(ROOT))] = sha(path)
        for entry in read(path)['data']['entries']:
            if entry:
                entities[entry['rcsb_id'].lower()] = entry.get('polymer_entities') or []
    decisions, selected = [], {}
    for record in records:
        sid = record['pdb']
        chains = set(record['antigen_chains'])
        matched = []
        covered = set()
        for entity in entities.get(sid, []):
            mapped = set((entity.get('rcsb_polymer_entity_container_identifiers') or {}).get('auth_asym_ids') or []) & chains
            if mapped:
                covered.update(mapped)
                matched.append(entity)
        eligible = bool(chains) and chains == covered and all(
            (e.get('rcsb_polymer_entity') or {}).get('pdbx_description', '').lower() in DESCRIPTIONS for e in matched)
        decisions.append(dict(sample_id=sid, preliminary_eligible=eligible,
                              reason='pending_taxonomy' if eligible else 'outside_explicit_scope_or_incomplete_mapping'))
        if eligible:
            selected[sid] = [e['rcsb_id'] for e in matched]
    targets = sorted({e for values in selected.values() for e in values})

    def fetch(eid):
        pdb, entity = eid.split('_')
        url = f'https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb}/{entity}'
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            value = response.json()
            dump(out/'metadata'/f'{eid}.json', value)
            tax = {s['ncbi_taxonomy_id'] for s in value.get('rcsb_entity_source_organism') or []}
            assert value['rcsb_id'] == eid
            description = value.get('rcsb_polymer_entity', {}).get('pdbx_description', '').lower()
            passed = tax == {9606} and description in DESCRIPTIONS
            return eid, dict(passed=passed, tax_ids=sorted(tax), description=description, url=url,
                             sha256=sha(out/'metadata'/f'{eid}.json'))
        except Exception as exc:
            return eid, dict(passed=False, error=repr(exc), url=url)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        taxonomy = dict(pool.map(fetch, targets))
    ids = [r['pdb'] for r in records if r['pdb'] in selected and all(taxonomy[e]['passed'] for e in selected[r['pdb']])]
    assert ids, 'No verified human validation targets'
    for decision in decisions:
        sid = decision['sample_id']
        decision['retained'] = sid in ids
        if sid in selected:
            decision['entities'] = selected[sid]
            decision['reason'] = 'verified_human_target' if sid in ids else 'taxonomy_or_metadata_not_verified'
    dump(out/'scope_audit.json', dict(decisions=decisions, taxonomy=taxonomy,
         allowed_descriptions=sorted(DESCRIPTIONS), allowed_taxonomy=[9606], selected_ids=ids,
         validation_source_sha256=sha(source), annotation_sha256=annotation_hashes,
         performance_used_for_selection=False))
    features = ROOT/'data/abgaze_binding_features_finalized_v1'
    fp = read(ROOT/'outputs/two_stage_binding60_gpu_v1/plan.json')
    assert sha(features/'plan.json') == pp['feature_plan_sha256']
    for name, digest in read(features/'plan.json')['code_sha256'].items():
        assert sha(ROOT/name) == digest, name
    cache_path = parent/'valid_cache.pt'
    raw = torch.load(cache_path, map_location='cpu', weights_only=False)
    assert [r[0] for r in raw] == pp['ids']['valid']
    raw = [r for r in raw if r[0] in ids]
    assert [r[0] for r in raw] == ids
    data = FeatureRows(raw, features, 'valid', fp['feature_sha256'])
    checkpoint = parent/'pair/best.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert state['plan_sha256'] == sha(parent/'plan.json')
    feature_hashes = {}
    for sid in ids:
        path = features/'valid'/sid/'features.pt'
        feature_hashes[sid] = sha(path)
        assert feature_hashes[sid] == fp['feature_sha256'][f'valid/{sid}']
        metadata = torch.load(path, map_location='cpu', weights_only=False)['metadata']
        assert metadata['antigen_only'] and not metadata['native_h3_used']
        assert not metadata['native_contact_crop']
        assert metadata['pdb_sha256'] == sha(ROOT/'data/abgaze_reconstructed_v1/pdb'/f'{sid}.pdb')
    plan = dict(n=len(ids), ids=ids, split='restricted_human_validation_subset', seed=2026,
        model_config=state['model_config'], projection_config=config, checkpoint_epoch=state['epoch'],
        checkpoint_sha256=sha(checkpoint), parent_plan_sha256=sha(parent/'plan.json'),
        cache_sha256=sha(cache_path), feature_sha256=feature_hashes,
        candidate_specification_sha256=sha(config_path), scope_audit_sha256=sha(out/'scope_audit.json'),
        code_sha256={f:sha(ROOT/f) for f in sorted(set(pp['code_sha256']) | {
            'peptide_geometry_refiner.py', 'eval/validate_peptide_geometry_v1.py',
            'eval/evaluate_external_temporal_v1.py', 'eval/audit_loop_pair_backbone.py',
            'train/compare_two_stage_binding.py'})},
        settings_selected_before_scoring=True, optimizer_steps=0, test_access=False,
        decision_rule='Report all metrics and all retained samples; no tuning, no best-case selection.',
        limitations=['One pre-existing checkpoint, fixed settings; no training or independent test.',
                    'Metadata-screened human validation targets, not representative of the full validation distribution.',
                    'Samples can be correlated; do not infer statistical significance from this pilot.',
                    'Projection may improve constraint energy without improving RMSD, contacts, or binding.',
                    'Same logits by construction; no AAR improvement expected without training.'])
    dump(out/'plan.json', plan)
    baseline = PersistentLoopPairRefiner(**state['model_config']).to(device).eval()
    candidate = PeptideConstrainedLoopPairRefiner(**state['model_config'], **config).to(device).eval()
    for model in [baseline, candidate]:
        model.load_state_dict(state['model'], strict=True)
    cases = []
    with torch.no_grad():
        for index in range(len(data)):
            sid, xc, yc, _ = data[index]
            x, y = move(xc, device), move(yc, device)
            predictions = {}
            timings = {}
            for name, model in [('baseline', baseline), ('candidate', candidate)]:
                torch.cuda.synchronize()
                tick = time.perf_counter()
                predictions[name] = model(x)
                torch.cuda.synchronize()
                timings[name] = time.perf_counter()-tick
            assert torch.equal(predictions['baseline']['sequence_logits'], predictions['candidate']['sequence_logits'])
            torch.testing.assert_close(predictions['baseline']['backbone'], predictions['candidate']['backbone_before_peptide_projection'])
            result = dict(sample_id=sid, length=len(y['bb']), seconds=timings)
            for name, p in predictions.items():
                _, metrics = objective(p, x, y)
                mask = y['mask'][:, 1]
                metrics['aligned_rmsd'] = aligned_rmsd(p['backbone'][mask, 1].cpu(), y['bb'][mask, 1].cpu())
                metrics['projection_energy'] = float(candidate.peptide_projection.energy(p['backbone'], x['anchors'], x['anchor_mask']))
                # The helper consumes in-memory coordinates only; it opens no external/test files.
                g = geometry([(sid, xc, yc)], [dict(backbone=p['backbone'].cpu().tolist())])['summary']
                metrics.update(g)
                assert all(math.isfinite(v) for v in metrics.values()), sid
                result[name] = metrics
            assert result['candidate']['projection_energy'] <= result['baseline']['projection_energy'] + 1e-6
            result['delta'] = {k:result['candidate'][k]-result['baseline'][k] for k in result['baseline']}
            cases.append(result)
            dump(out/'status.json', dict(phase='validating', completed=len(cases), n=len(ids), test_access=False))
    summary = {name:{k:statistics.mean(r[name][k] for r in cases) for k in cases[0][name]}
               for name in ['baseline', 'candidate', 'delta']}
    changes = {k:dict(lower=sum(r['delta'][k] < -1e-8 for r in cases),
                      equal=sum(abs(r['delta'][k]) <= 1e-8 for r in cases),
                      higher=sum(r['delta'][k] > 1e-8 for r in cases)) for k in summary['delta']}
    dump(out/'comparison.json', dict(n=len(ids), summary=summary, changes=changes, per_complex=cases,
        limitations=plan['limitations']))
    dump(out/'status.json', dict(phase='complete', n=len(ids), optimizer_steps=0, test_access=False))
    print(json.dumps(dict(n=len(ids), summary=summary, changes=changes), indent=2))


if __name__ == '__main__':
    main()
