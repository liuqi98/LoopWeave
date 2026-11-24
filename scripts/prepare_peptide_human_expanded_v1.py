"""Expand only the original training split using explicit benign human targets."""
import concurrent.futures
import json
from pathlib import Path
import sys
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.validate_peptide_geometry_v1 import DESCRIPTIONS, read
from train.train_abgaze_reconstructed import sha, dump

ALLOWED = DESCRIPTIONS | {
    'lysozyme c', 'epidermal growth factor receptor', 'vascular endothelial growth factor a',
    'receptor tyrosine-protein kinase erbb-2', 'receptor tyrosine-protein kinase erbb-3',
    'b-lymphocyte antigen cd20', 'leukocyte surface antigen cd47',
    'tyrosine-protein phosphatase non-receptor type substrate 1',
    't-cell surface glycoprotein cd4', 'cadherin-3', 'roundabout homolog 1',
    'butyrophilin subfamily 3 member a1', 'complement factor d',
    'granulocyte-macrophage colony-stimulating factor', 'transforming growth factor beta-2',
    'interleukin-2', 'interleukin-13', 'interleukin-17a', 'tumor necrosis factor',
    'tumor necrosis factor receptor superfamily member 5', 'stromal cell-derived factor 1',
    'insulin', 'insulin-degrading enzyme', 'insulin receptor domains l1-cr',
    'integrin beta-1', 'integrin beta-3', 'tissue factor',
    'urokinase plasminogen activator surface receptor', 'prostate-specific antigen',
    'ig epsilon chain c region', 'hla class i histocompatibility antigen, a-2 alpha chain',
    'histone h4', 'histone h2b type 1-j', 'histone h2a type 1-b/e',
    'histone chaperone asf1', 'e3 sumo-protein ligase ranbp2', 'beta-arrestin-1',
    'sodium-dependent serotonin transporter', 'sodium-dependent dopamine transporter',
    'gamma-aminobutyric acid receptor subunit alpha-1',
    'glutamate receptor ionotropic, nmda 2b', 'glutamate receptor ionotropic, nmda 1',
    'atp-binding cassette sub-family g member 2', 'atp-binding cassette sub-family g member 8',
    'atp-binding cassette sub-family g member 5', 'multidrug resistance protein 1',
    'adiponectin receptor protein 2', 'potassium channel subfamily k member 4',
    'isoform 2 of potassium channel subfamily k member 4', 'atypical chemokine receptor 3',
    'gastric inhibitory polypeptide receptor', 'isoform 2 of cystinosin',
    'sodium/bile acid cotransporter', 'isoform 2 of protein-serine o-palmitoleoyltransferase porcupine',
    'guanine nucleotide-binding protein g(i)/g(s)/g(t) subunit beta-1',
    'guanine nucleotide-binding protein g(i) subunit alpha-1',
    'guanine nucleotide-binding protein g(i)/g(s)/g(o) subunit gamma-2',
    'guanine nucleotide-binding protein g(o) subunit alpha',
    'guanine nucleotide-binding protein g(q) subunit alpha',
    'guanine nucleotide-binding protein g(i) subunit alpha-3',
}


def main():
    out = ROOT/'outputs/peptide_human_expanded_scope_v1'
    out.mkdir(exist_ok=False)
    (out/'metadata').mkdir()
    source = ROOT/'data/abgaze_reconstructed_v1/splits/train.json'
    records = [json.loads(line) for line in source.read_text().splitlines()]
    parent = read(ROOT/'outputs/full_pipeline_seed2026_v1/plan.json')
    assert [r['pdb'] for r in records] == parent['ids']['train']
    valid = read(ROOT/'outputs/peptide_geometry_human_valid_v1/plan.json')['ids']
    old_scope_path = ROOT/'outputs/peptide_geometry_human_train_pilot_v1/scope_audit.json'
    old_scope = read(old_scope_path)
    old_metadata = ROOT/'outputs/peptide_geometry_human_train_pilot_v1/metadata'
    entries, provenance = {}, {}
    for path in sorted((ROOT/'outputs/independent_domain_annotations_v1/batches').glob('*.json')):
        provenance[str(path.relative_to(ROOT))] = sha(path)
        for entry in read(path)['data']['entries']:
            if entry:
                entries[entry['rcsb_id'].lower()] = entry.get('polymer_entities') or []
    candidates, decisions = {}, []
    for r in records:
        sid, chains = r['pdb'], set(r['antigen_chains'])
        matched, covered = [], set()
        for e in entries.get(sid, []):
            overlap = chains & set((e.get('rcsb_polymer_entity_container_identifiers') or {}).get('auth_asym_ids') or [])
            if overlap:
                covered.update(overlap)
                matched.append(e)
        ok = bool(chains) and covered == chains and all(
            (e.get('rcsb_polymer_entity') or {}).get('pdbx_description', '').lower() in ALLOWED for e in matched)
        decisions.append(dict(sample_id=sid, preliminary_eligible=ok))
        if ok:
            candidates[sid] = [e['rcsb_id'] for e in matched]
    targets = sorted({eid for eids in candidates.values() for eid in eids})
    dump(out/'plan.json', dict(allowed_descriptions=sorted(ALLOWED), allowed_taxonomy=[9606],
        original_train_sha256=sha(source), original_scope_sha256=sha(old_scope_path),
        unchanged_valid_ids=valid, candidate_ids=list(candidates), entity_ids=targets,
        annotation_sha256=provenance, script_sha256=sha(Path(__file__)), performance_filter=False))

    def fetch(eid):
        pdb, entity = eid.split('_')
        url = f'https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb}/{entity}'
        try:
            cached = old_metadata/f'{eid}.json'
            if cached.exists():
                assert sha(cached) == old_scope['taxonomy'][eid]['sha256']
                value = read(cached)
            else:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                value = response.json()
            assert value['rcsb_id'] == eid
            dump(out/'metadata'/f'{eid}.json', value)
            tax = {s['ncbi_taxonomy_id'] for s in value.get('rcsb_entity_source_organism') or []}
            name = value.get('rcsb_polymer_entity', {}).get('pdbx_description', '').lower()
            chains = value['rcsb_polymer_entity_container_identifiers']['auth_asym_ids']
            return eid, dict(passed=tax == {9606} and name in ALLOWED, tax_ids=sorted(tax),
                description=name, chains=chains, url=url, sha256=sha(out/'metadata'/f'{eid}.json'))
        except Exception as exc:
            return eid, dict(passed=False, error=repr(exc), url=url)

    taxonomy = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch, eid):eid for eid in targets}
        for future in concurrent.futures.as_completed(futures):
            eid, result = future.result()
            taxonomy[eid] = result
            dump(out/'status.json', dict(phase='metadata_review', completed=len(taxonomy), total=len(targets)))
    ids = []
    for r, decision in zip(records, decisions):
        sid = r['pdb']
        ok = sid in candidates and all(taxonomy[e]['passed'] for e in candidates[sid])
        if ok:
            ok = set(r['antigen_chains']) <= {c for e in candidates[sid] for c in taxonomy[e]['chains']}
        decision['retained'] = ok
        decision['reason'] = 'verified_human_target' if ok else 'outside_scope_or_unverified'
        if sid in candidates:
            decision['entities'] = candidates[sid]
        if ok:
            ids.append(sid)
    assert set(old_scope['selected_ids']) <= set(ids)
    assert len(ids) > len(old_scope['selected_ids'])
    assert not set(ids) & set(valid)
    dump(out/'scope_audit.json', dict(selected_ids=ids, valid_ids=valid, decisions=decisions, taxonomy=taxonomy,
        original_count=len(old_scope['selected_ids']), expanded_count=len(ids),
        metadata_failures=sum('error' in v for v in taxonomy.values()),
        limitation='Explicit screened human target subset, not the full training distribution; original split membership retained.'))
    dump(out/'status.json', dict(phase='complete', train=len(ids), valid=len(valid),
        metadata_failures=sum('error' in v for v in taxonomy.values())))
    print((out/'status.json').read_text())


if __name__ == '__main__':
    main()
