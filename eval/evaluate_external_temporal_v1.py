"""Lock and evaluate a screened temporal cohort; no universal family-independence claim."""
import json
from pathlib import Path
import statistics
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from train.train_abgaze_reconstructed import sha,dump,move
from frame_relation_refiner import FrameRelationGraph,FrameRelationRefiner
from persistent_loop_pair_refiner import PersistentLoopPairRefiner
from eval.test_loop_pair_replicates_v1 import score,aggregate
from eval.audit_loop_pair_paper import paired
from eval.audit_loop_pair_backbone import torsion


def read(p):return json.loads(p.read_text())


def geometry(rows,generated):
    cases=[]
    for (sid,x,y),g in zip(rows,generated):
        bb=torch.tensor(g['backbone']);chain=torch.cat([x['anchors'][:1],bb,x['anchors'][1:]])
        native=torch.cat([x['anchors'][:1],y['bb'],x['anchors'][1:]])
        mask=torch.cat([x['anchor_mask'][:1],y['mask'],x['anchor_mask'][1:]])
        valid=mask[:-1,2]&mask[1:,0];errors=((chain[:-1,2]-chain[1:,0]).norm(dim=-1)-1.329).abs()
        m=dict(sample_id=sid,cn_error_gt_0_1_fraction=float((errors[valid]>.1).float().mean()),
            left_anchor_cn_error_A=float(errors[0]),right_anchor_cn_error_A=float(errors[-1]),
            internal_cn_error_A=float(errors[1:-1][valid[1:-1]].mean()))
        specs={'phi':[(slice(None,-2),2),(slice(1,-1),0),(slice(1,-1),1),(slice(1,-1),2)],
               'psi':[(slice(1,-1),0),(slice(1,-1),1),(slice(1,-1),2),(slice(2,None),0)],
               'omega':[(slice(None,-1),1),(slice(None,-1),2),(slice(1,None),0),(slice(1,None),1)]}
        for key,inds in specs.items():
            a,ok=torsion(*(chain[i,j] for i,j in inds));b,rok=torsion(*(native[i,j] for i,j in inds))
            v=ok&rok
            for i,j in inds:v=v&mask[i,j]
            m[key+'_native_abs_error_deg']=float(((a-b+180)%360-180).abs()[v].mean())
            if key=='omega':m['omega_planarity_deviation_deg']=float(torch.minimum(a.abs(),(180-a.abs()).abs())[v].mean())
        assert all(np.isfinite(v) for k,v in m.items() if k!='sample_id')
        cases.append(m)
    return dict(summary={k:statistics.mean(r[k] for r in cases) for k in cases[0] if k!='sample_id'},per_complex=cases)


def main():
    torch.set_num_threads(1);device=torch.device('cuda:0');torch.cuda.set_device(device)
    out=ROOT/'outputs/external_temporal_evaluation_v1';out.mkdir(exist_ok=False)
    data=ROOT/'data/independent_nonpathogen_inputs_v1'
    features=ROOT/'data/independent_nonpathogen_binding_v1'
    assert read(features/'status.json')['full_coverage']
    excluded={'pdb_000023vf-B-A':'Known CEACAM family overlap',
        'pdb_00009npg-A-B':'Known lipocalin family overlap',
        'pdb_000021rn-C-D':'Animal haem peroxidase family overlap IPR019791',
        'pdb_00009r1f-H-L':'MATE transporter family overlap PF01554 / IPR002528'}
    records=read(data/'records.json');ids=sorted(r['pdb'] for r in records if r['pdb'] not in excluded)
    assert len(ids)==20
    domain=ROOT/'outputs/independent_domain_annotations_v1/report.json';annotations=read(domain)
    lookup={r['instance']:r for r in annotations['candidates']}
    decisions=[]
    for r in records:
        sid=r['pdb'];ags=lookup[sid]['antigens']
        assert all(a['status']=='annotated' for a in ags)
        decisions.append(dict(instance=sid,decision='exclude' if sid in excluded else 'retain_with_scope_limits',
            reason=excluded.get(sid,'No additional established specific family overlap identified in available annotation review; generic folds and whole-entity annotations do not establish bound-domain overlap.'),
            shared_annotations=[dict(entity=a['entity'],id=b['annotation_id'],name=b['name'],type=b['type']) for a in ags for b in a.get('shared_annotations',[])]))
    groupfile=ROOT/'outputs/independent_candidate_groups_v1/report.json'
    groups={g:[s for s in ss if s in ids] for g,ss in read(groupfile)['components'].items()}
    groups={g:ss for g,ss in groups.items() if ss};membership={s:g for g,ss in groups.items() for s in ss}
    assert set(membership)==set(ids)
    model_lock=ROOT/'outputs/fixed_model_geometry_v1/lock.json'
    for artifacts in read(model_lock)['artifacts'].values():
        for path,digest in artifacts.items():assert sha(ROOT/path)==digest,path
    for name,digest in read(data/'manifest.json')['files'].items():assert sha(data/name)==digest,name
    coverage=read(features/'coverage.json')['success'];feature_hashes={s:coverage[s]['sha256'] for s in ids}
    for s,digest in feature_hashes.items():assert sha(features/'features'/s/'features.pt')==digest,s
    limitations=['Screened external temporal cohort, not certified universal antigen-family independence.',
        'Reference annotation gaps: 25 missing entries and 36 historical PDBs with unmatched antigen-chain mappings.',
        'Shared generic folds and non-bound whole-protein annotations retained with explicit limits.',
        'Antibody lineage screening is heuristic; local history audit does not certify every external source.',
        'Nonpathogen mammalian target scope; bound framework pose and H3 length supplied.',
        'Dependence groups are sequence-based, not curated independent antigen families.',
        'Five sampled sequences share one deterministic backbone; oracle AAR is auxiliary only.',
        'Backbone clash proxy and torsion errors are not all-atom MolProbity validation.']
    lock=dict(ids=ids,n=len(ids),groups=groups,seeds=[2026,2027,2028],excluded=excluded,decisions=decisions,
        model_lock_sha256=sha(model_lock),input_manifest_sha256=sha(data/'manifest.json'),
        domain_report_sha256=sha(domain),group_report_sha256=sha(groupfile),feature_sha256=feature_hashes,
        script_sha256=sha(Path(__file__)),K=5,temperature=.5,sampling_seed=2026,
        selection='Fixed old validation-selected checkpoints; all eligibility decisions made before first prediction.',
        aggregation='Per-complex macro mean; sample SD across training seeds; group-balanced means separately.',
        failure_policy='Abort on any failed or nonfinite case; no silent sample removal or denominator changes.',
        strict_family_independence_certified=False,limitations=limitations)
    dump(out/'evaluation_lock.json',lock)
    tensors={s:(s,x,y) for s,x,y in torch.load(data/'test.pt',map_location='cpu',weights_only=False)}
    rows=[tensors[s] for s in ids]
    graphs={i['sample_id']:g for i,g in torch.load(data/'graphs.pt',map_location='cpu',weights_only=False)}
    alphabet='ACDEFGHIKLMNPQRSTVWY';predictions={}
    for seed in lock['seeds']:
        run=ROOT/f'outputs/full_pipeline_seed{seed}_v1';plan=read(run/'plan.json')
        for name,digest in plan['code_sha256'].items():assert sha(ROOT/name)==digest,name
        gs=torch.load(run/'generator/best.pt',map_location='cpu',weights_only=False)
        ms=torch.load(run/'pair/best.pt',map_location='cpu',weights_only=False)
        generator=FrameRelationRefiner(**gs['model_config']).to(device);generator.load_state_dict(gs['model']);generator.eval()
        model=PersistentLoopPairRefiner(**ms['model_config']).to(device);model.load_state_dict(ms['model']);model.eval()
        rng=torch.Generator().manual_seed(2026);generated=[]
        with torch.no_grad():
            for sid,x,_ in rows:
                dump(out/'status.json',dict(phase='generating',seed=seed,sample=sid,completed=len(generated),total=len(ids)))
                item=torch.load(features/'features'/sid/'features.pt',map_location='cpu',weights_only=False)
                meta=item['metadata'];assert meta['antigen_only'] and not meta['native_h3_used'] and not meta['native_contact_crop']
                assert meta['pdb_sha256']==sha(data/'pdb'/f'{sid}.pdb')
                raw=generator(FrameRelationGraph(**graphs[sid]).to(device))
                inputs=move({**x,**item['features']},device)
                inputs.update(base_bb=raw['h3_backbone'],base_logits=torch.zeros_like(inputs['base_logits']))
                p=model(inputs);bb=p['backbone'].cpu();logits=p['sequence_logits'].cpu()
                assert torch.isfinite(bb).all() and torch.isfinite(logits).all(),sid
                samples=torch.multinomial((logits/.5).softmax(-1),5,replacement=True,generator=rng).T
                generated.append(dict(sample_id=sid,backbone=bb.tolist(),sequence_logits=logits.tolist(),
                    greedy=''.join(alphabet[i] for i in logits.argmax(-1).tolist()),
                    candidates=[''.join(alphabet[i] for i in seq.tolist()) for seq in samples]))
        dump(out/f'generated_seed{seed}.json',generated);predictions[seed]=generated
        del model,generator,gs,ms;torch.cuda.empty_cache()
    dump(out/'generation_audit.json',dict(all_seeds_committed_before_scoring=True,labels_not_passed_to_models=True,
        output_sha256={str(s):sha(out/f'generated_seed{s}.json') for s in lock['seeds']}))
    results={};geometries={};grouped={}
    for seed in lock['seeds']:
        results[seed]=score(rows,predictions[seed]);geometries[seed]=geometry(rows,predictions[seed])
        dump(out/f'metrics_seed{seed}.json',results[seed]);dump(out/f'geometry_seed{seed}.json',geometries[seed])
        byid={r['sample_id']:r for r in results[seed]['per_complex']}
        grouped[seed]={g:{k:statistics.mean(byid[s][k] for s in ss) for k in results[seed]['summary']} for g,ss in groups.items()}
    statistics_by_metric={}
    for key in results[2026]['summary']:
        means=np.mean([[r[key] for r in results[s]['per_complex']] for s in lock['seeds']],axis=0)
        ci=paired(means,[membership[s] for s in ids])
        statistics_by_metric[key]=dict(mean=float(means.mean()),cluster_bootstrap_ci95=ci['ci95'],n_groups=ci['n_groups'])
    worst=sorted([dict(sample_id=s,group=membership[s],mean_aligned_rmsd=statistics.mean(results[t]['per_complex'][i]['aligned_rmsd'] for t in lock['seeds']),
        mean_raw_rmsd=statistics.mean(results[t]['per_complex'][i]['rmsd'] for t in lock['seeds'])) for i,s in enumerate(ids)],key=lambda r:r['mean_aligned_rmsd'],reverse=True)
    dump(out/'comparison.json',dict(n=len(ids),n_groups=len(groups),seeds=lock['seeds'],
        by_seed={str(s):r['summary'] for s,r in results.items()},mean_std=aggregate([r['summary'] for r in results.values()]),
        geometry_mean_std=aggregate([r['summary'] for r in geometries.values()]),group_results=grouped,
        group_balanced_mean_std=aggregate([{k:statistics.mean(g[k] for g in grouped[s].values()) for k in results[s]['summary']} for s in lock['seeds']]),
        cluster_intervals_conditional_on_fitted_seeds=statistics_by_metric,worst_structure_cases=worst[:5],limitations=limitations))
    dump(out/'status.json',dict(phase='complete',n=len(ids),seeds=lock['seeds'],failures=0,strict_family_independence_certified=False))
    print(json.dumps(read(out/'comparison.json')['mean_std']))


if __name__=='__main__':
    try:main()
    except Exception as e:
        out=ROOT/'outputs/external_temporal_evaluation_v1'
        if out.exists():dump(out/'failure.json',dict(error=repr(e)))
        raise
