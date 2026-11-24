"""Fixed epoch10 primary and epoch3 secondary on the previously seen external20.

No checkpoint selection on this cohort. This is diagnostic re-evaluation, not
independent confirmation, because earlier cohort results informed development.
"""
import json
from pathlib import Path
import statistics as st
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from train.train_abgaze_reconstructed import sha,dump,move
from frame_relation_refiner import FrameRelationGraph,FrameRelationRefiner
from peptide_geometry_refiner import PeptideConstrainedLoopPairRefiner
from eval.evaluate_external_temporal_v1 import geometry
from eval.test_loop_pair_replicates_v1 import score
from eval.audit_loop_pair_paper import paired


def read(p):return json.loads(p.read_text())


def main():
    torch.set_num_threads(1);torch.manual_seed(2026)
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    live=ROOT/'outputs/peptide_geometry_380_continue_v1'
    assert read(live/'status.json')['phase']=='complete'
    selection=read(live/'selection.json')
    assert selection['best_guarded_epoch']==10 and selection['best_structure_epoch']==3
    plan=read(live/'plan.json')
    for name,digest in plan['code_sha256'].items():assert sha(ROOT/name)==digest,name
    prior=ROOT/'outputs/external_temporal_evaluation_v1'
    lock=read(prior/'evaluation_lock.json');ids=lock['ids']
    assert len(ids)==20
    data=ROOT/'data/independent_nonpathogen_inputs_v1'
    features=ROOT/'data/independent_nonpathogen_binding_v1'
    assert sha(data/'manifest.json')==lock['input_manifest_sha256']
    for name,digest in read(data/'manifest.json')['files'].items():assert sha(data/name)==digest,name
    for sid,digest in lock['feature_sha256'].items():assert sha(features/'features'/sid/'features.pt')==digest,sid
    generator_path=ROOT/'outputs/full_pipeline_seed2026_v1/generator/best.pt'
    original_lock=read(ROOT/'outputs/fixed_model_geometry_v1/lock.json')
    for artifacts in original_lock['artifacts'].values():
        for path,digest in artifacts.items():assert sha(ROOT/path)==digest,path
    base_path=prior/'generated_seed2026.json'
    assert sha(base_path)==read(prior/'generation_audit.json')['output_sha256']['2026']
    candidates={'primary_epoch10':live/'checkpoint_010.pt','secondary_epoch3':live/'checkpoint_003.pt'}
    states={}
    for name,path in candidates.items():
        state=torch.load(path,map_location='cpu',weights_only=False)
        assert state['plan_sha256']==sha(live/'plan.json')
        assert state['epoch']==(10 if name=='primary_epoch10' else 3)
        states[name]=state
    out=ROOT/'outputs/peptide_continuation_external_test_v1';out.mkdir(exist_ok=False)
    limitations=lock['limitations']+[
        'Previously inspected external cohort; its earlier results informed development. Not a fresh independent test.',
        'Single seed2026 for original, primary epoch10 and secondary epoch3; do not compare as if these were three-seed means.',
        'Epoch10 has36-sample training for epochs1-3 and380-sample continuation for epochs4-10.',
        'Epoch3 is an explicitly secondary diagnostic comparator, not a test-selected replacement.',
        'No full-distribution retraining or matched continued-training control.']
    dump(out/'evaluation_lock.json',dict(ids=ids,groups=lock['groups'],seed=2026,
        primary='primary_epoch10',secondary='secondary_epoch3',K=5,temperature=.5,sampling_seed=2026,
        external_lock_sha256=sha(prior/'evaluation_lock.json'),training_plan_sha256=sha(live/'plan.json'),
        validation_selection_sha256=sha(live/'selection.json'),generator_sha256=sha(generator_path),
        baseline_generated_sha256=sha(base_path),checkpoints={k:dict(path=str(p),sha256=sha(p)) for k,p in candidates.items()},
        code_sha256={f:sha(ROOT/f) for f in sorted(set(plan['code_sha256'])|{
            'eval/test_peptide_continuation_external_v1.py','eval/test_loop_pair_replicates_v1.py',
            'eval/evaluate_external_temporal_v1.py','eval/audit_loop_pair_paper.py'})},
        failure_policy='Abort; no dropped samples or settings changes.',limitations=limitations))
    lookup={s:(s,x,y) for s,x,y in torch.load(data/'test.pt',map_location='cpu',weights_only=False)}
    rows=[lookup[sid] for sid in ids]
    graphs={item['sample_id']:graph for item,graph in torch.load(data/'graphs.pt',map_location='cpu',weights_only=False)}
    gs=torch.load(generator_path,map_location='cpu',weights_only=False)
    generator=FrameRelationRefiner(**gs['model_config']).to(device);generator.load_state_dict(gs['model']);generator.eval()
    alphabet='ACDEFGHIKLMNPQRSTVWY';predictions={'original_seed2026':read(base_path)}
    for name,state in states.items():
        model=PeptideConstrainedLoopPairRefiner(**state['model_config']).to(device)
        model.load_state_dict(state['model']);model.eval();rng=torch.Generator().manual_seed(2026)
        generated=[]
        with torch.no_grad():
            for sid,x,_ in rows:
                item=torch.load(features/'features'/sid/'features.pt',map_location='cpu',weights_only=False)
                meta=item['metadata'];assert meta['antigen_only'] and not meta['native_h3_used'] and not meta['native_contact_crop']
                assert meta['pdb_sha256']==sha(data/'pdb'/f'{sid}.pdb')
                raw=generator(FrameRelationGraph(**graphs[sid]).to(device))
                inputs=move({**x,**item['features']},device)
                inputs.update(base_bb=raw['h3_backbone'],base_logits=torch.zeros_like(inputs['base_logits']))
                pred=model(inputs);bb=pred['backbone'].cpu();logits=pred['sequence_logits'].cpu()
                assert torch.isfinite(bb).all() and torch.isfinite(logits).all(),sid
                samples=torch.multinomial((logits/.5).softmax(-1),5,replacement=True,generator=rng).T
                generated.append(dict(sample_id=sid,backbone=bb.tolist(),sequence_logits=logits.tolist(),
                    greedy=''.join(alphabet[i] for i in logits.argmax(-1).tolist()),
                    candidates=[''.join(alphabet[i] for i in seq.tolist()) for seq in samples]))
                dump(out/'status.json',dict(phase='generating',candidate=name,completed=len(generated),n=20))
        dump(out/f'generated_{name}.json',generated);predictions[name]=generated
        del model;torch.cuda.empty_cache()
    dump(out/'generation_audit.json',dict(all_candidates_committed_before_scoring=True,labels_not_passed_to_models=True,
        sha256={name:sha(out/f'generated_{name}.json') for name in candidates}))
    results={};geometries={}
    for name,generated in predictions.items():
        assert [g['sample_id'] for g in generated]==ids
        results[name]=score(rows,generated);geometries[name]=geometry(rows,generated)
        dump(out/f'metrics_{name}.json',results[name]);dump(out/f'geometry_{name}.json',geometries[name])
    old=read(prior/'metrics_seed2026.json')['summary']
    assert all(abs(results['original_seed2026']['summary'][k]-v)<1e-7 for k,v in old.items())
    membership={sid:group for group,ss in lock['groups'].items() for sid in ss}
    contrasts={}
    for name in candidates:
        contrasts[name]={}
        for k in results[name]['summary']:
            delta=np.array([a[k]-b[k] for a,b in zip(results[name]['per_complex'],results['original_seed2026']['per_complex'])])
            contrasts[name][k]=paired(delta,[membership[s] for s in ids])
    grouped={name:{k:st.mean(st.mean(r[k] for r in result['per_complex'] if r['sample_id'] in ss)
                      for ss in lock['groups'].values()) for k in result['summary']} for name,result in results.items()}
    dump(out/'comparison.json',dict(n=20,n_groups=len(lock['groups']),seed=2026,
        summary={k:v['summary'] for k,v in results.items()},geometry_summary={k:v['summary'] for k,v in geometries.items()},
        group_balanced=grouped,paired_contrasts_vs_original=contrasts,limitations=limitations,
        primary='primary_epoch10',secondary='secondary_epoch3',checkpoint_selection_changed=False))
    dump(out/'status.json',dict(phase='complete',n=20,failures=0,checkpoint_selection_changed=False))
    print(json.dumps(read(out/'comparison.json')['summary'],indent=2))


if __name__=='__main__':main()
