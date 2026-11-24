"""All three fixed RAAD replicas on the already locked temporal cohort."""
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from train.train_abgaze_reconstructed import sha,dump
from eval.test_loop_pair_replicates_v1 import score,aggregate
from eval.evaluate_external_temporal_v1 import geometry
from eval.audit_loop_pair_paper import paired


def read(p):return json.loads(p.read_text())


def main():
    torch.set_num_threads(1)
    out=ROOT/'outputs/external_raad_comparison_v1';out.mkdir(exist_ok=False)
    own=ROOT/'outputs/external_temporal_evaluation_v1';external=read(own/'evaluation_lock.json')
    assert read(own/'status.json')['phase']=='complete'
    runs={2026:ROOT/'outputs/raad_abgaze_reconstructed_seed2026_v1',
        2027:ROOT/'outputs/raad_reconstructed_seed2027_v2',2028:ROOT/'outputs/raad_reconstructed_seed2028_v2'}
    locks={}
    for s,p in runs.items():
        status=read(p/'status.json');assert status['phase']=='complete'
        state=torch.load(p/'best.pt',map_location='cpu',weights_only=False)
        old=ROOT/('outputs/raad_abgaze_reconstructed_test_v1' if s==2026 else f'outputs/raad_reconstructed_seed{s}_test_v2')
        assert sha(p/'best.pt')==read(old/'plan.json')['checkpoint_sha256']
        plan=read(p/'plan.json')
        ownplan=read(ROOT/f'outputs/full_pipeline_seed{s}_v1/plan.json')
        for split in ('train','valid'):assert plan['ids'][split]==ownplan['ids'][split]
        locks[str(s)]=dict(run=str(p),epoch=state['epoch'],checkpoint_sha256=sha(p/'best.pt'),plan_sha256=sha(p/'plan.json'))
    dataset=ROOT/'data/independent_nonpathogen_inputs_v1';ids=external['ids'];groups=external['groups']
    manifest=read(dataset/'manifest.json')
    for name,digest in manifest['files'].items():assert sha(dataset/name)==digest
    lock=dict(ids=ids,groups=groups,runs=locks,graphs_sha256=sha(dataset/'graphs.pt'),
        test_sha256=sha(dataset/'test.pt'),external_lock_sha256=sha(own/'evaluation_lock.json'),
        code_sha256={f:sha(ROOT/f) for f in ['baselines/generate_external_raad_v1.py','eval/compare_external_raad_v1.py']},
        sampling_seed=2026,K=5,temperature=.5,failure_policy='Abort, never silently drop samples',
        limitations=external['limitations']+['Same split and retained backbone context; ours additionally uses surface/atom features and different training compute.'])
    dump(out/'evaluation_lock.json',lock)
    for s in runs:
        dump(out/'status.json',dict(phase='generating',seed=s))
        with (out/f'seed{s}.log').open('x') as log:
            subprocess.run([sys.executable,str(ROOT/'baselines/generate_external_raad_v1.py'),'--lock',str(out/'evaluation_lock.json'),'--seed',str(s)],
                cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
    dump(out/'generation_audit.json',dict(all_three_outputs_committed_before_scoring=True,
        generated_sha256={str(s):sha(out/f'generated_seed{s}.json') for s in runs}))
    rowsbyid={s:(s,x,y) for s,x,y in torch.load(dataset/'test.pt',map_location='cpu',weights_only=False)}
    rows=[rowsbyid[s] for s in ids];results={};geom={};ours={}
    for s in runs:
        generated=read(out/f'generated_seed{s}.json');results[s]=score(rows,generated);geom[s]=geometry(rows,generated)
        dump(out/f'metrics_seed{s}.json',results[s]);dump(out/f'geometry_seed{s}.json',geom[s])
        ours[s]=read(own/f'metrics_seed{s}.json')
        assert [r['sample_id'] for r in ours[s]['per_complex']]==ids
    membership={sid:g for g,ss in groups.items() for sid in ss};contrasts={}
    for k in results[2026]['summary']:
        delta=np.mean([[a[k]-b[k] for a,b in zip(ours[s]['per_complex'],results[s]['per_complex'])] for s in runs],axis=0)
        contrasts[k]=dict(ours_minus_raad=paired(delta,[membership[s] for s in ids]),
            per_seed_delta={str(s):ours[s]['summary'][k]-results[s]['summary'][k] for s in runs})
    group_results={}
    for name,models in [('ours',ours),('raad',results)]:
        group_results[name]={}
        for s,m in models.items():
            lookup={r['sample_id']:r for r in m['per_complex']}
            group_results[name][str(s)]={g:{k:statistics.mean(lookup[i][k] for i in ss) for k in m['summary']} for g,ss in groups.items()}
    case_results=[]
    for i,sid in enumerate(ids):
        r=dict(sample_id=sid,group=membership[sid])
        for name,models in [('ours',ours),('raad',results)]:
            r[name]={k:statistics.mean(m['per_complex'][i][k] for m in models.values()) for k in ('aar','aligned_rmsd','rmsd','contact_f1','cn_mae')}
        case_results.append(r)
    dump(out/'comparison.json',dict(n=len(ids),n_groups=len(groups),seeds=list(runs),
        ours_mean_std=aggregate([m['summary'] for m in ours.values()]),raad_mean_std=aggregate([m['summary'] for m in results.values()]),
        raad_geometry_mean_std=aggregate([m['summary'] for m in geom.values()]),
        paired_contrasts=contrasts,group_results=group_results,per_complex_seed_means=case_results,
        limitations=lock['limitations']+['Cluster intervals condition on these fitted seeds; small cohort, no universal superiority claim.']))
    dump(out/'status.json',dict(phase='complete',n=len(ids),seeds=list(runs),failures=0))
    print(json.dumps(read(out/'comparison.json')['raad_mean_std']))


if __name__=='__main__':main()
