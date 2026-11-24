"""Frozen three-seed backbone audit; no training or model selection."""
import json
from pathlib import Path
import sys
import statistics
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from train.train_abgaze_reconstructed import sha,dump
from eval.audit_loop_pair_backbone import angle,torsion,describe


def main():
    torch.set_num_threads(1)
    out=ROOT/'outputs/fixed_model_geometry_v1';out.mkdir(exist_ok=False)
    paths={str(s):ROOT/f'outputs/full_pipeline_seed{s}_v1/test/generated.json' for s in (2026,2027,2028)}
    locks={}
    for s,p in paths.items():
        run=p.parent.parent
        locks[s]={str(f.relative_to(ROOT)):sha(f) for f in [run/'generator/best.pt',run/'pair/best.pt',run/'plan.json',p]}
    dump(out/'lock.json',dict(architecture='unchanged full model',seeds=[2026,2027,2028],artifacts=locks,
        policy='No training, relaxation, best-seed selection or architecture changes.',script_sha256=sha(Path(__file__))))
    data=ROOT/'data/abgaze_reconstructed_v1/tensors/test.pt'
    rows=torch.load(data,map_location='cpu',weights_only=False)
    generated={s:{r['sample_id']:r for r in json.loads(p.read_text())} for s,p in paths.items()}
    for g in generated.values():assert set(g)=={r[0] for r in rows}
    report={}
    for name in ['native']+list(paths):
        cases=[];pooled={}
        for sid,x,y in rows:
            bb=y['bb'] if name=='native' else torch.tensor(generated[name][sid]['backbone'])
            assert torch.isfinite(bb).all() and bb.shape==y['bb'].shape
            chain=torch.cat([x['anchors'][:1],bb,x['anchors'][1:]])
            native=torch.cat([x['anchors'][:1],y['bb'],x['anchors'][1:]])
            mask=torch.cat([x['anchor_mask'][:1],y['mask'],x['anchor_mask'][1:]])
            n,ca,c=chain[:,0],chain[:,1],chain[:,2]
            rn,rca,rc=native[:,0],native[:,1],native[:,2]
            values={}
            for key,a,b,ideal in [('n_ca',0,1,1.458),('ca_c',1,2,1.525),('c_o',2,3,1.231)]:
                valid=y['mask'][:,a]&y['mask'][:,b]
                values[key+'_ideal_abs_error_A']=((bb[:,a]-bb[:,b]).norm(dim=-1)-ideal).abs()[valid].tolist()
            valid=mask[:-1,2]&mask[1:,0]
            dist=(c[:-1]-n[1:]).norm(dim=-1)
            err=(dist-1.329).abs()
            values['cn_ideal_abs_error_A']=err[valid].tolist()
            values['cn_error_gt_0_1_fraction']=(err[valid]>.1).float().tolist()
            values['left_anchor_cn_error_A']=err[:1][valid[:1]].tolist()
            values['right_anchor_cn_error_A']=err[-1:][valid[-1:]].tolist()
            values['internal_cn_error_A']=err[1:-1][valid[1:-1]].tolist()
            specs=[('n_ca_c_angle',(n[1:-1],ca[1:-1],c[1:-1]),(rn[1:-1],rca[1:-1],rc[1:-1]),mask[1:-1,:3].all(-1)),
                ('ca_c_n_angle',(ca[:-1],c[:-1],n[1:]),(rca[:-1],rc[:-1],rn[1:]),mask[:-1,1]&mask[:-1,2]&mask[1:,0]),
                ('c_n_ca_angle',(c[:-1],n[1:],ca[1:]),(rc[:-1],rn[1:],rca[1:]),mask[:-1,2]&mask[1:,0]&mask[1:,1]),
                ('phi',(c[:-2],n[1:-1],ca[1:-1],c[1:-1]),(rc[:-2],rn[1:-1],rca[1:-1],rc[1:-1]),mask[:-2,2]&mask[1:-1,:3].all(-1)),
                ('psi',(n[1:-1],ca[1:-1],c[1:-1],n[2:]),(rn[1:-1],rca[1:-1],rc[1:-1],rn[2:]),mask[1:-1,:3].all(-1)&mask[2:,0]),
                ('omega',(ca[:-1],c[:-1],n[1:],ca[1:]),(rca[:-1],rc[:-1],rn[1:],rca[1:]),mask[:-1,1]&mask[:-1,2]&mask[1:,0]&mask[1:,1])]
            for key,pred,truth,valid in specs:
                if len(pred)==3:
                    a,b=angle(*pred),angle(*truth)
                    for coords in (pred,truth):
                        valid=valid&((coords[0]-coords[1]).norm(dim=-1)>1e-6)&((coords[2]-coords[1]).norm(dim=-1)>1e-6)
                    error=(a-b).abs()
                else:
                    a,ok=torsion(*pred);b,rok=torsion(*truth);valid=valid&ok&rok
                    error=((a-b+180)%360-180).abs()
                    values[key+'_deg']=a[valid].tolist()
                    if key=='omega':
                        deviation=torch.minimum(a.abs(),(180-a.abs()).abs())[valid]
                        values['omega_planarity_deviation_deg']=deviation.tolist()
                        values['omega_deviation_gt_20_fraction']=(deviation>20).float().tolist()
                values[key+'_native_abs_error_deg']=error[valid].tolist()
            for k,v in values.items():pooled.setdefault(k,[]).extend(v)
            cases.append(dict(sample_id=sid,length=len(bb),metrics={k:describe(v) for k,v in values.items()}))
        report[name]=dict(pooled={k:describe(v) for k,v in pooled.items()},
            macro_mean={k:statistics.mean(c['metrics'][k]['mean'] for c in cases if c['metrics'][k]['n']) for k in pooled},per_complex=cases)
    dump(out/'metrics.json',report)
    summary={k:dict(mean=statistics.mean(report[s]['macro_mean'][k] for s in paths),
        sample_std=statistics.stdev(report[s]['macro_mean'][k] for s in paths),native=report['native']['macro_mean'][k]) for k in report['native']['macro_mean']}
    dump(out/'summary.json',dict(metrics=summary,test_sha256=sha(data),limitations=[
        'Backbone only; no full-atom MolProbity clashscore or Ramachandran classification.',
        'Native-relative torsion errors are not Ramachandran outlier rates.',
        '0.1 A bond and 20 degree planarity thresholds are descriptive, not calibrated validation criteria.',
        'Anchor linkage includes both junctions; original native missing-atom masks used.',
        'Existing development test, not independent generalization evidence.']))
    dump(out/'status.json',dict(phase='complete',n=len(rows),seeds=list(paths)))
    print(json.dumps(summary))


if __name__=='__main__':main()
