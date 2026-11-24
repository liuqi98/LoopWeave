"""Masked native-relative angles/torsions and peptide planarity; descriptive, not MolProbity."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from train.train_abgaze_reconstructed import dump,sha


def angle(a,b,c):
    return torch.rad2deg(torch.acos((F.normalize(a-b,dim=-1)*F.normalize(c-b,dim=-1)).sum(-1).clamp(-1,1)))


def torsion(a,b,c,d):
    axis=F.normalize(c-b,dim=-1)
    u=a-b;v=d-c
    u=u-(u*axis).sum(-1,keepdim=True)*axis
    v=v-(v*axis).sum(-1,keepdim=True)*axis
    valid=(u.norm(dim=-1)>1e-6)&(v.norm(dim=-1)>1e-6)&((c-b).norm(dim=-1)>1e-6)
    return torch.rad2deg(torch.atan2((torch.cross(axis,u,dim=-1)*v).sum(-1),(u*v).sum(-1))),valid


def describe(values):
    if not values:return dict(n=0)
    a=np.array(values)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),p95=float(np.quantile(a,.95)))


def main():
    torch.set_num_threads(1)
    out=ROOT/'outputs/loop_pair_backbone_audit_v1';out.mkdir(exist_ok=False)
    data=ROOT/'data/abgaze_reconstructed_v1/tensors/test.pt'
    rows=torch.load(data,map_location='cpu',weights_only=False)
    runs=dict(source='two_stage_binding_epoch60_gpu_test_v1',new='persistent_loop_pair_test_v1')
    paths={k:ROOT/'outputs'/v/'generated.json' for k,v in runs.items()}
    generated={k:{r['sample_id']:r for r in json.loads(p.read_text())} for k,p in paths.items()}
    report={}
    for name in ('native','source','new'):
        totals={k:[] for k in ('n_ca_c_angle_native_error_deg','ca_c_n_angle_native_error_deg',
            'c_n_ca_angle_native_error_deg','phi_native_error_deg','psi_native_error_deg',
            'omega_native_error_deg','omega_cis_or_trans_deviation_deg')}
        cases=[]
        for sid,x,y in rows:
            bb=y['bb'] if name=='native' else torch.tensor(generated[name][sid]['backbone'])
            chain=torch.cat([x['anchors'][:1],bb,x['anchors'][1:]])
            native=torch.cat([x['anchors'][:1],y['bb'],x['anchors'][1:]])
            mask=torch.cat([x['anchor_mask'][:1],y['mask'],x['anchor_mask'][1:]])
            n,ca,c=chain[:,0],chain[:,1],chain[:,2]
            rn,rca,rc=native[:,0],native[:,1],native[:,2]
            specs=[('n_ca_c_angle_native_error_deg',(n[1:-1],ca[1:-1],c[1:-1]),(rn[1:-1],rca[1:-1],rc[1:-1]),mask[1:-1,:3].all(-1)),
                ('ca_c_n_angle_native_error_deg',(ca[:-1],c[:-1],n[1:]),(rca[:-1],rc[:-1],rn[1:]),mask[:-1,1]&mask[:-1,2]&mask[1:,0]),
                ('c_n_ca_angle_native_error_deg',(c[:-1],n[1:],ca[1:]),(rc[:-1],rn[1:],rca[1:]),mask[:-1,2]&mask[1:,0]&mask[1:,1]),
                ('phi_native_error_deg',(c[:-2],n[1:-1],ca[1:-1],c[1:-1]),(rc[:-2],rn[1:-1],rca[1:-1],rc[1:-1]),mask[:-2,2]&mask[1:-1,:3].all(-1)),
                ('psi_native_error_deg',(n[1:-1],ca[1:-1],c[1:-1],n[2:]),(rn[1:-1],rca[1:-1],rc[1:-1],rn[2:]),mask[1:-1,:3].all(-1)&mask[2:,0]),
                ('omega_native_error_deg',(ca[:-1],c[:-1],n[1:],ca[1:]),(rca[:-1],rc[:-1],rn[1:],rca[1:]),mask[:-1,1]&mask[:-1,2]&mask[1:,0]&mask[1:,1])]
            per={}
            for key,pred,truth,valid in specs:
                if len(pred)==3:
                    v=(angle(*pred)-angle(*truth)).abs()
                    for coords in (pred,truth):
                        valid=valid&((coords[0]-coords[1]).norm(dim=-1)>1e-6)&((coords[2]-coords[1]).norm(dim=-1)>1e-6)
                else:
                    a,ok=torsion(*pred);b,refok=torsion(*truth);valid=valid&ok&refok
                    v=((a-b+180)%360-180).abs()
                    if key.startswith('omega'):
                        dev=torch.minimum(a.abs(),(180-a.abs()).abs())[valid].tolist()
                        totals['omega_cis_or_trans_deviation_deg'].extend(dev)
                        per['omega_cis_or_trans_deviation_deg']=describe(dev)
                values=v[valid].tolist();totals[key].extend(values);per[key]=describe(values)
            cases.append(dict(sample_id=sid,metrics=per))
        report[name]=dict(pooled={k:describe(v) for k,v in totals.items()},per_complex=cases)
    dump(out/'metrics.json',report)
    dump(out/'plan.json',dict(script_sha256=sha(Path(__file__)),test_sha256=sha(data),
        generated_sha256={k:sha(p) for k,p in paths.items()},
        limitations=['Existing inspected test, descriptive only; not for hyperparameter selection.',
                     'Native-relative torsion errors are NOT Ramachandran outlier rates.',
                     'Peptide cis/trans planarity is not a full stereochemical validation.',
                     'Pooled angle counts; separate per-complex records provided.']))
    print(json.dumps({k:v['pooled'] for k,v in report.items()}))


if __name__=='__main__':main()
