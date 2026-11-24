"""Matched IMGT backbone-context DiffAb adapter (not the unchanged official protocol).

Missing target frames have deterministic proxy geometry and no rotation loss.
Missing CA atoms have no position loss. Sequence supervision keeps every sample.
"""
import hashlib
import json
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from easydict import EasyDict

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = Path('/media/xd/code2/DiffCDRH3/external/diffab')
sys.path.insert(0, str(UPSTREAM))
from diffab.models.diffab import DiffusionAntibodyDesign
from diffab.modules.common.so3 import rotation_to_so3vec, so3vec_to_rotation
from diffab.modules.diffusion.dpm_full import rotation_matrix_cosine_loss
from diffab.utils.protein.constants import AA

ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
CONFIG = dict(resolution='backbone+CB', res_feat_dim=128, pair_feat_dim=64,
              diffusion=dict(num_steps=100, eps_net_opt=dict(num_layers=6)),
              train_structure=True, train_sequence=True)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def convert(item, graph, x, y, record):
    """Use sequence positions, not IMGT numeric labels, for gap-aware adjacency."""
    l = len(y['sequence'])
    fw_ca = torch.as_tensor(item['framework_ca']).float()
    fw_index = torch.cdist(x['context_bb'][~x['antigen_mask'], 1], fw_ca,
                          compute_mode='donot_use_mm_for_euclid_dist').argmin(-1)
    h3 = list(map(int, item['heavy_h3_indices']))
    heavy = record['heavy_chain_seq']
    assert ''.join(heavy[i] for i in h3) == item['native_h3_sequence']
    heavy_kept = [i for i in range(len(heavy)) if i not in set(h3)]
    fw_meta = [(0, i+1, 1, heavy[i]) for i in heavy_kept]
    fw_meta += [(1, i+1, 2, a) for i,a in enumerate(record['light_chain_seq'])]
    ag_meta = [(j+2, i+1, 3, a) for j,s in enumerate(record['antigen_seqs']) for i,a in enumerate(s)]
    meta = [fw_meta[i] for i in fw_index.tolist()]
    meta += [ag_meta[int(i)] for i in graph['antigen_original_indices']]
    assert len(meta) == len(x['context_bb'])
    # Exact correspondence of sequence identities independently checks the mapping.
    assert ''.join(m[3] for m in meta) == ''.join(ALPHABET[i] for i in x['context_features'][:,:20].argmax(-1))
    meta = [(0,i+1,1,heavy[i]) for i in h3] + meta
    order = sorted(range(len(meta)), key=lambda i: meta[i][:2])
    bb = torch.cat([y['bb'], x['context_bb']]).float()
    observed = torch.cat([y['mask'], x['context_mask']]).bool()
    generate = torch.arange(len(bb)) < l
    origin = x['anchors'][:,1].mean(0)
    fraction = torch.arange(1,l+1).float()/(l+1)
    proxy_ca = x['anchors'][0,1] + fraction[:,None]*(x['anchors'][1,1]-x['anchors'][0,1])
    ca = torch.where(observed[:,1,None],bb[:,1], origin)
    ca[:l] = torch.where(observed[:l,1,None], bb[:l,1], proxy_ca)
    # Only proxy frames use fixed local offsets. Never treat them as labels.
    offsets = bb.new_tensor([[-.525,1.363,0],[0,0,0],[1.526,0,0],[2.15,1.05,0]])
    cross = torch.cross(bb[:,2]-bb[:,1], bb[:,0]-bb[:,1],dim=-1).norm(dim=-1)
    rot_valid = observed[:,:3].all(-1) & (cross > 1e-4)
    repaired = torch.where(observed[:,:,None] & rot_valid[:,None,None],bb,ca[:,None]+offsets)
    pos = torch.zeros(len(bb),5,3);pos[:,:4] = repaired-origin
    atom_mask = torch.zeros(len(bb),5,dtype=torch.bool);atom_mask[:,:4] = observed
    # Target existence must not leak native atom availability to the conditioner.
    atom_mask[generate,:4] = True
    out = dict(pos_heavyatom=pos,mask_heavyatom=atom_mask,
        aa=torch.tensor([int(AA(m[3])) for m in meta]),
        chain_nb=torch.tensor([m[0] for m in meta]),res_nb=torch.tensor([m[1] for m in meta]),
        fragment_type=torch.tensor([m[2] for m in meta]),generate_flag=generate,
        rot_valid=rot_valid,pos_valid=observed[:,1],mask=torch.ones(len(bb),dtype=torch.bool))
    out = {k:v[order] for k,v in out.items()}
    out['sample_id'] = item['sample_id'];out['origin'] = origin
    # Fully target-free inference initialization; diffusion randomizes it again.
    proxy = proxy_ca[:,None]+offsets-origin
    out['initial_target'] = proxy
    return out


def batch(rows, device, inference=False):
    n=max(len(r['aa']) for r in rows)
    keys=[k for k,v in rows[0].items() if torch.is_tensor(v) and k not in ('origin','initial_target')]
    result={}
    for k in keys:
        v=rows[0][k]; padded=v.new_zeros((len(rows),n)+v.shape[1:])
        if k=='aa': padded.fill_(21)
        for j,r in enumerate(rows):padded[j,:len(r[k])]=r[k]
        result[k]=padded.to(device)
    if inference:
        for j,r in enumerate(rows):
            g=result['generate_flag'][j]
            result['aa'][j,g]=int(AA.UNK)
            result['pos_heavyatom'][j,g]=0
            result['pos_heavyatom'][j,g,:4]=r['initial_target'].to(device)
    return result


class MaskedDiffAb(DiffusionAntibodyDesign):
    def __init__(self):super().__init__(EasyDict(CONFIG))

    def forward(self,b):
        d=self.diffusion;g=b['generate_flag']
        rf,pf,r,p=self.encode(b,True,True)
        v=rotation_to_so3vec(r);s=b['aa']
        t=torch.randint(0,d.num_steps,(len(s),),device=s.device)
        p=d._normalize_position(p);r=so3vec_to_rotation(v)
        vn,_=d.trans_rot.add_noise(v,g,t)
        pn,eps=d.trans_pos.add_noise(p,g,t)
        _,sn=d.trans_seq.add_noise(s,g,t)
        _,rp,ep,c=d.eps_net(vn,pn,sn,rf,pf,d.trans_pos.var_sched.betas[t],g,b['mask'])
        def avg(v,m):return (v*m).sum()/(m.sum().float()+1e-8)
        kl=F.kl_div(torch.log(d.trans_seq.posterior(sn,c,t)+1e-8),
                    d.trans_seq.posterior(sn,s,t),reduction='none',log_target=False).sum(-1)
        return dict(rot=avg(rotation_matrix_cosine_loss(rp,r),g & b['rot_valid']),
                    pos=avg(F.mse_loss(ep,eps,reduction='none').sum(-1),g & b['pos_valid']),
                    seq=avg(kl,g))


def prepare():
    torch.set_num_threads(1)
    dest=ROOT/'data/diffab_reconstructed_v3';dest.mkdir(exist_ok=False)
    report={}
    for split in ('train','valid','test'):
        source=ROOT/f'data/raad_abgaze_reconstructed_v1/{split}.pt'
        targets=ROOT/f'data/abgaze_reconstructed_v1/tensors/{split}.pt'
        records=[json.loads(s) for s in (ROOT/f'data/abgaze_reconstructed_v1/splits/{split}.json').read_text().splitlines() if s]
        raw=torch.load(source,weights_only=False);xy=torch.load(targets,weights_only=False)
        assert len(raw)==len(xy)==len(records)
        result=[]
        for (item,graph),(sid,x,y),rec in zip(raw,xy,records):
            assert sid==item['sample_id']==rec['pdb']
            row=convert(item,graph,x,y,rec)
            if not bool(y['mask'].all()):
                changed=dict(y,bb=y['bb'].clone())
                changed['bb'][~y['mask']]=98765
                alternative=convert(item,graph,x,changed,rec)
                for k in row:
                    if torch.is_tensor(row[k]):assert torch.equal(row[k],alternative[k]),(sid,k)
            result.append(row)
        torch.save(result,dest/f'{split}.pt')
        report[split]=dict(n=len(result),ids=[r['sample_id'] for r in result],
            incomplete_frames=sum(bool((r['generate_flag'] & ~r['rot_valid']).any()) for r in result),
            sha256=sha(dest/f'{split}.pt'),source_sha256=sha(source),targets_sha256=sha(targets))
        print(split,report[split]['n'],report[split]['incomplete_frames'],flush=True)
    dump(dest/'plan.json',dict(splits=report,adapter_sha256=sha(__file__),
        missing_label_placeholder_invariance=True,
        protocol='IMGT H3; exact retained backbone context; no sidechains; masked structure supervision; all samples retained',
        caveat='Adapted DiffAb, not unchanged official configuration; missing frame proxies affect noisy training states.'))


if __name__=='__main__':prepare()
