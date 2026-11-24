"""Geometry-only v21 continuation, H3 messaging and train-time residue geometry."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from v21_atomic_refiner import AtomicPairRefiner
from v22_joint_refiner import H3Message
from train.train_v21_atomic import objective as original_objective


def residue_geometry(bb):
    """Differentiable residue reconstruction; not hard inter-residue closure.

    Preserve CA and predicted frame; fix N-CA, CA-C, C-O and two intra-residue
    angles. Oxygen retains a free rotation around CA-C. No native information.
    """
    ca=bb[:,1];x=F.normalize(bb[:,2]-ca,dim=-1)
    v=bb[:,0]-ca;y=F.normalize(v-(v*x).sum(-1,keepdim=True)*x,dim=-1)
    n=ca+1.458*(math.cos(math.radians(111.2))*x+math.sin(math.radians(111.2))*y)
    c=ca+1.525*x
    oxygen=bb[:,3]-bb[:,2]
    perpendicular=oxygen-(oxygen*x).sum(-1,keepdim=True)*x
    perpendicular=torch.where(perpendicular.norm(dim=-1,keepdim=True)>1e-6,perpendicular,-y)
    perpendicular=F.normalize(perpendicular,dim=-1)
    o=c+1.231*(-math.cos(math.radians(120.8))*x+math.sin(math.radians(120.8))*perpendicular)
    return torch.stack([n,ca,c,o],1)


class StructureRefiner(AtomicPairRefiner):
    def __init__(self,width=64,layers=4,variant='control'):
        super().__init__(width,layers)
        if variant not in ('control','message','constrained'):raise ValueError(variant)
        self.variant=variant
        self.h3_messages=nn.ModuleList([H3Message(width) for _ in range(layers)] if variant!='control' else [])

    def load_v21(self,state):
        result=self.load_state_dict(state,strict=False)
        if result.unexpected_keys or any(not k.startswith('h3_messages.') for k in result.missing_keys):
            raise ValueError(result)

    def refine_h3(self,h,bb,frame,index):
        return self.h3_messages[index](h,bb,frame) if self.h3_messages else h

    def forward(self,x):
        out=super().forward(x)
        if self.variant=='constrained':out['backbone']=residue_geometry(out['backbone'])
        return out


def mean_mask(value,mask):
    return value[mask].mean() if mask.any() else value.sum()*0


def geometry_quality(bb,x):
    chain=torch.cat([x['anchors'][:1],bb,x['anchors'][1:]])
    valid=torch.cat([x['anchor_mask'][:1],torch.ones_like(bb[:,:,0],dtype=torch.bool),x['anchor_mask'][1:]])
    n,ca,c,o=chain.unbind(1)
    mask=valid[:-1,2]&valid[1:,0]
    cn=(c[:-1]-n[1:]).norm(dim=-1)
    bond=mean_mask(F.smooth_l1_loss(cn,torch.full_like(cn,1.329),reduction='none'),mask)
    def angle(a,b,c,target,m):
        cosine=(F.normalize(a-b,dim=-1)*F.normalize(c-b,dim=-1)).sum(-1)
        return mean_mask((cosine-math.cos(math.radians(target))).square(),m)
    angles=angle(ca[:-1],c[:-1],n[1:],116.2,mask&valid[:-1,1])
    angles=angles+angle(c[:-1],n[1:],ca[1:],121.7,mask&valid[1:,1])
    # Planarity allows cis/trans; do not impose trans on potentially cis loops.
    normal=F.normalize(torch.cross(ca[:-1]-c[:-1],n[1:]-c[:-1],dim=-1),dim=-1)
    plane=mean_mask((F.normalize(o[:-1]-c[:-1],dim=-1)*normal).sum(-1).square(),mask&valid[:-1,1]&valid[:-1,3])
    axis=F.normalize(n[1:]-c[:-1],dim=-1)
    a=ca[:-1]-c[:-1];b=ca[1:]-n[1:]
    a=F.normalize(a-(a*axis).sum(-1,keepdim=True)*axis,dim=-1)
    b=F.normalize(b-(b*axis).sum(-1,keepdim=True)*axis,dim=-1)
    plane=plane+mean_mask(1-(a*b).sum(-1).square(),mask&valid[:-1,1]&valid[1:,1])
    # Conservative non-neighbor backbone overlap proxy, not all-atom steric energy.
    length=len(bb);d=torch.cdist(bb.flatten(0,1),bb.flatten(0,1))
    positions=torch.arange(length,device=bb.device).repeat_interleave(4)
    nonneighbor=(positions[:,None]-positions[None,:]).abs()>1
    internal=mean_mask(F.relu(2.-d).square(),nonneighbor)
    context=x['context_bb'];cm=x['context_mask']
    dcontext=torch.cdist(bb.flatten(0,1),context.flatten(0,1))
    allowed=cm.flatten()[None].expand(length*4,-1).clone()
    # Remove covalently adjacent anchor residues from terminal-residue clash check.
    for loop_index,anchor in [(0,x['anchors'][0,1]),(length-1,x['anchors'][1,1])]:
        match=(context[:,1]-anchor).norm(dim=-1)<1e-3
        allowed[loop_index*4:(loop_index+1)*4,match.repeat_interleave(4)]=False
    external=mean_mask(F.relu(2.-dcontext).square(),allowed)
    clash=internal+external
    return dict(bond=bond,angles=angles,plane=plane,clash=clash,
                cn_mae=mean_mask((cn-1.329).abs(),mask),
                cn_outlier=mean_mask(((cn-1.329).abs()>.1).float(),mask))


def objective(out,x,y,variant):
    loss,metrics=original_objective(out,x,y)
    quality=geometry_quality(out['backbone'],x)
    if variant=='constrained':loss=loss+.5*quality['bond']+.2*quality['angles']+.1*quality['plane']+.2*quality['clash']
    with torch.no_grad():
        pred=torch.cdist(out['backbone'][:,1],x['context_bb'][x['antigen_mask'],1])<=8
        truth=torch.cdist(y['bb'][:,1],x['context_bb'][x['antigen_mask'],1])<=8
        tp=(pred&truth).sum()
        metrics.update(loss=float(loss.detach()),contact_precision=float(tp/pred.sum().clamp_min(1)),
                       contact_recall=float(tp/truth.sum().clamp_min(1)),
                       **{k:float(quality[k].detach()) for k in ('cn_mae','cn_outlier','clash')})
    return loss,metrics


def eligible(m,baseline):
    return (m['rmsd']<=baseline['rmsd']+.05 and m['aligned_rmsd']<=baseline['aligned_rmsd']+.05
            and m['contact_f1']>=baseline['contact_f1']-.02 and m['cn_mae']<=baseline['cn_mae']+.02
            and m['clash']<=baseline['clash']+.01)


def score(m):return m['contact_f1']-.2*m['rmsd']-.2*m['aligned_rmsd']
