"""Peptide-link and antigen-relative corrections via rigid residue updates."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from structure_refinement_ablation import StructureRefiner,objective as previous_objective,mean_mask


def rotation_matrix(vector):
    """Stable Rodrigues map with finite gradients at zero rotation."""
    a,b,c=vector.unbind(-1);z=torch.zeros_like(a)
    skew=torch.stack([z,-c,b,c,z,-a,-b,a,z],-1).reshape(*vector.shape[:-1],3,3)
    theta=vector.norm(dim=-1)
    first=torch.sinc(theta/math.pi)[...,None,None]
    second=(.5*torch.sinc(theta/(2*math.pi)).square())[...,None,None]
    return torch.eye(3,device=vector.device,dtype=vector.dtype)+first*skew+second*(skew@skew)


def frames(bb):
    x=F.normalize(bb[:,2]-bb[:,1],dim=-1)
    v=bb[:,0]-bb[:,1];y=F.normalize(v-(v*x).sum(-1,keepdim=True)*x,dim=-1)
    return torch.stack([x,y,torch.cross(x,y,dim=-1)],-1)


def known_frame(x):
    for i in (0,1):
        if x['anchor_mask'][i,:3].all():return frames(x['anchors'][i:i+1])[0]
    valid=x['context_mask'][:,:3].all(-1)
    if valid.any():return frames(x['context_bb'][valid][:1])[0]
    # CA-only structures have no complete residue frames. Use known anchor CA
    # direction and a non-collinear known-context CA, never a missing N/C atom.
    points=x['context_bb'][x['context_mask'][:,1],1]
    if len(points)<3:raise ValueError('Need at least three known context CAs for a CA-only frame')
    if x['anchor_mask'][:,1].all():
        origin=x['anchors'][0,1];direction=x['anchors'][1,1]-origin
    else:
        origin=points[0];offset=points-origin;direction=offset[offset.norm(dim=-1).argmax()]
    if direction.norm()<1e-6:raise ValueError('Degenerate known CA axis')
    axis=F.normalize(direction,dim=-1);offset=points-origin
    perpendicular=offset-(offset*axis).sum(-1,keepdim=True)*axis
    vector=perpendicular[perpendicular.norm(dim=-1).argmax()]
    if vector.norm()<1e-6:raise ValueError('All known context CAs are collinear')
    second=F.normalize(vector,dim=-1)
    return torch.stack([axis,second,torch.cross(axis,second,dim=-1)],-1)


class LinkInterfaceRefiner(StructureRefiner):
    def __init__(self,width=64,layers=4,mode='link_pose',steps=3):
        super().__init__(width,layers,variant='constrained')
        if mode not in ('control','link','link_pose'):raise ValueError(mode)
        self.mode=mode;self.steps=steps
        self.link_encoder=nn.Sequential(nn.Linear(2*width+12,width),nn.SiLU(),nn.LayerNorm(width),nn.Linear(width,width))
        self.link_update=nn.Sequential(nn.Linear(3*width,width),nn.SiLU(),nn.Linear(width,6))
        self.pose_encoder=nn.Sequential(nn.Linear(width+24+3,width),nn.SiLU(),nn.LayerNorm(width))
        self.pose_attention=nn.Linear(width,1)
        self.pose_update=nn.Linear(width,6)
        for layer in (self.link_update[-1],self.pose_update):
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
        self.register_buffer('pose_centers',torch.tensor([2.,4.,6.,8.,12.,16.]))

    def load_previous(self,state):
        result=self.load_state_dict(state,strict=False)
        prefixes=('link_encoder.','link_update.','pose_encoder.','pose_attention.','pose_update.','pose_centers')
        if result.unexpected_keys or any(not k.startswith(prefixes) for k in result.missing_keys):raise ValueError(result)

    def link_messages(self,bb,h,x,frame):
        chain=torch.cat([x['anchors'][:1],bb,x['anchors'][1:]])
        valid=torch.cat([x['anchor_mask'][:1],torch.ones_like(bb[:,:,0],dtype=torch.bool),x['anchor_mask'][1:]])
        n,ca,c,_=chain.unbind(1)
        hidden=torch.cat([torch.zeros_like(h[:1]),h,torch.zeros_like(h[:1])])
        cn=n[1:]-c[:-1];cac=ca[1:]-ca[:-1]
        cosine_left=(F.normalize(ca[:-1]-c[:-1],dim=-1)*F.normalize(cn,dim=-1)).sum(-1)
        cosine_right=(F.normalize(-cn,dim=-1)*F.normalize(ca[1:]-n[1:],dim=-1)).sum(-1)
        mask=valid[:-1,2]&valid[1:,0]&valid[:-1,1]&valid[1:,1]
        boundary=torch.zeros_like(cosine_left);boundary[0]=1;boundary[-1]=1
        features=torch.cat([cn@frame/3,cac@frame/5,
            (cn.norm(dim=-1)-1.329)[:,None],cosine_left[:,None],cosine_right[:,None],
            mask.float()[:,None],boundary[:,None],torch.ones_like(boundary[:,None])*len(bb)/30],-1)
        features=features*mask[:,None]
        edge=self.link_encoder(torch.cat([hidden[:-1],hidden[1:],features],-1))*mask[:,None]
        return self.link_update(torch.cat([h,edge[:-1],edge[1:]],-1))

    def conditioning_update(self,bb,h,x,frame,out):
        return 0

    def forward(self,x):
        out=super().forward(x)
        if self.mode=='control':return out
        bb=out['backbone'];h=out['h3_hidden'];frame=known_frame(x)
        ag=x['antigen_mask'];context=x['context_bb'][ag];cm=x['context_mask'][ag]
        for _ in range(self.steps):
            update=self.link_messages(bb,h,x,frame)
            update=update+self.conditioning_update(bb,h,x,frame,out)
            if self.mode=='link_pose' and ag.any():
                dist=(bb[:,1,None,None,:]-context[None]).norm(dim=-1)
                rbf=torch.exp(-.5*((dist[...,None]-self.pose_centers)/2)**2)*cm[None,:,:,None]
                direction=F.normalize(context[None,:,1]-bb[:,None,1],dim=-1)@frame
                pair=self.pose_encoder(torch.cat([out['pair_hidden'][:,ag],rbf.flatten(-2),direction],-1))
                ca_distance=dist[:,:,1];chosen=ca_distance.topk(min(32,len(context)),dim=-1,largest=False).indices
                local=pair.gather(1,chosen[...,None].expand(-1,-1,pair.shape[-1]))
                score=self.pose_attention(local).squeeze(-1)-ca_distance.gather(1,chosen)/4
                pose=(score.softmax(-1)[...,None]*local).sum(1)
                update=update+self.pose_update(pose)
            translation=.2*update[:,:3].tanh()@frame.T
            rotation=rotation_matrix(.15*update[:,3:].tanh())
            local=(bb-bb[:,1:2])@frame
            bb=(local@rotation.transpose(-1,-2))@frame.T+bb[:,1:2]+translation[:,None]
        out['backbone']=bb
        return out


def extra_terms(bb,x,y):
    native=y['bb'];valid=y['mask']
    chain=torch.cat([x['anchors'][:1],bb,x['anchors'][1:]])
    target=torch.cat([x['anchors'][:1],native,x['anchors'][1:]])
    mask=torch.cat([x['anchor_mask'][:1],valid,x['anchor_mask'][1:]])
    links=mask[:-1,2]&mask[1:,0]
    pred_vector=chain[1:,0]-chain[:-1,2];true_vector=target[1:,0]-target[:-1,2]
    link_vector=mean_mask(F.smooth_l1_loss(pred_vector,true_vector,reduction='none').mean(-1),links)
    errors=(pred_vector.norm(dim=-1)-1.329).abs()
    boundary=torch.zeros_like(links);boundary[0]=True;boundary[-1]=True
    ag=x['antigen_mask'];context=x['context_bb'][ag,1]
    distance=torch.cdist(native[:,1],context)
    contact=(distance<=8).any(-1)&valid[:,1]
    position=mean_mask(F.smooth_l1_loss(bb[:,1],native[:,1],reduction='none').mean(-1),contact)
    rmsd=mean_mask((bb[:,1]-native[:,1]).square().sum(-1),contact).sqrt()
    # H3-local frames make the target sensitive to interface orientation, not just distances.
    predicted_local=torch.einsum('lri,lij->lrj',context[None]-bb[:,None,1],frames(bb))
    target_local=torch.einsum('lri,lij->lrj',context[None]-native[:,None,1],frames(native))
    near=(distance<=12)&valid[:,:3].all(-1)[:,None]
    fape=mean_mask((predicted_local-target_local).norm(dim=-1).clamp_max(10)/10,near)
    return dict(link_vector=link_vector,interface_position=position,interface_frame=fape,
                contact_position_rmsd=rmsd,has_contact=contact.any().float(),
                endpoint_cn_mae=mean_mask(errors,links&boundary))


def objective(out,x,y,mode):
    if not x['antigen_mask'].any():raise ValueError('Geometry supervision requires nonempty antigen context')
    loss,metrics=previous_objective(out,x,y,'constrained')
    terms=extra_terms(out['backbone'],x,y)
    if mode!='control':loss=loss+.5*terms['link_vector']
    if mode=='link_pose':loss=loss+.5*terms['interface_position']+.3*terms['interface_frame']
    metrics.update(loss=float(loss.detach()),**{k:float(v.detach()) for k,v in terms.items()})
    return loss,metrics


def eligible(m,b):
    return (m['rmsd']<=b['rmsd']+.03 and m['aligned_rmsd']<=b['aligned_rmsd']+.03
            and m['contact_position_rmsd']<=b['contact_position_rmsd']+.02
            and m['cn_mae']<=b['cn_mae']+.005 and m['endpoint_cn_mae']<=b['endpoint_cn_mae']+.01
            and m['contact_f1']>=b['contact_f1']-.01 and m['clash']<=b['clash']*1.1+1e-5)


def score(m):
    return m['contact_f1']-.15*m['rmsd']-.15*m['aligned_rmsd']-.2*m['cn_mae']-.1*m['contact_position_rmsd']
