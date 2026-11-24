"""RAAD-inspired typed messages with shared gates for features and geometry.

An interleaved extension of the constrained baseline, not a RAAD reproduction.
No contact labels or additional contact-head loss; sequence logits stay frozen.
"""
import torch
from torch import nn
from torch.nn import functional as F
from structure_refinement_ablation import StructureRefiner,objective as base_objective
from link_interface_refiner import known_frame,rotation_matrix,extra_terms


class RelationLayer(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.message=nn.Sequential(nn.Linear(3*width+83,width),nn.SiLU(),nn.LayerNorm(width))
        self.types=nn.Embedding(5,width)
        self.gates=nn.ModuleList([nn.Linear(width,1) for _ in range(5)])
        self.forces=nn.ModuleList([nn.Linear(width,4) for _ in range(5)])
        self.node=nn.Sequential(nn.Linear(6*width,width),nn.SiLU(),nn.Linear(width,width))
        self.edge=nn.Linear(width,width)
        for module in [*self.forces,self.node[-1],self.edge]:
            nn.init.zeros_(module.weight);nn.init.zeros_(module.bias)
        self.register_buffer('centers',torch.tensor([3.,6.,10.,16.]))

    def forward(self,h,bb,pair,c,x,coupled):
        length=len(bb);context=x['context_bb'];frame=known_frame(x)
        coordinates=torch.cat([bb,context]);hidden=torch.cat([h,c])
        valid=torch.cat([torch.ones_like(bb[:,:,0],dtype=torch.bool),x['context_mask']])
        delta=coordinates[None,:,None,:,:]-bb[:,None,:,None,:]
        distance=delta.norm(dim=-1)
        atom_mask=valid[None,:,None,:].expand(length,-1,4,-1)
        radial=(torch.exp(-.5*((distance[...,None]-self.centers)/2)**2)*atom_mask[...,None]).flatten(-3)
        direction=F.normalize(coordinates[None,:,1]-bb[:,None,1],dim=-1)@frame
        edges=torch.cat([pair.new_zeros(length,length,pair.shape[-1]),pair],1)
        feature=torch.cat([h[:,None].expand(-1,len(hidden),-1),hidden[None].expand(length,-1,-1),
                           edges,radial,atom_mask.float().flatten(-2),direction],-1)
        message=self.message(feature)
        ca=distance[:,:,1,1];indices=torch.arange(length,device=bb.device)
        separation=(indices[:,None]-indices[None]).abs()
        masks=[]
        for offset in (1,2):
            mask=torch.zeros_like(ca,dtype=torch.bool);mask[:,:length]=separation==offset;masks.append(mask)
        # Spatial CDR, known framework, and antigen each get their own relation.
        for kind in range(3):
            allowed=torch.zeros_like(ca,dtype=torch.bool)
            if kind==0:allowed[:,:length]=separation>2
            else:allowed[:,length:]=(x['antigen_mask'] if kind==2 else ~x['antigen_mask'])[None]&valid[None,length:,1]
            d=ca.masked_fill(~allowed,float('inf'))
            nearest=torch.zeros_like(allowed)
            chosen=d.topk(min(16 if kind==2 else 8,d.shape[-1]),largest=False,dim=-1).indices
            nearest.scatter_(1,chosen,True)
            masks.append(allowed&(nearest|(d<(8 if kind==0 else 12))))
        aggregates=[];force=bb.new_zeros(length,4,3);edge_update=torch.zeros_like(message)
        atom_direction=F.normalize(coordinates[None]-bb[:,None],dim=-1)*valid[None,:,:,None]
        for i,mask in enumerate(masks):
            m=F.silu(message+self.types.weight[i])
            gate=self.gates[i](m).squeeze(-1).sigmoid() if coupled else torch.ones_like(ca)
            weight=gate*mask;weight=weight/(weight.sum(-1,keepdim=True)+1e-8)
            aggregates.append((weight[...,None]*m).sum(1))
            force=force+(weight[...,None,None]*self.forces[i](m).tanh()[...,None]*atom_direction).sum(1)/5
            edge_update=edge_update+weight[...,None]*self.edge(m)/5
        h=h+self.node(torch.cat([h,*aggregates],-1))
        # Convert atom forces to a bounded residue-rigid update, retaining local geometry.
        relative=bb-bb[:,1:2]
        translation=.15*(force.mean(1)@frame).tanh()@frame.T
        torque=torch.cross(relative,force-force.mean(1,keepdim=True),dim=-1).sum(1)
        torque=torque/relative.square().sum((1,2)).clamp_min(1e-6)[:,None]
        rotation=rotation_matrix(.1*(torque@frame).tanh())
        bb=((relative@frame)@rotation.transpose(-1,-2))@frame.T+bb[:,1:2]+translation[:,None]
        return h,bb,pair+edge_update[:,length:]


class RelationCoupledRefiner(StructureRefiner):
    def __init__(self,width=64,layers=4,variant='constrained',relation_mode='coupled'):
        super().__init__(width,layers,variant)
        if relation_mode not in ('control','typed','coupled'):raise ValueError(relation_mode)
        self.relation_mode=relation_mode
        self.relation_layers=nn.ModuleList([RelationLayer(width) for _ in range(layers)])

    def load_source(self,state):
        result=self.load_state_dict(state,strict=False)
        if result.unexpected_keys or any(not k.startswith('relation_layers.') for k in result.missing_keys):raise ValueError(result)

    def refine_relations(self,h,bb,pair,c,x,layer_index):
        if self.relation_mode=='control':return h,bb,pair
        return self.relation_layers[layer_index](h,bb,pair,c,x,self.relation_mode=='coupled')


def objective(out,x,y,mode):
    # Exactly the same geometry objective in every arm. Extra terms are metrics only.
    loss,metrics=base_objective(out,x,y,'constrained')
    with torch.no_grad():
        terms=extra_terms(out['backbone'],x,y)
        metrics.update({k:float(terms[k]) for k in ('contact_position_rmsd','endpoint_cn_mae')})
    return loss,metrics


def eligible(m,b):
    return (all(m[k]>=b[k]-.002 for k in ('contact_f1','contact_precision','contact_recall'))
            and all(m[k]<=b[k]+.02 for k in ('rmsd','aligned_rmsd','contact_position_rmsd'))
            and m['cn_mae']<=b['cn_mae']+.01 and m['endpoint_cn_mae']<=b['endpoint_cn_mae']+.01
            and m['clash']<=b['clash']*1.1+1e-5)


def score(m):return m['contact_f1']+.2*m['contact_recall']-.1*m['aligned_rmsd']-.05*m['contact_position_rmsd']
