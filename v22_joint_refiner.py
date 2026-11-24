"""Geometry-guided sequence decoding and H3-internal equivariant messaging."""
import torch
from torch import nn
from torch.nn import functional as F
from v21_atomic_refiner import AtomicPairRefiner,anchor_frame


class H3Message(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.message=nn.Sequential(nn.Linear(2*width+10,width),nn.SiLU(),nn.Linear(width,width))
        self.attention=nn.Linear(width,1)
        self.output=nn.Linear(width,width)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,h,bb,frame):
        length=len(h)
        if length==1:return h
        delta=bb[None,:,1]-bb[:,None,1]
        distance=delta.norm(dim=-1)
        offset=torch.arange(length,device=h.device)
        offset=offset[None]-offset[:,None]
        unit=F.normalize(delta,dim=-1)@frame
        centers=h.new_tensor([2.,4.,6.,8.])
        rbf=torch.exp(-.5*((distance[...,None]-centers)/2)**2)
        neighbor=(offset.abs()==1)|((distance<=10)&(offset!=0))
        # Adjacent pairs ensure nonempty attention even with poor initial geometry.
        features=torch.cat([rbf,unit,(offset/length)[...,None],
                            (offset.abs()==1)[...,None],(offset.abs()==2)[...,None]],-1)
        pair=self.message(torch.cat([h[:,None].expand(-1,length,-1),
                                     h[None].expand(length,-1,-1),features],-1))
        weight=self.attention(pair).squeeze(-1).masked_fill(~neighbor,-torch.inf).softmax(-1)
        return h+self.output((weight[...,None]*pair).sum(1))


class JointRefiner(AtomicPairRefiner):
    NEW_PREFIXES=('h3_messages.','sequence_fusion.','sequence_delta.',
                  'sequence_gate.','decode_attention.','contact_residual.','distance_residual.')

    def __init__(self,width=64,layers=4,h3_messages=True):
        super().__init__(width,layers)
        self.h3_messages=nn.ModuleList([H3Message(width) for _ in range(layers)] if h3_messages else [])
        # Persistent pair states augmented by distances from final coordinates.
        self.decode_attention=nn.Linear(width+4,1)
        self.sequence_fusion=nn.Sequential(nn.Linear(3*width+3,2*width),nn.SiLU(),
                                           nn.LayerNorm(2*width))
        self.sequence_delta=nn.Linear(2*width,20)
        self.sequence_gate=nn.Linear(2*width,1)
        self.contact_residual=nn.Linear(width,1)
        self.distance_residual=nn.Linear(width,1)
        for layer in (self.sequence_delta,self.contact_residual,self.distance_residual):
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)

    def load_v21(self,state):
        result=self.load_state_dict(state,strict=False)
        if result.unexpected_keys or any(not k.startswith(self.NEW_PREFIXES) for k in result.missing_keys):
            raise ValueError(f'Unexpected v21 initialization mismatch: {result}')

    def refine_h3(self,hidden,backbone,frame,layer_index):
        return self.h3_messages[layer_index](hidden,backbone,frame) if self.h3_messages else hidden

    def forward(self,inputs):
        out=super().forward(inputs)
        bb,pair=out['backbone'],out['pair_hidden']
        distance=torch.cdist(bb[:,1],inputs['context_bb'][:,1])
        rbf=torch.exp(-.5*((distance[...,None]-self.centers)/2)**2)
        score=self.decode_attention(torch.cat([pair,rbf],-1)).squeeze(-1)
        streams=[]
        for mask in (~inputs['antigen_mask'],inputs['antigen_mask']):
            streams.append((score[:,mask].softmax(-1)[...,None]*pair[:,mask]).sum(1)
                           if mask.any() else torch.zeros_like(out['h3_hidden']))
        n= len(bb);t=torch.linspace(0,1,n,device=bb.device)[:,None]
        hidden=self.sequence_fusion(torch.cat([out['h3_hidden'],*streams,t,1-t,torch.ones_like(t)*n/30],-1))
        out['sequence_logits']=inputs['base_logits']+self.sequence_gate(hidden).sigmoid()*self.sequence_delta(hidden)
        ag=inputs['antigen_mask'];pairs=pair[:,ag]
        out['contact_logits']=(8-distance[:,ag])/.75+self.contact_residual(pairs).squeeze(-1)
        out['pair_distances']=distance[:,ag]+2*self.distance_residual(pairs).squeeze(-1).tanh()
        return out


def configure_stage(model,stage):
    if stage not in ('A','B','C'):raise ValueError(stage)
    for name,p in model.named_parameters():
        # Stage A isolates sequence learning: internal H3 updates remain frozen
        # because they feed coordinate heads and could otherwise move geometry.
        if stage=='A':
            enabled=name.startswith(('sequence_','decode_attention.','contact_residual.','distance_residual.'))
        elif stage=='B':enabled=not name.startswith(('radial.','local.'))
        else:enabled=True
        p.requires_grad_(enabled)
    return [p for p in model.parameters() if p.requires_grad]
