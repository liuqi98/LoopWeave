"""Persistent H3-H3 pair states interleaved with the existing binding refiner.

One shallow-copied input dictionary carries state within a single forward only;
no tensors persist between samples. Zero-initialized readouts preserve source.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from binding_interface_refiner import BindingInterfaceRefiner
from orientation_interface_refiner import local_frames
from link_interface_refiner import rotation_matrix


class LoopPairLayer(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.offset=nn.Embedding(65,8)
        self.input=nn.Sequential(nn.Linear(2*width+76,width),nn.SiLU(),nn.Linear(width,width))
        self.norm=nn.LayerNorm(width)
        self.left=nn.Linear(width,width);self.right=nn.Linear(width,width)
        self.triangle=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width))
        self.triangle_gate=nn.Linear(width,width)
        self.attention=nn.Linear(width,1)
        self.node=nn.Linear(width,width)
        self.rigid=nn.Linear(width,6)
        for layer in (self.node,self.rigid):nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
        self.register_buffer('centers',torch.linspace(0,20,16))

    def forward(self,h,bb,z,probabilities):
        length=len(bb);origin,frame,valid=local_frames(bb,torch.ones_like(bb[...,0],dtype=torch.bool))
        delta=origin[None]-origin[:,None];distance=delta.norm(dim=-1)
        rbf=torch.exp(-.5*((distance[...,None]-self.centers)/1.5).square())
        direction=torch.einsum('ijc,icd->ijd',F.normalize(delta,dim=-1),frame)
        rotation=torch.einsum('ica,jcb->ijab',frame,frame).flatten(-2)
        index=torch.arange(length,device=bb.device)
        offsets=self.offset((index[None]-index[:,None]).clamp(-32,32)+32)
        pair=self.input(torch.cat([h[:,None].expand(-1,length,-1),h[None].expand(length,-1,-1),
            rbf,rotation,direction,offsets,probabilities[:,None].expand(-1,length,-1),
            probabilities[None].expand(length,-1,-1)],-1))
        z=self.norm(z+pair)
        triangle=torch.einsum('ikd,jkd->ijd',self.left(z),self.right(z))/math.sqrt(length)
        z=self.norm(z+self.triangle_gate(z).sigmoid()*self.triangle(triangle))
        weight=self.attention(z).squeeze(-1).sigmoid()*(index[:,None]!=index[None])
        weight=weight/weight.sum(-1,keepdim=True).clamp_min(1e-8)
        aggregate=(weight[...,None]*z).sum(1)
        h=h+self.node(aggregate)
        raw=self.rigid(aggregate)
        translation=.1*raw[:,:3].tanh()*valid[:,None]
        rotation=rotation_matrix(.05*raw[:,3:].tanh()*valid[:,None])
        relative=bb-origin[:,None]
        local=torch.einsum('lac,lcd->lad',relative,frame)
        change=torch.einsum('lac,ldc->lad',local,rotation-torch.eye(3,device=bb.device,dtype=bb.dtype))
        change=torch.einsum('lad,lcd->lac',change+translation[:,None],frame)
        return h,bb+change,z


class PersistentLoopPairRefiner(BindingInterfaceRefiner):
    def __init__(self,width=64,layers=4,**kwargs):
        super().__init__(width,layers,**kwargs)
        self.loop_pair_layers=nn.ModuleList([LoopPairLayer(width) for _ in range(layers)])
        self.loop_width=width

    def load_binding(self,state):
        result=self.load_state_dict(state,strict=False)
        if result.unexpected_keys or any(not key.startswith('loop_pair_layers.') for key in result.missing_keys):
            raise ValueError(result)

    def forward(self,x):
        length=len(x['base_bb'])
        private={**x,'_loop_pair_state':x['base_bb'].new_zeros(length,length,self.loop_width)}
        out=super().forward(private)
        out['loop_pair_state']=private['_loop_pair_state']
        return out

    def refine_relations(self,h,bb,pair,c,x,layer_index):
        h,bb,pair=super().refine_relations(h,bb,pair,c,x,layer_index)
        probabilities=(x['base_logits']+self.native_head(h)).softmax(-1)
        h,bb,z=self.loop_pair_layers[layer_index](h,bb,x['_loop_pair_state'],probabilities)
        x['_loop_pair_state']=z
        return h,bb,pair
