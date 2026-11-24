"""Target-free four-atom refinement with persistent H3-context pair states."""
import torch
from torch import nn
from torch.nn import functional as F


def context_inputs(item, graph, prediction):
    """Native-pose protocol only. H3 labels are never read here."""
    length = graph.h3_length
    context_ca = graph.coords[length:]
    ag = graph.antigen_mask[length:]
    fw_indices = torch.cdist(context_ca[~ag].cpu(),
                            torch.as_tensor(item['framework_ca']).float(),
                            compute_mode='donot_use_mm_for_euclid_dist').argmin(-1)
    fw = torch.as_tensor(item['framework_backbone']).float()[fw_indices]
    fm = torch.as_tensor(item['framework_backbone_mask']).bool()[fw_indices]
    ai = graph.antigen_original_indices.cpu()
    ab = torch.as_tensor(item['antigen_backbone']).float()[ai]
    am = torch.as_tensor(item['antigen_backbone_mask']).bool()[ai]
    bb, mask = torch.cat([fw, ab]), torch.cat([fm, am])
    if not torch.allclose(bb[:,1], context_ca.cpu(), atol=1e-3):
        raise ValueError('Context mapping mismatch; this adapter supports native poses only')
    bb = torch.where(mask[...,None], bb, bb[:,1:2])
    anchors = torch.stack([torch.as_tensor(item[k]).float()
                           for k in ('h3_n_anchor_frame','h3_c_anchor_frame')])
    anchor_indices = torch.cdist(anchors[:,1],torch.as_tensor(item['framework_ca']).float(),
                                 compute_mode='donot_use_mm_for_euclid_dist').argmin(-1)
    anchor_mask = torch.as_tensor(item['framework_backbone_mask']).bool()[anchor_indices]
    anchors = torch.where(anchor_mask[...,None],anchors,anchors[:,1:2])
    return dict(context_bb=bb, context_mask=mask,
                context_features=graph.node_features[length:].cpu(),
                antigen_mask=ag.cpu(), anchors=anchors, anchor_mask=anchor_mask,
                base_bb=prediction['h3_backbone'].detach().cpu(),
                base_logits=prediction['sequence_logits'].detach().cpu())


def anchor_frame(anchors):
    n, ca, c = anchors[0,:3]
    x = F.normalize(c-ca,dim=-1)
    y = n-ca; y = F.normalize(y-(x*y).sum()*x,dim=-1)
    return torch.stack([x,y,torch.cross(x,y,dim=-1)],-1)


class AtomicPairRefiner(nn.Module):
    def __init__(self, width=64, layers=4):
        super().__init__()
        self.query = nn.Linear(23,width)
        self.context = nn.Linear(31,width)
        # 4x4 atom distances, pair validity masks, and two anchor local vectors
        # for every H3 atom; anchor relative rotation and translation.
        self.geometry = nn.Sequential(nn.Linear(16*4+16+24+12+32+8,width),nn.SiLU())
        self.pair_updates = nn.ModuleList([
            nn.Sequential(nn.Linear(width*4,width),nn.SiLU(),nn.Linear(width,width))
            for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.node_updates = nn.ModuleList([
            nn.Sequential(nn.Linear(width*2,width),nn.SiLU(),nn.Linear(width,width))
            for _ in range(layers)])
        self.attention = nn.ModuleList([nn.Linear(width,1) for _ in range(layers)])
        self.radial = nn.ModuleList([nn.Linear(width,4) for _ in range(layers)])
        self.local = nn.ModuleList([nn.Linear(width,12) for _ in range(layers)])
        for layer in [*self.radial,*self.local]:
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
        self.register_buffer('centers',torch.tensor([3.,6.,10.,16.]))

    def refine_h3(self, hidden, backbone, frame, layer_index):
        """Identity hook; v22 overrides this without changing v21 weights."""
        return hidden

    def refine_relations(self,hidden,backbone,pair,context_hidden,inputs,layer_index):
        """Identity extension point for interleaved relation-coordinate updates."""
        return hidden,backbone,pair

    def forward(self, inputs):
        bb, context, anchors = inputs['base_bb'],inputs['context_bb'],inputs['anchors']
        mask = inputs['context_mask']
        length = len(bb)
        t = torch.linspace(0,1,length,device=bb.device)[:,None]
        h = self.query(torch.cat([inputs['base_logits'].softmax(-1),t,1-t,
                                  torch.ones_like(t)*length/30],-1))
        c = self.context(torch.cat([inputs['context_features'],mask.float()],-1))
        frame = anchor_frame(anchors)
        cframe = anchor_frame(anchors.flip(0))
        boundary = torch.cat([(frame.T@cframe).flatten(),
                              (anchors[1,1]-anchors[0,1])@frame / 10])
        pair = bb.new_zeros(length,len(context),h.size(-1))
        for layer_index,(update,norm,node,att,rad,local) in enumerate(zip(self.pair_updates,self.norms,
                self.node_updates,self.attention,self.radial,self.local)):
            difference = context[None,None,:,:,:]-bb[:, :,None,None,:]
            distances = difference.norm(dim=-1).permute(0,2,1,3)
            valid = mask[None,:,None,:].expand(length,-1,4,-1)
            rbf = torch.exp(-.5*((distances[...,None]-self.centers)/2)**2)
            rbf = (rbf*valid[...,None]).flatten(-3)
            anchor_vectors = ((anchors[None,:,1,None,:]-bb[:,None,:,:])@frame).flatten(1)/10
            anchor_distances = (bb[:,None,:,None,:]-anchors[None,:,None,:,:]).norm(dim=-1)
            anchor_distances = (anchor_distances.clamp_max(20)/20 *
                                inputs['anchor_mask'][None,:,None,:]).flatten(1)
            geom = self.geometry(torch.cat([rbf,valid.float().flatten(-2),
                anchor_vectors[:,None].expand(-1,len(context),-1),
                boundary[None,None].expand(length,len(context),-1),
                anchor_distances[:,None].expand(-1,len(context),-1),
                inputs['anchor_mask'].float().flatten()[None,None].expand(length,len(context),-1)],-1))
            pair = norm(pair+update(torch.cat([pair,geom,
                h[:,None].expand(-1,len(context),-1),
                c[None].expand(length,-1,-1)],-1)))
            # Separate framework/antigen normalization prevents one stream
            # overwhelming the other solely through its number of residues.
            weight = torch.zeros_like(pair[...,0])
            for selected in (inputs['antigen_mask'],~inputs['antigen_mask']):
                if selected.any():
                    weight[:,selected] = att(pair[:,selected]).squeeze(-1).softmax(-1)/2
            aggregate = (pair*weight[...,None]).sum(1)
            h = h+node(torch.cat([h,aggregate],-1))
            h = self.refine_h3(h,bb,frame,layer_index)
            direction = context[None,:,1,None,:]-bb[:,None,:,:]
            direction = F.normalize(direction,dim=-1)
            radial = (weight[...,None,None]*rad(pair).tanh()[...,None]*direction).sum(1)
            local_step = local(h).view(length,4,3).tanh()@frame.T
            bb = bb + radial + local_step
            h,bb,pair = self.refine_relations(h,bb,pair,c,inputs,layer_index)
        return {'backbone':bb,'sequence_logits':inputs['base_logits'],
                'h3_hidden':h,'pair_hidden':pair}
