"""Missing-atom-aware supervision for the reconstructed H3 pilot."""
import torch
from torch.nn import functional as F
from orientation_interface_refiner import interface_fape
from structure_refinement_ablation import geometry_quality


def objective(out,x,y,fape_weight=.5,sequence_weight=1.):
    bb,target,mask=out['backbone'],y['bb'],y['mask']
    zero=bb.sum()*0
    # Mask before computing geometry, so arbitrary missing-label placeholders
    # cannot affect either the loss or its derivatives.
    error=(bb[mask]-target[mask]).norm(dim=-1)
    atomic=F.smooth_l1_loss(error,torch.zeros_like(error)) if mask.any() else zero
    ca_valid=mask[:,1]
    ca,truth=bb[ca_valid,1],target[ca_valid,1]
    context_valid=x['antigen_mask'] & x['context_mask'][:,1]
    context=x['context_bb'][context_valid,1]
    distances,truth_distances=torch.cdist(ca,context),torch.cdist(truth,context)
    near=truth_distances<=12
    interface=F.smooth_l1_loss(distances[near],truth_distances[near]) if near.any() else zero
    distance=F.smooth_l1_loss(distances.clamp_max(16),truth_distances.clamp_max(16)) if distances.numel() else zero
    contacts=truth_distances<=8
    soft=torch.sigmoid((8-distances)/.75)
    soft_f1=(2*(soft*contacts).sum()+1)/(soft.sum()+contacts.sum()+1)
    shape=F.smooth_l1_loss(torch.cdist(ca,ca),torch.cdist(truth,truth)) if len(ca)>1 else zero
    quality=geometry_quality(bb,x)
    # FAPE already masks missing target frames and observed context atoms.
    fape=interface_fape(bb,x,y)
    ce=F.cross_entropy(out['sequence_logits'],y['sequence'])
    geometry=atomic+.5*interface+.25*distance+.5*(1-soft_f1)+.1*shape
    geometry=geometry+.5*quality['bond']+.2*quality['angles']+.1*quality['plane']+.2*quality['clash']
    loss=geometry+fape_weight*fape+sequence_weight*ce
    with torch.no_grad():
        predicted=distances<=8
        tp=(predicted&contacts).sum()
        f1=2*tp/(predicted.sum()+contacts.sum()).clamp_min(1)
        rmsd=(ca-truth).square().sum(-1).mean().sqrt() if len(ca) else zero
    return loss,dict(loss=float(loss.detach()),geometry_loss=float(geometry.detach()),
        sequence_ce=float(ce.detach()),interface_fape=float(fape.detach()),
        aar=float((out['sequence_logits'].argmax(-1)==y['sequence']).float().mean()),
        rmsd=float(rmsd),contact_f1=float(f1),
        cn_mae=float(quality['cn_mae'].detach()),clash=float(quality['clash'].detach()))
