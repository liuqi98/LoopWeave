import torch
import pytest
from link_interface_refiner import LinkInterfaceRefiner,rotation_matrix,objective,known_frame
from structure_refinement_ablation import StructureRefiner


def inputs(length=5):
    torch.manual_seed(81)
    return dict(base_bb=torch.randn(length,4,3),base_logits=torch.randn(length,20),
                anchors=torch.randn(2,4,3),anchor_mask=torch.ones(2,4,dtype=torch.bool),
                context_bb=torch.randn(10,4,3),context_mask=torch.ones(10,4,dtype=torch.bool),
                context_features=torch.randn(10,27),antigen_mask=torch.tensor([False]*5+[True]*5))


def test_zero_rotation_gradients_and_orthogonality():
    v=torch.zeros(3,3,requires_grad=True);r=rotation_matrix(v)
    assert torch.equal(r,torch.eye(3).expand(3,-1,-1))
    (r*torch.randn_like(r)).sum().backward();assert torch.isfinite(v.grad).all()
    r=rotation_matrix(torch.randn(4,3)*.1)
    assert torch.allclose(r@r.transpose(-1,-2),torch.eye(3).expand(4,-1,-1),atol=1e-6)


def test_initialization_preserves_previous_and_rigid_distances():
    x=inputs();old=StructureRefiner(width=16,layers=2,variant='constrained')
    for mode in ('control','link','link_pose'):
        m=LinkInterfaceRefiner(width=16,layers=2,mode=mode);m.load_previous(old.state_dict())
        before=old(x)['backbone'];after=m(x)['backbone']
        assert torch.allclose(before,after,atol=2e-5)
        assert torch.equal(m(x)['sequence_logits'],x['base_logits'])
        with torch.no_grad():m.link_update[-1].weight.normal_(std=.03);m.pose_update.weight.normal_(std=.03)
        after=m(x)['backbone']
        assert torch.allclose(torch.cdist(before,before),torch.cdist(after,after),atol=2e-5)


def test_equivariance_and_trainable_heads():
    x=inputs();m=LinkInterfaceRefiner(width=16,layers=2)
    with torch.no_grad():m.link_update[-1].weight.normal_(std=.02);m.pose_update.weight.normal_(std=.02)
    rotation=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    moved=dict(x)
    for key in ('base_bb','anchors','context_bb'):moved[key]=x[key]@rotation.T+3
    assert torch.allclose(m(moved)['backbone'],m(x)['backbone']@rotation.T+3,atol=3e-5)
    y=dict(bb=torch.randn(5,4,3),mask=torch.ones(5,4,dtype=torch.bool))
    loss,metrics=objective(m(x),x,y,'link_pose');loss.backward()
    for p in (m.link_update[-1].weight,m.pose_update.weight,m.pose_encoder[0].weight):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0


def test_missing_anchor_frames_and_singleton():
    x=inputs(1);x['anchor_mask'][:]=False
    f=known_frame(x);assert torch.allclose(f.T@f,torch.eye(3),atol=2e-6)
    x['antigen_mask'][:]=False
    m=LinkInterfaceRefiner(width=16,layers=2)
    out=m(x);assert torch.isfinite(out['backbone']).all()
    y=dict(bb=torch.randn(1,4,3),mask=torch.ones(1,4,dtype=torch.bool))
    with pytest.raises(ValueError,match='nonempty antigen'):objective(out,x,y,'link_pose')
    x['antigen_mask'][5:]=True
    loss,metrics=objective(out,x,y,'link_pose');assert torch.isfinite(loss)


def test_ca_only_frame_is_equivariant_and_ignores_missing_atoms():
    x=inputs();x['anchor_mask'][:,[0,2,3]]=False;x['context_mask'][:,[0,2,3]]=False
    before=known_frame(x)
    modified={**x,'anchors':x['anchors'].clone(),'context_bb':x['context_bb'].clone()}
    modified['anchors'][:,[0,2,3]]+=999;modified['context_bb'][:,[0,2,3]]-=999
    assert torch.equal(before,known_frame(modified))
    rot=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    moved={**x,'anchors':x['anchors']@rot.T+3,'context_bb':x['context_bb']@rot.T+3}
    assert torch.allclose(known_frame(moved),rot@before,atol=2e-6)
    assert torch.allclose(before.T@before,torch.eye(3),atol=2e-6)
