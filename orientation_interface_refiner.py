"""Backbone-local orientation inputs and optional interface FAPE.

AbGaze-inspired geometric conditioning, NOT its surface implementation.
No surface normals, native H3 coordinates, or native contacts enter forward().
"""
import torch
from torch import nn
from torch.nn import functional as F
from native_sequence_coupled_refiner import NativeSequenceCoupledRefiner, objective as native_objective


def local_frames(bb, mask):
    valid = mask[:, :3].all(-1) & torch.isfinite(bb[:, :3]).all((-1, -2))
    clean = torch.where(mask[..., None] & torch.isfinite(bb), bb, torch.zeros_like(bb))
    origin = clean[:, 1]
    a = clean[:, 2] - origin
    e1 = F.normalize(a, dim=-1, eps=1e-6)
    b = clean[:, 0] - origin
    perpendicular = b - (b * e1).sum(-1, keepdim=True) * e1
    valid = valid & (a.norm(dim=-1) > 1e-4) & (perpendicular.norm(dim=-1) > 1e-4)
    e2 = F.normalize(perpendicular, dim=-1, eps=1e-6)
    frame = torch.stack([e1, e2, torch.cross(e1, e2, dim=-1)], -1)
    return origin, torch.where(valid[:, None, None], frame, torch.zeros_like(frame)), valid


def orientation_features(bb, context, context_mask):
    origin, frame, valid = local_frames(bb, torch.ones_like(bb[..., 0], dtype=torch.bool))
    co, cf, cv = local_frames(context, context_mask)
    ca_valid = context_mask[:, 1] & torch.isfinite(context[:, 1]).all(-1)
    direction = F.normalize(co[None] - origin[:, None], dim=-1, eps=1e-6)
    forward = torch.einsum('lcj,ljk->lck', direction, frame)
    reverse = torch.einsum('lcj,cjk->lck', -direction, cf)
    rotation = torch.einsum('lji,cjk->lcik', frame, cf).flatten(-2)
    both = valid[:, None] & cv[None]
    forward = torch.where((valid[:, None] & ca_valid[None])[..., None], forward, 0.)
    reverse = torch.where(both[..., None], reverse, 0.)
    flags = torch.stack([valid[:, None] & ca_valid[None], both], -1).to(bb.dtype)
    return torch.cat([forward, reverse, rotation, flags], -1)  # 3 + 3 + 9 + 2


class OrientationInterfaceRefiner(NativeSequenceCoupledRefiner):
    def __init__(self, width=64, layers=4, use_orientation=True, **kwargs):
        super().__init__(width=width, layers=layers, **kwargs)
        self.use_orientation = use_orientation
        self.orientation_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(17, width), nn.SiLU(), nn.Linear(width, width, bias=False))
            for _ in range(layers)])
        for layer in self.orientation_layers:
            nn.init.zeros_(layer[-1].weight)

    def load_source(self, state):
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or any(not k.startswith(('native_head.', 'orientation_layers.')) for k in result.missing_keys):
            raise ValueError(result)

    def refine_relations(self, h, bb, pair, c, x, layer_index):
        if self.use_orientation:
            features = orientation_features(bb, x['context_bb'], x['context_mask'])
            update = self.orientation_layers[layer_index](features)
            pair = pair + update * features[..., -2:-1]
        return super().refine_relations(h, bb, pair, c, x, layer_index)


def interface_fape(prediction, x, y, clamp=10.):
    """H3-frame error of fixed antigen backbone points, normalized by 10 A.

    Equal average over valid H3 frames/antigen atoms. No native-contact crop.
    Native coordinates are supervision only; empty/degenerate frames yield zero.
    """
    po, pf, pv = local_frames(prediction, torch.ones_like(prediction[..., 0], dtype=torch.bool))
    to, tf, tv = local_frames(y['bb'], y['mask'])
    context = x['context_bb']
    mask = x['context_mask'] & x['antigen_mask'][:, None] & torch.isfinite(context).all(-1)
    points = torch.where(mask[..., None], context, 0.).flatten(0, 1)
    plocal = torch.einsum('lpj,ljk->lpk', points[None] - po[:, None], pf)
    tlocal = torch.einsum('lpj,ljk->lpk', points[None] - to[:, None], tf)
    error = ((plocal - tlocal).square().sum(-1) + 1e-8).sqrt().clamp(max=clamp) / clamp
    valid = (pv & tv)[:, None] & mask.flatten()[None]
    return error[valid].mean() if valid.any() else prediction.sum() * 0.


def objective(out, x, y, fape_weight=0., sequence_weight=1.):
    loss, metrics = native_objective(out, x, y)
    ce = F.cross_entropy(out['sequence_logits'], y['sequence'])
    fape = interface_fape(out['backbone'], x, y)
    loss = loss + (sequence_weight - 1.) * ce + fape_weight * fape
    metrics.update(interface_fape=float(fape.detach()), loss=float(loss.detach()))
    return loss, metrics
