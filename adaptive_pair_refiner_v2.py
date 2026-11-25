"""A + B (confidence-gated pair memory) and A + B + C (pair-to-geometry feedback)."""
import math
import torch
from torch import nn
from torch.nn import functional as F

from adaptive_pair_refiner import BridgeLoopPairLayer
from orientation_interface_refiner import local_frames
from link_interface_refiner import rotation_matrix


class BridgeGatedLayer(BridgeLoopPairLayer):
    def __init__(self, width, coordinated_bound=None, local_bound=None, geo_feedback=False):
        super().__init__(width, coordinated_bound=coordinated_bound, local_bound=local_bound)
        self.geo_feedback = geo_feedback
        self.mem_gate = nn.Linear(width + 2, 1, bias=True)
        nn.init.zeros_(self.mem_gate.weight)
        nn.init.constant_(self.mem_gate.bias, -20.0)   # start ~baseline (retention gate open)
        if geo_feedback:
            self.pair_mlp = nn.Sequential(nn.Linear(width * 2, width), nn.SiLU(),
                                          nn.Linear(width, width))
            self.rigid_rot = nn.Linear(width, 3)
            self.rigid_trans = nn.Linear(width, 3)
            for m in (self.pair_mlp, self.rigid_rot, self.rigid_trans):
                for p in m.parameters():
                    nn.init.zeros_(p)

    def forward(self, h, bb, z, probabilities):
        length = len(bb)
        origin, frame, valid = local_frames(bb, torch.ones_like(bb[..., 0], dtype=torch.bool))
        delta = origin[None] - origin[:, None]
        distance = delta.norm(dim=-1)
        rbf = torch.exp(-.5 * ((distance[..., None] - self.centers) / 1.5).square())
        direction = torch.einsum('ijc,icd->ijd', F.normalize(delta, dim=-1), frame)
        rotation = torch.einsum('ica,jcb->ijab', frame, frame).flatten(-2)
        index = torch.arange(length, device=bb.device)
        offsets = self.offset((index[None] - index[:, None]).clamp(-32, 32) + 32)
        pair_feat = self.input(torch.cat([
            h[:, None].expand(-1, length, -1), h[None].expand(length, -1, -1), rbf, rotation,
            direction, offsets, probabilities[:, None].expand(-1, length, -1),
            probabilities[None].expand(length, -1, -1)], -1))
        z = self.norm(z + pair_feat)
        z_before = z
        left, right = self.left(z), self.right(z)
        omega = self.bridge_weights(z, h, valid)
        update = self.triangle(torch.einsum('ikd,jkd,ijk->ijd', left, right, omega) / math.sqrt(length))
        # B: entropy-gated retention of the persistent pair memory
        ent = -(probabilities.clamp_min(1e-8) * probabilities.clamp_min(1e-8).log()).sum(-1, keepdim=True)
        ent_i = ent[:, None, :].expand(length, length, 1)
        ent_j = ent[None, :, :].expand(length, length, 1)
        s = torch.cat([z, ent_i, ent_j], dim=-1)
        r = torch.sigmoid(self.mem_gate(s))
        z = self.norm(z + (1.0 - r) * self.triangle_gate(z_before).sigmoid() * update)
        # node aggregation (unchanged)
        weight = self.attention(z).squeeze(-1).sigmoid() * (index[:, None] != index[None])
        weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-8)
        aggregate = (weight[..., None] * z).sum(1)
        h = h + self.node(aggregate)
        # coordinates (two-scale, unchanged) + optional C: pair-to-geometry feedback
        coefficients = self.coordinated(aggregate).view(length, self.MODES, 3)
        basis = self.smooth_basis(length, bb.device, bb.dtype)
        coordinated = torch.einsum('lk,lkc->lc', basis, coefficients)
        coordinated = self.coordinated_bound * torch.tanh(coordinated) * valid[:, None]
        local = self.local(aggregate).view(length, 4, 3)
        local = self.local_bound * torch.tanh(local) * valid[:, None, None]
        rotation_update = rotation_matrix(.05 * self.rotation_head(aggregate).tanh() * valid[:, None])
        if self.geo_feedback:
            dz = z - z_before
            q = torch.einsum('ij,ijd->id', weight, self.pair_mlp(torch.cat([z, dz], dim=-1)))
            rotation_update = rotation_update @ rotation_matrix(.05 * self.rigid_rot(q).tanh())
            local = local + 0.1 * self.rigid_trans(q).tanh()[:, None, :]
        relative = bb - origin[:, None]
        local_coordinates = torch.einsum('lac,lcd->lad', relative, frame)
        rotated = torch.einsum('lac,ldc->lad', local_coordinates,
                               rotation_update - torch.eye(3, device=bb.device, dtype=bb.dtype))
        combined = rotated + coordinated[:, None, :] + local
        return h, bb + torch.einsum('lad,lcd->lac', combined, frame), z


def build_v2_model(base, width, bounds, geo_feedback=False):
    from two_scale_pair_refiner import TwoScalePairRefiner
    base.loop_pair_layers = nn.ModuleList([
        BridgeGatedLayer(width, coordinated_bound=bounds, local_bound=0.20,
                         geo_feedback=geo_feedback)
        for _ in range(len(base.loop_pair_layers))])
    return base
