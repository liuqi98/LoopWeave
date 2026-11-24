"""Experimental rigid-residue peptide constraint output layer.

No native coordinates, antigen labels, or predicted sequence are used. This is
not exact loop closure, an all-atom relaxation, or a guarantee of better RMSD.
The plane term permits cis and trans; it does not select the correct isomer.
Backtracking gives nonincreasing constraint energy, not nonincreasing test error.
"""
import math

import torch
from torch import nn

from persistent_loop_pair_refiner import PersistentLoopPairRefiner


class PeptideGeometryProjection(nn.Module):
    def __init__(self, steps=3, step_size=0.05, plane_weight=0.2,
                 max_translation=0.03, max_rotation=0.03, backtracks=6):
        super().__init__()
        if not isinstance(steps, int) or steps < 0:
            raise ValueError('steps must be a nonnegative integer')
        if not isinstance(backtracks, int) or backtracks < 1:
            raise ValueError('backtracks must be a positive integer')
        values = (step_size, plane_weight, max_translation, max_rotation)
        if not all(math.isfinite(v) for v in values) or min(values) < 0 or step_size == 0:
            raise ValueError('finite nonnegative settings and positive step_size required')
        self.steps, self.step_size = steps, step_size
        self.plane_weight = plane_weight
        self.max_translation, self.max_rotation = max_translation, max_rotation
        self.backtracks = backtracks

    def energy(self, bb, anchors, anchor_mask):
        chain = torch.cat((anchors[:1], bb, anchors[1:]))
        mask = torch.cat((anchor_mask[:1], torch.ones_like(bb[..., 0], dtype=torch.bool), anchor_mask[1:]))
        # The four atoms defining peptide omega: CA_i, C_i, N_i+1, CA_i+1.
        a, b, c, d = chain[:-1, 1], chain[:-1, 2], chain[1:, 0], chain[1:, 1]
        valid_link = mask[:-1, 2] & mask[1:, 0]
        cn = (((c-b).square().sum(-1) + 1e-12).sqrt() - 1.329).square()
        n1, n2 = torch.cross(b-a, c-b, dim=-1), torch.cross(c-b, d-c, dim=-1)
        valid_plane = valid_link & mask[:-1, 1] & mask[1:, 1]
        valid_plane = valid_plane & (n1.norm(dim=-1) > 1e-6) & (n2.norm(dim=-1) > 1e-6)
        # Smooth denominators also keep second derivatives finite for masked or
        # degenerate links; norm().clamp_min() alone is insufficient at zero.
        unit1 = n1 / (n1.square().sum(-1, keepdim=True) + 1e-12).sqrt()
        unit2 = n2 / (n2.square().sum(-1, keepdim=True) + 1e-12).sqrt()
        cosine = (unit1 * unit2).sum(-1)
        # sin^2(omega) treats both cis and trans as planar.
        plane = (1-cosine.square()).clamp_min(0)
        cn_loss = (cn*valid_link).sum()/valid_link.sum().clamp_min(1)
        plane_loss = (plane*valid_plane).sum()/valid_plane.sum().clamp_min(1)
        return cn_loss + self.plane_weight*plane_loss

    @staticmethod
    def bounded(vector, maximum):
        norm = (vector.square().sum(-1, keepdim=True) + 1e-24).sqrt()
        return vector * (maximum/norm).clamp(max=1)

    @staticmethod
    def rigid_update(bb, translation, rotation):
        x, y, z = rotation.unbind(-1)
        zero = torch.zeros_like(x)
        skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(-1, 3, 3)
        matrix = torch.matrix_exp(skew)
        center = bb[:, 1:2]
        return (bb-center) @ matrix.transpose(-1, -2) + center + translation[:, None]

    def forward(self, bb, anchors, anchor_mask):
        if torch.is_inference_mode_enabled():
            raise RuntimeError('Projection needs autograd internally; use torch.no_grad(), not inference_mode().')
        if bb.ndim != 3 or bb.shape[1:] != (4, 3) or len(bb) < 1:
            raise ValueError('Expected nonempty [L,4,3] N/CA/C/O backbone')
        if anchors.shape != (2, 4, 3) or anchor_mask.shape != (2, 4) or anchor_mask.dtype != torch.bool:
            raise ValueError('Expected two anchors and Boolean atom masks')
        if not torch.isfinite(bb).all() or not torch.isfinite(anchors[anchor_mask]).all():
            raise ValueError('Nonfinite valid coordinates')
        # Missing anchor atoms must not contaminate norms or gradients with NaN.
        anchors = torch.where(anchor_mask[..., None], anchors, torch.zeros_like(anchors)).detach()
        outer_grad = torch.is_grad_enabled()
        with torch.enable_grad():
            current = bb if outer_grad else bb.detach()
            for _ in range(self.steps):
                if not current.requires_grad:
                    current = current.detach().requires_grad_(True)
                value = self.energy(current, anchors, anchor_mask)
                force, = torch.autograd.grad(value, current, create_graph=outer_grad)
                translation = self.bounded(-self.step_size*force.sum(1), self.max_translation)
                torque = torch.cross(current-current[:, 1:2], force, dim=-1).sum(1)
                rotation = self.bounded(-self.step_size*torque, self.max_rotation)
                # Each candidate is a rigid residue transform. Anchors never move.
                for i in range(self.backtracks):
                    scale = 0.5**i
                    candidate = self.rigid_update(current, scale*translation, scale*rotation)
                    candidate_value = self.energy(candidate, anchors, anchor_mask)
                    if torch.isfinite(candidate_value) and candidate_value.detach() <= value.detach():
                        current = candidate
                        break
                if not outer_grad:
                    current = current.detach()
        return current if outer_grad else current.detach()


class PeptideConstrainedLoopPairRefiner(PersistentLoopPairRefiner):
    """Opt-in candidate; baseline implementation and checkpoints stay unchanged.

    Sequence logits are unchanged by this final geometry layer. Coordinate loss
    gradients pass through its accepted updates during training. There is no
    claim that the frozen sequence head is consistent with the updated geometry.
    """
    def __init__(self, width=64, layers=4, geometry_steps=3,
                 geometry_step_size=0.05, geometry_plane_weight=0.2,
                 geometry_max_translation=0.03, geometry_max_rotation=0.03,
                 geometry_backtracks=6, **kwargs):
        super().__init__(width=width, layers=layers, **kwargs)
        self.peptide_projection = PeptideGeometryProjection(
            steps=geometry_steps, step_size=geometry_step_size,
            plane_weight=geometry_plane_weight,
            max_translation=geometry_max_translation, max_rotation=geometry_max_rotation,
            backtracks=geometry_backtracks)

    def forward(self, x):
        out = super().forward(x)
        out['backbone_before_peptide_projection'] = out['backbone']
        out['backbone'] = self.peptide_projection(out['backbone'], x['anchors'], x['anchor_mask'])
        return out
