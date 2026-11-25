"""Stage-A exploration: geometry-conditioned bridge attention inside the two-scale pair layer."""
import math
import torch
from torch import nn
from torch.nn import functional as F

from two_scale_pair_refiner import TwoScaleLoopPairLayer
from orientation_interface_refiner import local_frames
from link_interface_refiner import rotation_matrix


class BridgeLoopPairLayer(TwoScaleLoopPairLayer):
    """Triangle update with adaptive third-residue bridge weights.

    omega_{ijk} = L * softmax_k( score(Z_ik, Z_jk, h_k, E_ik, E_jk) )
    T_ij = sum_k omega_{ijk} (W_L Z_ik * W_R Z_jk) / sqrt(L)
    Zero-initialised scoring -> uniform omega (=1) -> exactly recovers the baseline.
    """

    def __init__(self, width, coordinated_bound=None, local_bound=None, bridge_width=None):
        super().__init__(width, coordinated_bound=coordinated_bound, local_bound=local_bound)
        w = bridge_width or width
        self.bq = nn.Linear(width, w, bias=False)
        self.bk = nn.Linear(width, w, bias=False)
        self.bh = nn.Linear(width, w, bias=False)
        self.bw = nn.Linear(w, 1, bias=True)
        for m in (self.bq, self.bk, self.bh, self.bw):
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def bridge_weights(self, z, h, valid):
        length = len(z)
        q, k = self.bq(self.left(z)), self.bk(self.right(z))
        score = torch.einsum('ikd,jkd->ijk', q, k) / math.sqrt(q.shape[-1])
        score = score + self.bw(self.bh(h)).squeeze(-1)[None, None, :]
        return torch.softmax(score, dim=-1) * length

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
        pair = self.input(torch.cat([
            h[:, None].expand(-1, length, -1), h[None].expand(length, -1, -1), rbf, rotation,
            direction, offsets, probabilities[:, None].expand(-1, length, -1),
            probabilities[None].expand(length, -1, -1)], -1))
        z = self.norm(z + pair)
        left, right = self.left(z), self.right(z)
        omega = self.bridge_weights(z, h, valid)
        bridge = torch.einsum('ikd,jkd,ijk->ijd', left, right, omega) / math.sqrt(length)
        z = self.norm(z + self.triangle_gate(z).sigmoid() * self.triangle(bridge))
        weight = self.attention(z).squeeze(-1).sigmoid() * (index[:, None] != index[None])
        weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-8)
        aggregate = (weight[..., None] * z).sum(1)
        h = h + self.node(aggregate)
        coefficients = self.coordinated(aggregate).view(length, self.MODES, 3)
        basis = self.smooth_basis(length, bb.device, bb.dtype)
        coordinated = torch.einsum('lk,lkc->lc', basis, coefficients)
        coordinated = self.coordinated_bound * torch.tanh(coordinated) * valid[:, None]
        local = self.local(aggregate).view(length, 4, 3)
        local = self.local_bound * torch.tanh(local) * valid[:, None, None]
        rotation_update = rotation_matrix(.05 * self.rotation_head(aggregate).tanh() * valid[:, None])
        relative = bb - origin[:, None]
        local_coordinates = torch.einsum('lac,lcd->lad', relative, frame)
        rotated = torch.einsum('lac,ldc->lad', local_coordinates,
                               rotation_update - torch.eye(3, device=bb.device, dtype=bb.dtype))
        combined = rotated + coordinated[:, None, :] + local
        change = torch.einsum('lad,lcd->lac', combined, frame)
        return h, bb + change, z


def build_bridge_model(base, width, bounds):
    """Replace the pair layers of an already-built TwoScalePairRefiner in place."""
    from two_scale_pair_refiner import TwoScalePairRefiner
    if not isinstance(base, TwoScalePairRefiner):
        raise TypeError('expected TwoScalePairRefiner')
    base.loop_pair_layers = nn.ModuleList([
        BridgeLoopPairLayer(width, coordinated_bound=bounds, local_bound=0.20)
        for _ in range(len(base.loop_pair_layers))])
    return base
