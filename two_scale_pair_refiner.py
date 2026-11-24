"""Matched continuation variant with an explicit two-scale coordinate update.

Hypothesis (to be tested): the current layer emits one residue-level translation, so a smooth
bend of the whole loop and a local correction of one residue are produced by the same readout and
cannot be separated. This variant splits the translation into

    Delta x = Delta x_coordinated + Delta x_local

* ``coordinated``: a low-frequency field over the loop index, built from three cosine modes and
  bounded to +-0.3 A, so it can bend or shift contiguous stretches but cannot make a residue jump.
* ``local``: a per-atom displacement, bounded to +-0.15 A, free in frequency, so single-residue
  corrections remain possible.

The residue rotation readout is unchanged. Both branches are separately bounded, so neither can
absorb the other's role, and their contributions can be ablated one at a time.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from link_interface_refiner import rotation_matrix
from orientation_interface_refiner import local_frames
from persistent_loop_pair_refiner import PersistentLoopPairRefiner


class TwoScaleLoopPairLayer(nn.Module):
    """Loop-pair layer whose translation is split into smooth and local components."""

    MODES = 3
    COORDINATED_BOUND = 0.30
    LOCAL_BOUND = 0.15

    def __init__(self, width, coordinated_bound=None, local_bound=None):
        super().__init__()
        # Instance-level bounds allow a per-layer schedule; the class attributes remain the
        # defaults so checkpoints trained before the schedule existed stay reproducible.
        self.coordinated_bound = (type(self).COORDINATED_BOUND if coordinated_bound is None
                                  else float(coordinated_bound))
        self.local_bound = (type(self).LOCAL_BOUND if local_bound is None
                            else float(local_bound))
        self.offset = nn.Embedding(65, 8)
        self.input = nn.Sequential(nn.Linear(2 * width + 76, width), nn.SiLU(),
                                   nn.Linear(width, width))
        self.norm = nn.LayerNorm(width)
        self.left = nn.Linear(width, width)
        self.right = nn.Linear(width, width)
        self.triangle = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width))
        self.triangle_gate = nn.Linear(width, width)
        self.attention = nn.Linear(width, 1)
        self.node = nn.Linear(width, width)
        self.coordinated = nn.Linear(width, self.MODES * 3)
        self.local = nn.Linear(width, 12)
        self.rotation_head = nn.Linear(width, 3)
        for layer in (self.node, self.coordinated, self.local, self.rotation_head):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.register_buffer('centers', torch.linspace(0, 20, 16))

    def smooth_basis(self, length, device, dtype):
        index = torch.arange(length, device=device, dtype=dtype)
        phase = math.pi * index.unsqueeze(-1) / max(length - 1, 1)
        modes = torch.arange(self.MODES, device=device, dtype=dtype)
        return torch.cos(phase * modes)  # (L, MODES)

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
        triangle = torch.einsum('ikd,jkd->ijd', self.left(z), self.right(z)) / math.sqrt(length)
        z = self.norm(z + self.triangle_gate(z).sigmoid() * self.triangle(triangle))
        weight = self.attention(z).squeeze(-1).sigmoid() * (index[:, None] != index[None])
        weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-8)
        aggregate = (weight[..., None] * z).sum(1)
        h = h + self.node(aggregate)

        # Coordinated, low-frequency translation shared across contiguous residues.
        coefficients = self.coordinated(aggregate).view(length, self.MODES, 3)
        basis = self.smooth_basis(length, bb.device, bb.dtype)
        coordinated = torch.einsum('lk,lkc->lc', basis, coefficients)
        coordinated = self.coordinated_bound * torch.tanh(coordinated) * valid[:, None]
        # Local, per-atom displacement.
        local = self.local(aggregate).view(length, 4, 3)
        local = self.local_bound * torch.tanh(local) * valid[:, None, None]
        # Residue rotation readout, unchanged from the baseline layer.
        rotation_update = rotation_matrix(
            .05 * self.rotation_head(aggregate).tanh() * valid[:, None])

        relative = bb - origin[:, None]
        local_coordinates = torch.einsum('lac,lcd->lad', relative, frame)
        rotated = torch.einsum('lac,ldc->lad', local_coordinates,
                               rotation_update - torch.eye(3, device=bb.device, dtype=bb.dtype))
        combined = rotated + coordinated[:, None, :] + local
        change = torch.einsum('lad,lcd->lac', combined, frame)
        return h, bb + change, z


class TwoScalePairRefiner(PersistentLoopPairRefiner):
    """Same refiner whose per-layer translation is split into coordinated and local parts."""

    def __init__(self, width=64, layers=4, coordinated_bounds=None, local_bounds=None, **kwargs):
        super().__init__(width=width, layers=layers, **kwargs)
        def schedule(values):
            if values is None:
                return [None] * layers
            values = list(values)
            if len(values) != layers:
                raise ValueError(f'expected {layers} bounds, got {len(values)}')
            return values
        coordinated, local = schedule(coordinated_bounds), schedule(local_bounds)
        self.loop_pair_layers = nn.ModuleList([
            TwoScaleLoopPairLayer(width, coordinated_bound=coordinated[i], local_bound=local[i])
            for i in range(layers)])

    def forward(self, x):
        length = len(x['base_bb'])
        private = {**x, '_loop_pair_state': x['base_bb'].new_zeros(length, length, self.loop_width)}
        out = super(PersistentLoopPairRefiner, self).forward(private)
        out['loop_pair_state'] = private['_loop_pair_state']
        return out

    def refine_relations(self, h, bb, pair, c, x, layer_index):
        h, bb, pair = super(PersistentLoopPairRefiner, self).refine_relations(
            h, bb, pair, c, x, layer_index)
        probabilities = (x['base_logits'] + self.native_head(h)).softmax(-1)
        h, bb, z = self.loop_pair_layers[layer_index](
            h, bb, x['_loop_pair_state'], probabilities)
        x['_loop_pair_state'] = z
        return h, bb, pair
