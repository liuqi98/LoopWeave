"""Matched continuation variant that exposes the initial backbone state to every layer.

Hypothesis (to be tested, not assumed): the loop-pair layers currently see only the current
backbone, so a residue cannot tell how far it has already moved. Every layer therefore receives

    h_i^l,  x_i^l,  R_i^{0\\top}(x_i^l - x_i^0)

where the last term is the cumulative displacement expressed in the residue's *initial* local
frame, plus its magnitude. The quantity is provided only as an input feature; no penalty pulls the
prediction back towards ``x^0``, so the model may deviate as far as the data support.

Everything else is identical to ``PersistentLoopPairRefiner``: same four layers, same width, same
rigid readout, and the extra input projection is the only new tensor.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from link_interface_refiner import rotation_matrix
from orientation_interface_refiner import local_frames
from persistent_loop_pair_refiner import PersistentLoopPairRefiner


class InitStateLoopPairLayer(nn.Module):
    """Loop-pair layer with an additional initial-state displacement channel."""

    EXTRA = 7  # displacement in the initial frame (3), its norm (1), and per-atom offsets (3)

    def __init__(self, width):
        super().__init__()
        self.offset = nn.Embedding(65, 8)
        self.input = nn.Sequential(
            nn.Linear(2 * width + 76 + 2 * self.EXTRA, width), nn.SiLU(), nn.Linear(width, width))
        self.norm = nn.LayerNorm(width)
        self.left = nn.Linear(width, width)
        self.right = nn.Linear(width, width)
        self.triangle = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width))
        self.triangle_gate = nn.Linear(width, width)
        self.attention = nn.Linear(width, 1)
        self.node = nn.Linear(width, width)
        self.rigid = nn.Linear(width, 6)
        for layer in (self.node, self.rigid):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.register_buffer('centers', torch.linspace(0, 20, 16))

    @staticmethod
    def initial_state_features(origin, frame, base_bb, bb):
        """Cumulative displacement from the initial backbome, in the initial local frame.

        Three channels, all expressed in the residue's *initial* frame R^0:
          * the C-alpha displacement (rigid part),
          * the mean four-atom displacement (which also carries shape change),
          * the magnitude of the C-alpha displacement.
        """
        base_origin, base_frame, _ = local_frames(
            base_bb, torch.ones_like(base_bb[..., 0], dtype=torch.bool))
        ca_displacement = torch.einsum('lc,lcd->ld', origin - base_origin, base_frame)
        atom_displacement = torch.einsum('lc,lcd->ld', (bb - base_bb).mean(1), base_frame)
        magnitude = ca_displacement.norm(dim=-1, keepdim=True)
        return torch.cat([ca_displacement, atom_displacement, magnitude], -1)

    def forward(self, h, bb, z, probabilities, base_bb=None):
        length = len(bb)
        origin, frame, valid = local_frames(bb, torch.ones_like(bb[..., 0], dtype=torch.bool))
        delta = origin[None] - origin[:, None]
        distance = delta.norm(dim=-1)
        rbf = torch.exp(-.5 * ((distance[..., None] - self.centers) / 1.5).square())
        direction = torch.einsum('ijc,icd->ijd', F.normalize(delta, dim=-1), frame)
        rotation = torch.einsum('ica,jcb->ijab', frame, frame).flatten(-2)
        index = torch.arange(length, device=bb.device)
        offsets = self.offset((index[None] - index[:, None]).clamp(-32, 32) + 32)
        if base_bb is None:
            state = origin.new_zeros(length, self.EXTRA)
        else:
            state = self.initial_state_features(origin, frame, base_bb, bb)
        pair = self.input(torch.cat([
            h[:, None].expand(-1, length, -1), h[None].expand(length, -1, -1), rbf, rotation,
            direction, offsets, probabilities[:, None].expand(-1, length, -1),
            probabilities[None].expand(length, -1, -1),
            state[:, None].expand(-1, length, -1), state[None].expand(length, -1, -1)], -1))
        z = self.norm(z + pair)
        triangle = torch.einsum('ikd,jkd->ijd', self.left(z), self.right(z)) / math.sqrt(length)
        z = self.norm(z + self.triangle_gate(z).sigmoid() * self.triangle(triangle))
        weight = self.attention(z).squeeze(-1).sigmoid() * (index[:, None] != index[None])
        weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-8)
        aggregate = (weight[..., None] * z).sum(1)
        h = h + self.node(aggregate)
        raw = self.rigid(aggregate)
        translation = .1 * raw[:, :3].tanh() * valid[:, None]
        rotation = rotation_matrix(.05 * raw[:, 3:].tanh() * valid[:, None])
        relative = bb - origin[:, None]
        local = torch.einsum('lac,lcd->lad', relative, frame)
        change = torch.einsum('lac,ldc->lad', local, rotation - torch.eye(3, device=bb.device,
                                                                          dtype=bb.dtype))
        change = torch.einsum('lad,lcd->lac', change + translation[:, None], frame)
        return h, bb + change, z


class InitStatePairRefiner(PersistentLoopPairRefiner):
    """Same refiner with initial-state features injected in every loop-pair layer."""

    def __init__(self, width=64, layers=4, **kwargs):
        super().__init__(width=width, layers=layers, **kwargs)
        self.loop_pair_layers = nn.ModuleList(
            [InitStateLoopPairLayer(width) for _ in range(layers)])

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
            h, bb, x['_loop_pair_state'], probabilities, base_bb=x['base_bb'])
        x['_loop_pair_state'] = z
        return h, bb, pair
