"""Matched continuation variant with explicit dual-anchor conditioning.

Hypothesis (to be tested): knowing where the antigen is does not by itself tell the model how the
loop must connect the two fixed framework anchors. The layer-1 diagnostic showed that the first
pair layer mostly translates the loop (raw RMSD improves more than aligned RMSD) and that 38% of
complexes get worse there, which is the symptom of missing anchor constraints.

Every loop-pair layer therefore receives, for each H3 residue ``i`` and each flanking anchor
``a in {left, right}``:

    R_a^T (o_i - o_a)           relative position in the anchor frame   (3)
    R_a^T unit(o_i - o_a)       relative direction in the anchor frame   (3)
    ||o_i - o_a||               anchor distance                          (1)
    d_seq(i, a) / (L - 1)       normalised chain distance to the anchor  (1)

plus one anchor-pair term shared by all residues, ``vec(R_left^T R_right)`` (9). Invalid anchors
are masked to zero. The features are inputs only; no positional restraint or loss is added.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from link_interface_refiner import rotation_matrix
from orientation_interface_refiner import local_frames
from persistent_loop_pair_refiner import PersistentLoopPairRefiner


class AnchorLoopPairLayer(nn.Module):
    """Loop-pair layer with per-residue dual-anchor geometric conditioning."""

    PER_ANCHOR = 8                      # 3 local position + 3 local direction + 1 distance + 1 seq
    PER_RESIDUE = 2 * PER_ANCHOR + 9    # plus the anchor-pair relative orientation
    EXTRA = PER_RESIDUE

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
    def anchor_features(origin, anchors, anchor_mask):
        """Dual-anchor descriptor for every H3 residue (L, PER_RESIDUE)."""
        origin = origin
        length = len(origin)
        anchor_origin, anchor_frame, anchor_valid = local_frames(anchors, anchor_mask)
        anchor_valid = anchor_valid.bool()
        index = torch.arange(length, device=origin.device).float()
        denominator = max(length - 1, 1)
        blocks = []
        for slot in range(2):
            delta = origin - anchor_origin[slot]
            distance = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            local_position = torch.einsum('lc,cd->ld', delta, anchor_frame[slot])
            local_direction = torch.einsum('lc,cd->ld', delta / distance, anchor_frame[slot])
            sequence_distance = ((index - (-1 if slot == 0 else length)).abs() / denominator
                                 ).unsqueeze(-1)
            block = torch.cat([local_position, local_direction, distance, sequence_distance], -1)
            blocks.append(block * anchor_valid[slot].to(block.dtype))
        relative_orientation = torch.einsum('ac,cb->ab', anchor_frame[0], anchor_frame[1]).flatten()
        relative_orientation = (relative_orientation.unsqueeze(0).expand(length, -1) *
                                (anchor_valid[0] & anchor_valid[1]).to(relative_orientation.dtype))
        return torch.cat(blocks + [relative_orientation], -1)

    def forward(self, h, bb, z, probabilities, anchors=None, anchor_mask=None):
        length = len(bb)
        origin, frame, valid = local_frames(bb, torch.ones_like(bb[..., 0], dtype=torch.bool))
        delta = origin[None] - origin[:, None]
        distance = delta.norm(dim=-1)
        rbf = torch.exp(-.5 * ((distance[..., None] - self.centers) / 1.5).square())
        direction = torch.einsum('ijc,icd->ijd', F.normalize(delta, dim=-1), frame)
        rotation = torch.einsum('ica,jcb->ijab', frame, frame).flatten(-2)
        index = torch.arange(length, device=bb.device)
        offsets = self.offset((index[None] - index[:, None]).clamp(-32, 32) + 32)
        if anchors is None or anchor_mask is None:
            state = origin.new_zeros(length, self.EXTRA)
        else:
            state = self.anchor_features(origin, anchors, anchor_mask)
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
        rotation_matrix_update = rotation_matrix(.05 * raw[:, 3:].tanh() * valid[:, None])
        relative = bb - origin[:, None]
        local = torch.einsum('lac,lcd->lad', relative, frame)
        change = torch.einsum('lac,ldc->lad', local,
                              rotation_matrix_update - torch.eye(3, device=bb.device,
                                                                 dtype=bb.dtype))
        change = torch.einsum('lad,lcd->lac', change + translation[:, None], frame)
        return h, bb + change, z


class AnchorPairRefiner(PersistentLoopPairRefiner):
    """Same refiner with dual-anchor conditioning in every loop-pair layer."""

    def __init__(self, width=64, layers=4, **kwargs):
        super().__init__(width=width, layers=layers, **kwargs)
        self.loop_pair_layers = nn.ModuleList([AnchorLoopPairLayer(width) for _ in range(layers)])

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
            h, bb, x['_loop_pair_state'], probabilities,
            anchors=x.get('anchors'), anchor_mask=x.get('anchor_mask'))
        x['_loop_pair_state'] = z
        return h, bb, pair
