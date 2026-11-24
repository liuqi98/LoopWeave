"""Antigen-only SES/electrostatic and heavy-atom inputs for joint H3 design.

Features are rebuilt around the CURRENT predicted loop at every layer. No
native contacts/loop coordinates are accepted by this branch. Precomputed
vectors remain in the complex frame; learnable operations use local frames.
"""
import torch
from torch import nn
from torch.nn import functional as F
from orientation_interface_refiner import OrientationInterfaceRefiner, local_frames


SURFACE_SCALARS = ('curvature', 'normal_dispersion', 'concavity', 'exposure',
                   'hydrophobic', 'donor', 'acceptor', 'potential', 'charge', 'radius', 'modeled_atom')
ATOM_SCALARS = ('radius', 'charge', 'donor', 'acceptor', 'hydrophobic',
                'exposure', 'sidechain', 'donor_direction_valid', 'acceptor_direction_valid', 'modeled_atom')


def neighborhood(bb, x, kind, k):
    points = x[f'binding_{kind}_xyz']
    scalars = x[f'binding_{kind}_scalars']
    vectors = x[f'binding_{kind}_vectors']
    origin, frame, valid = local_frames(bb, torch.ones_like(bb[..., 0], dtype=torch.bool))
    distance, indices = torch.cdist(origin, points).topk(min(k, len(points)), largest=False)
    delta = points[indices] - origin[:, None]
    direction = F.normalize(delta, dim=-1, eps=1e-6)
    local_direction = torch.einsum('lkj,lji->lki', direction, frame)
    local_vectors = torch.einsum('lkvj,lji->lkvi', vectors[indices], frame).flatten(-2)
    align = (direction[:, :, None] * vectors[indices]).sum(-1)
    centers = torch.arange(8, device=bb.device, dtype=bb.dtype) * 2.
    radial = torch.exp(-.5 * ((distance[..., None] - centers) / 2.).square())
    atom_distances = (points[indices, None] - bb[:, None]).norm(dim=-1).clamp(max=20.) / 10.
    features = torch.cat([scalars[indices], local_vectors, local_direction,
                          radial, atom_distances, align], -1)
    mask = (distance < 12.) & valid[:, None]
    return features, mask, frame


class BindingLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.messages = nn.ModuleDict({
            'surface': nn.Sequential(nn.Linear(width + 42, width), nn.SiLU(), nn.LayerNorm(width)),
            'atom': nn.Sequential(nn.Linear(width + 33, width), nn.SiLU(), nn.LayerNorm(width))})
        self.gates = nn.ModuleDict({k: nn.Linear(width, 1) for k in self.messages})
        self.hidden = nn.Linear(2 * width, width)
        self.translation = nn.Linear(2 * width, 3)
        for layer in (self.hidden, self.translation):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, h, bb, x):
        summaries = []
        any_neighbor = torch.zeros(len(bb), dtype=torch.bool, device=bb.device)
        for kind, k in (('surface', 48), ('atom', 24)):
            features, mask, frame = neighborhood(bb, x, kind, k)
            message = self.messages[kind](torch.cat([h[:, None].expand(-1, features.shape[1], -1), features], -1))
            weight = self.gates[kind](message).squeeze(-1).sigmoid() * mask
            weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-8)
            summaries.append((message * weight[..., None]).sum(1))
            any_neighbor |= mask.any(-1)
        summary = torch.cat(summaries, -1)
        h = h + self.hidden(summary) * any_neighbor[:, None]
        # Bounded residue-rigid translation preserves within-residue geometry.
        shift = .15 * self.translation(summary).tanh() * any_neighbor[:, None]
        shift = torch.einsum('li,lji->lj', shift, frame)
        return h, bb + shift[:, None]


class BindingInterfaceRefiner(OrientationInterfaceRefiner):
    def __init__(self, width=64, layers=4, **kwargs):
        super().__init__(width=width, layers=layers, **kwargs)
        self.binding_layers = nn.ModuleList([BindingLayer(width) for _ in range(layers)])

    def load_source(self, state):
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or any(not k.startswith('binding_layers.') for k in result.missing_keys):
            raise ValueError(result)

    def forward(self, x):
        for kind, ns, nv in (('surface', 11, 4), ('atom', 10, 2)):
            pts, sc, vec = [x[f'binding_{kind}_{suffix}'] for suffix in ('xyz', 'scalars', 'vectors')]
            if not len(pts) or pts.shape != (len(pts), 3) or sc.shape != (len(pts), ns) or vec.shape != (len(pts), nv, 3):
                raise ValueError(f'Invalid/missing {kind} binding features')
            if not all(torch.isfinite(t).all() for t in (pts, sc, vec)):
                raise ValueError(f'Nonfinite {kind} binding features; no silent zero filling')
        return super().forward(x)

    def refine_relations(self, h, bb, pair, c, x, layer_index):
        h, bb = self.binding_layers[layer_index](h, bb, x)
        return super().refine_relations(h, bb, pair, c, x, layer_index)
