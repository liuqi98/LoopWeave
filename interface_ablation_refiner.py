"""Capacity-preserving interface ablations for the binding branch.

Two orthogonal knobs, usable at inference time and during matched training:

``feature_groups``
    Which binding *input* blocks reach the branch. ``joint`` is the original
    behaviour. ``surface_geometry`` and ``electrostatics_chemistry`` zero the
    columns that carry the other block; ``none`` zeroes every scalar/vector
    column. Masking happens on the RAW per-kind scalars/vectors *before* any
    derived quantity (local vectors, alignment) is built, so excluded
    information cannot leak through a derived channel. Parameter count,
    module shapes, losses and update counts are unchanged; only the visible
    inputs differ. This tests dependence, not learnable value.

``neighborhood``
    ``dynamic`` re-selects the antigen neighbor *members* from the CURRENT loop
    conformation at every layer (the original behaviour). ``static`` freezes
    the member index set and membership mask from the initial loop
    conformation (``x['base_bb']`` once per forward) while still recomputing
    every feature VALUE from the current conformation. Feature definitions,
    network, and the number of updates are identical, isolating the value of
    re-selecting interface members as the loop moves.

Defaults reproduce the archived ``BindingInterfaceRefiner`` /
``PersistentLoopPairRefiner`` exactly, and the resulting state dicts load with
``strict=True`` in both directions.
"""
import torch
from torch import nn
from torch.nn import functional as F
from binding_interface_refiner import BindingInterfaceRefiner, SURFACE_SCALARS, ATOM_SCALARS
from persistent_loop_pair_refiner import LoopPairLayer
from orientation_interface_refiner import local_frames

SURFACE_VECTORS = ('normal', 'electric_field', 'donor_direction', 'acceptor_direction')
ATOM_VECTORS = ('donor_direction', 'acceptor_direction')

# Name-based definitions keep the masks robust to column reordering upstream.
FEATURE_GROUPS = {
    'joint': {
        'surface': {'scalars': SURFACE_SCALARS, 'vectors': SURFACE_VECTORS},
        'atom': {'scalars': ATOM_SCALARS, 'vectors': ATOM_VECTORS}},
    'surface_geometry': {
        'surface': {'scalars': ('curvature', 'normal_dispersion', 'concavity', 'exposure',
                                'radius', 'modeled_atom'),
                    'vectors': ('normal',)},
        'atom': {'scalars': ('radius', 'exposure', 'sidechain', 'modeled_atom'),
                 'vectors': ()}},
    'electrostatics_chemistry': {
        'surface': {'scalars': ('hydrophobic', 'donor', 'acceptor', 'potential', 'charge'),
                    'vectors': ('electric_field', 'donor_direction', 'acceptor_direction')},
        'atom': {'scalars': ('charge', 'donor', 'acceptor', 'hydrophobic',
                             'donor_direction_valid', 'acceptor_direction_valid'),
                 'vectors': ('donor_direction', 'acceptor_direction')}},
    'none': {
        'surface': {'scalars': (), 'vectors': ()},
        'atom': {'scalars': (), 'vectors': ()}},
}

# Neighbor positions/distances/rbf/alignment are the shared geometric scaffold
# of the branch and are never zeroed by ``feature_groups``; see README/report.
NEIGHBORHOODS = ('dynamic', 'static')
NEIGHBOR_COUNT = {'surface': 48, 'atom': 24}
NEIGHBOR_RADIUS = 12.


def names(kind):
    return (SURFACE_SCALARS, SURFACE_VECTORS) if kind == 'surface' else (ATOM_SCALARS, ATOM_VECTORS)


def keep_mask(columns, selected):
    chosen = set(selected)
    if not chosen <= set(columns):
        raise ValueError(f'Unknown feature name(s): {sorted(chosen - set(columns))}')
    return torch.tensor([1. if name in chosen else 0. for name in columns])


def masked_raw(kind, scalars, vectors, feature_groups):
    """Zero excluded scalar/vector columns before any derived feature is built."""
    if feature_groups not in FEATURE_GROUPS:
        raise ValueError(f'Unknown feature_groups: {feature_groups}')
    scalar_names, vector_names = names(kind)
    if scalars.shape[-1] != len(scalar_names) or vectors.shape[-2] != len(vector_names):
        raise ValueError(f'{kind} feature width does not match the declared name lists')
    spec = FEATURE_GROUPS[feature_groups][kind]
    smask = keep_mask(scalar_names, spec['scalars']).to(scalars.dtype).to(scalars.device)
    vmask = keep_mask(vector_names, spec['vectors']).to(vectors.dtype).to(vectors.device)
    return scalars * smask, vectors * vmask[None, :, None]


def member_selection(bb, x, kind, k):
    """Nearest-k antigen points of ``bb`` within the radius, with membership mask."""
    points = x[f'binding_{kind}_xyz']
    origin, _, valid = local_frames(bb, torch.ones_like(bb[..., 0], dtype=torch.bool))
    distance, indices = torch.cdist(origin, points).topk(min(k, len(points)), largest=False)
    return indices, (distance < NEIGHBOR_RADIUS) & valid[:, None]


def binding_neighborhood(bb, x, kind, feature_groups='joint', frozen=None):
    """Original neighborhood, optionally over a frozen member set."""
    k = NEIGHBOR_COUNT[kind]
    points = x[f'binding_{kind}_xyz']
    scalars = x[f'binding_{kind}_scalars']
    vectors = x[f'binding_{kind}_vectors']
    origin, frame, valid = local_frames(bb, torch.ones_like(bb[..., 0], dtype=torch.bool))
    scalars, vectors = masked_raw(kind, scalars, vectors, feature_groups)
    if frozen is None:
        distance, indices = torch.cdist(origin, points).topk(min(k, len(points)), largest=False)
        mask = (distance < NEIGHBOR_RADIUS) & valid[:, None]
    else:
        indices, frozen_mask = frozen
        distance = (points[indices] - origin[:, None]).norm(dim=-1)
        mask = frozen_mask & valid[:, None]
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
    return features, mask, frame


class AblatedBindingLayer(nn.Module):
    """Identical parameters to :class:`BindingLayer`; masked inputs / frozen members."""

    def __init__(self, width, feature_groups='joint', neighborhood='dynamic'):
        super().__init__()
        if neighborhood not in NEIGHBORHOODS:
            raise ValueError(f'Unknown neighborhood: {neighborhood}')
        self.feature_groups = feature_groups
        self.neighborhood = neighborhood
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
        for kind in ('surface', 'atom'):
            frozen = None
            if self.neighborhood == 'static':
                key = f'_static_members_{kind}'
                if key not in x:
                    x[key] = member_selection(x['base_bb'], x, kind, NEIGHBOR_COUNT[kind])
                frozen = x[key]
            features, mask, frame = binding_neighborhood(bb, x, kind, self.feature_groups, frozen)
            message = self.messages[kind](torch.cat([h[:, None].expand(-1, features.shape[1], -1), features], -1))
            weight = self.gates[kind](message).squeeze(-1).sigmoid() * mask
            weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-8)
            summaries.append((message * weight[..., None]).sum(1))
            any_neighbor |= mask.any(-1)
        summary = torch.cat(summaries, -1)
        h = h + self.hidden(summary) * any_neighbor[:, None]
        shift = .15 * self.translation(summary).tanh() * any_neighbor[:, None]
        shift = torch.einsum('li,lji->lj', shift, frame)
        return h, bb + shift[:, None]


class AblatedBindingInterfaceRefiner(BindingInterfaceRefiner):
    def __init__(self, width=64, layers=4, feature_groups='joint', neighborhood='dynamic', **kwargs):
        super().__init__(width=width, layers=layers, **kwargs)
        self.feature_groups = feature_groups
        self.neighborhood = neighborhood
        self.binding_layers = nn.ModuleList(
            [AblatedBindingLayer(width, feature_groups, neighborhood) for _ in range(layers)])


class AblatedPersistentLoopPairRefiner(AblatedBindingInterfaceRefiner):
    """Binding ablation plus the in-loop H3-H3 pair branch of the full model.

    ``stage_trace`` may be set to a list to record the post-layer backbone after
    every layer (layer 0 first); used for per-layer contact diagnostics.
    """

    def __init__(self, width=64, layers=4, **kwargs):
        super().__init__(width=width, layers=layers, **kwargs)
        self.loop_pair_layers = nn.ModuleList([LoopPairLayer(width) for _ in range(layers)])
        self.loop_width = width
        self.stage_trace = None

    def load_binding(self, state):
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or any(not key.startswith('loop_pair_layers.') for key in result.missing_keys):
            raise ValueError(result)

    def forward(self, x):
        length = len(x['base_bb'])
        private = {**x, '_loop_pair_state': x['base_bb'].new_zeros(length, length, self.loop_width)}
        out = super().forward(private)
        out['loop_pair_state'] = private['_loop_pair_state']
        return out

    def refine_relations(self, h, bb, pair, c, x, layer_index):
        h, bb, pair = super().refine_relations(h, bb, pair, c, x, layer_index)
        probabilities = (x['base_logits'] + self.native_head(h)).softmax(-1)
        h, bb, z = self.loop_pair_layers[layer_index](h, bb, x['_loop_pair_state'], probabilities)
        x['_loop_pair_state'] = z
        if self.stage_trace is not None:
            self.stage_trace.append(bb.detach().clone())
        return h, bb, pair
