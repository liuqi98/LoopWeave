"""Frame-aware relation-equivariant CDRH3 refinement.

Unlike the CA-only v1 refiner, edge messages contain invariant local-frame
directions and relative rotations derived from N/CA/C backbone frames.  Native
H3 frames are never inputs: candidate H3 frames are reconstructed from its CA
trace, the specified epitope direction and the target-free N-anchor frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from dygf import backbone_frame
from relation_interface_refiner import (
    RELATION_COUNT, _directed_edges, _knn_mask,
    RelationGraph,
    apply_pose,
    build_relation_graph,
    parse_json_array,
)


def apply_pose_about(coords, framework_ca, rotation, translation):
    coords = torch.as_tensor(coords, dtype=torch.float32)
    framework_ca = torch.as_tensor(framework_ca, dtype=torch.float32)
    rotation = torch.as_tensor(rotation, dtype=torch.float32)
    translation = torch.as_tensor(translation, dtype=torch.float32)
    pivot = framework_ca.mean(0)
    return (coords - pivot) @ rotation.T + pivot + translation


def trace_frames(coords: torch.Tensor, reference: torch.Tensor,
                 fallback_axes: torch.Tensor) -> torch.Tensor:
    """Construct an equivariant frame from a CA trace without native H3 atoms."""
    if len(coords) < 2:
        return fallback_axes.unsqueeze(0)
    previous = torch.cat([2 * coords[:1] - coords[1:2], coords[:-1]], dim=0)
    following = torch.cat([coords[1:], 2 * coords[-1:] - coords[-2:-1]], dim=0)
    x_axis = F.normalize(following - previous, dim=-1)
    toward_reference = reference.unsqueeze(0) - coords
    y_axis = toward_reference - (
        toward_reference * x_axis
    ).sum(-1, keepdim=True) * x_axis
    fallback_y = fallback_axes[:, 1].unsqueeze(0).expand_as(y_axis)
    fallback_y = fallback_y - (fallback_y * x_axis).sum(-1, keepdim=True) * x_axis
    fallback_z = fallback_axes[:, 2].unsqueeze(0).expand_as(y_axis)
    fallback_z = fallback_z - (fallback_z * x_axis).sum(-1, keepdim=True) * x_axis
    y_axis = torch.where(
        y_axis.norm(dim=-1, keepdim=True) > 1e-4, y_axis,
        torch.where(fallback_y.norm(dim=-1, keepdim=True) > 1e-4,
                    fallback_y, fallback_z),
    )
    y_axis = F.normalize(y_axis, dim=-1)
    z_axis = F.normalize(torch.cross(x_axis, y_axis, dim=-1), dim=-1)
    y_axis = torch.cross(z_axis, x_axis, dim=-1)
    return torch.stack([x_axis, y_axis, z_axis], dim=-1)


@dataclass
class FrameRelationGraph:
    node_features: torch.Tensor
    coords: torch.Tensor
    context_frames: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    design_mask: torch.Tensor
    antigen_mask: torch.Tensor
    epitope_mask: torch.Tensor
    h3_length: int
    antigen_original_indices: torch.Tensor
    frame_reference: torch.Tensor
    fallback_frame: torch.Tensor

    def to(self, device):
        return FrameRelationGraph(**{
            name: value.to(device) if torch.is_tensor(value) else value
            for name, value in self.__dict__.items()
        })


def recurrent_graph_update(graph: FrameRelationGraph, h3_coords: torch.Tensor,
                           sequence_logits: torch.Tensor, same_cutoff=8.0,
                           interface_cutoff=12.0, k=8) -> FrameRelationGraph:
    """Build the next differentiable recurrent state and refresh spatial edges.

    The forward value of the H3 sequence is a hard argmax, matching inference,
    while the backward value follows the soft probabilities.  Covalent d1/d2
    relations are preserved; all geometry-dependent relations are rebuilt from
    the updated coordinates.
    """
    length = graph.h3_length
    if h3_coords.shape != graph.coords[:length].shape:
        raise ValueError("recurrent H3 coordinate shape mismatch")
    if sequence_logits.shape != (length, 20):
        raise ValueError("recurrent H3 sequence-logit shape mismatch")
    probabilities = sequence_logits.softmax(-1)
    hard = F.one_hot(probabilities.argmax(-1), 20).to(probabilities.dtype)
    straight_through = hard + probabilities - probabilities.detach()
    node_features = torch.cat([
        torch.cat([straight_through, graph.node_features[:length, 20:]], dim=-1),
        graph.node_features[length:],
    ], dim=0)
    coords = torch.cat([h3_coords, graph.coords[length:]], dim=0)

    # Component channels are fixed metadata: H3, heavy framework, light
    # framework, antigen.  They remain discrete across recurrent steps.
    component = graph.node_features[:, 20:24].argmax(-1)
    eye = torch.eye(len(coords), dtype=torch.bool, device=coords.device)
    same_component = (component[:, None] == component[None, :]) & ~eye
    different_component = ~same_component & ~eye
    distances = torch.cdist(coords, coords)
    relation_masks = (
        same_component & (distances <= same_cutoff),
        _knn_mask(distances, same_component, k),
        different_component & (distances <= interface_cutoff),
        _knn_mask(distances, different_component, k),
    )
    sequential = graph.edge_type < 2
    edges = [graph.edge_index[:, sequential]]
    types = [graph.edge_type[sequential]]
    for relation, mask in enumerate(relation_masks, 2):
        edge = _directed_edges(mask)
        if edge.numel():
            edges.append(edge)
            types.append(torch.full(
                (edge.size(1),), relation, dtype=torch.long, device=coords.device
            ))
    values = dict(graph.__dict__)
    values.update({
        "node_features": node_features,
        "coords": coords,
        "edge_index": torch.cat(edges, dim=1),
        "edge_type": torch.cat(types),
    })
    return FrameRelationGraph(**values)


def build_frame_relation_graph(item, candidate_row, max_framework=96,
                               max_antigen=128):
    required = {
        "framework_backbone", "framework_backbone_mask",
        "antigen_backbone", "antigen_backbone_mask",
    }
    missing = required - set(item)
    if missing:
        raise KeyError(f"frame-aware context lacks {sorted(missing)}")
    base: RelationGraph = build_relation_graph(
        item, candidate_row, max_framework=max_framework, max_antigen=max_antigen
    )
    h3_coords = base.coords[:base.h3_length]
    framework_ca = torch.as_tensor(item["framework_ca"], dtype=torch.float32)
    framework_distance = torch.cdist(framework_ca, h3_coords).amin(-1)
    framework_keep = torch.topk(
        framework_distance, min(max_framework, len(framework_ca)), largest=False
    ).indices.sort().values
    rotation = parse_json_array(candidate_row["pose_rotation"])
    translation = parse_json_array(candidate_row["pose_translation"])
    framework_backbone = apply_pose_about(
        item["framework_backbone"], item["framework_ca"], rotation, translation
    )[framework_keep]
    framework_mask = torch.as_tensor(
        item["framework_backbone_mask"], dtype=torch.bool
    )[framework_keep]
    antigen_keep = base.antigen_original_indices
    antigen_backbone = torch.as_tensor(
        item["antigen_backbone"], dtype=torch.float32
    )[antigen_keep]
    antigen_mask = torch.as_tensor(
        item["antigen_backbone_mask"], dtype=torch.bool
    )[antigen_keep]
    epitope_coords = base.coords[base.epitope_mask]
    reference = epitope_coords.mean(0)
    if "h3_n_anchor_frame" in item:
        n_anchor_frame = apply_pose_about(
            item["h3_n_anchor_frame"], item["framework_ca"], rotation, translation
        )
        fallback = backbone_frame(n_anchor_frame.unsqueeze(0))[0]
    else:
        n_anchor = apply_pose_about(
            item["h3_n_anchor"], item["framework_ca"], rotation, translation
        )
        c_anchor = apply_pose_about(
            item["h3_c_anchor"], item["framework_ca"], rotation, translation
        )
        x_axis = F.normalize(c_anchor - n_anchor, dim=-1)
        y_axis = reference - n_anchor
        y_axis = y_axis - (y_axis * x_axis).sum() * x_axis
        if float(y_axis.norm()) < 1e-4:
            y_axis = base.coords[base.h3_length:].mean(0) - n_anchor
            y_axis = y_axis - (y_axis * x_axis).sum() * x_axis
        y_axis = F.normalize(y_axis, dim=-1)
        z_axis = F.normalize(torch.cross(x_axis, y_axis, dim=-1), dim=-1)
        fallback = torch.stack([x_axis, y_axis, z_axis], dim=-1)
    context_backbone = torch.cat([framework_backbone, antigen_backbone], dim=0)
    context_mask = torch.cat([framework_mask, antigen_mask], dim=0)
    measured_frames = backbone_frame(context_backbone)
    fallback_frames = trace_frames(
        base.coords[base.h3_length:], reference, fallback
    )
    complete_frames = context_mask[:, :3].all(-1)
    context_frames = torch.where(
        complete_frames[:, None, None], measured_frames, fallback_frames
    )
    return FrameRelationGraph(
        node_features=base.node_features,
        coords=base.coords,
        context_frames=context_frames,
        edge_index=base.edge_index,
        edge_type=base.edge_type,
        design_mask=base.design_mask,
        antigen_mask=base.antigen_mask,
        epitope_mask=base.epitope_mask,
        h3_length=base.h3_length,
        antigen_original_indices=base.antigen_original_indices,
        frame_reference=reference,
        fallback_frame=fallback,
    )


class FrameRelationLayer(nn.Module):
    def __init__(self, hidden_dim=128, relation_dim=16, rbf_count=16,
                 coordinate_step_scale=1.0):
        super().__init__()
        self.coordinate_step_scale = float(coordinate_step_scale)
        self.relation = nn.Embedding(RELATION_COUNT, relation_dim)
        edge_dim = hidden_dim * 2 + relation_dim + rbf_count + 3 + 9
        self.message = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.gate = nn.Linear(hidden_dim, 1)
        self.radial_coordinate = nn.Linear(hidden_dim, 1, bias=False)
        self.local_coordinate = nn.Linear(hidden_dim, 3, bias=False)
        nn.init.zeros_(self.radial_coordinate.weight)
        nn.init.zeros_(self.local_coordinate.weight)
        self.node = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.register_buffer("centers", torch.linspace(0.0, 20.0, rbf_count))

    def forward(self, hidden, coords, frames, edge_index, edge_type, design_mask):
        receiver, sender = edge_index
        difference = coords[sender] - coords[receiver]
        distance = difference.norm(dim=-1).clamp_min(1e-6)
        unit = difference / distance[:, None]
        local_direction = torch.bmm(
            unit.unsqueeze(1), frames[receiver]
        ).squeeze(1)
        relative_rotation = torch.matmul(
            frames[receiver].transpose(-1, -2), frames[sender]
        ).flatten(1)
        rbf = torch.exp(-0.5 * ((distance[:, None] - self.centers) / 1.5) ** 2)
        message = self.message(torch.cat([
            hidden[receiver], hidden[sender], self.relation(edge_type), rbf,
            local_direction, relative_rotation,
        ], dim=-1))
        message = message * torch.sigmoid(self.gate(message))
        aggregate = torch.zeros_like(hidden)
        aggregate.index_add_(0, receiver, message)
        degree = torch.zeros(len(hidden), 1, device=hidden.device, dtype=hidden.dtype)
        degree.index_add_(0, receiver, torch.ones(
            len(receiver), 1, device=hidden.device, dtype=hidden.dtype
        ))
        aggregate = aggregate / degree.clamp_min(1.0)
        hidden = self.norm(hidden + self.node(torch.cat([hidden, aggregate], dim=-1)))

        radial = 0.5 * torch.tanh(self.radial_coordinate(message)) * unit
        local = 0.25 * torch.tanh(self.local_coordinate(message))
        local_global = torch.bmm(
            local.unsqueeze(1), frames[receiver].transpose(-1, -2)
        ).squeeze(1)
        displacement = radial + local_global
        update = torch.zeros_like(coords)
        update.index_add_(0, receiver, displacement)
        update = update / degree.clamp_min(1.0)
        coords = coords + self.coordinate_step_scale * update * design_mask[:, None]
        return hidden, coords


class DynamicRelationLayer(nn.Module):
    """RAAD-style layer for a trainable branch alongside the frozen v10 trunk.

    Cross-component H3--antigen messages receive a learned, layer-specific
    interaction gate. Relation-specific projections keep sequential,
    intra-component and interface messages distinct, while pooled component
    states act as inexpensive virtual global nodes. Coordinate heads are
    zero-initialized so adding the branch preserves the frozen geometry.
    """

    def __init__(self, hidden_dim=128, relation_dim=16, rbf_count=16,
                 coordinate_step_scale=1.0):
        super().__init__()
        self.coordinate_step_scale = float(coordinate_step_scale)
        self.relation = nn.Embedding(RELATION_COUNT, relation_dim)
        edge_dim = hidden_dim * 2 + relation_dim + rbf_count + 3 + 9
        self.message = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.relation_updates = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(RELATION_COUNT)
        ])
        self.interaction_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.attention = nn.Linear(hidden_dim, 1)
        self.node = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.global_update = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.radial_coordinate = nn.Linear(hidden_dim, 1, bias=False)
        self.local_coordinate = nn.Linear(hidden_dim, 3, bias=False)
        nn.init.zeros_(self.radial_coordinate.weight)
        nn.init.zeros_(self.local_coordinate.weight)
        self.norm = nn.LayerNorm(hidden_dim)
        self.global_norm = nn.LayerNorm(hidden_dim)
        self.register_buffer("centers", torch.linspace(0.0, 20.0, rbf_count))

    def forward(self, hidden, coords, frames, edge_index, edge_type,
                design_mask, antigen_mask, component):
        receiver, sender = edge_index
        difference = coords[sender] - coords[receiver]
        distance = difference.norm(dim=-1).clamp_min(1e-6)
        unit = difference / distance[:, None]
        local_direction = torch.bmm(
            unit.unsqueeze(1), frames[receiver]
        ).squeeze(1)
        relative_rotation = torch.matmul(
            frames[receiver].transpose(-1, -2), frames[sender]
        ).flatten(1)
        rbf = torch.exp(-0.5 * ((distance[:, None] - self.centers) / 1.5) ** 2)
        message = self.message(torch.cat([
            hidden[receiver], hidden[sender], self.relation(edge_type), rbf,
            local_direction, relative_rotation,
        ], dim=-1))
        relation_message = torch.zeros_like(message)
        for relation, transform in enumerate(self.relation_updates):
            selected = edge_type == relation
            relation_message[selected] = transform(message[selected])
        message = relation_message

        # Only H3--antigen relations are dynamically pruned. Other relations
        # retain a unit gate and provide stable framework/chain context.
        interface = (
            (design_mask[receiver] & antigen_mask[sender])
            | (antigen_mask[receiver] & design_mask[sender])
        )
        learned_gate = torch.sigmoid(self.interaction_gate(message))
        gate = torch.where(interface[:, None], learned_gate,
                           torch.ones_like(learned_gate))
        message = message * gate * torch.sigmoid(self.attention(message))

        # Normalize each relation independently before merging them.  A single
        # global degree normalization makes the many antigen--antigen edges
        # drown out the much rarer interface messages, even when their learned
        # gates are informative.
        aggregate = torch.zeros_like(hidden)
        active_relations = torch.zeros(
            len(hidden), 1, device=hidden.device, dtype=hidden.dtype
        )
        relation_degrees = []
        for relation in range(RELATION_COUNT):
            selected = edge_type == relation
            degree = torch.zeros(
                len(hidden), 1, device=hidden.device, dtype=hidden.dtype
            )
            relation_aggregate = torch.zeros_like(hidden)
            selected_receiver = receiver[selected]
            relation_aggregate.index_add_(
                0, selected_receiver, message[selected]
            )
            degree.index_add_(0, selected_receiver, torch.ones(
                selected_receiver.numel(), 1, device=hidden.device,
                dtype=hidden.dtype,
            ))
            aggregate = aggregate + relation_aggregate / degree.clamp_min(1.0)
            active_relations = active_relations + (degree > 0).to(hidden.dtype)
            relation_degrees.append(degree)
        aggregate = aggregate / active_relations.clamp_min(1.0)
        hidden = self.norm(hidden + self.node(torch.cat([hidden, aggregate], dim=-1)))

        # Four virtual component nodes (H3, heavy framework, light framework,
        # antigen) plus a complex-wide node provide long-range communication.
        pooled = hidden.new_zeros(4, hidden.size(-1))
        pooled.index_add_(0, component, hidden)
        component_counts = hidden.new_zeros(4, 1)
        component_counts.index_add_(
            0, component, hidden.new_ones(len(component), 1)
        )
        pooled = pooled / component_counts.clamp_min(1.0)
        component_context = pooled[component]
        global_context = pooled.mean(0, keepdim=True).expand_as(hidden)
        hidden = self.global_norm(hidden + self.global_update(torch.cat([
            hidden, component_context, global_context,
        ], dim=-1)))

        radial = 0.25 * torch.tanh(self.radial_coordinate(message)) * unit
        local = 0.15 * torch.tanh(self.local_coordinate(message))
        local_global = torch.bmm(
            local.unsqueeze(1), frames[receiver].transpose(-1, -2)
        ).squeeze(1)
        displacement = radial + local_global
        update = torch.zeros_like(coords)
        coordinate_relations = torch.zeros_like(active_relations)
        for relation, degree in enumerate(relation_degrees):
            selected = edge_type == relation
            relation_update = torch.zeros_like(coords)
            relation_update.index_add_(
                0, receiver[selected], displacement[selected]
            )
            update = update + relation_update / degree.clamp_min(1.0)
            coordinate_relations = coordinate_relations + (
                degree > 0
            ).to(hidden.dtype)
        update = update / coordinate_relations.clamp_min(1.0)
        coords = coords + self.coordinate_step_scale * update * design_mask[:, None]
        return hidden, coords


class DynamicRelationBranch(nn.Module):
    """End-to-end conditional residual branch initialized as an identity."""

    def __init__(self, base_hidden_dim=128, hidden_dim=128, layers=4,
                 max_displacement=0.35, max_prior_adjustment=2.0,
                 candidate_base_strength=0.0, dense_interface=False,
                 coordinate_step_scale=1.0):
        super().__init__()
        self.max_displacement = float(max_displacement)
        self.max_prior_adjustment = float(max_prior_adjustment)
        self.candidate_base_strength = float(candidate_base_strength)
        self.dense_interface = bool(dense_interface)
        self.input = nn.Sequential(
            nn.Linear(27 + base_hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList([
            DynamicRelationLayer(hidden_dim=hidden_dim,
                                 coordinate_step_scale=coordinate_step_scale)
            for _ in range(layers)
        ])
        self.sequence_delta = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 20),
        )
        self.prior_delta = nn.Linear(hidden_dim, 1)
        self.contact_delta = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 16, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.sequence_projection = nn.Linear(hidden_dim, 128)
        self.epitope_projection = nn.Linear(hidden_dim, 128)
        self.register_buffer("contact_centers", torch.linspace(0.0, 20.0, 16))
        for module in (self.sequence_delta[-1], self.prior_delta,
                       self.contact_delta[-1]):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, graph, base_hidden, base_coords, base_sequence_logits,
                base_contact_logits):
        hidden = self.input(torch.cat([
            graph.node_features, base_hidden.detach(),
        ], dim=-1))
        coords = base_coords
        component = graph.node_features[:, 20:24].argmax(-1)
        edge_index, edge_type = graph.edge_index, graph.edge_type
        if self.dense_interface:
            # RAAD starts from dense antigen--antibody relations and learns to
            # prune them.  The legacy generic cross-component kNN is usually
            # consumed by nearby framework residues: 98% of the RAbD training
            # graphs consequently contain no direct H3--antigen edge.  Replace
            # any accidental H3--antigen edges by the complete bipartite set.
            receiver, sender = edge_index
            old_interface = (
                (graph.design_mask[receiver] & graph.antigen_mask[sender])
                | (graph.antigen_mask[receiver] & graph.design_mask[sender])
            )
            h3_nodes = torch.nonzero(
                graph.design_mask, as_tuple=False
            ).flatten()
            antigen_nodes = torch.nonzero(
                graph.antigen_mask, as_tuple=False
            ).flatten()
            h3_receiver = h3_nodes[:, None].expand(
                -1, len(antigen_nodes)
            ).reshape(-1)
            antigen_sender = antigen_nodes[None, :].expand(
                len(h3_nodes), -1
            ).reshape(-1)
            dense_edges = torch.stack([
                torch.cat([h3_receiver, antigen_sender]),
                torch.cat([antigen_sender, h3_receiver]),
            ])
            edge_index = torch.cat([
                edge_index[:, ~old_interface], dense_edges
            ], dim=1)
            edge_type = torch.cat([
                edge_type[~old_interface],
                torch.full(
                    (dense_edges.size(1),), 5, dtype=edge_type.dtype,
                    device=edge_type.device,
                ),
            ])
        for layer in self.layers:
            h3_frames = trace_frames(
                coords[:graph.h3_length], graph.frame_reference,
                graph.fallback_frame,
            )
            frames = torch.cat([h3_frames, graph.context_frames], dim=0)
            hidden, coords = layer(
                hidden, coords, frames, edge_index, edge_type,
                graph.design_mask, graph.antigen_mask, component,
            )
        h3_hidden = hidden[:graph.h3_length]
        base_h3_coords = base_coords[:graph.h3_length]
        raw_delta = coords[:graph.h3_length] - base_h3_coords
        if self.max_displacement > 0:
            coordinate_delta = self.max_displacement * torch.tanh(
                raw_delta / self.max_displacement
            )
        else:
            coordinate_delta = torch.zeros_like(raw_delta)
        h3_coords = base_h3_coords + coordinate_delta

        candidate_one_hot = graph.node_features[:graph.h3_length, :20]
        prior_adjustment = self.max_prior_adjustment * torch.tanh(
            self.prior_delta(h3_hidden)
        )
        sequence_base = base_sequence_logits
        if self.candidate_base_strength > 0:
            sequence_base = self.candidate_base_strength * candidate_one_hot
        sequence_logits = (
            sequence_base + self.sequence_delta(h3_hidden)
            + prior_adjustment * candidate_one_hot
        )

        antigen_hidden = hidden[graph.antigen_mask]
        antigen_coords = base_coords[graph.antigen_mask]
        distances = torch.cdist(h3_coords, antigen_coords)
        rbf = torch.exp(-0.5 * (
            (distances[..., None] - self.contact_centers) / 1.5
        ) ** 2)
        pair = torch.cat([
            h3_hidden[:, None].expand(-1, len(antigen_hidden), -1),
            antigen_hidden[None].expand(len(h3_hidden), -1, -1), rbf,
        ], dim=-1)
        contact_logits = base_contact_logits + self.contact_delta(pair).squeeze(-1)

        antigen_epitope = graph.epitope_mask[graph.antigen_mask]
        positive_hidden = antigen_hidden[antigen_epitope]
        negative_hidden = antigen_hidden[~antigen_epitope]
        if not bool(positive_hidden.numel()):
            positive_hidden = antigen_hidden
        if not bool(negative_hidden.numel()):
            negative_hidden = antigen_hidden
        return {
            "hidden": hidden,
            "sequence_logits": sequence_logits,
            "h3_coords": h3_coords,
            "contact_logits": contact_logits,
            "prior_adjustment": prior_adjustment.squeeze(-1),
            "sequence_embedding": F.normalize(
                self.sequence_projection(h3_hidden.mean(0)), dim=-1
            ),
            "epitope_embedding": F.normalize(
                self.epitope_projection(positive_hidden.mean(0)), dim=-1
            ),
            "negative_epitope_embedding": F.normalize(
                self.epitope_projection(negative_hidden.mean(0)), dim=-1
            ),
        }


class FrameRelationRefiner(nn.Module):
    def __init__(self, hidden_dim=128, layers=4, projection_dim=128,
                 sequence_adapter_dim=0, dynamic_relation_layers=0,
                 dynamic_relation_dim=128, dynamic_max_displacement=0.35,
                 dynamic_prior_adjustment=2.0,
                 dynamic_candidate_base_strength=0.0,
                 dynamic_dense_interface=False,
                 dynamic_coordinate_step_scale=1.0,
                 coordinate_step_scale=1.0):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(27, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.layers = nn.ModuleList([
            FrameRelationLayer(
                hidden_dim=hidden_dim,
                coordinate_step_scale=coordinate_step_scale,
            ) for _ in range(layers)
        ])
        self.sequence_head = nn.Linear(hidden_dim, 20)
        self.sequence_adapter = None
        if sequence_adapter_dim > 0:
            self.sequence_adapter = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, sequence_adapter_dim), nn.SiLU(),
                nn.Linear(sequence_adapter_dim, 20),
            )
            nn.init.zeros_(self.sequence_adapter[-1].weight)
            nn.init.zeros_(self.sequence_adapter[-1].bias)
        self.edit_gate = nn.Linear(hidden_dim, 1)
        self.backbone_delta = nn.Linear(hidden_dim, 9)
        self.contact_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 16, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.sequence_projection = nn.Linear(hidden_dim, projection_dim)
        self.epitope_projection = nn.Linear(hidden_dim, projection_dim)
        self.dynamic_branch = None
        if dynamic_relation_layers > 0:
            self.dynamic_branch = DynamicRelationBranch(
                base_hidden_dim=hidden_dim,
                hidden_dim=dynamic_relation_dim,
                layers=dynamic_relation_layers,
                max_displacement=dynamic_max_displacement,
                max_prior_adjustment=dynamic_prior_adjustment,
                candidate_base_strength=dynamic_candidate_base_strength,
                dense_interface=dynamic_dense_interface,
                coordinate_step_scale=dynamic_coordinate_step_scale,
            )
        self.register_buffer("contact_centers", torch.linspace(0.0, 20.0, 16))
        self.register_buffer("ideal_offsets", torch.tensor([
            [-1.20, 0.80, 0.00], [1.525, 0.00, 0.00], [2.10, -1.00, 0.00]
        ]))

    def forward(self, graph: FrameRelationGraph):
        hidden = self.input(graph.node_features)
        coords = graph.coords
        context_frames = graph.context_frames
        for layer in self.layers:
            h3_frames = trace_frames(
                coords[:graph.h3_length], graph.frame_reference, graph.fallback_frame
            )
            frames = torch.cat([h3_frames, context_frames], dim=0)
            hidden, coords = layer(
                hidden, coords, frames, graph.edge_index, graph.edge_type,
                graph.design_mask,
            )
        h3_hidden = hidden[:graph.h3_length]
        h3_coords = coords[:graph.h3_length]
        h3_frames = trace_frames(
            h3_coords, graph.frame_reference, graph.fallback_frame
        )
        candidate_one_hot = graph.node_features[:graph.h3_length, :20]
        edit_gate = torch.sigmoid(self.edit_gate(h3_hidden))
        sequence_logits = self.sequence_head(h3_hidden)
        sequence_adapter_delta = torch.zeros_like(sequence_logits)
        if self.sequence_adapter is not None:
            sequence_adapter_delta = self.sequence_adapter(h3_hidden)
            sequence_logits = sequence_logits + sequence_adapter_delta
        sequence_logits = sequence_logits + (1.0 - edit_gate) * candidate_one_hot

        local_offsets = self.ideal_offsets.unsqueeze(0) + 0.5 * torch.tanh(
            self.backbone_delta(h3_hidden).view(-1, 3, 3)
        )
        global_offsets = torch.bmm(local_offsets, h3_frames.transpose(-1, -2))
        backbone = torch.stack([
            h3_coords + global_offsets[:, 0], h3_coords,
            h3_coords + global_offsets[:, 1], h3_coords + global_offsets[:, 2],
        ], dim=1)
        antigen_hidden = hidden[graph.antigen_mask]
        antigen_coords = coords[graph.antigen_mask]
        distances = torch.cdist(h3_coords, antigen_coords)
        rbf = torch.exp(-0.5 * (
            (distances[..., None] - self.contact_centers) / 1.5
        ) ** 2)
        pair = torch.cat([
            h3_hidden[:, None].expand(-1, len(antigen_hidden), -1),
            antigen_hidden[None].expand(len(h3_hidden), -1, -1), rbf,
        ], dim=-1)
        epitope_hidden = hidden[graph.epitope_mask]
        contact_logits = self.contact_head(pair).squeeze(-1)
        negative_epitope_embedding = None
        if self.dynamic_branch is not None:
            dynamic = self.dynamic_branch(
                graph, hidden, coords, sequence_logits, contact_logits
            )
            sequence_logits = dynamic["sequence_logits"]
            h3_coords = dynamic["h3_coords"]
            contact_logits = dynamic["contact_logits"]
            h3_frames = trace_frames(
                h3_coords, graph.frame_reference, graph.fallback_frame
            )
            local_offsets = self.ideal_offsets.unsqueeze(0) + 0.5 * torch.tanh(
                self.backbone_delta(h3_hidden).view(-1, 3, 3)
            )
            global_offsets = torch.bmm(local_offsets, h3_frames.transpose(-1, -2))
            backbone = torch.stack([
                h3_coords + global_offsets[:, 0], h3_coords,
                h3_coords + global_offsets[:, 1], h3_coords + global_offsets[:, 2],
            ], dim=1)
            sequence_embedding = dynamic["sequence_embedding"]
            epitope_embedding = dynamic["epitope_embedding"]
            negative_epitope_embedding = dynamic["negative_epitope_embedding"]
            prior_adjustment = dynamic["prior_adjustment"]
        else:
            sequence_embedding = F.normalize(
                self.sequence_projection(h3_hidden.mean(0)), dim=-1
            )
            epitope_embedding = F.normalize(
                self.epitope_projection(epitope_hidden.mean(0)), dim=-1
            )
            prior_adjustment = sequence_logits.new_zeros(graph.h3_length)
        return {
            "sequence_logits": sequence_logits,
            "sequence_adapter_delta": sequence_adapter_delta,
            "edit_gate": edit_gate.squeeze(-1),
            "h3_coords": h3_coords,
            "h3_backbone": backbone,
            "h3_frames": h3_frames,
            "contact_logits": contact_logits,
            "sequence_embedding": sequence_embedding,
            "epitope_embedding": epitope_embedding,
            "negative_epitope_embedding": negative_epitope_embedding,
            "prior_adjustment": prior_adjustment,
        }
