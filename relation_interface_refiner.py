"""Relation-aware equivariant refinement for generated CDRH3 candidates.

The module is deliberately downstream of a frozen generator.  It consumes only
candidate sequence/coordinates and target-free antibody--antigen context, then
predicts sequence logits, an H3 coordinate correction and an H3--antigen
contact map.  Native H3 fields are used only by the training script as labels.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: index for index, aa in enumerate(AMINO_ACIDS)}
NODE_FEATURE_DIM = 27  # AA20 + component4 + epitope + design + relative position
RELATION_COUNT = 6


def sequence_one_hot(sequence: str) -> torch.Tensor:
    result = torch.zeros(len(sequence), 20)
    for index, amino_acid in enumerate(sequence):
        # ``X`` is used by target-free, from-scratch design as a true masked
        # residue.  Treating it as alanine leaks an arbitrary sequence prior
        # into every design position and makes an ablation against RAAD's
        # sequence-free initialization needlessly ambiguous.
        if amino_acid in AA_TO_INDEX:
            result[index, AA_TO_INDEX[amino_acid]] = 1.0
    return result


def apply_pose(coords, rotation, translation):
    coords = torch.as_tensor(coords, dtype=torch.float32)
    rotation = torch.as_tensor(rotation, dtype=torch.float32)
    translation = torch.as_tensor(translation, dtype=torch.float32)
    pivot = coords.mean(0, keepdim=True)
    return (coords - pivot) @ rotation.T + pivot + translation


def parse_json_array(value):
    return np.asarray(json.loads(value), dtype=np.float32)


@dataclass
class RelationGraph:
    node_features: torch.Tensor
    coords: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    design_mask: torch.Tensor
    antigen_mask: torch.Tensor
    epitope_mask: torch.Tensor
    h3_length: int
    antigen_original_indices: torch.Tensor

    def to(self, device):
        return RelationGraph(**{
            name: value.to(device) if torch.is_tensor(value) else value
            for name, value in self.__dict__.items()
        })


def _directed_edges(mask: torch.Tensor):
    return torch.nonzero(mask, as_tuple=False).T.contiguous()


def _knn_mask(distances: torch.Tensor, eligible: torch.Tensor, k: int):
    """Vectorized row-wise kNN restricted to an eligibility mask."""
    output = torch.zeros_like(eligible)
    if distances.numel() == 0 or k <= 0:
        return output
    masked = distances.masked_fill(~eligible, torch.inf)
    values, selected = torch.topk(
        masked, min(k, distances.size(1)), dim=1, largest=False
    )
    output.scatter_(1, selected, torch.isfinite(values))
    return output


def build_relation_graph(item, candidate_row, max_framework=96, max_antigen=128,
                         same_cutoff=8.0, interface_cutoff=12.0, k=8):
    """Build six RAAD-inspired relation types around one generated H3."""
    h3_sequence = str(candidate_row["generated_cdrh3"])
    h3_coords = torch.as_tensor(
        parse_json_array(candidate_row["ca_coords"]), dtype=torch.float32
    )
    length = min(len(h3_sequence), len(h3_coords))
    h3_sequence, h3_coords = h3_sequence[:length], h3_coords[:length]

    framework_sequence = str(item["framework_seq"])
    framework_coords = apply_pose(
        item["framework_ca"], parse_json_array(candidate_row["pose_rotation"]),
        parse_json_array(candidate_row["pose_translation"]),
    )
    framework_distance = torch.cdist(framework_coords, h3_coords).amin(-1)
    framework_keep = torch.topk(
        framework_distance, min(max_framework, len(framework_coords)), largest=False
    ).indices.sort().values
    framework_coords = framework_coords[framework_keep]
    framework_sequence = "".join(framework_sequence[int(i)] for i in framework_keep)

    antigen_sequence = str(item["antigen_seq"])
    antigen_coords_all = torch.as_tensor(item["antigen_ca"], dtype=torch.float32)
    epitope_all = torch.as_tensor(item["antigen_epitope_mask"], dtype=torch.bool)
    antigen_distance = torch.cdist(antigen_coords_all, h3_coords).amin(-1)
    epitope_indices = torch.nonzero(epitope_all, as_tuple=False).flatten()
    if epitope_indices.numel() > max_antigen:
        epitope_indices = epitope_indices[torch.topk(
            antigen_distance[epitope_indices], max_antigen, largest=False
        ).indices]
    remaining = max_antigen - epitope_indices.numel()
    non_epitope = torch.nonzero(~epitope_all, as_tuple=False).flatten()
    if remaining > 0 and non_epitope.numel() > 0:
        fill = non_epitope[torch.topk(
            antigen_distance[non_epitope], min(remaining, non_epitope.numel()),
            largest=False,
        ).indices]
        antigen_keep = torch.cat([epitope_indices, fill]).unique().sort().values
    else:
        antigen_keep = epitope_indices.unique().sort().values
    antigen_coords = antigen_coords_all[antigen_keep]
    antigen_sequence = "".join(antigen_sequence[int(i)] for i in antigen_keep)
    antigen_epitope = epitope_all[antigen_keep]

    heavy_length = len(str(item.get("heavy_sequence_without_h3", framework_sequence)))
    framework_component = torch.where(
        framework_keep < heavy_length,
        torch.ones_like(framework_keep), torch.full_like(framework_keep, 2),
    )
    component = torch.cat([
        torch.zeros(length, dtype=torch.long), framework_component,
        torch.full((len(antigen_keep),), 3, dtype=torch.long),
    ])
    # Restore heavy-chain residue numbering across the removed H3 span.  This
    # prevents a false sequential edge directly joining the two H3 flanks and
    # lets d1/d2 relations connect each generated endpoint to its true anchor.
    h3_native_indices = item.get("heavy_h3_indices")
    if h3_native_indices is not None:
        h3_original = torch.as_tensor(h3_native_indices, dtype=torch.long)[:length]
        total_heavy = heavy_length + len(torch.as_tensor(h3_native_indices))
        heavy_keep_mask = torch.ones(total_heavy, dtype=torch.bool)
        heavy_keep_mask[torch.as_tensor(h3_native_indices, dtype=torch.long)] = False
        heavy_original_all = torch.arange(total_heavy)[heavy_keep_mask]
        framework_original = torch.where(
            framework_keep < heavy_length,
            heavy_original_all[framework_keep.clamp_max(max(heavy_length - 1, 0))],
            framework_keep - heavy_length,
        )
    else:
        h3_original = torch.arange(length)
        framework_original = framework_keep
    original_index = torch.cat([h3_original, framework_original, antigen_keep])
    chain = torch.where(
        component <= 1, torch.zeros_like(component), component - 1
    )
    sequence = h3_sequence + framework_sequence + antigen_sequence
    coords = torch.cat([h3_coords, framework_coords, antigen_coords], dim=0)
    design_mask = component == 0
    antigen_mask = component == 3
    epitope_mask = torch.zeros(len(coords), dtype=torch.bool)
    epitope_mask[antigen_mask] = antigen_epitope

    aa = sequence_one_hot(sequence)
    component_one_hot = F.one_hot(component, 4).float()
    position = torch.zeros(len(coords), 1)
    for value in range(4):
        selected = component == value
        count = int(selected.sum())
        if count:
            position[selected, 0] = torch.linspace(0.0, 1.0, count)
    node_features = torch.cat([
        aa, component_one_hot, epitope_mask[:, None].float(),
        design_mask[:, None].float(), position,
    ], dim=-1)

    distances = torch.cdist(coords, coords)
    eye = torch.eye(len(coords), dtype=torch.bool)
    same_component = component[:, None] == component[None, :]
    same_component &= ~eye
    different_component = ~same_component & ~eye
    same_chain = chain[:, None] == chain[None, :]
    same_chain &= ~eye
    index_delta = (original_index[:, None] - original_index[None, :]).abs()
    relation_masks = [
        same_chain & (index_delta == 1),
        same_chain & (index_delta == 2),
        same_component & (distances <= same_cutoff),
        _knn_mask(distances, same_component, k),
        different_component & (distances <= interface_cutoff),
        _knn_mask(distances, different_component, k),
    ]
    edges, types = [], []
    for relation, mask in enumerate(relation_masks):
        edge = _directed_edges(mask)
        if edge.numel():
            edges.append(edge)
            types.append(torch.full((edge.size(1),), relation, dtype=torch.long))
    if not edges:
        raise ValueError("relation graph contains no edges")
    return RelationGraph(
        node_features=node_features, coords=coords,
        edge_index=torch.cat(edges, dim=1), edge_type=torch.cat(types),
        design_mask=design_mask, antigen_mask=antigen_mask,
        epitope_mask=epitope_mask, h3_length=length,
        antigen_original_indices=antigen_keep,
    )


class RelationLayer(nn.Module):
    def __init__(self, hidden_dim=128, relation_dim=16, rbf_count=16):
        super().__init__()
        self.relation = nn.Embedding(RELATION_COUNT, relation_dim)
        edge_dim = hidden_dim * 2 + relation_dim + rbf_count
        self.message = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.gate = nn.Linear(hidden_dim, 1)
        self.coordinate = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.zeros_(self.coordinate.weight)
        self.node = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.register_buffer("centers", torch.linspace(0.0, 20.0, rbf_count))

    def forward(self, hidden, coords, edge_index, edge_type, design_mask):
        receiver, sender = edge_index
        difference = coords[sender] - coords[receiver]
        distance = torch.linalg.vector_norm(difference, dim=-1).clamp_min(1e-6)
        rbf = torch.exp(-0.5 * ((distance[:, None] - self.centers) / 1.5) ** 2)
        message = self.message(torch.cat([
            hidden[receiver], hidden[sender], self.relation(edge_type), rbf,
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

        scalar = 0.1 * torch.tanh(self.coordinate(message))
        displacement = scalar * difference / distance[:, None]
        coordinate_update = torch.zeros_like(coords)
        coordinate_update.index_add_(0, receiver, displacement)
        coordinate_update = coordinate_update / degree.clamp_min(1.0)
        coords = coords + coordinate_update * design_mask[:, None]
        return hidden, coords


class RelationInterfaceRefiner(nn.Module):
    def __init__(self, hidden_dim=128, layers=3, projection_dim=128):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(NODE_FEATURE_DIM, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList([
            RelationLayer(hidden_dim=hidden_dim) for _ in range(layers)
        ])
        self.sequence_head = nn.Linear(hidden_dim, 20)
        self.contact_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 16, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.sequence_projection = nn.Linear(hidden_dim, projection_dim)
        self.epitope_projection = nn.Linear(hidden_dim, projection_dim)
        self.register_buffer("contact_centers", torch.linspace(0.0, 20.0, 16))

    def forward(self, graph: RelationGraph):
        hidden = self.input(graph.node_features)
        coords = graph.coords
        for layer in self.layers:
            hidden, coords = layer(
                hidden, coords, graph.edge_index, graph.edge_type, graph.design_mask
            )
        h3_hidden = hidden[:graph.h3_length]
        h3_coords = coords[:graph.h3_length]
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
        contact_logits = self.contact_head(pair).squeeze(-1)
        epitope_hidden = hidden[graph.epitope_mask]
        return {
            "sequence_logits": self.sequence_head(h3_hidden),
            "h3_coords": h3_coords,
            "contact_logits": contact_logits,
            "sequence_embedding": F.normalize(
                self.sequence_projection(h3_hidden.mean(0)), dim=-1
            ),
            "epitope_embedding": F.normalize(
                self.epitope_projection(epitope_hidden.mean(0)), dim=-1
            ),
        }


def symmetric_masked_infonce(sequence_embeddings, epitope_embeddings, pdb_ids,
                             temperature=0.2):
    """Symmetric in-batch InfoNCE while excluding same-PDB false negatives."""
    if len(sequence_embeddings) < 2:
        return sequence_embeddings[0].new_zeros(())
    sequence = torch.stack(sequence_embeddings)
    epitope = torch.stack(epitope_embeddings)
    logits = sequence @ epitope.T / temperature
    allowed = torch.ones_like(logits, dtype=torch.bool)
    for row, pdb_row in enumerate(pdb_ids):
        for column, pdb_column in enumerate(pdb_ids):
            if row != column and pdb_row == pdb_column:
                allowed[row, column] = False
    logits = logits.masked_fill(~allowed, -torch.inf)
    target = torch.arange(len(sequence), device=sequence.device)
    return 0.5 * (F.cross_entropy(logits, target)
                  + F.cross_entropy(logits.T, target))
