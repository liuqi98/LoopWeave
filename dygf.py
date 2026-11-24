"""Canonical dyGF model components shared by training and inference.

The flow convention in this module is always

    t=0: Gaussian prior       t=1: encoded CDRH3 latent.

Dynamic geometry conditioning is recomputed at every flow evaluation, while
input-only encodings are cached during sampling. A supplied shadow loop must
be expressed in (or approximately initialized in) the antigen coordinate
frame; the model predicts a time-dependent residual rigid pose before building
the epitope--CDR distance bias.
"""

import math
from collections.abc import Mapping
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


VOCAB_SIZE = 22
PAD_IDX = 0
MAX_LEN = 30
D_MODEL = 256
NHEAD = 8
ENC_LAYERS = 4
DEC_LAYERS = 2
LATENT_DIM = 128
COND_DIM = 256
FLOW_HIDDEN = 512
TIME_DIM = 128
LEN_EMB_DIM = 32
ATOM_FEATURE_DIM = 12
ATOM_HIDDEN_DIM = 128

# Atom feature layout shared with data/augment_all_atom_interfaces.py:
# C, N, O, S, H, other, donor, acceptor, aromatic, charge, vdw radius,
# backbone indicator.
ATOM_DONOR = 6
ATOM_ACCEPTOR = 7
ATOM_AROMATIC = 8
ATOM_CHARGE = 9
ATOM_RADIUS = 10


REDESIGN_CONTEXT_POLICY = "fully_masked_h3_v1"


def masked_redesign_atom_inputs(antigen_inputs, shadow_coords):
    """Build unbatched atom inputs without reading any native H3 atom fields.

    All H3 sites are identical carbon alpha atoms. This deliberately drops
    side-chain chemistry; it is an antigen-atom/CA interface, not all-atom H3.
    """
    features = shadow_coords.new_zeros((len(shadow_coords), ATOM_FEATURE_DIM))
    features[:, 0] = 1.0
    features[:, ATOM_RADIUS] = 1.7
    features[:, 11] = 1.0
    antigen_coords = antigen_inputs["antigen_atom_coords"]
    return {
        "antigen_atom_coords": antigen_coords,
        "antigen_atom_features": antigen_inputs["antigen_atom_features"],
        "antigen_atom_mask": torch.ones(len(antigen_coords), dtype=torch.bool,
                                        device=antigen_coords.device),
        "cdr_atom_coords": shadow_coords,
        "cdr_atom_features": features,
        "cdr_atom_mask": torch.ones(len(shadow_coords), dtype=torch.bool,
                                    device=shadow_coords.device),
    }


def checkpoint_state_dict(checkpoint, key: Optional[str] = None):
    """Return a model state dict from either a raw or wrapped checkpoint.

    Training scripts in this project historically used several wrapper keys.
    Keeping the unwrapping in one place prevents train/inference incompatibility
    while still rejecting optimizer-only or otherwise malformed checkpoints.
    """
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"checkpoint must be a mapping, got {type(checkpoint).__name__}")
    if key is not None and key in checkpoint:
        candidate = checkpoint[key]
    else:
        candidate = checkpoint
        # Current exact-resume training checkpoints use ``model`` while older
        # training and external checkpoints use the other two names.
        for wrapper_key in ("model", "model_state_dict", "state_dict"):
            if wrapper_key in checkpoint:
                candidate = checkpoint[wrapper_key]
                break
    if not isinstance(candidate, Mapping) or not candidate:
        raise ValueError("checkpoint does not contain a non-empty model state dict")
    if not all(isinstance(name, str) for name in candidate):
        raise ValueError("checkpoint state-dict keys must be strings")
    if not all(torch.is_tensor(value) for value in candidate.values()):
        raise ValueError(
            "checkpoint mapping contains metadata; specify the model wrapper key"
        )
    return candidate


def shift_right(tokens: torch.Tensor) -> torch.Tensor:
    """Create causal decoder inputs without exposing the current target token."""
    shifted = torch.full_like(tokens, PAD_IDX)
    shifted[:, 1:] = tokens[:, :-1]
    return shifted


def normalize_surface_features(features: torch.Tensor) -> torch.Tensor:
    """Normalize learned surface inputs without changing physical coordinates.

    The returned copy is translation- and uniform-scale-normalized in xyz;
    callers still use the original first three channels for distance matrices.
    Feature-v2 scalar descriptors are mostly bounded, with a final clamp as a
    guard against malformed surfaces and legacy outliers.
    """
    normalized = torch.nan_to_num(features.clone(), nan=0.0, posinf=10.0, neginf=-10.0)
    xyz = normalized[..., :3]
    centered = xyz - xyz.mean(dim=1, keepdim=True)
    scale = torch.sqrt(centered.square().sum(-1).mean(1, keepdim=True).clamp_min(1e-6))
    normalized[..., :3] = centered / scale.unsqueeze(-1)
    normalized[..., 3:] = normalized[..., 3:].clamp(-5.0, 5.0)
    return normalized


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=PAD_IDX)
        self.pos = PositionalEncoding(D_MODEL)
        enc_layer = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=NHEAD, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, ENC_LAYERS)
        self.cond_proj = nn.Linear(COND_DIM, D_MODEL)
        self.fc_mu = nn.Linear(D_MODEL, LATENT_DIM)
        self.fc_logvar = nn.Linear(D_MODEL, LATENT_DIM)
        self.latent_proj = nn.Linear(LATENT_DIM, D_MODEL)
        dec_layer = nn.TransformerDecoderLayer(d_model=D_MODEL, nhead=NHEAD, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, DEC_LAYERS)
        self.out = nn.Linear(D_MODEL, VOCAB_SIZE)

    def encode(self, x: torch.Tensor, mask: torch.Tensor, cond: torch.Tensor):
        h = self.pos(self.embedding(x)) + self.cond_proj(cond).unsqueeze(1)
        h = self.encoder(h, src_key_padding_mask=(mask == 0))
        mask_f = mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * mask_f).sum(1) / mask_f.sum(1).clamp_min(1.0)
        return self.fc_mu(pooled), self.fc_logvar(pooled)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z: torch.Tensor, decoder_input: torch.Tensor) -> torch.Tensor:
        tgt = self.pos(self.embedding(decoder_input))
        memory = self.latent_proj(z).unsqueeze(1)
        length = decoder_input.size(1)
        causal_mask = torch.triu(
            torch.ones(length, length, device=z.device, dtype=torch.bool), diagonal=1
        )
        return self.out(self.decoder(tgt, memory, tgt_mask=causal_mask))

    def forward(self, x: torch.Tensor, mask: torch.Tensor, cond: torch.Tensor):
        mu, logvar = self.encode(x, mask, cond)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, shift_right(x)), mu, logvar


class PointEncoder(nn.Module):
    def __init__(self, input_dim: int = 48, emb_dim: int = COND_DIM):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 256), nn.ReLU()
        )
        self.mlp2 = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256))
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(256, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.mlp2(self.mlp1(normalize_surface_features(x))).transpose(1, 2)
        return F.normalize(self.fc(self.pool(h).squeeze(-1)), dim=-1)


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int = TIME_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


class LengthEmbedding(nn.Module):
    def __init__(self, max_len: int = MAX_LEN, emb_dim: int = LEN_EMB_DIM):
        super().__init__()
        self.max_len = max_len
        self.embed = nn.Embedding(max_len + 1, emb_dim)

    def forward(self, lengths: torch.Tensor) -> torch.Tensor:
        return self.embed(lengths.clamp(0, self.max_len))


class FlowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.time = TimeEmbedding(TIME_DIM)
        self.len_emb = LengthEmbedding(MAX_LEN, LEN_EMB_DIM)
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM + TIME_DIM + COND_DIM + LEN_EMB_DIM, FLOW_HIDDEN),
            nn.SiLU(),
            nn.Linear(FLOW_HIDDEN, FLOW_HIDDEN),
            nn.SiLU(),
            nn.Linear(FLOW_HIDDEN, LATENT_DIM),
        )

    def forward(self, zt: torch.Tensor, t: torch.Tensor, cond: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([zt, self.time(t), cond, self.len_emb(lengths)], dim=-1))


class EuclideanDistanceBiasedAttention(nn.Module):
    """Epitope queries attending to CDR residues with a monotone distance prior."""

    def __init__(self, d_epi: int = 48, d_cdr: int = 23, d_model: int = 128):
        super().__init__()
        self.d_model = d_model
        self.epi_proj = nn.Linear(d_epi, d_model)
        self.cdr_proj = nn.Linear(d_cdr, d_model)
        self.log_distance_scale = nn.Parameter(torch.tensor(0.0))
        self.dist_mlp = nn.Sequential(nn.Linear(16, 32), nn.SiLU(), nn.Linear(32, 1))
        self.register_buffer("rbf_centers", torch.linspace(2.0, 20.0, 16))
        self.fusion = nn.Linear(d_model + d_epi, d_epi)

    def _distance_bias(self, distances: torch.Tensor) -> torch.Tensor:
        rbf = torch.exp(-0.5 * ((distances.unsqueeze(-1) - self.rbf_centers) / 1.5) ** 2)
        learned = self.dist_mlp(rbf).squeeze(-1)
        monotone = -F.softplus(self.log_distance_scale) * distances
        return learned + monotone

    def forward(self, epi_feat: torch.Tensor, cdr_feat: torch.Tensor,
                cdr_coords: torch.Tensor, dist_matrix: torch.Tensor):
        learned_epi = normalize_surface_features(epi_feat)
        epi_h = self.epi_proj(learned_epi)
        cdr_h = self.cdr_proj(torch.cat([cdr_feat, cdr_coords], dim=-1))
        logits = torch.bmm(epi_h, cdr_h.transpose(1, 2)) / math.sqrt(self.d_model)
        logits = logits + self._distance_bias(dist_matrix)
        weights = F.softmax(logits, dim=-1)
        attended = torch.bmm(weights, cdr_h)
        return self.fusion(torch.cat([learned_epi, attended], dim=-1)), weights


# Backward-friendly public name used by the scripts.
CrossModalAttention = EuclideanDistanceBiasedAttention


class AllAtomInterfaceEncoder(nn.Module):
    """Encode a local antigen--H3 interface without discarding atom identity.

    The learned atom attention is invariant to global rigid transformations:
    coordinates enter only through pair distances.  A small set of bounded,
    energy-inspired summaries exposes steric overlap, charge complementarity,
    donor/acceptor contacts, aromatic packing and general contact density to
    the conditional flow.  These are differentiable geometric proxies, not a
    replacement for side-chain repacking or an atomistic force field.
    """

    def __init__(self, feature_dim: int = ATOM_FEATURE_DIM,
                 hidden_dim: int = ATOM_HIDDEN_DIM,
                 output_dim: int = COND_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.antigen_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cdr_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.register_buffer("rbf_centers", torch.linspace(2.0, 12.0, 16))
        self.distance_mlp = nn.Sequential(
            nn.Linear(16, 32), nn.SiLU(), nn.Linear(32, 1)
        )
        self.context_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim), nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )
        self.physics_proj = nn.Sequential(
            nn.Linear(5, 64), nn.SiLU(), nn.Linear(64, output_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim), nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    @staticmethod
    def _masked_mean(value: torch.Tensor, mask: torch.Tensor,
                     dimensions) -> torch.Tensor:
        mask_f = mask.to(value.dtype)
        numerator = (value * mask_f).sum(dim=dimensions)
        denominator = mask_f.sum(dim=dimensions).clamp_min(1.0)
        return numerator / denominator

    def forward(self, antigen_features: torch.Tensor,
                antigen_coords: torch.Tensor, antigen_mask: torch.Tensor,
                cdr_features: torch.Tensor, cdr_coords: torch.Tensor,
                cdr_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        antigen_mask = antigen_mask.bool()
        cdr_mask = cdr_mask.bool()
        pair_mask = antigen_mask.unsqueeze(2) & cdr_mask.unsqueeze(1)
        distances = torch.cdist(antigen_coords, cdr_coords).clamp_min(0.5)

        antigen_h = self.antigen_proj(antigen_features)
        cdr_h = self.cdr_proj(cdr_features)
        logits = torch.bmm(antigen_h, cdr_h.transpose(1, 2)) / math.sqrt(self.hidden_dim)
        rbf = torch.exp(-0.5 * ((distances.unsqueeze(-1) - self.rbf_centers) / 0.75) ** 2)
        logits = logits + self.distance_mlp(rbf).squeeze(-1) - 0.15 * distances
        logits = logits.masked_fill(~pair_mask, -1e4)
        attention = F.softmax(logits, dim=-1) * pair_mask.to(logits.dtype)
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-8)
        attended = torch.bmm(attention, cdr_h)
        atom_context = self._masked_mean(
            torch.cat([antigen_h, attended], dim=-1), antigen_mask.unsqueeze(-1), 1
        )

        ag_radius = antigen_features[..., ATOM_RADIUS].unsqueeze(2)
        cdr_radius = cdr_features[..., ATOM_RADIUS].unsqueeze(1)
        # A reduced van der Waals envelope flags hard clashes without treating
        # normal polar contacts (often around 2.8--3.0 A) as overlaps.
        overlap = F.relu(0.80 * (ag_radius + cdr_radius) - distances).square()
        contact = torch.exp(-0.5 * ((distances - 3.8) / 1.5).square())
        charge = (
            antigen_features[..., ATOM_CHARGE].unsqueeze(2)
            * cdr_features[..., ATOM_CHARGE].unsqueeze(1)
            / (distances + 1.0)
            * torch.sigmoid(8.0 - distances)
        )
        donor_acceptor = (
            antigen_features[..., ATOM_DONOR].unsqueeze(2)
            * cdr_features[..., ATOM_ACCEPTOR].unsqueeze(1)
            + antigen_features[..., ATOM_ACCEPTOR].unsqueeze(2)
            * cdr_features[..., ATOM_DONOR].unsqueeze(1)
        )
        hydrogen_bond = donor_acceptor * torch.exp(-0.5 * ((distances - 2.9) / 0.6).square())
        aromatic = (
            antigen_features[..., ATOM_AROMATIC].unsqueeze(2)
            * cdr_features[..., ATOM_AROMATIC].unsqueeze(1)
            * torch.exp(-0.5 * ((distances - 4.5) / 1.0).square())
        )
        physics = torch.stack([
            self._masked_mean(contact, pair_mask, (1, 2)),
            self._masked_mean(overlap, pair_mask, (1, 2)),
            self._masked_mean(charge, pair_mask, (1, 2)),
            self._masked_mean(hydrogen_bond, pair_mask, (1, 2)),
            self._masked_mean(aromatic, pair_mask, (1, 2)),
        ], dim=-1)
        condition = self.fusion(torch.cat([
            self.context_proj(atom_context), self.physics_proj(physics)
        ], dim=-1))
        return {
            "condition": condition,
            "attention": attention,
            "distances": distances,
            "pair_mask": pair_mask,
            "physics": physics,
            "steric_overlap": physics[:, 1].mean(),
        }


class TransformPredictor(nn.Module):
    """Predict a time- and latent-dependent residual rigid transformation."""

    def __init__(self, input_dim: int = LATENT_DIM + TIME_DIM + COND_DIM + 6,
                 hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.rot_head = nn.Linear(hidden_dim, 6)
        self.trans_head = nn.Linear(hidden_dim, 3)
        nn.init.zeros_(self.trans_head.weight)
        nn.init.zeros_(self.trans_head.bias)
        with torch.no_grad():
            self.rot_head.weight.zero_()
            self.rot_head.bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    @staticmethod
    def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack([b1, b2, b3], dim=-1)

    def forward(self, feat: torch.Tensor):
        h = self.net(feat)
        return self.rotation_6d_to_matrix(self.rot_head(h)), self.trans_head(h)


def skew_matrix(vector: torch.Tensor) -> torch.Tensor:
    """Map (..., 3) rotation vectors to skew-symmetric matrices."""
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1).reshape(
        *vector.shape[:-1], 3, 3
    )


def so3_exp_map(rotation_vector: torch.Tensor) -> torch.Tensor:
    """Differentiable exponential map with a stable small-angle branch."""
    theta = torch.linalg.vector_norm(rotation_vector, dim=-1, keepdim=True)
    k = skew_matrix(rotation_vector)
    theta2 = theta.square()
    # ``where`` evaluates both branches, so every denominator must be safe;
    # otherwise the unused 0/0 branch still poisons gradients at identity.
    a = torch.where(theta > 1e-4, torch.sin(theta) / theta.clamp_min(1e-8),
                    1.0 - theta2 / 6.0 + theta2.square() / 120.0)
    b = torch.where(theta > 1e-4, (1.0 - torch.cos(theta)) / theta2.clamp_min(1e-12),
                    0.5 - theta2 / 24.0 + theta2.square() / 720.0)
    eye = torch.eye(3, device=rotation_vector.device, dtype=rotation_vector.dtype)
    return eye.expand(*rotation_vector.shape[:-1], 3, 3) + a.unsqueeze(-1) * k + b.unsqueeze(-1) * (k @ k)


def so3_log_map(rotation: torch.Tensor) -> torch.Tensor:
    """Logarithm map for proper rotations; accurate for the bounded pose curriculum."""
    cosine = ((rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cosine)
    vee = torch.stack([
        rotation[..., 2, 1] - rotation[..., 1, 2],
        rotation[..., 0, 2] - rotation[..., 2, 0],
        rotation[..., 1, 0] - rotation[..., 0, 1],
    ], dim=-1)
    scale = torch.where(theta > 1e-4, theta / (2.0 * torch.sin(theta).clamp_min(1e-8)),
                        0.5 + theta.square() / 12.0)
    return vee * scale.unsqueeze(-1)


class PoseTwistPredictor(nn.Module):
    """Predict an SE(3) tangent velocity instead of an endpoint transform."""

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 6),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor):
        twist = self.net(features)
        return twist[..., :3], twist[..., 3:]


def coordinate_summary(coords: torch.Tensor) -> torch.Tensor:
    center = coords.mean(dim=1)
    radius = torch.sqrt(((coords - center.unsqueeze(1)) ** 2).sum(-1).mean(1).clamp_min(1e-8))
    end_distance = torch.linalg.vector_norm(coords[:, -1] - coords[:, 0], dim=-1)
    return torch.cat([center, radius.unsqueeze(-1), end_distance.unsqueeze(-1),
                      torch.ones_like(radius).unsqueeze(-1)], dim=-1)


class DynamicGeometryFlow(nn.Module):
    """Close the GAM -> dynamic distance -> EDBA -> velocity loop."""

    def __init__(self, flow_model: FlowModel, point_encoder: PointEncoder,
                 cross_attn: EuclideanDistanceBiasedAttention,
                 atom_interface: Optional[AllAtomInterfaceEncoder] = None):
        super().__init__()
        self.flow = flow_model
        self.point_encoder = point_encoder
        self.cross_attn = cross_attn
        self.cond_proj = nn.Linear(48, COND_DIM)
        self.time_for_pose = TimeEmbedding(TIME_DIM)
        self.transform_predictor = TransformPredictor()
        self.atom_interface = atom_interface
        self.atom_gate = nn.Parameter(torch.tensor(0.0)) if atom_interface is not None else None

    def forward(self, zt: torch.Tensor, t: torch.Tensor, epi_cloud: torch.Tensor,
                cdr_feat: torch.Tensor, shadow_coords: torch.Tensor,
                lengths: torch.Tensor, atom_inputs: Optional[Dict[str, torch.Tensor]] = None
                ) -> Dict[str, torch.Tensor]:
        base_cond = self.point_encoder(epi_cloud)
        pose_input = torch.cat(
            [zt, self.time_for_pose(t), base_cond, coordinate_summary(shadow_coords)], dim=-1
        )
        rotation, translation = self.transform_predictor(pose_input)
        anchored = torch.bmm(shadow_coords, rotation.transpose(1, 2)) + translation.unsqueeze(1)

        distances = torch.cdist(epi_cloud[..., :3], anchored)
        enhanced_epi, attention = self.cross_attn(epi_cloud, cdr_feat, anchored, distances)
        condition = base_cond + self.cond_proj(enhanced_epi.mean(dim=1))
        atom_output = None
        if self.atom_interface is not None and atom_inputs is not None:
            anchored_atoms = (
                torch.bmm(atom_inputs["cdr_atom_coords"], rotation.transpose(1, 2))
                + translation.unsqueeze(1)
            )
            atom_output = self.atom_interface(
                atom_inputs["antigen_atom_features"],
                atom_inputs["antigen_atom_coords"],
                atom_inputs["antigen_atom_mask"],
                atom_inputs["cdr_atom_features"], anchored_atoms,
                atom_inputs["cdr_atom_mask"],
            )
            condition = condition + torch.sigmoid(self.atom_gate) * atom_output["condition"]
        velocity = self.flow(zt, t, condition, lengths)
        result = {
            "velocity": velocity,
            "attention": attention,
            "anchored_coords": anchored,
            "rotation": rotation,
            "translation": translation,
            "distances": distances,
            "condition": condition,
        }
        if atom_output is not None:
            result.update({
                "anchored_atom_coords": anchored_atoms,
                "atom_attention": atom_output["attention"],
                "atom_distances": atom_output["distances"],
                "atom_pair_mask": atom_output["pair_mask"],
                "atom_physics": atom_output["physics"],
                "steric_overlap": atom_output["steric_overlap"],
            })
        return result


EnhancedFlowModelWithShadow = DynamicGeometryFlow


class FixedInterfaceStateAttention(nn.Module):
    """Attend from a target epitope to a fixed, anchor-labelled H3 backbone.

    This application head deliberately does not predict or transform loop
    coordinates.  Residue identity is visible only at protected contact
    anchors; all redesignable positions receive a uniform amino-acid context.
    """

    RESIDUE_DIM = 20 + 1 + 1 + 3  # sequence context, anchor, distance, local xyz

    def __init__(self, d_model: int = 128, output_dim: int = COND_DIM):
        super().__init__()
        self.d_model = d_model
        self.epitope_projection = nn.Linear(48, d_model)
        self.residue_projection = nn.Linear(self.RESIDUE_DIM, d_model)
        self.register_buffer("rbf_centers", torch.linspace(2.0, 20.0, 16))
        self.distance_projection = nn.Sequential(
            nn.Linear(16, 32), nn.SiLU(), nn.Linear(32, 1)
        )
        self.output = nn.Sequential(
            nn.Linear(d_model + 48, output_dim), nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    @staticmethod
    def local_coordinates(epitope_xyz: torch.Tensor,
                          loop_coords: torch.Tensor) -> torch.Tensor:
        midpoint = 0.5 * (loop_coords[:, 0] + loop_coords[:, -1])
        x_axis = F.normalize(loop_coords[:, -1] - loop_coords[:, 0], dim=-1)
        direction = epitope_xyz.mean(1) - midpoint
        y_axis = direction - (direction * x_axis).sum(-1, keepdim=True) * x_axis
        fallback = torch.cross(
            x_axis,
            torch.tensor([1.0, 0.0, 0.0], device=x_axis.device,
                         dtype=x_axis.dtype).expand_as(x_axis),
            dim=-1,
        )
        second = torch.cross(
            x_axis,
            torch.tensor([0.0, 1.0, 0.0], device=x_axis.device,
                         dtype=x_axis.dtype).expand_as(x_axis),
            dim=-1,
        )
        y_axis = torch.where(
            torch.linalg.vector_norm(y_axis, dim=-1, keepdim=True) > 1e-4,
            y_axis,
            torch.where(
                torch.linalg.vector_norm(fallback, dim=-1, keepdim=True) > 1e-4,
                fallback, second,
            ),
        )
        y_axis = F.normalize(y_axis, dim=-1)
        z_axis = F.normalize(torch.cross(x_axis, y_axis, dim=-1), dim=-1)
        frame = torch.stack([x_axis, y_axis, z_axis], dim=-1)
        return torch.bmm(loop_coords - midpoint[:, None], frame) / 20.0

    def forward(self, epitope: torch.Tensor, context: torch.Tensor,
                loop_coords: torch.Tensor, anchor_mask: torch.Tensor):
        distances = torch.cdist(epitope[..., :3], loop_coords)
        minimum = distances.amin(1)
        local = self.local_coordinates(epitope[..., :3], loop_coords)
        residue_features = torch.cat([
            context, anchor_mask.unsqueeze(-1).to(context.dtype),
            (minimum / 20.0).unsqueeze(-1), local,
        ], dim=-1)
        epitope_features = normalize_surface_features(epitope)
        query = self.epitope_projection(epitope_features)
        key = self.residue_projection(residue_features)
        logits = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(self.d_model)
        rbf = torch.exp(
            -0.5 * ((distances.unsqueeze(-1) - self.rbf_centers) / 1.5).square()
        )
        logits = logits + self.distance_projection(rbf).squeeze(-1) - 0.10 * distances
        attention = F.softmax(logits, dim=-1)
        attended = torch.bmm(attention, key)
        enhanced = self.output(torch.cat([epitope_features, attended], dim=-1))
        return enhanced.mean(1), attention, distances, minimum, local


class FixedInterfaceRedesignFlow(nn.Module):
    """dyGF application model for redesign on an already accepted interface.

    Unlike ``DeNovoGeometryFlow``, this model never generates a new H3 pose.
    It conditions latent sequence flow on the fixed H3 backbone, target
    epitope, and protected contact-anchor identities.  This matches the
    known-complex-to-VHH use case where pose preservation is an explicit gate.
    """

    FORMAT_VERSION = 30

    def __init__(self, flow_model: Optional[FlowModel] = None,
                 point_encoder: Optional[PointEncoder] = None,
                 interface_attention: Optional[FixedInterfaceStateAttention] = None):
        super().__init__()
        self.flow = flow_model or FlowModel()
        self.point_encoder = point_encoder or PointEncoder()
        self.interface_attention = interface_attention or FixedInterfaceStateAttention()
        self.interface_summary = nn.Sequential(
            nn.Linear(8, 128), nn.SiLU(), nn.Linear(128, COND_DIM)
        )
        self.anchor_identity = nn.Sequential(
            nn.Linear(20, 128), nn.SiLU(), nn.Linear(128, COND_DIM)
        )
        self.condition_fusion = nn.Sequential(
            nn.Linear(COND_DIM * 4, COND_DIM), nn.SiLU(),
            nn.Linear(COND_DIM, COND_DIM),
        )

    def encode_condition(self, epitope: torch.Tensor, context: torch.Tensor,
                         loop_coords: torch.Tensor,
                         anchor_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        base = self.point_encoder(epitope)
        interface, attention, distances, minimum, local = self.interface_attention(
            epitope, context, loop_coords, anchor_mask
        )
        anchor_weight = anchor_mask.to(context.dtype)
        anchor_context = (
            context * anchor_weight.unsqueeze(-1)
        ).sum(1) / anchor_weight.sum(1, keepdim=True).clamp_min(1.0)
        summary_values = torch.stack([
            minimum.amin(1) / 20.0,
            minimum.mean(1) / 20.0,
            (minimum < 4.0).to(context.dtype).mean(1),
            (minimum < 6.0).to(context.dtype).mean(1),
            (minimum < 8.0).to(context.dtype).mean(1),
            (minimum < 2.5).to(context.dtype).mean(1),
            anchor_weight.mean(1),
            torch.sqrt(local.square().sum(-1).mean(1).clamp_min(1e-8)),
        ], dim=-1)
        condition = self.condition_fusion(torch.cat([
            base, interface, self.interface_summary(summary_values),
            self.anchor_identity(anchor_context),
        ], dim=-1))
        return {
            "condition": condition,
            "attention": attention,
            "distances": distances,
            "minimum_distances": minimum,
            "local_loop_coordinates": local,
            "interface_summary": summary_values,
        }

    def forward(self, zt: torch.Tensor, t: torch.Tensor,
                epitope: torch.Tensor, context: torch.Tensor,
                loop_coords: torch.Tensor, anchor_mask: torch.Tensor,
                lengths: torch.Tensor,
                static_condition: Optional[Dict[str, torch.Tensor]] = None
                ) -> Dict[str, torch.Tensor]:
        encoded = (
            self.encode_condition(epitope, context, loop_coords, anchor_mask)
            if static_condition is None else static_condition
        )
        return {
            **encoded,
            "velocity": self.flow(zt, t, encoded["condition"], lengths),
            "fixed_loop_coords": loop_coords,
            "anchor_mask": anchor_mask,
        }


def sample_fixed_interface_flow(model: FixedInterfaceRedesignFlow,
                                epitope: torch.Tensor,
                                context: torch.Tensor,
                                loop_coords: torch.Tensor,
                                anchor_mask: torch.Tensor,
                                lengths: torch.Tensor,
                                num_steps: int = 50,
                                latent_scale: float = 1.0,
                                initial_latent: Optional[torch.Tensor] = None
                                ) -> torch.Tensor:
    """Integrate sequence latent flow while keeping interface geometry fixed."""
    latent = (
        latent_scale * torch.randn(
            epitope.size(0), LATENT_DIM,
            device=epitope.device, dtype=epitope.dtype,
        )
        if initial_latent is None else initial_latent.clone()
    )
    static = model.encode_condition(epitope, context, loop_coords, anchor_mask)
    grid = torch.linspace(0.0, 1.0, num_steps + 1, device=epitope.device)
    for index in range(num_steps):
        time = grid[index].expand(epitope.size(0), 1)
        output = model(
            latent, time, epitope, context, loop_coords, anchor_mask,
            lengths, static,
        )
        latent = latent + output["velocity"] * (grid[index + 1] - grid[index])
    return latent


def flow_matching_batch(model: DynamicGeometryFlow, data_latent: torch.Tensor,
                        epi_cloud: torch.Tensor, cdr_feat: torch.Tensor,
                        shadow_coords: torch.Tensor, lengths: torch.Tensor,
                        t: Optional[torch.Tensor] = None,
                        atom_inputs: Optional[Dict[str, torch.Tensor]] = None):
    """Sample a prior-to-data conditional flow-matching training tuple."""
    batch = data_latent.size(0)
    if t is None:
        t = torch.rand(batch, 1, device=data_latent.device)
    prior = torch.randn_like(data_latent)
    zt = (1.0 - t) * prior + t * data_latent
    target_velocity = data_latent - prior
    output = model(zt, t, epi_cloud, cdr_feat, shadow_coords, lengths, atom_inputs)
    return output, target_velocity, t


def contact_attention_loss(attention: torch.Tensor, distances: torch.Tensor,
                           threshold: float = 6.0) -> torch.Tensor:
    contacts = (distances < threshold).to(attention.dtype)
    valid = contacts.any(dim=-1)
    if not valid.any():
        return attention.sum() * 0.0
    contact_mass = (attention * contacts).sum(dim=-1)
    return F.binary_cross_entropy(contact_mass[valid].clamp(1e-6, 1.0 - 1e-6),
                                  torch.ones_like(contact_mass[valid]))


def anchor_loss(predicted: torch.Tensor, target: torch.Tensor,
                t: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """Emphasize endpoint geometry without forcing a random early state to be final."""
    per_sample = F.smooth_l1_loss(predicted, target, reduction="none").mean(dim=(1, 2))
    return ((t.squeeze(-1) ** gamma) * per_sample).mean()


def sample_flow(model: DynamicGeometryFlow, epi_cloud: torch.Tensor,
                cdr_feat: torch.Tensor, shadow_coords: torch.Tensor,
                lengths: torch.Tensor, num_steps: int = 50,
                latent_scale: float = 1.0,
                atom_inputs: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
    """Euler sample from the prior at t=0 to the data manifold at t=1."""
    z = latent_scale * torch.randn(epi_cloud.size(0), LATENT_DIM, device=epi_cloud.device)
    grid = torch.linspace(0.0, 1.0, num_steps + 1, device=epi_cloud.device)
    for index in range(num_steps):
        t = grid[index].expand(epi_cloud.size(0), 1)
        dt = grid[index + 1] - grid[index]
        z = z + model(
            z, t, epi_cloud, cdr_feat, shadow_coords, lengths, atom_inputs
        )["velocity"] * dt
    return z


class FrameworkEncoder(nn.Module):
    """Encode the fixed VH/VL scaffold after removing CDRH3.

    Inputs are residue one-hot features and CA coordinates in the antigen
    coordinate frame. Pairwise geometry is summarized relative to the H3
    flanking anchors, avoiding any use of native H3 residues or coordinates.
    """

    def __init__(self, hidden_dim: int = 256, output_dim: int = COND_DIM):
        super().__init__()
        self.residue_mlp = nn.Sequential(
            nn.Linear(20 + 3 + 2, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.output = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, residue_features: torch.Tensor, coords: torch.Tensor,
                n_anchor: torch.Tensor, c_anchor: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        d_n = torch.linalg.vector_norm(coords - n_anchor.unsqueeze(1), dim=-1, keepdim=True)
        d_c = torch.linalg.vector_norm(coords - c_anchor.unsqueeze(1), dim=-1, keepdim=True)
        local_coords = coords - 0.5 * (n_anchor + c_anchor).unsqueeze(1)
        h = self.residue_mlp(torch.cat([residue_features, local_coords, d_n, d_c], dim=-1))
        if mask is None:
            mean = h.mean(dim=1)
            maximum = h.max(dim=1).values
        else:
            mask_f = mask.unsqueeze(-1).to(h.dtype)
            mean = (h * mask_f).sum(1) / mask_f.sum(1).clamp_min(1.0)
            maximum = h.masked_fill(~mask.bool().unsqueeze(-1), -torch.inf).max(1).values
            maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        return F.normalize(self.output(torch.cat([mean, maximum], dim=-1)), dim=-1)


class FullAntigenContextEncoder(nn.Module):
    """Encode every antigen residue while retaining an explicit epitope mask.

    This is a residue-level full-antigen context (sequence + CA geometry), not
    a cropped epitope peptide.  The higher-resolution 48D surface patch remains
    a separate input and supplies local surface chemistry.
    """

    def __init__(self, hidden_dim: int = 256, output_dim: int = COND_DIM):
        super().__init__()
        self.residue_mlp = nn.Sequential(
            nn.Linear(20 + 3 + 2 + 1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.output = nn.Linear(hidden_dim * 4, output_dim)

    @staticmethod
    def pool(hidden: torch.Tensor, mask: torch.Tensor):
        weight = mask.unsqueeze(-1).to(hidden.dtype)
        mean = (hidden * weight).sum(1) / weight.sum(1).clamp_min(1.0)
        maximum = hidden.masked_fill(~mask.bool().unsqueeze(-1), -torch.inf).max(1).values
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        return mean, maximum

    def forward(self, residue_features: torch.Tensor, coords: torch.Tensor,
                epitope_mask: torch.Tensor, n_anchor: torch.Tensor,
                c_anchor: torch.Tensor,
                residue_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if residue_mask is None:
            residue_mask = torch.ones(coords.shape[:2], dtype=torch.bool, device=coords.device)
        epitope_mask = epitope_mask.bool() & residue_mask.bool()
        if not epitope_mask.any(dim=1).all():
            raise ValueError("every full-antigen input must mark at least one epitope residue")
        midpoint = 0.5 * (n_anchor + c_anchor)
        x_axis = F.normalize(c_anchor - n_anchor, dim=-1)
        epi_weight = epitope_mask.unsqueeze(-1).to(coords.dtype)
        epi_center = (coords * epi_weight).sum(1) / epi_weight.sum(1).clamp_min(1.0)
        y_axis = epi_center - midpoint
        y_axis = y_axis - (y_axis * x_axis).sum(-1, keepdim=True) * x_axis
        y_axis = F.normalize(y_axis, dim=-1)
        z_axis = F.normalize(torch.cross(x_axis, y_axis, dim=-1), dim=-1)
        y_axis = torch.cross(z_axis, x_axis, dim=-1)
        frame = torch.stack([x_axis, y_axis, z_axis], dim=-1)
        local = torch.bmm(coords - midpoint.unsqueeze(1), frame) / 20.0
        d_n = torch.linalg.vector_norm(coords - n_anchor.unsqueeze(1), dim=-1, keepdim=True) / 20.0
        d_c = torch.linalg.vector_norm(coords - c_anchor.unsqueeze(1), dim=-1, keepdim=True) / 20.0
        hidden = self.residue_mlp(torch.cat([
            residue_features, local, d_n, d_c, epitope_mask.unsqueeze(-1).to(coords.dtype)
        ], dim=-1))
        all_mean, all_max = self.pool(hidden, residue_mask.bool())
        epi_mean, epi_max = self.pool(hidden, epitope_mask)
        return F.normalize(
            self.output(torch.cat([all_mean, all_max, epi_mean, epi_max], dim=-1)), dim=-1
        )


class RelativePoseEncoder(nn.Module):
    """Expose cross-body geometry needed to infer docking displacement.

    The separate epitope/framework encoders intentionally discard some global
    position information. Their pooled embeddings alone cannot identify a
    translated framework, so the pose field also receives explicit relative
    centers, radii, anchor displacement and nearest/mean cross distances.
    """

    def __init__(self, output_dim: int = COND_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 128), nn.SiLU(), nn.Linear(128, output_dim)
        )

    def forward(self, epi_xyz: torch.Tensor, framework_coords: torch.Tensor,
                n_anchor: torch.Tensor, c_anchor: torch.Tensor,
                framework_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if framework_mask is None:
            framework_center = framework_coords.mean(1)
        else:
            weight = framework_mask.to(framework_coords.dtype).unsqueeze(-1)
            framework_center = (framework_coords * weight).sum(1) / weight.sum(1).clamp_min(1.0)
        epitope_center = epi_xyz.mean(1)
        anchor_midpoint = 0.5 * (n_anchor + c_anchor)
        framework_radius = torch.sqrt(
            (framework_coords - framework_center[:, None]).square().sum(-1).mean(1).clamp_min(1e-8)
        )
        epitope_radius = torch.sqrt(
            (epi_xyz - epitope_center[:, None]).square().sum(-1).mean(1).clamp_min(1e-8)
        )
        cross = torch.cdist(framework_coords, epi_xyz)
        features = torch.cat([
            (framework_center - epitope_center) / 20.0,
            (anchor_midpoint - epitope_center) / 20.0,
            framework_radius[:, None] / 20.0,
            epitope_radius[:, None] / 20.0,
            cross.amin(dim=(1, 2))[:, None] / 20.0,
            cross.mean(dim=(1, 2))[:, None] / 20.0,
        ], dim=-1)
        return F.normalize(self.net(features), dim=-1)


class LoopPoseFeedbackEncoder(nn.Module):
    """Encode current H3--epitope forces for closed-loop pose refinement.

    The generated loop is evaluated in the current antibody pose. Attraction
    and torque toward its nearest epitope points make joint pose velocity depend
    on the actual evolving H3 geometry, a signal unavailable to serial docking.
    All vectors are expressed in the antibody-local frame for SE(3) invariance.
    """

    FEATURE_DIM = 14

    def __init__(self, output_dim: int = COND_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.FEATURE_DIM, 128), nn.SiLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, loop_coords: torch.Tensor, loop_mask: torch.Tensor,
                epi_cloud: torch.Tensor, framework_coords: torch.Tensor,
                local_frame: torch.Tensor) -> torch.Tensor:
        epi_xyz = epi_cloud[..., :3]
        distance = torch.cdist(loop_coords, epi_xyz)
        minimum, closest_index = distance.min(-1)
        closest_epi = epi_xyz.gather(
            1, closest_index.unsqueeze(-1).expand(-1, -1, 3)
        )
        valid = loop_mask.to(loop_coords.dtype)
        normalizer = valid.sum(1, keepdim=True).clamp_min(1.0)
        loop_center = (loop_coords * valid.unsqueeze(-1)).sum(1) / normalizer
        epi_center = epi_xyz.mean(1)
        displacement = closest_epi - loop_coords
        weight = torch.exp(-minimum / 6.0) * valid
        weight = weight / weight.sum(1, keepdim=True).clamp_min(1e-8)
        force = (weight.unsqueeze(-1) * displacement).sum(1)
        pivot = framework_coords.mean(1)
        torque = (
            weight.unsqueeze(-1)
            * torch.cross(loop_coords - pivot[:, None], displacement, dim=-1)
        ).sum(1)
        to_local = lambda vector: torch.bmm(
            vector.unsqueeze(1), local_frame
        ).squeeze(1)
        masked_minimum = minimum.masked_fill(~loop_mask, torch.inf)
        mean_minimum = (minimum * valid).sum(1) / normalizer.squeeze(1)
        contact_fractions = torch.stack([
            (((minimum < cutoff).to(valid.dtype) * valid).sum(1)
             / normalizer.squeeze(1)) for cutoff in (4.0, 6.0, 8.0)
        ], dim=-1)
        features = torch.cat([
            to_local(loop_center - epi_center) / 20.0,
            to_local(force) / 20.0,
            to_local(torque) / 400.0,
            masked_minimum.amin(1, keepdim=True) / 20.0,
            mean_minimum[:, None] / 20.0,
            contact_fractions,
        ], dim=-1)
        return self.net(features)


class LocalPoseGeometryEncoder(nn.Module):
    """Direction-sensitive cross-body geometry in an antibody-local frame.

    Scalar distance summaries cannot identify a framework rotation about its
    own center.  This encoder retains local epitope directions, normals and
    nearest-neighbour force/torque proxies.  Expressing them in a backbone
    frame makes the condition invariant to a joint global rigid transform,
    while remaining sensitive to motion of the framework relative to antigen.
    """

    FEATURE_DIM = 23

    def __init__(self, output_dim: int = COND_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.FEATURE_DIM, 128), nn.SiLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, epi_cloud: torch.Tensor, framework_coords: torch.Tensor,
                n_anchor: torch.Tensor, c_anchor: torch.Tensor,
                local_frame: torch.Tensor,
                framework_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        epi_xyz = epi_cloud[..., :3]
        midpoint = 0.5 * (n_anchor + c_anchor)
        epi_local = torch.bmm(
            epi_xyz - midpoint[:, None], local_frame
        )
        epi_centered = epi_local - epi_local.mean(1, keepdim=True)
        nearest_index = epi_local.square().sum(-1).argmin(-1)
        nearest_local = epi_local[
            torch.arange(epi_local.size(0), device=epi_local.device), nearest_index
        ]
        if epi_cloud.size(-1) >= 6:
            normal_local = torch.bmm(epi_cloud[..., 3:6], local_frame).mean(1)
        else:
            normal_local = torch.zeros_like(midpoint)
        anchor_local = torch.bmm(
            (c_anchor - n_anchor).unsqueeze(1), local_frame
        ).squeeze(1)

        cross = torch.cdist(framework_coords, epi_xyz)
        minimum, closest_index = cross.min(-1)
        closest_epi = epi_xyz.gather(
            1, closest_index.unsqueeze(-1).expand(-1, -1, 3)
        )
        displacement = closest_epi - framework_coords
        if framework_mask is None:
            weight = torch.exp(-minimum / 10.0)
            framework_center = framework_coords.mean(1)
        else:
            valid = framework_mask.to(framework_coords.dtype)
            weight = torch.exp(-minimum / 10.0) * valid
            framework_center = (
                (framework_coords * valid.unsqueeze(-1)).sum(1)
                / valid.sum(1, keepdim=True).clamp_min(1.0)
            )
        weight = weight / weight.sum(1, keepdim=True).clamp_min(1e-8)
        force = (weight.unsqueeze(-1) * displacement).sum(1)
        torque = (
            weight.unsqueeze(-1)
            * torch.cross(framework_coords - framework_center[:, None],
                          displacement, dim=-1)
        ).sum(1)
        force_local = torch.bmm(force.unsqueeze(1), local_frame).squeeze(1)
        torque_local = torch.bmm(torque.unsqueeze(1), local_frame).squeeze(1)
        features = torch.cat([
            epi_local.mean(1) / 20.0,
            nearest_local / 20.0,
            torch.sqrt(epi_centered.square().mean(1).clamp_min(1e-8)) / 20.0,
            normal_local,
            anchor_local / 20.0,
            cross.amin(dim=(1, 2))[:, None] / 20.0,
            cross.mean(dim=(1, 2))[:, None] / 20.0,
            force_local / 20.0,
            torque_local / 400.0,
        ], dim=-1)
        # Preserve response magnitude: unlike a semantic embedding, pose
        # geometry must distinguish a small residual from a large displacement.
        return self.net(features)


class LearnedPoseCorrespondenceEncoder(nn.Module):
    """Learn soft framework--epitope correspondences for an SE(3) pose field.

    Pair scores use only invariant residue/surface descriptors and distances.
    Their weighted displacement and torque vectors are then expressed in the
    antibody-local frame, retaining the direction needed for docking while
    remaining equivariant to a joint rigid transform of the complex.
    """

    def __init__(self, heads: int = 4, hidden_dim: int = 64,
                 output_dim: int = COND_DIM):
        super().__init__()
        self.heads = heads
        self.framework_proj = nn.Sequential(
            nn.Linear(22, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.epitope_proj = nn.Sequential(
            nn.Linear(48, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.register_buffer("rbf_centers", torch.linspace(2.0, 30.0, 12))
        self.pair_score = nn.Sequential(
            nn.Linear(hidden_dim + 12, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, heads),
        )
        self.output = nn.Sequential(
            nn.Linear(heads * 7, 128), nn.SiLU(), nn.Linear(128, output_dim)
        )

    def forward(self, epi_cloud: torch.Tensor, framework_features: torch.Tensor,
                framework_coords: torch.Tensor, n_anchor: torch.Tensor,
                c_anchor: torch.Tensor, local_frame: torch.Tensor,
                framework_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        epi_xyz = epi_cloud[..., :3]
        d_n = torch.linalg.vector_norm(
            framework_coords - n_anchor[:, None], dim=-1, keepdim=True
        ) / 20.0
        d_c = torch.linalg.vector_norm(
            framework_coords - c_anchor[:, None], dim=-1, keepdim=True
        ) / 20.0
        framework_h = self.framework_proj(torch.cat([framework_features, d_n, d_c], -1))
        epitope_h = self.epitope_proj(normalize_surface_features(epi_cloud))
        pair_h = torch.tanh(framework_h[:, :, None] + epitope_h[:, None, :])
        distances = torch.cdist(framework_coords, epi_xyz)
        rbf = torch.exp(-0.5 * (
            (distances[..., None] - self.rbf_centers) / 2.0
        ).square())
        logits = self.pair_score(torch.cat([
            pair_h, rbf
        ], dim=-1))
        if framework_mask is not None:
            logits = logits.masked_fill(
                ~framework_mask.bool()[:, :, None, None], -1e4
            )
        weights = F.softmax(logits.flatten(1, 2), dim=1).view_as(logits)
        displacement = epi_xyz[:, None, :, :] - framework_coords[:, :, None, :]
        if framework_mask is None:
            center = framework_coords.mean(1)
        else:
            valid = framework_mask.to(framework_coords.dtype)[..., None]
            center = (framework_coords * valid).sum(1) / valid.sum(1).clamp_min(1.0)
        lever = framework_coords - center[:, None]
        force = torch.einsum("bfeh,bfec->bhc", weights, displacement)
        torque_pair = torch.cross(
            lever[:, :, None, :].expand_as(displacement), displacement, dim=-1
        )
        torque = torch.einsum("bfeh,bfec->bhc", weights, torque_pair)
        mean_distance = torch.einsum("bfeh,bfe->bh", weights, distances)
        to_local = lambda vectors: torch.matmul(vectors, local_frame)
        features = torch.cat([
            (to_local(force) / 20.0).flatten(1),
            (to_local(torque) / 400.0).flatten(1),
            mean_distance / 20.0,
        ], dim=-1)
        return self.output(features)


def backbone_frame(backbone_atoms: torch.Tensor) -> torch.Tensor:
    """Build an orthonormal residue frame from N, CA and C coordinates.

    ``backbone_atoms`` is (..., 4, 3) in N/CA/C/O order. O is retained in the
    serialized format for future peptide-plane losses but is not needed here.
    """
    n_atom, ca_atom, c_atom = (backbone_atoms[..., index, :] for index in range(3))
    x_axis = F.normalize(c_atom - ca_atom, dim=-1)
    n_direction = n_atom - ca_atom
    y_axis = F.normalize(
        n_direction - (n_direction * x_axis).sum(-1, keepdim=True) * x_axis,
        dim=-1,
    )
    z_axis = torch.cross(x_axis, y_axis, dim=-1)
    return torch.stack([x_axis, y_axis, z_axis], dim=-1)


class AnchorFrameEncoder(nn.Module):
    """Encode the relative orientation and separation of the two H3 flanks."""

    def __init__(self, output_dim: int = COND_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 128), nn.SiLU(), nn.Linear(128, output_dim)
        )

    def forward(self, n_frame_atoms: torch.Tensor,
                c_frame_atoms: torch.Tensor) -> torch.Tensor:
        n_rotation = backbone_frame(n_frame_atoms)
        c_rotation = backbone_frame(c_frame_atoms)
        relative_rotation = torch.matmul(n_rotation.transpose(-1, -2), c_rotation)
        separation = torch.linalg.vector_norm(
            c_frame_atoms[..., 1, :] - n_frame_atoms[..., 1, :], dim=-1,
            keepdim=True,
        ) / 20.0
        return F.normalize(
            self.net(torch.cat([relative_rotation.flatten(1), separation], dim=-1)),
            dim=-1,
        )


class AnchorLoopInitializer(nn.Module):
    """Construct a target-independent smooth bridge between H3 flank anchors."""

    def __init__(self, max_len: int = MAX_LEN):
        super().__init__()
        self.max_len = max_len

    def forward(self, n_anchor: torch.Tensor, c_anchor: torch.Tensor,
                lengths: torch.Tensor, framework_coords: torch.Tensor) -> torch.Tensor:
        batch = n_anchor.size(0)
        output = n_anchor.new_zeros(batch, self.max_len, 3)
        framework_center = framework_coords.mean(dim=1)
        axis = F.normalize(c_anchor - n_anchor, dim=-1)
        midpoint = 0.5 * (n_anchor + c_anchor)
        outward = midpoint - framework_center
        outward = outward - (outward * axis).sum(-1, keepdim=True) * axis
        fallback = torch.cross(axis, torch.tensor([1.0, 0.0, 0.0], device=axis.device).expand_as(axis), dim=-1)
        second_fallback = torch.cross(axis, torch.tensor([0.0, 1.0, 0.0], device=axis.device).expand_as(axis), dim=-1)
        outward = torch.where(
            (torch.linalg.vector_norm(outward, dim=-1, keepdim=True) > 1e-4), outward,
            torch.where((torch.linalg.vector_norm(fallback, dim=-1, keepdim=True) > 1e-4),
                        fallback, second_fallback),
        )
        outward = F.normalize(outward, dim=-1)
        for item in range(batch):
            length = int(lengths[item].item())
            u = torch.arange(1, length + 1, device=axis.device, dtype=axis.dtype) / (length + 1)
            line = (1.0 - u[:, None]) * n_anchor[item] + u[:, None] * c_anchor[item]
            chord = torch.linalg.vector_norm(c_anchor[item] - n_anchor[item])
            desired_contour = 3.8 * (length + 1)
            bow = torch.sqrt((desired_contour ** 2 - chord.square()).clamp_min(0.0)) / math.pi
            output[item, :length] = line + (bow * torch.sin(math.pi * u))[:, None] * outward[item]
        return output


class LoopStructureDecoder(nn.Module):
    """Decode a global flow latent into residue states and a deformable loop."""

    def __init__(self, max_len: int = MAX_LEN, hidden_dim: int = 256):
        super().__init__()
        self.max_len = max_len
        self.position = nn.Embedding(max_len, 32)
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM + TIME_DIM + 32 + COND_DIM, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.offset = nn.Linear(hidden_dim, 3)
        self.residue_state = nn.Linear(hidden_dim, 20)

    @staticmethod
    def project_chain(raw: torch.Tensor, n_anchor: torch.Tensor,
                      c_anchor: torch.Tensor, lengths: torch.Tensor,
                      iterations: int = 4, bond_length: float = 3.8) -> torch.Tensor:
        """Differentiable FABRIK projection onto a closed fixed-CA-length chain."""
        projected = raw.clone()
        for batch_index, length_tensor in enumerate(lengths):
            length = int(length_tensor.item())
            points = [raw[batch_index, index] for index in range(length)]
            for _ in range(iterations):
                forward = []
                previous = n_anchor[batch_index]
                for point in points:
                    point = previous + bond_length * F.normalize(point - previous, dim=-1)
                    forward.append(point)
                    previous = point
                backward = [None] * length
                following = c_anchor[batch_index]
                for index in range(length - 1, -1, -1):
                    point = following + bond_length * F.normalize(
                        forward[index] - following, dim=-1
                    )
                    backward[index] = point
                    following = point
                points = backward
            projected[batch_index, :length] = torch.stack(points)
        return projected

    def forward(self, zt: torch.Tensor, time_embedding: torch.Tensor,
                framework_condition: torch.Tensor, initial_coords: torch.Tensor,
                lengths: torch.Tensor, n_anchor: torch.Tensor,
                c_anchor: torch.Tensor):
        batch = zt.size(0)
        positions = torch.arange(self.max_len, device=zt.device)
        pos = self.position(positions).unsqueeze(0).expand(batch, -1, -1)
        expanded = torch.cat([
            zt.unsqueeze(1).expand(-1, self.max_len, -1),
            time_embedding.unsqueeze(1).expand(-1, self.max_len, -1),
            pos,
            framework_condition.unsqueeze(1).expand(-1, self.max_len, -1),
        ], dim=-1)
        hidden = self.net(expanded)
        residue_state = F.softmax(self.residue_state(hidden), dim=-1)
        index = torch.arange(self.max_len, device=zt.device).unsqueeze(0)
        mask = index < lengths.unsqueeze(1)
        # The sine envelope suppresses endpoint-adjacent deformation and makes
        # closure learnable without revealing native H3 coordinates.
        u = (index.to(zt.dtype) + 1.0) / (lengths.unsqueeze(1).to(zt.dtype) + 1.0)
        envelope = torch.sin(math.pi * u).clamp_min(0.0) * mask
        raw_coords = initial_coords + self.offset(hidden) * envelope.unsqueeze(-1)
        coords = self.project_chain(raw_coords, n_anchor, c_anchor, lengths)
        return coords, residue_state, mask, raw_coords


class DeNovoGeometryFlow(nn.Module):
    """CDRH3 flow conditioned only on framework anchors and an epitope.

    No native/WT CDRH3 sequence or coordinates are accepted by this API.
    The framework is supplied in a coarse antigen-aligned pose. In joint mode
    its SE(3) state is integrated together with the H3 latent; serial mode uses
    the same pose flow without H3 feedback and fixed mode is the identity.
    """

    def __init__(self, flow_model: Optional[FlowModel] = None,
                 point_encoder: Optional[PointEncoder] = None,
                 cross_attn: Optional[EuclideanDistanceBiasedAttention] = None,
                 pose_mode: str = "joint", pose_feature_version: int = 17,
                 pose_time_conditioning: bool = True,
                 joint_h3_feedback: bool = True,
                 pose_correspondence: bool = True,
                 pose_gate_enabled: bool = True,
                 full_antigen_context: bool = False):
        super().__init__()
        if pose_mode not in {"fixed", "serial", "joint"}:
            raise ValueError("pose_mode must be fixed, serial or joint")
        self.pose_mode = pose_mode
        if pose_feature_version not in {10, 11, 12, 13, 14, 15, 16, 17}:
            raise ValueError("pose_feature_version must be 10--17")
        self.pose_feature_version = pose_feature_version
        self.pose_time_conditioning = bool(pose_time_conditioning)
        self.joint_h3_feedback = bool(joint_h3_feedback)
        self.pose_correspondence = bool(pose_correspondence)
        self.pose_gate_enabled = bool(pose_gate_enabled)
        self.full_antigen_context = bool(full_antigen_context)
        self.flow = flow_model or FlowModel()
        self.point_encoder = point_encoder or PointEncoder()
        self.framework_encoder = FrameworkEncoder()
        self.antigen_context_encoder = (
            FullAntigenContextEncoder() if self.full_antigen_context else None
        )
        self.relative_pose_encoder = LocalPoseGeometryEncoder()
        self.pose_correspondence_encoder = (
            LearnedPoseCorrespondenceEncoder() if pose_feature_version >= 14 else None
        )
        self.loop_pose_feedback = LoopPoseFeedbackEncoder()
        self.anchor_frame_encoder = AnchorFrameEncoder()
        self.initializer = AnchorLoopInitializer()
        self.loop_decoder = LoopStructureDecoder()
        self.cross_attn = cross_attn or EuclideanDistanceBiasedAttention()
        self.time_for_structure = TimeEmbedding(TIME_DIM)
        self.time_for_pose = TimeEmbedding(TIME_DIM)
        self.pose_refiner = PoseTwistPredictor(
            input_dim=LATENT_DIM + TIME_DIM + COND_DIM * (
                6 if pose_feature_version >= 14 else 5 if pose_feature_version >= 13 else 4
            )
        )
        # v16 factorizes docking into an H3-independent base field and a
        # strictly additive joint-feedback residual.  The zero-initialized
        # residual makes a joint model identical to its serial base before
        # fine-tuning and prevents candidate feedback from replacing the
        # stable docking solution wholesale.
        self.joint_pose_residual = (
            PoseTwistPredictor(input_dim=TIME_DIM + COND_DIM * 2)
            if pose_feature_version >= 16 else None
        )
        # A deterministic no-op gate is intentionally independent of the H3
        # latent.  It can suppress exact/native-pose updates without allowing
        # candidate noise to decide whether docking should move at all.
        self.pose_gate = (nn.Sequential(
            nn.Linear(COND_DIM * (4 if pose_feature_version >= 14 else 3), COND_DIM),
            nn.SiLU(), nn.Linear(COND_DIM, 2)
        ) if pose_feature_version >= 13 else None)
        if pose_feature_version >= 14:
            nn.init.zeros_(self.pose_gate[-1].weight)
            nn.init.constant_(self.pose_gate[-1].bias, -2.0)
        self.cond_proj = nn.Linear(48, COND_DIM)
        self.condition_fusion = nn.Sequential(
            nn.Linear(COND_DIM * (5 if self.full_antigen_context else 4), COND_DIM),
            nn.SiLU(), nn.Linear(COND_DIM, COND_DIM)
        )

    def forward(self, zt: torch.Tensor, t: torch.Tensor, epi_cloud: torch.Tensor,
                framework_features: torch.Tensor, framework_coords: torch.Tensor,
                n_anchor: torch.Tensor, c_anchor: torch.Tensor,
                lengths: torch.Tensor, framework_mask: Optional[torch.Tensor] = None,
                n_anchor_frame: Optional[torch.Tensor] = None,
                c_anchor_frame: Optional[torch.Tensor] = None,
                pose_state_rotation: Optional[torch.Tensor] = None,
                pose_state_translation: Optional[torch.Tensor] = None,
                static_conditions: Optional[Tuple[torch.Tensor,
                                                  torch.Tensor]] = None,
                antigen_features: Optional[torch.Tensor] = None,
                antigen_coords: Optional[torch.Tensor] = None,
                antigen_epitope_mask: Optional[torch.Tensor] = None,
                antigen_mask: Optional[torch.Tensor] = None):
        if static_conditions is None:
            epitope_condition = self.point_encoder(epi_cloud)
            initial_framework_condition = self.framework_encoder(
                framework_features, framework_coords, n_anchor, c_anchor, framework_mask
            )
        else:
            epitope_condition, initial_framework_condition = static_conditions
        pivot = framework_coords.mean(dim=1)

        def apply_pose(coords, rotation, translation):
            centered = coords - pivot.unsqueeze(1)
            return (torch.bmm(centered, rotation.transpose(1, 2))
                    + pivot.unsqueeze(1) + translation.unsqueeze(1))

        if self.pose_mode == "fixed":
            pose_rotation = torch.eye(3, device=zt.device, dtype=zt.dtype).expand(
                zt.size(0), -1, -1
            )
            pose_translation = torch.zeros(zt.size(0), 3, device=zt.device, dtype=zt.dtype)
            angular_velocity = torch.zeros_like(pose_translation)
            translation_velocity = torch.zeros_like(pose_translation)
        else:
            if pose_state_rotation is None:
                pose_rotation = torch.eye(3, device=zt.device, dtype=zt.dtype).expand(
                    zt.size(0), -1, -1
                )
            else:
                pose_rotation = pose_state_rotation
            pose_translation = (torch.zeros(zt.size(0), 3, device=zt.device, dtype=zt.dtype)
                                if pose_state_translation is None else pose_state_translation)
        refined_framework = apply_pose(framework_coords, pose_rotation, pose_translation)
        refined_n_anchor = apply_pose(
            n_anchor.unsqueeze(1), pose_rotation, pose_translation
        ).squeeze(1)
        refined_c_anchor = apply_pose(
            c_anchor.unsqueeze(1), pose_rotation, pose_translation
        ).squeeze(1)
        if n_anchor_frame is not None and c_anchor_frame is not None:
            refined_n_frame = apply_pose(n_anchor_frame, pose_rotation, pose_translation)
            refined_c_frame = apply_pose(c_anchor_frame, pose_rotation, pose_translation)
            local_frame = backbone_frame(refined_n_frame)
        else:
            # Equivariant fallback from the anchor axis and antigen direction.
            x_axis = F.normalize(refined_c_anchor - refined_n_anchor, dim=-1)
            epi_direction = epi_cloud[..., :3].mean(1) - refined_n_anchor
            y_axis = epi_direction - (epi_direction * x_axis).sum(-1, keepdim=True) * x_axis
            y_axis = F.normalize(y_axis, dim=-1)
            z_axis = torch.cross(x_axis, y_axis, dim=-1)
            local_frame = torch.stack([x_axis, y_axis, z_axis], dim=-1)
        relative_pose_condition = self.relative_pose_encoder(
            epi_cloud, refined_framework, refined_n_anchor, refined_c_anchor,
            local_frame, framework_mask
        )
        correspondence_condition = (
            self.pose_correspondence_encoder(
                epi_cloud, framework_features, refined_framework,
                refined_n_anchor, refined_c_anchor, local_frame, framework_mask
            ) if (self.pose_correspondence_encoder is not None
                  and self.pose_correspondence)
            else torch.zeros_like(relative_pose_condition)
        )
        framework_condition = self.framework_encoder(
            framework_features, refined_framework, refined_n_anchor,
            refined_c_anchor, framework_mask
        )
        if self.antigen_context_encoder is not None:
            if (antigen_features is None or antigen_coords is None
                    or antigen_epitope_mask is None):
                raise ValueError(
                    "full-antigen model requires antigen_features, antigen_coords, "
                    "and antigen_epitope_mask"
                )
            antigen_condition = self.antigen_context_encoder(
                antigen_features, antigen_coords, antigen_epitope_mask,
                refined_n_anchor, refined_c_anchor, antigen_mask,
            )
        else:
            antigen_condition = None
        if n_anchor_frame is not None and c_anchor_frame is not None:
            anchor_condition = self.anchor_frame_encoder(refined_n_frame, refined_c_frame)
        else:
            anchor_condition = torch.zeros_like(framework_condition)
        initial = self.initializer(
            refined_n_anchor, refined_c_anchor, lengths, refined_framework
        )
        loop_coords, residue_state, loop_mask, raw_loop_coords = self.loop_decoder(
            zt, self.time_for_structure(t), framework_condition, initial, lengths,
            refined_n_anchor, refined_c_anchor
        )
        distances = torch.cdist(epi_cloud[..., :3], loop_coords)
        # Padded residues must never receive attention mass.
        distances = distances.masked_fill(~loop_mask.unsqueeze(1), 1e4)
        enhanced, attention = self.cross_attn(epi_cloud, residue_state, loop_coords, distances)
        interface_condition = self.cond_proj(enhanced.mean(dim=1))
        if self.pose_mode == "joint":
            loop_pose_condition = self.loop_pose_feedback(
                loop_coords, loop_mask, epi_cloud, refined_framework, local_frame
            )
            pose_interface_condition = interface_condition
        else:
            # A strict serial dock must not inspect a candidate H3.
            loop_pose_condition = torch.zeros_like(relative_pose_condition)
            pose_interface_condition = torch.zeros_like(relative_pose_condition)
        if self.pose_mode != "fixed":
            # v15 keeps joint feedback through the decoded loop/interface but
            # removes the raw stochastic latent shortcut. Pose targets do not
            # depend on candidate noise, and v14 showed that this shortcut
            # caused exact-P0 drift without improving late-path direction.
            pose_latent = (
                zt if self.pose_mode == "joint" and self.pose_feature_version < 15
                else torch.zeros_like(zt)
            )
            # v17 exposes integration time to the deterministic base in both
            # serial and joint modes.  With the flow-matching linear path, the
            # current geometry alone does not identify how much perturbation
            # remains; a time-blind autonomous field therefore has conflicting
            # velocity targets at the same apparent state.
            pose_time = (
                self.time_for_pose(t)
                if (self.pose_time_conditioning
                    and (self.pose_mode == "joint" or self.pose_feature_version >= 17))
                else torch.zeros(zt.size(0), TIME_DIM, device=zt.device, dtype=zt.dtype)
            )
            base_pose_time = (
                torch.zeros_like(pose_time)
                if self.pose_feature_version == 16 else pose_time
            )
            base_loop_condition = (
                torch.zeros_like(loop_pose_condition)
                if self.pose_feature_version >= 16 else loop_pose_condition
            )
            base_interface_condition = (
                torch.zeros_like(pose_interface_condition)
                if self.pose_feature_version >= 16 else pose_interface_condition
            )
            pose_parts = [
                pose_latent, base_pose_time, epitope_condition,
                initial_framework_condition, relative_pose_condition,
                base_loop_condition
            ]
            if self.pose_feature_version >= 13:
                pose_parts.append(base_interface_condition)
            if self.pose_feature_version >= 14:
                pose_parts.append(correspondence_condition)
            pose_input = torch.cat(pose_parts, dim=-1)
            local_angular_velocity, local_translation_velocity = self.pose_refiner(pose_input)
            if (self.joint_pose_residual is not None and self.pose_mode == "joint"
                    and self.joint_h3_feedback):
                residual_angular, residual_translation = self.joint_pose_residual(
                    torch.cat([
                        pose_time, loop_pose_condition, pose_interface_condition
                    ], dim=-1)
                )
                local_angular_velocity = local_angular_velocity + residual_angular
                local_translation_velocity = local_translation_velocity + residual_translation
            else:
                residual_angular = torch.zeros_like(local_angular_velocity)
                residual_translation = torch.zeros_like(local_translation_velocity)
            if self.pose_gate is not None and self.pose_gate_enabled:
                gate_parts = [epitope_condition, initial_framework_condition,
                              relative_pose_condition]
                if self.pose_feature_version >= 14:
                    gate_parts.append(correspondence_condition)
                pose_gate = torch.sigmoid(self.pose_gate(torch.cat(gate_parts, dim=-1)))
                local_angular_velocity = local_angular_velocity * pose_gate[:, :1]
                local_translation_velocity = local_translation_velocity * pose_gate[:, 1:2]
            else:
                pose_gate = torch.ones(zt.size(0), 2, device=zt.device, dtype=zt.dtype)
            angular_velocity = torch.bmm(
                local_angular_velocity.unsqueeze(1), local_frame.transpose(1, 2)
            ).squeeze(1)
            translation_velocity = torch.bmm(
                local_translation_velocity.unsqueeze(1), local_frame.transpose(1, 2)
            ).squeeze(1)
        remaining = (1.0 - t).clamp_min(0.0)
        endpoint_rotation = torch.matmul(
            so3_exp_map(angular_velocity * remaining), pose_rotation
        )
        endpoint_translation = pose_translation + translation_velocity * remaining
        endpoint_framework = apply_pose(
            framework_coords, endpoint_rotation, endpoint_translation
        )
        # ``loop_coords`` lives in the current pose frame.  Map it back through
        # the current state and forward through the predicted endpoint state so
        # endpoint interface losses optimize the generated complex, rather than
        # a teacher-forced intermediate frame.  Row-vector convention matches
        # ``apply_pose`` above.
        loop_in_input_frame = torch.bmm(
            loop_coords - pivot.unsqueeze(1) - pose_translation.unsqueeze(1),
            pose_rotation,
        ) + pivot.unsqueeze(1)
        endpoint_loop = torch.bmm(
            loop_in_input_frame - pivot.unsqueeze(1),
            endpoint_rotation.transpose(1, 2),
        ) + pivot.unsqueeze(1) + endpoint_translation.unsqueeze(1)
        condition_parts = [
            epitope_condition, framework_condition, anchor_condition,
            interface_condition,
        ]
        if antigen_condition is not None:
            condition_parts.append(antigen_condition)
        condition = self.condition_fusion(torch.cat(condition_parts, dim=-1))
        velocity = self.flow(zt, t, condition, lengths)
        return {
            "velocity": velocity,
            "loop_coords": loop_coords,
            "raw_loop_coords": raw_loop_coords,
            "initial_coords": initial,
            "residue_state": residue_state,
            "loop_mask": loop_mask,
            "attention": attention,
            "distances": distances,
            "condition": condition,
            "antigen_condition": antigen_condition,
            "loop_pose_condition": loop_pose_condition,
            "pose_interface_condition": pose_interface_condition,
            "pose_correspondence_condition": correspondence_condition,
            "pose_gate": (pose_gate if self.pose_mode != "fixed" else
                          torch.zeros(zt.size(0), 2, device=zt.device, dtype=zt.dtype)),
            "pose_rotation": pose_rotation,
            "pose_translation": pose_translation,
            "pose_angular_velocity": angular_velocity,
            "pose_translation_velocity": translation_velocity,
            "pose_residual_angular_velocity": (
                residual_angular if self.pose_mode != "fixed"
                else torch.zeros_like(angular_velocity)
            ),
            "pose_residual_translation_velocity": (
                residual_translation if self.pose_mode != "fixed"
                else torch.zeros_like(translation_velocity)
            ),
            "pose_endpoint_rotation": endpoint_rotation,
            "pose_endpoint_translation": endpoint_translation,
            "pose_endpoint_framework_coords": endpoint_framework,
            "pose_endpoint_loop_coords": endpoint_loop,
            "refined_framework_coords": refined_framework,
            "refined_n_anchor": refined_n_anchor,
            "refined_c_anchor": refined_c_anchor,
        }


def de_novo_flow_matching_batch(model: DeNovoGeometryFlow, data_latent: torch.Tensor,
                                epi_cloud: torch.Tensor, framework_features: torch.Tensor,
                                framework_coords: torch.Tensor, n_anchor: torch.Tensor,
                                c_anchor: torch.Tensor, lengths: torch.Tensor,
                                framework_mask: Optional[torch.Tensor] = None,
                                n_anchor_frame: Optional[torch.Tensor] = None,
                                c_anchor_frame: Optional[torch.Tensor] = None,
                                target_pose_rotation: Optional[torch.Tensor] = None,
                                target_pose_translation: Optional[torch.Tensor] = None,
                                antigen_features: Optional[torch.Tensor] = None,
                                antigen_coords: Optional[torch.Tensor] = None,
                                antigen_epitope_mask: Optional[torch.Tensor] = None,
                                antigen_mask: Optional[torch.Tensor] = None):
    batch = data_latent.size(0)
    t = torch.rand(batch, 1, device=data_latent.device)
    prior = torch.randn_like(data_latent)
    zt = (1.0 - t) * prior + t * data_latent
    pose_rotation = pose_translation = None
    if target_pose_rotation is not None and target_pose_translation is not None:
        target_rotation_vector = so3_log_map(target_pose_rotation)
        pose_rotation = so3_exp_map(target_rotation_vector * t)
        pose_translation = target_pose_translation * t
    output = model(
        zt, t, epi_cloud, framework_features, framework_coords, n_anchor,
        c_anchor, lengths, framework_mask, n_anchor_frame, c_anchor_frame,
        pose_rotation, pose_translation, None,
        antigen_features, antigen_coords, antigen_epitope_mask, antigen_mask,
    )
    # Expose the sampled flow state so auxiliary objectives can evaluate the
    # same CDRH3/pose state under a wrong epitope.  Re-sampling here would turn
    # a conditionality test into a noise-comparison test.
    output["flow_latent_state"] = zt
    output["flow_time"] = t
    if target_pose_rotation is not None:
        output["target_pose_angular_velocity"] = target_rotation_vector
        output["target_pose_translation_velocity"] = target_pose_translation
    return output, data_latent - prior, t


def sample_de_novo_flow(model: DeNovoGeometryFlow, epi_cloud: torch.Tensor,
                        framework_features: torch.Tensor, framework_coords: torch.Tensor,
                        n_anchor: torch.Tensor, c_anchor: torch.Tensor,
                        lengths: torch.Tensor, framework_mask: Optional[torch.Tensor] = None,
                        num_steps: int = 50, latent_scale: float = 1.0,
                        n_anchor_frame: Optional[torch.Tensor] = None,
                        c_anchor_frame: Optional[torch.Tensor] = None,
                        antigen_features: Optional[torch.Tensor] = None,
                        antigen_coords: Optional[torch.Tensor] = None,
                        antigen_epitope_mask: Optional[torch.Tensor] = None,
                        antigen_mask: Optional[torch.Tensor] = None):
    z = latent_scale * torch.randn(epi_cloud.size(0), LATENT_DIM, device=epi_cloud.device)
    # These encodings are invariant throughout the ODE trajectory. Caching is
    # especially important for the framework--epitope cdist in P0--P4 sweeps.
    static_conditions = (
        model.point_encoder(epi_cloud),
        model.framework_encoder(
            framework_features, framework_coords, n_anchor, c_anchor, framework_mask
        ),
    )
    pose_rotation = torch.eye(3, device=epi_cloud.device, dtype=epi_cloud.dtype).expand(
        epi_cloud.size(0), -1, -1
    ).clone()
    pose_translation = torch.zeros(epi_cloud.size(0), 3, device=epi_cloud.device,
                                   dtype=epi_cloud.dtype)
    grid = torch.linspace(0.0, 1.0, num_steps + 1, device=epi_cloud.device)
    output = None
    for index in range(num_steps):
        t = grid[index].expand(epi_cloud.size(0), 1)
        output = model(
            z, t, epi_cloud, framework_features, framework_coords, n_anchor,
            c_anchor, lengths, framework_mask, n_anchor_frame, c_anchor_frame,
            pose_rotation, pose_translation, static_conditions,
            antigen_features, antigen_coords, antigen_epitope_mask, antigen_mask,
        )
        dt = grid[index + 1] - grid[index]
        z = z + output["velocity"] * dt
        if model.pose_mode != "fixed":
            pose_rotation = torch.matmul(
                so3_exp_map(output["pose_angular_velocity"] * dt), pose_rotation
            )
            pose_translation = pose_translation + output["pose_translation_velocity"] * dt
    final_t = torch.ones(epi_cloud.size(0), 1, device=epi_cloud.device)
    output = model(
        z, final_t, epi_cloud, framework_features, framework_coords, n_anchor,
        c_anchor, lengths, framework_mask, n_anchor_frame, c_anchor_frame,
        pose_rotation, pose_translation, static_conditions,
        antigen_features, antigen_coords, antigen_epitope_mask, antigen_mask,
    )
    return z, output


def loop_geometry_loss(predicted: torch.Tensor, target: torch.Tensor,
                       lengths: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    index = torch.arange(predicted.size(1), device=predicted.device).unsqueeze(0)
    mask = index < lengths.unsqueeze(1)
    coordinate = F.smooth_l1_loss(predicted, target, reduction="none").mean(-1)
    coordinate = (coordinate * mask).sum(-1) / mask.sum(-1).clamp_min(1)
    pred_dist = torch.cdist(predicted, predicted)
    target_dist = torch.cdist(target, target)
    pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
    distance = ((pred_dist - target_dist).abs() * pair_mask).sum(dim=(1, 2))
    distance = distance / pair_mask.sum(dim=(1, 2)).clamp_min(1)
    return ((t.squeeze(-1) ** 2) * (coordinate + 0.25 * distance)).mean()


def framework_clash_loss(loop_coords: torch.Tensor, framework_coords: torch.Tensor,
                         lengths: torch.Tensor, cutoff: float = 2.5) -> torch.Tensor:
    distances = torch.cdist(loop_coords, framework_coords)
    mask = torch.arange(loop_coords.size(1), device=loop_coords.device).unsqueeze(0) < lengths.unsqueeze(1)
    penalty = F.relu(cutoff - distances).square() * mask.unsqueeze(-1)
    return penalty.sum() / mask.sum().clamp_min(1)
