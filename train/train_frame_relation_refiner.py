#!/usr/bin/env python3
"""Train the frame-aware relation refiner with interface-heavy objectives."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from dygf import backbone_frame  # noqa: E402
from frame_relation_refiner import (  # noqa: E402
    FrameRelationGraph, FrameRelationRefiner, recurrent_graph_update,
)
from relation_interface_refiner import (  # noqa: E402
    AA_TO_INDEX, symmetric_masked_infonce,
)


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GraphDataset(Dataset):
    def __init__(self, path):
        self.items = torch.load(path, map_location="cpu", weights_only=False)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def rotation_loss(predicted, target):
    relative = torch.matmul(predicted.transpose(-1, -2), target)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1, 1)
    return (1.0 - cosine).mean()


def kabsch_align(predicted, target):
    """Align row-vector coordinates with a proper, differentiable rotation."""
    predicted_centered = predicted - predicted.mean(0, keepdim=True)
    target_centered = target - target.mean(0, keepdim=True)
    covariance = predicted_centered.transpose(0, 1) @ target_centered
    left, _, right_t = torch.linalg.svd(covariance)
    sign = torch.where(
        torch.det(left @ right_t) < 0,
        predicted.new_tensor(-1.0), predicted.new_tensor(1.0),
    )
    correction = torch.diag(torch.stack([
        predicted.new_tensor(1.0), predicted.new_tensor(1.0), sign,
    ]))
    rotation = left @ correction @ right_t
    return predicted_centered @ rotation + target.mean(0, keepdim=True)


def frame_aligned_point_error(predicted_coords, predicted_frames,
                              target_coords, target_frames, frame_mask):
    """Rigid-invariant all-pairs CA error in each available residue frame."""
    if not bool(frame_mask.any()):
        return predicted_coords.new_zeros(())
    pred_origins = predicted_coords[frame_mask]
    target_origins = target_coords[frame_mask]
    pred_local = torch.matmul(
        predicted_coords.unsqueeze(0) - pred_origins.unsqueeze(1),
        predicted_frames[frame_mask],
    )
    target_local = torch.matmul(
        target_coords.unsqueeze(0) - target_origins.unsqueeze(1),
        target_frames,
    )
    error = (pred_local - target_local).norm(dim=-1).clamp(max=10.0)
    return error.mean()


def expected_liabilities(logits):
    probabilities = logits.softmax(-1)
    value = probabilities[:, AA_TO_INDEX["M"]].sum()
    for motif in ("NG", "DG", "DS", "DD", "NS", "NT"):
        value = value + (
            probabilities[:-1, AA_TO_INDEX[motif[0]]]
            * probabilities[1:, AA_TO_INDEX[motif[1]]]
        ).sum()
    return value


def soft_interface_f1_losses(predicted_distances, contact_target,
                             specified_epitope, cutoff=8.0,
                             temperature=0.5):
    """Differentiable surrogates for CHIMERA Fnat and epitope F1.

    Pairwise probabilities approximate the native-contact set used by Fnat.
    A smooth minimum over H3 residues turns them into antigen-residue contact
    probabilities for the epitope precision/recall objective.
    """
    pair_probabilities = torch.sigmoid(
        (cutoff - predicted_distances) / temperature
    )
    pair_target = contact_target.to(pair_probabilities.dtype)
    pair_true_positive = (pair_probabilities * pair_target).sum()
    pair_f1 = 2.0 * pair_true_positive / (
        pair_probabilities.sum() + pair_target.sum()
    ).clamp_min(1e-6)

    soft_minimum = -temperature * torch.logsumexp(
        -predicted_distances / temperature, dim=0
    )
    antigen_contact = torch.sigmoid((cutoff - soft_minimum) / temperature)
    epitope_target = specified_epitope.to(antigen_contact.dtype)
    epitope_true_positive = (antigen_contact * epitope_target).sum()
    epitope_f1 = 2.0 * epitope_true_positive / (
        antigen_contact.sum() + epitope_target.sum()
    ).clamp_min(1e-6)
    return 1.0 - pair_f1, 1.0 - epitope_f1, pair_f1, epitope_f1


def _sample_objective_step(model, item, graph, device, weights, teacher=None):
    target_indices = item["target_indices"].to(device)
    target_coords = item["target_coords"].to(device)
    target_backbone = item["target_backbone"].to(device)
    target_mask = item["target_backbone_mask"].to(device)
    contact_target = item["contact_target"].to(device)
    output = model(graph)
    is_pose_augmented = "__dockaug" in str(item.get("sample_id", ""))
    distillation = output["sequence_logits"].new_zeros(())
    geometry_distillation = distillation
    if teacher is not None and not is_pose_augmented:
        with torch.no_grad():
            teacher_output = teacher(graph)
        temperature = weights.get("distillation_temperature", 1.0)
        distillation = F.kl_div(
            F.log_softmax(output["sequence_logits"] / temperature, dim=-1),
            F.softmax(
                teacher_output["sequence_logits"] / temperature, dim=-1
            ),
            reduction="batchmean",
        ) * temperature ** 2
        geometry_distillation = F.smooth_l1_loss(
            output["h3_coords"], teacher_output["h3_coords"]
        )
    label_smoothing = weights.get("label_smoothing", 0.0)
    class_weights = weights.get("class_weights")
    if class_weights is not None:
        class_weights = torch.as_tensor(
            class_weights, device=device, dtype=output["sequence_logits"].dtype
        )
    sequence = F.cross_entropy(
        output["sequence_logits"], target_indices,
        weight=class_weights,
        label_smoothing=label_smoothing,
    )
    contact_positions = contact_target.any(-1)
    contact_sequence = (
        F.cross_entropy(
            output["sequence_logits"][contact_positions],
            target_indices[contact_positions],
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
        if bool(contact_positions.any()) else sequence.new_zeros(())
    )
    coordinate = F.smooth_l1_loss(output["h3_coords"], target_coords)
    # Independent antibody folding is already a strong geometry prior.  This
    # term lets distribution-matched training learn only the antigen-driven
    # correction instead of replacing the starting loop wholesale.
    initial_coords = graph.coords[:graph.h3_length]
    residual = F.smooth_l1_loss(output["h3_coords"], initial_coords)
    aligned_coords = kabsch_align(output["h3_coords"], target_coords)
    aligned_coordinate = F.smooth_l1_loss(aligned_coords, target_coords)
    backbone = F.smooth_l1_loss(
        output["h3_backbone"][target_mask], target_backbone[target_mask]
    )
    pair_distance = F.smooth_l1_loss(
        torch.cdist(output["h3_coords"], output["h3_coords"]),
        torch.cdist(target_coords, target_coords),
    )
    complete = target_mask[:, :3].all(-1)
    if bool(complete.any()):
        target_frames = backbone_frame(target_backbone[complete])
        frame = rotation_loss(output["h3_frames"][complete], target_frames)
        fape = frame_aligned_point_error(
            output["h3_coords"], output["h3_frames"], target_coords,
            target_frames, complete,
        )
    else:
        # Some deposited loops have CA-only coordinates. An empty mean would
        # poison the complete validation epoch even though all other labels
        # remain usable.
        frame = sequence.new_zeros(())
        fape = sequence.new_zeros(())
    if len(target_coords) >= 2:
        predicted_bonds = torch.diff(output["h3_coords"], dim=0).norm(dim=-1)
        target_bonds = torch.diff(target_coords, dim=0).norm(dim=-1)
        bond = F.smooth_l1_loss(predicted_bonds, target_bonds)
    else:
        bond = sequence.new_zeros(())
    if len(target_coords) >= 3:
        predicted_curvature = torch.diff(output["h3_coords"], n=2, dim=0)
        target_curvature = torch.diff(target_coords, n=2, dim=0)
        curvature = F.smooth_l1_loss(predicted_curvature, target_curvature)
    else:
        # RAbD's SAbDab source snapshot contains one valid two-residue IMGT
        # H3.  A mean over its empty second difference is NaN.
        curvature = sequence.new_zeros(())
    antigen_coords = graph.coords[graph.antigen_mask]
    predicted_interface = torch.cdist(output["h3_coords"], antigen_coords)
    target_interface = torch.cdist(target_coords, antigen_coords)
    interface_mask = target_interface <= 12.0
    interface = F.smooth_l1_loss(
        predicted_interface[interface_mask], target_interface[interface_mask]
    ) if bool(interface_mask.any()) else sequence.new_zeros(())
    # Match the complete local interface while saturating distant pairs.  The
    # positive-only term above attracts native contacts but cannot discourage
    # a loop from making additional contacts with the wrong antigen patch.
    interface_all = F.smooth_l1_loss(
        predicted_interface.clamp_max(12.0), target_interface.clamp_max(12.0)
    )
    specified_epitope = graph.epitope_mask[graph.antigen_mask]
    geometry_contact, epitope_geometry, soft_geometry_contact_f1, \
        soft_epitope_f1 = soft_interface_f1_losses(
            predicted_interface, contact_target, specified_epitope
        )
    positives = contact_target.sum().clamp_min(1)
    negatives = (~contact_target).sum().clamp_min(1)
    contact = F.binary_cross_entropy_with_logits(
        output["contact_logits"], contact_target.float(),
        # Native contacts are only about 2% of dense H3--antigen pairs in the
        # RAAD split, so the previous cap of ten underweighted positives.
        pos_weight=(negatives / positives).clamp(1.0, 50.0),
    )
    edit = output["edit_gate"].mean()
    liability = expected_liabilities(output["sequence_logits"])
    adapter_delta = output.get("sequence_adapter_delta")
    native_preservation = (
        adapter_delta.square().mean()
        if adapter_delta is not None and not is_pose_augmented
        else sequence.new_zeros(())
    )
    negative_epitope = output.get("negative_epitope_embedding")
    if negative_epitope is None:
        hard_negative_specificity = sequence.new_zeros(())
    else:
        positive_similarity = torch.dot(
            output["sequence_embedding"], output["epitope_embedding"]
        )
        negative_similarity = torch.dot(
            output["sequence_embedding"], negative_epitope
        )
        hard_negative_specificity = F.relu(
            weights.get("specificity_margin", 0.2)
            - positive_similarity + negative_similarity
        )
    objective = (
        weights["sequence"] * sequence
        + weights["contact_sequence"] * contact_sequence
        + weights["coordinate"] * coordinate
        + weights["residual"] * residual
        + weights["aligned_coordinate"] * aligned_coordinate
        + weights["backbone"] * backbone
        + weights["pair_distance"] * pair_distance
        + weights["bond"] * bond
        + weights["curvature"] * curvature
        + weights["fape"] * fape
        + weights["interface"] * interface
        + weights["interface_all"] * interface_all
        + weights.get("geometry_contact", 0.0) * geometry_contact
        + weights.get("epitope_geometry", 0.0) * epitope_geometry
        + weights["frame"] * frame
        + weights["contact"] * contact
        + weights["edit"] * edit
        + weights["liability"] * liability
        + weights.get("native_preservation", 0.0) * native_preservation
        + weights.get("distillation", 0.0) * distillation
        + weights.get("geometry_distillation", 0.0)
        * geometry_distillation
        + weights.get("hard_negative_specificity", 0.0)
        * hard_negative_specificity
    )
    predicted = output["sequence_logits"].argmax(-1)
    aar = (predicted == target_indices).float().mean()
    caar = ((predicted[contact_positions] == target_indices[contact_positions]).float().mean()
            if bool(contact_positions.any()) else aar.new_zeros(()))
    rmsd = torch.sqrt((output["h3_coords"] - target_coords).square().sum(-1).mean())
    aligned_rmsd = torch.sqrt(
        (aligned_coords - target_coords).square().sum(-1).mean()
    )
    predicted_contacts = output["contact_logits"] > 0
    tp = (predicted_contacts & contact_target).sum().float()
    contact_f1 = 2 * tp / (predicted_contacts.sum() + contact_target.sum()).clamp_min(1)
    geometry_contacts = predicted_interface <= 8.0
    geometry_tp = (geometry_contacts & contact_target).sum().float()
    geometry_contact_f1 = 2 * geometry_tp / (
        geometry_contacts.sum() + contact_target.sum()
    ).clamp_min(1)
    contacted_antigen = geometry_contacts.any(0)
    epitope_tp = (contacted_antigen & specified_epitope).sum().float()
    epitope_f1 = 2 * epitope_tp / (
        contacted_antigen.sum() + specified_epitope.sum()
    ).clamp_min(1)
    return objective, output, {
        "sequence": sequence, "contact_sequence": contact_sequence,
        "coordinate": coordinate, "residual": residual,
        "aligned_coordinate": aligned_coordinate,
        "backbone": backbone, "pair_distance": pair_distance, "bond": bond,
        "curvature": curvature, "fape": fape, "interface": interface,
        "interface_all": interface_all,
        "geometry_contact": geometry_contact,
        "epitope_geometry": epitope_geometry,
        "soft_geometry_contact_f1": soft_geometry_contact_f1,
        "soft_epitope_f1": soft_epitope_f1,
        "frame": frame, "contact": contact,
        "edit": edit, "liability": liability,
        "native_preservation": native_preservation,
        "distillation": distillation,
        "geometry_distillation": geometry_distillation,
        "hard_negative_specificity": hard_negative_specificity,
        "aar": aar, "caar": caar, "rmsd": rmsd,
        "aligned_rmsd": aligned_rmsd,
        "contact_f1": contact_f1,
        "geometry_contact_f1": geometry_contact_f1,
        "epitope_f1": epitope_f1,
    }


def sample_objective(model, item, device, weights, unroll_steps=1,
                     unroll_decay=0.3, teacher=None):
    """Train the same hard recurrent trajectory used during inference."""
    if unroll_steps < 1:
        raise ValueError("unroll_steps must be positive")
    graph = FrameRelationGraph(**item["graph"]).to(device)
    objectives, output, metrics = [], None, None
    for step in range(unroll_steps):
        objective, output, metrics = _sample_objective_step(
            model, item, graph, device, weights, teacher=teacher
        )
        objectives.append((unroll_decay ** (unroll_steps - step - 1)) * objective)
        if step + 1 < unroll_steps:
            graph = recurrent_graph_update(
                graph, output["h3_coords"], output["sequence_logits"]
            )
    return torch.stack(objectives).sum(), output, metrics


def run_epoch(model, loader, device, weights, optimizer=None, temperature=0.2,
              unroll_steps=1, unroll_decay=0.3, teacher=None):
    training = optimizer is not None
    model.train(training)
    names = (
        "loss", "sequence", "contact_sequence", "coordinate", "residual",
        "aligned_coordinate", "backbone", "pair_distance", "bond",
        "curvature", "fape", "interface", "interface_all", "frame", "contact",
        "geometry_contact", "epitope_geometry", "soft_geometry_contact_f1",
        "soft_epitope_f1", "edit", "liability",
        "native_preservation",
        "distillation", "geometry_distillation",
        "specificity", "hard_negative_specificity", "composition",
        "aar", "caar", "rmsd", "aligned_rmsd", "contact_f1",
        "geometry_contact_f1", "epitope_f1",
    )
    totals = {name: 0.0 for name in names}; samples_seen = 0
    with (torch.enable_grad() if training else torch.no_grad()):
        for batch in loader:
            if training:
                optimizer.zero_grad(set_to_none=True)
            objectives, sequence_embeddings, epitope_embeddings, pdb_ids = [], [], [], []
            sequence_probabilities = []
            metrics_list = []
            for item in batch:
                objective, output, metrics = sample_objective(
                    model, item, device, weights, unroll_steps, unroll_decay,
                    teacher=teacher,
                )
                objectives.append(objective); metrics_list.append(metrics)
                sequence_embeddings.append(output["sequence_embedding"])
                epitope_embeddings.append(output["epitope_embedding"])
                sequence_probabilities.append(output["sequence_logits"].softmax(-1))
                pdb_ids.append(item["pdb"])
            specificity = symmetric_masked_infonce(
                sequence_embeddings, epitope_embeddings, pdb_ids, temperature
            )
            composition = specificity.new_zeros(())
            composition_prior = weights.get("composition_prior")
            if composition_prior is not None:
                predicted_prior = torch.cat(sequence_probabilities).mean(0).clamp_min(1e-7)
                target_prior = torch.as_tensor(
                    composition_prior, device=device, dtype=predicted_prior.dtype
                ).clamp_min(1e-7)
                composition = 0.5 * (
                    (predicted_prior * (predicted_prior.log() - target_prior.log())).sum()
                    + (target_prior * (target_prior.log() - predicted_prior.log())).sum()
                )
            loss = (
                torch.stack(objectives).mean()
                + weights["specificity"] * specificity
                + weights.get("composition", 0.0) * composition
            )
            if training:
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            count = len(batch); samples_seen += count
            totals["loss"] += float(loss.detach()) * count
            totals["specificity"] += float(specificity.detach()) * count
            totals["composition"] += float(composition.detach()) * count
            for metrics in metrics_list:
                for name, value in metrics.items():
                    totals[name] += float(value.detach())
    return {name: value / max(samples_seen, 1) for name, value in totals.items()}


def configure_trainable_parameters(model, mode, pose_tune_layers=2):
    """Freeze model components for a modular optimization stage."""
    if mode == "all":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    elif mode == "geometry_coordinates":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for layer in model.layers:
            for module in (layer.radial_coordinate, layer.local_coordinate):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
    elif mode == "sequence_adapter":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for module in (model.sequence_head, model.edit_gate):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    elif mode == "sequence_mlp_adapter":
        if model.sequence_adapter is None:
            raise ValueError("sequence_mlp_adapter requires --sequence_adapter_dim > 0")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.sequence_adapter.parameters():
            parameter.requires_grad_(True)
    elif mode == "dynamic_relation_branch":
        if model.dynamic_branch is None:
            raise ValueError(
                "dynamic_relation_branch requires --dynamic_relation_layers > 0"
            )
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.dynamic_branch.parameters():
            parameter.requires_grad_(True)
    elif mode in ("interface_geometry", "interface_sequence"):
        if model.dynamic_branch is None:
            raise ValueError("two-stage training requires a dynamic branch")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        branch = model.dynamic_branch
        modules = ([branch.input, branch.layers, branch.contact_delta]
                   if mode == "interface_geometry" else
                   [branch.sequence_delta])
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    elif mode == "interface_pose_tune":
        if model.dynamic_branch is None:
            raise ValueError(
                "interface_pose_tune requires --dynamic_relation_layers > 0"
            )
        if pose_tune_layers < 1 or pose_tune_layers > len(model.layers):
            raise ValueError("pose_tune_layers must select existing trunk layers")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        modules = [
            *model.layers[-pose_tune_layers:], model.sequence_head,
            model.edit_gate, model.backbone_delta, model.contact_head,
            model.sequence_projection, model.epitope_projection,
            model.dynamic_branch,
        ]
        if model.sequence_adapter is not None:
            modules.append(model.sequence_adapter)
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    else:
        raise ValueError(f"unknown trainable mode: {mode}")
    trainable = [parameter for parameter in model.parameters()
                 if parameter.requires_grad]
    if not trainable:
        raise ValueError(f"trainable mode {mode} selected no parameters")
    return trainable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics_jsonl")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--coordinate_step_scale", type=float, default=1.0)
    parser.add_argument("--sequence_adapter_dim", type=int, default=0)
    parser.add_argument("--dynamic_relation_layers", type=int, default=0)
    parser.add_argument("--dynamic_relation_dim", type=int, default=128)
    parser.add_argument("--dynamic_max_displacement", type=float, default=0.35)
    parser.add_argument("--dynamic_prior_adjustment", type=float, default=2.0)
    parser.add_argument("--dynamic_candidate_base_strength", type=float, default=0.0)
    parser.add_argument("--dynamic_dense_interface", action="store_true")
    parser.add_argument("--dynamic_coordinate_step_scale", type=float, default=1.0)
    parser.add_argument("--unroll_steps", type=int, default=1)
    parser.add_argument("--unroll_decay", type=float, default=0.3)
    parser.add_argument("--sequence_weight", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--contact_sequence_weight", type=float, default=1.0)
    parser.add_argument("--specificity_weight", type=float, default=0.3)
    parser.add_argument("--hard_negative_specificity_weight", type=float, default=0.0)
    parser.add_argument("--specificity_margin", type=float, default=0.2)
    parser.add_argument("--composition_weight", type=float, default=0.0)
    parser.add_argument("--class_balance_power", type=float, default=0.0)
    parser.add_argument("--coordinate_weight", type=float, default=0.5)
    parser.add_argument("--residual_weight", type=float, default=0.0)
    parser.add_argument("--aligned_coordinate_weight", type=float, default=0.0)
    parser.add_argument("--backbone_weight", type=float, default=0.2)
    parser.add_argument("--pair_distance_weight", type=float, default=0.2)
    parser.add_argument("--bond_weight", type=float, default=0.0)
    parser.add_argument("--curvature_weight", type=float, default=0.0)
    parser.add_argument("--fape_weight", type=float, default=0.0)
    parser.add_argument("--interface_weight", type=float, default=0.0)
    parser.add_argument("--interface_all_weight", type=float, default=0.0)
    parser.add_argument("--geometry_contact_weight", type=float, default=0.0)
    parser.add_argument("--epitope_geometry_weight", type=float, default=0.0)
    parser.add_argument("--liability_weight", type=float, default=0.0)
    parser.add_argument("--native_preservation_weight", type=float, default=0.0)
    parser.add_argument("--distillation_weight", type=float, default=0.0)
    parser.add_argument("--geometry_distillation_weight", type=float, default=0.0)
    parser.add_argument("--distillation_temperature", type=float, default=1.0)
    parser.add_argument("--frame_weight", type=float, default=0.1)
    parser.add_argument("--contact_weight", type=float, default=0.3)
    parser.add_argument("--edit_weight", type=float, default=0.02)
    parser.add_argument(
        "--trainable_mode",
        choices=("all", "geometry_coordinates", "sequence_adapter",
                 "sequence_mlp_adapter", "dynamic_relation_branch",
                 "interface_pose_tune", "interface_geometry", "interface_sequence"),
        default="all",
    )
    parser.add_argument("--pose_tune_layers", type=int, default=2)
    parser.add_argument("--init_checkpoint")
    parser.add_argument("--teacher_checkpoint")
    parser.add_argument(
        "--preserve_init_best", action="store_true",
        help="score the initialization on validation and keep it unless training improves it",
    )
    parser.add_argument(
        "--last_output",
        help="optional checkpoint written every epoch for exact interrupted-run resumption",
    )
    parser.add_argument(
        "--resume_optimizer", action="store_true",
        help="restore optimizer state and counters when available in init_checkpoint",
    )
    parser.add_argument(
        "--save_epoch_every", type=int, default=0,
        help="save a validation-selection candidate every N epochs (0 disables)",
    )
    parser.add_argument("--selection_aligned_rmsd_weight", type=float, default=0.0)
    parser.add_argument("--selection_rmsd_weight", type=float, default=0.0)
    parser.add_argument("--selection_caar_weight", type=float, default=1.0)
    parser.add_argument("--selection_aar_weight", type=float, default=0.1)
    parser.add_argument("--selection_contact_f1_weight", type=float, default=0.0)
    parser.add_argument("--selection_geometry_contact_f1_weight", type=float, default=0.0)
    parser.add_argument("--selection_epitope_f1_weight", type=float, default=0.0)
    parser.add_argument("--selection_liability_weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    args = parser.parse_args(); seed_everything(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train = GraphDataset(args.train_data); val = GraphDataset(args.val_data)
    generator = torch.Generator().manual_seed(args.seed)
    collate = lambda values: values
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, generator=generator,
                              num_workers=args.workers)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=args.workers)
    model = FrameRelationRefiner(
        args.hidden_dim, args.layers,
        sequence_adapter_dim=args.sequence_adapter_dim,
        dynamic_relation_layers=args.dynamic_relation_layers,
        dynamic_relation_dim=args.dynamic_relation_dim,
        dynamic_max_displacement=args.dynamic_max_displacement,
        dynamic_prior_adjustment=args.dynamic_prior_adjustment,
        dynamic_candidate_base_strength=args.dynamic_candidate_base_strength,
        dynamic_dense_interface=args.dynamic_dense_interface,
        dynamic_coordinate_step_scale=args.dynamic_coordinate_step_scale,
        coordinate_step_scale=args.coordinate_step_scale,
    ).to(device)
    model_config = {
        "hidden_dim": args.hidden_dim, "layers": args.layers,
        "sequence_adapter_dim": args.sequence_adapter_dim,
        "dynamic_relation_layers": args.dynamic_relation_layers,
        "dynamic_relation_dim": args.dynamic_relation_dim,
        "dynamic_max_displacement": args.dynamic_max_displacement,
        "dynamic_prior_adjustment": args.dynamic_prior_adjustment,
        "dynamic_candidate_base_strength": args.dynamic_candidate_base_strength,
        "dynamic_dense_interface": args.dynamic_dense_interface,
        "dynamic_coordinate_step_scale": args.dynamic_coordinate_step_scale,
        "coordinate_step_scale": args.coordinate_step_scale,
    }
    initial = None
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        allow_new_modules = (
            args.sequence_adapter_dim > 0 or args.dynamic_relation_layers > 0
        )
        incompatible = model.load_state_dict(
            initial["model"], strict=not allow_new_modules
        )
        if allow_new_modules:
            unexpected = list(incompatible.unexpected_keys)
            missing = list(incompatible.missing_keys)
            allowed_prefixes = []
            if args.sequence_adapter_dim > 0:
                allowed_prefixes.append("sequence_adapter.")
            if args.dynamic_relation_layers > 0:
                allowed_prefixes.append("dynamic_branch.")
            if unexpected or any(
                not any(key.startswith(prefix) for prefix in allowed_prefixes)
                for key in missing
            ):
                raise ValueError(
                    f"unexpected modular initialization mismatch: "
                    f"missing={missing}, unexpected={unexpected}"
                )
    teacher = None
    if args.teacher_checkpoint:
        teacher_state = torch.load(
            args.teacher_checkpoint, map_location="cpu", weights_only=False
        )
        teacher = FrameRelationRefiner(**teacher_state["model_config"]).to(device)
        teacher.load_state_dict(teacher_state["model"])
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    trainable_parameters = configure_trainable_parameters(
        model, args.trainable_mode, args.pose_tune_layers
    )
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=1e-4)
    if args.resume_optimizer and initial is not None and "optimizer" in initial:
        optimizer.load_state_dict(initial["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
    target_counts = torch.zeros(20, dtype=torch.float64)
    for item in train.items:
        target_counts += torch.bincount(item["target_indices"], minlength=20)
    composition_prior = (target_counts / target_counts.sum()).float()
    class_weights = (composition_prior + 1e-4).pow(-args.class_balance_power)
    class_weights = class_weights / class_weights.mean()
    weights = {
        "sequence": args.sequence_weight,
        "label_smoothing": args.label_smoothing,
        "contact_sequence": args.contact_sequence_weight,
        "coordinate": args.coordinate_weight,
        "residual": args.residual_weight,
        "aligned_coordinate": args.aligned_coordinate_weight,
        "backbone": args.backbone_weight,
        "pair_distance": args.pair_distance_weight,
        "bond": args.bond_weight, "curvature": args.curvature_weight,
        "fape": args.fape_weight, "interface": args.interface_weight,
        "interface_all": args.interface_all_weight,
        "geometry_contact": args.geometry_contact_weight,
        "epitope_geometry": args.epitope_geometry_weight,
        "liability": args.liability_weight,
        "native_preservation": args.native_preservation_weight,
        "distillation": args.distillation_weight,
        "geometry_distillation": args.geometry_distillation_weight,
        "distillation_temperature": args.distillation_temperature,
        "frame": args.frame_weight, "contact": args.contact_weight,
        "edit": args.edit_weight,
        "specificity": args.specificity_weight,
        "hard_negative_specificity": args.hard_negative_specificity_weight,
        "specificity_margin": args.specificity_margin,
        "composition": args.composition_weight,
        "composition_prior": composition_prior.tolist(),
        "class_weights": class_weights.tolist(),
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = (Path(args.metrics_jsonl) if args.metrics_jsonl else
                    output.with_suffix(".metrics.jsonl"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True); metrics_path.write_text("")

    def selection_key(val_metrics):
        selection_score = (
            args.selection_caar_weight * val_metrics["caar"]
            + args.selection_aar_weight * val_metrics["aar"]
            + args.selection_contact_f1_weight * val_metrics["contact_f1"]
            + args.selection_geometry_contact_f1_weight
            * val_metrics["geometry_contact_f1"]
            + args.selection_epitope_f1_weight * val_metrics["epitope_f1"]
            - args.selection_aligned_rmsd_weight * val_metrics["aligned_rmsd"]
            - args.selection_rmsd_weight * val_metrics["rmsd"]
            - args.selection_liability_weight * val_metrics["liability"]
        )
        return (selection_score, val_metrics["caar"], val_metrics["aar"])

    best_key = None; best_epoch = None; stale = 0; start_epoch = 1
    if args.resume_optimizer and initial is not None and "optimizer" in initial:
        start_epoch = int(initial.get("epoch", 0)) + 1
        best_key_value = initial.get("best_key", initial.get("selection_key"))
        best_key = tuple(best_key_value) if best_key_value is not None else None
        best_epoch = initial.get("best_epoch", initial.get("epoch"))
        stale = int(initial.get("stale", 0))
    elif args.preserve_init_best and initial is not None:
        initial_val_metrics = run_epoch(
            model, val_loader, device, weights,
            unroll_steps=args.unroll_steps, unroll_decay=args.unroll_decay,
            teacher=teacher,
        )
        best_key = selection_key(initial_val_metrics)
        best_epoch = 0
        torch.save({
            "model": model.state_dict(), "epoch": 0,
            "model_config": model_config,
            "loss_weights": weights, "selection_key": best_key,
            "train_metrics": None, "val_metrics": initial_val_metrics,
            "config": vars(args), "format": "frame_relation_refiner_v4",
        }, output)
        initial_record = {
            "epoch": 0, "best_epoch": 0, "stale": 0,
            **{f"val_{k}": v for k, v in initial_val_metrics.items()},
        }
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(initial_record, sort_keys=True) + "\n")
        print(json.dumps(initial_record, sort_keys=True), flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, device, weights, optimizer,
            unroll_steps=args.unroll_steps, unroll_decay=args.unroll_decay,
            teacher=teacher,
        )
        val_metrics = run_epoch(
            model, val_loader, device, weights,
            unroll_steps=args.unroll_steps, unroll_decay=args.unroll_decay,
            teacher=teacher,
        )
        key = selection_key(val_metrics)
        if best_key is None or key > best_key:
            best_key, best_epoch, stale = key, epoch, 0
            torch.save({
                "model": model.state_dict(), "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "model_config": model_config,
                "loss_weights": weights, "selection_key": key,
                "best_key": key, "best_epoch": best_epoch, "stale": stale,
                "train_metrics": train_metrics, "val_metrics": val_metrics,
                "config": vars(args),
                "format": ("frame_relation_refiner_v4" if args.init_checkpoint
                           else "frame_relation_refiner_v2"),
            }, output)
        else:
            stale += 1
        record = {"epoch": epoch, "best_epoch": best_epoch, "stale": stale,
                  **{f"train_{k}": v for k, v in train_metrics.items()},
                  **{f"val_{k}": v for k, v in val_metrics.items()}}
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if args.last_output:
            last_output = Path(args.last_output)
            last_output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "best_key": best_key,
                "best_epoch": best_epoch, "stale": stale,
                "model_config": model_config,
                "loss_weights": weights, "selection_key": key,
                "train_metrics": train_metrics, "val_metrics": val_metrics,
                "config": vars(args), "format": "frame_relation_refiner_v4",
            }, last_output)
        if args.save_epoch_every > 0 and epoch % args.save_epoch_every == 0:
            epoch_output = output.with_name(
                f"{output.stem}.epoch{epoch}{output.suffix}"
            )
            torch.save({
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "best_key": best_key,
                "best_epoch": best_epoch, "stale": stale,
                "model_config": model_config,
                "loss_weights": weights, "selection_key": key,
                "train_metrics": train_metrics, "val_metrics": val_metrics,
                "config": vars(args), "format": "frame_relation_refiner_v4",
            }, epoch_output)
        print(json.dumps(record, sort_keys=True), flush=True)
        if epoch >= args.min_epochs and stale >= args.patience:
            break


if __name__ == "__main__":
    main()
