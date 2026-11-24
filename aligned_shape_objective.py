"""Rigid-body-aligned backbone loss, complementary to absolute-coordinate supervision.

The existing objective already contains

  * ``atomic``  — absolute per-atom distance to the native coordinates, which
                  penalises *where* the loop sits in the global frame;
  * ``shape``   — internal CA-CA distance-matrix error, which is invariant but
                  insensitive to handedness and never constrains Cartesian
                  position directly.

Neither term measures "is the loop's own conformation right, independent of where
it was placed". This module adds that third view: superimpose the predicted
backbone on the native backbone with a differentiable Kabsch rotation and compare
coordinates in the shared frame. It is therefore complementary to ``atomic`` by
construction — ``atomic`` pays for global placement error, the aligned term pays
only for residual conformational shape error after placement is factored out.

Nothing else about the objective changes; the wrapper calls the unchanged
``abgaze_reconstructed_objective.objective`` and adds ``weight * aligned``.

Supervision stays missing-atom aware: only atoms whose native label is observed
enter both the alignment and the error, so arbitrary placeholders never leak.
"""
import torch
from torch.nn import functional as F
from abgaze_reconstructed_objective import objective as base_objective

ATOM_SETS = {'backbone': (0, 1, 2, 3), 'ca': (1,)}


def kabsch_align(prediction, target):
    """Rotate/translate ``prediction`` (N,3) onto ``target`` (N,3), differentiably.

    Gradients flow through both the rotation and the coordinates, so the loss
    trains the predicted conformation rather than a detached alignment.
    """
    pc = prediction - prediction.mean(dim=0, keepdim=True)
    tc = target - target.mean(dim=0, keepdim=True)
    covariance = pc.transpose(0, 1) @ tc
    u, _, vt = torch.linalg.svd(covariance)
    v = vt.transpose(0, 1)
    # Reflection guard: force a proper rotation, never a mirror.
    sign = torch.sign(torch.linalg.det(v @ u.transpose(0, 1)))
    correction = torch.diag(torch.stack([torch.ones_like(sign), torch.ones_like(sign), sign]))
    rotation = v @ correction @ u.transpose(0, 1)
    return pc @ rotation.transpose(0, 1) + target.mean(dim=0, keepdim=True)


def aligned_errors(backbone, target, mask, atom_set='backbone'):
    """Per-atom aligned errors and the aligned CA RMSD.

    The refiner operates on one complex at a time, so ``backbone``/``target`` are
    ``(L, 4, 3)`` and ``mask`` is ``(L, 4)``. A leading batch dimension is also
    accepted so the same code can be reused by batched callers.
    """
    atoms = ATOM_SETS[atom_set]
    batched = backbone.dim() == 4
    if not batched:
        backbone, target, mask = backbone[None], target[None], mask[None]
    errors, ca_rmsd = [], []
    for index in range(len(backbone)):
        keep = mask[index][:, list(atoms)].flatten()
        if int(keep.sum()) < 3:
            continue
        pred = backbone[index][:, list(atoms)].reshape(-1, 3)[keep]
        truth = target[index][:, list(atoms)].reshape(-1, 3)[keep]
        if not (torch.isfinite(pred).all() and torch.isfinite(truth).all()):
            continue
        aligned = kabsch_align(pred, truth)
        errors.append((aligned - truth).norm(dim=-1))
        ca_keep = mask[index][:, 1]
        if int(ca_keep.sum()) >= 3:
            ca_pred = kabsch_align(backbone[index][ca_keep, 1], target[index][ca_keep, 1])
            ca_rmsd.append((ca_pred - target[index][ca_keep, 1]).square().sum(-1).mean().sqrt())
    return errors, ca_rmsd


def aligned_backbone_loss(backbone, target, mask, atom_set='backbone'):
    """smooth-L1 on aligned per-atom residuals, matching the ``atomic`` functional."""
    errors, ca_rmsd = aligned_errors(backbone, target, mask, atom_set)
    zero = backbone.sum() * 0
    if not errors:
        return zero, zero, zero
    stacked = torch.cat(errors)
    loss = F.smooth_l1_loss(stacked, torch.zeros_like(stacked))
    ca = torch.stack(ca_rmsd).mean() if ca_rmsd else zero
    return loss, stacked.mean(), ca


def objective(out, x, y, fape_weight=.5, sequence_weight=1., aligned_weight=0.,
              aligned_atom_set='backbone'):
    """Unchanged base objective plus ``aligned_weight`` times the aligned term."""
    loss, metrics = base_objective(out, x, y, fape_weight, sequence_weight)
    aligned_loss, aligned_mae, aligned_ca_rmsd = aligned_backbone_loss(
        out['backbone'], y['bb'], y['mask'], aligned_atom_set)
    loss = loss + aligned_weight * aligned_loss
    metrics = dict(metrics)
    metrics.update(aligned_loss=float(aligned_loss.detach()),
                   aligned_mae=float(aligned_mae.detach()) if torch.is_tensor(aligned_mae) else 0.,
                   aligned_ca_rmsd=float(aligned_ca_rmsd.detach())
                   if torch.is_tensor(aligned_ca_rmsd) else 0.,
                   loss=float(loss.detach()))
    return loss, metrics
