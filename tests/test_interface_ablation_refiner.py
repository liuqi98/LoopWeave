"""Synthetic tests for capacity-preserving interface ablations."""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from binding_interface_refiner import BindingInterfaceRefiner, SURFACE_SCALARS, ATOM_SCALARS
from interface_ablation_refiner import (ATOM_VECTORS, FEATURE_GROUPS, SURFACE_VECTORS,
                                        AblatedBindingInterfaceRefiner,
                                        AblatedPersistentLoopPairRefiner,
                                        binding_neighborhood, masked_raw, member_selection)


def synthetic(length=5, context=6, antigens=9, seed=0):
    torch.manual_seed(seed)
    x = dict(base_bb=torch.randn(length, 4, 3), base_logits=torch.zeros(length, 20),
             context_bb=torch.randn(context, 4, 3), context_mask=torch.ones(context, 4, dtype=torch.bool),
             context_features=torch.randn(context, 27), anchors=torch.randn(2, 4, 3),
             anchor_mask=torch.ones(2, 4, dtype=torch.bool),
             antigen_mask=torch.tensor([True] * (context // 2) + [False] * (context - context // 2)))
    x['binding_surface_xyz'] = torch.randn(antigens, 3)
    x['binding_surface_scalars'] = torch.randn(antigens, 11)
    x['binding_surface_vectors'] = torch.randn(antigens, 4, 3)
    x['binding_atom_xyz'] = torch.randn(antigens, 3)
    x['binding_atom_scalars'] = torch.randn(antigens, 10)
    x['binding_atom_vectors'] = torch.randn(antigens, 2, 3)
    return x


def test_feature_groups_partition_the_declared_names():
    assert len(SURFACE_SCALARS) == 11 and len(ATOM_SCALARS) == 10
    assert len(SURFACE_VECTORS) == 4 and len(ATOM_VECTORS) == 2
    for kind, scalars, vectors in (('surface', SURFACE_SCALARS, SURFACE_VECTORS),
                                   ('atom', ATOM_SCALARS, ATOM_VECTORS)):
        geometry = set(FEATURE_GROUPS['surface_geometry'][kind]['scalars']) | set(
            FEATURE_GROUPS['surface_geometry'][kind]['vectors'])
        chemistry = set(FEATURE_GROUPS['electrostatics_chemistry'][kind]['scalars']) | set(
            FEATURE_GROUPS['electrostatics_chemistry'][kind]['vectors'])
        assert geometry | chemistry == set(scalars) | set(vectors)
        assert not (geometry & chemistry)
        assert FEATURE_GROUPS['none'][kind]['scalars'] == ()


def test_masking_zeroes_only_the_excluded_columns():
    torch.manual_seed(1)
    scalars = torch.randn(4, 11) + 1.
    vectors = torch.randn(4, 4, 3) + 1.
    masked, masked_vectors = masked_raw('surface', scalars, vectors, 'surface_geometry')
    keep = [SURFACE_SCALARS.index(n) for n in FEATURE_GROUPS['surface_geometry']['surface']['scalars']]
    drop = [i for i in range(11) if i not in keep]
    assert torch.allclose(masked[:, keep], scalars[:, keep])
    assert torch.equal(masked[:, drop], torch.zeros(4, len(drop)))
    assert torch.equal(masked_vectors[:, 1:], torch.zeros(4, 3, 3))
    none_scalars, none_vectors = masked_raw('surface', scalars, vectors, 'none')
    assert torch.equal(none_scalars, torch.zeros_like(scalars))
    assert torch.equal(none_vectors, torch.zeros_like(vectors))
    with pytest.raises(ValueError):
        masked_raw('surface', scalars, vectors, 'not_a_group')


def test_defaults_reproduce_the_original_module():
    x = synthetic()
    state = BindingInterfaceRefiner(width=8, layers=2).state_dict()
    original = BindingInterfaceRefiner(width=8, layers=2)
    original.load_state_dict(state)
    ablated = AblatedBindingInterfaceRefiner(width=8, layers=2)
    assert not ablated.load_state_dict(state).unexpected_keys
    original.eval()
    ablated.eval()
    with torch.no_grad():
        a, b = original(dict(x)), ablated(dict(x))
    assert torch.equal(a['backbone'], b['backbone'])
    assert torch.equal(a['sequence_logits'], b['sequence_logits'])
    assert ablated.binding_layers[0].messages['surface'][0].in_features == 8 + 42


def test_masking_changes_output_but_not_capacity():
    x = synthetic()
    source = AblatedPersistentLoopPairRefiner(width=8, layers=2)
    # Zero-initialized readouts make the branch identity at step 0, so activate
    # them first; otherwise every arm would agree regardless of its inputs.
    with torch.no_grad():
        for layer in source.binding_layers:
            layer.hidden.weight.normal_(generator=torch.Generator().manual_seed(3))
            layer.translation.weight.normal_(generator=torch.Generator().manual_seed(4))
    state = source.state_dict()
    full = AblatedPersistentLoopPairRefiner(width=8, layers=2)
    full.load_state_dict(state)
    full.eval()
    with torch.no_grad():
        reference = full(dict(x))['backbone']
    for groups in ('surface_geometry', 'electrostatics_chemistry', 'none'):
        model = AblatedPersistentLoopPairRefiner(width=8, layers=2, feature_groups=groups)
        assert not model.load_state_dict(state).unexpected_keys
        model.eval()
        with torch.no_grad():
            out = model(dict(x))
        assert torch.isfinite(out['backbone']).all()
        assert not torch.equal(out['backbone'], reference)
        assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in full.parameters())


def test_static_membership_is_frozen_from_the_initial_conformation():
    x = synthetic()
    base = x['base_bb']
    perturbed = base + 3.
    indices, mask = member_selection(base, x, 'surface', 48)
    _, dynamic_mask = member_selection(perturbed, x, 'surface', 48)
    frozen_features, frozen_mask, _ = binding_neighborhood(perturbed, x, 'surface',
                                                           frozen=(indices, mask))
    dynamic_features, dynamic_mask, _ = binding_neighborhood(perturbed, x, 'surface')
    assert frozen_mask.shape == dynamic_mask.shape
    # Frozen members keep the base mask; values still come from the current conformation.
    assert torch.equal(frozen_mask, mask & torch.ones_like(mask))
    assert not torch.equal(frozen_features, dynamic_features)
    same = binding_neighborhood(base, x, 'surface', frozen=(indices, mask))[0]
    assert torch.allclose(same, binding_neighborhood(base, x, 'surface')[0], atol=1e-5)
    model = AblatedPersistentLoopPairRefiner(width=8, layers=2, neighborhood='static')
    model.load_state_dict(AblatedPersistentLoopPairRefiner(width=8, layers=2).state_dict())
    model.eval()
    with torch.no_grad():
        out = model(dict(x))
    assert torch.isfinite(out['backbone']).all()


def test_masked_gradients_are_finite_and_branch_is_trainable():
    x = synthetic()
    model = AblatedPersistentLoopPairRefiner(width=8, layers=2, feature_groups='electrostatics_chemistry')
    out = model(dict(x))
    (out['backbone'].square().mean() + out['sequence_logits'].square().mean()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_stage_trace_records_every_layer():
    x = synthetic()
    model = AblatedPersistentLoopPairRefiner(width=8, layers=3)
    model.eval()
    model.stage_trace = []
    with torch.no_grad():
        model(dict(x))
    assert len(model.stage_trace) == 3
    assert all(t.shape == (len(x['base_bb']), 4, 3) for t in model.stage_trace)
