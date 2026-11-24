"""Matched-Protocol ablation training entry (v3).

Extends ``train_matched_ablation_v2.py`` without modifying it. Every variant
starts from the parent replica's own frozen generator cache and features, keeps
the parent's split, optimizer, loss, guards and validation selection, and
differs only in the branch/visible input that is removed.

Variants
--------
no_pair, no_binding, no_sequence
    Unchanged v2 semantics, reproduced here for one entry point.
no_pair_no_binding
    The missing 2x2 "base" cell: binding branch bypassed for the 60-epoch
    binding stage, both branches bypassed for the 30-epoch pair stage. This is
    the double removal required for the factorial interaction estimate.
features_surface_geometry, features_electrostatics_chemistry, features_none
    Capacity-preserving input masking. The binding layer keeps every parameter
    and all derived geometric channels; only the selected raw scalar/vector
    columns are zeroed, for both training stages. Masks are applied before any
    derived feature is computed, so excluded information cannot leak.
static_neighborhood
    Interface members frozen from the initial loop conformation while feature
    values are still recomputed from the current conformation. Same features,
    same network, same update count as the dynamic default.

Budget and source follow the parent full pipeline exactly (100 generator +
30 refiner + 60 binding + 30 pair); the generator/refiner stages are inherited
as frozen caches, so each variant trains only the stages listed in
``plan['epochs']``.

This entry never reads the evaluation test tensors. Scoring is a separate,
locked step.
"""
import argparse
import json
import random
import sys
from pathlib import Path
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train.train_full_pipeline_replica_v1 import train_later
from train.train_abgaze_reconstructed import sha, dump, move
from train.compare_two_stage_binding import FeatureRows
from binding_interface_refiner import BindingInterfaceRefiner
from loop_pair_ablation_refiner import AblatedLoopPairRefiner
from two_scale_pair_refiner import TwoScalePairRefiner
from interface_ablation_refiner import (AblatedBindingInterfaceRefiner,
                                        AblatedPersistentLoopPairRefiner,
                                        FEATURE_GROUPS, NEIGHBORHOODS)

DEFAULT_WORKSPACE = '/media/xd/code1/dyGF'
FEATURE_VARIANTS = {'features_surface_geometry': 'surface_geometry',
                    'features_electrostatics_chemistry': 'electrostatics_chemistry',
                    'features_none': 'none'}

SPEC = {
    'no_pair':            dict(disable=('pair',), source='binding/last.pt'),
    'no_binding':         dict(disable=('binding',), source='refiner/best.pt'),
    'no_pair_no_binding': dict(disable=('binding', 'pair'), source='refiner/best.pt'),
    'no_sequence':        dict(disable=(), source='binding/last.pt', loop='no_sequence'),
    'static_neighborhood': dict(disable=(), source='refiner/best.pt', neighborhood='static'),
}
for name, groups in FEATURE_VARIANTS.items():
    SPEC[name] = dict(disable=(), source='refiner/best.pt', feature_groups=groups)
VARIANT_ORDER = ('no_binding', 'no_pair', 'no_pair_no_binding', 'no_sequence',
                 'features_surface_geometry', 'features_electrostatics_chemistry',
                 'features_none', 'static_neighborhood')


class BypassBinding(nn.Module):
    def forward(self, h, bb, x):
        return h, bb


class BypassPair(nn.Module):
    def forward(self, h, bb, z, probabilities):
        return h, bb, z


def configure(model, disable):
    """Replace disabled branches after construction so retained init is untouched."""
    if 'binding' in disable:
        model.binding_layers = nn.ModuleList([BypassBinding() for _ in model.binding_layers])
    if 'pair' in disable and hasattr(model, 'loop_pair_layers'):
        model.loop_pair_layers = nn.ModuleList([BypassPair() for _ in model.loop_pair_layers])
    return model


def plan_epochs(variant, smoke):
    """Budget matched to the parent full pipeline (60 binding + 30 pair)."""
    disable = set(SPEC[variant]['disable'])
    trains_binding = ('binding' in disable or variant in FEATURE_VARIANTS
                      or variant == 'static_neighborhood')
    return dict(binding=(1 if smoke else 60) if trains_binding else 0,
                pair=1 if smoke else 30)


def build_pair_model(variant, config, device):
    """Pair stage on the two-scale layer; branch ablation is applied by module substitution."""
    spec = SPEC[variant]
    if variant == 'no_sequence':
        model = AblatedLoopPairRefiner(**config, variant='no_sequence').to(device)
        return model
    model = TwoScalePairRefiner(**config, coordinated_bounds=[0.80] * 4,
                                local_bounds=[0.20] * 4).to(device)
    return configure(model, spec['disable'])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workspace', type=Path, default=Path(DEFAULT_WORKSPACE),
                        help='Read-only parent-run workspace containing outputs/ and data/')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--variant', choices=VARIANT_ORDER, required=True)
    parser.add_argument('--output', type=Path, required=True,
                        help='New run directory; refused if it already exists')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke', action='store_true', help='Two samples, one epoch per stage')
    args = parser.parse_args()
    torch.set_num_threads(1)
    workspace = args.workspace.resolve()
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    spec = SPEC[args.variant]
    disable = set(spec['disable'])
    parent = workspace / f'outputs/full_pipeline_seed{args.seed}_v1'
    pp = json.loads((parent / 'plan.json').read_text())
    assert json.loads((parent / 'status.json').read_text())['phase'] == 'complete'
    for name, digest in pp['code_sha256'].items():
        path = workspace / name
        assert path.is_file() and sha(path) == digest, f'parent code mismatch: {name}'
    features = workspace / 'data/abgaze_binding_features_finalized_v1'
    binding_plan_path = workspace / 'outputs/two_stage_binding60_gpu_v1/plan.json'
    binding_plan = json.loads(binding_plan_path.read_text())
    assert sha(features / 'plan.json') == pp['feature_plan_sha256']
    feature_hashes = binding_plan['feature_sha256']

    rows, hashes = {}, {}
    for split in ('train', 'valid'):
        path = parent / f'{split}_cache.pt'
        hashes[split] = sha(path)
        raw = torch.load(path, map_location='cpu', weights_only=False)
        assert [r[0] for r in raw] == pp['ids'][split]
        if args.smoke:
            raw = raw[:2]
        rows[split] = FeatureRows(raw, features, split, feature_hashes)
    source_path = parent / spec['source']
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    assert source['plan_sha256'] == sha(parent / 'plan.json')
    if spec['source'] == 'binding/last.pt':
        assert source['epoch'] == 60
    config = source['model_config']
    epochs = plan_epochs(args.variant, args.smoke)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    this_file = Path(__file__).resolve()
    ablation_file = ROOT / 'interface_ablation_refiner.py'
    plan = dict(seed=args.seed, variant=args.variant, smoke=args.smoke,
                parent=str(parent), parent_plan_sha256=sha(parent / 'plan.json'),
                source=str(source_path), source_sha256=sha(source_path),
                source_epoch=source['epoch'], cache_sha256=hashes,
                generator_sha256=sha(parent / 'generator/best.pt'), model_config=config,
                feature_plan_sha256=sha(features / 'plan.json'), epochs=epochs,
                disabled=dict(binding='binding' in disable, pair='pair' in disable),
                feature_groups=spec.get('feature_groups', 'joint'),
                neighborhood=spec.get('neighborhood', 'dynamic'),
                feature_group_definition=FEATURE_GROUPS[
                    spec.get('feature_groups', 'joint')] if spec.get('feature_groups', 'joint')
                in FEATURE_GROUPS else None,
                selection='Unchanged validation aligned RMSD with guards; epoch0 eligible; no early stopping',
                objective='Unchanged geometry + 0.5 interface FAPE + CE',
                test_access=False,
                code_sha256={name: sha(workspace / name) for name in sorted(pp['code_sha256'])},
                ablation_code={str(this_file): sha(this_file), str(ablation_file): sha(ablation_file)},
                caveats=['Matched-epoch comparisons, not matched FLOPs or parameter counts.',
                         'Input masking tests dependence on a feature block, not its learnable value; '
                         'a masked arm shares the architecture and still spends the full budget.',
                         'Neighbor positions/distances/rbf channels are the shared geometric scaffold '
                         'and are not removed by feature_groups.',
                         'A static-neighborhood arm freezes membership from the initial loop but keeps '
                         'recomputing feature values, isolating member re-selection.',
                         'Runs are sequential and not bitwise reproducible across CUDA versions.'])
    dump(out / 'plan.json', plan)
    plan_sha = sha(out / 'plan.json')

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    state = source
    if epochs['binding'] > 0:
        model = AblatedBindingInterfaceRefiner(
            **config, feature_groups=spec.get('feature_groups', 'joint'),
            neighborhood=spec.get('neighborhood', 'dynamic')).to(device)
        model.load_source(source['model'])
        if 'binding' in disable:
            configure(model, disable)
        state = train_later(model, config, rows, out, 'binding', epochs['binding'],
                            args.seed, device, plan_sha)
        del model

    torch.manual_seed(args.seed)
    model = build_pair_model(args.variant, config, device)
    if 'binding' in disable:
        configure(model, {'binding'})
    model.load_binding(state['model'])
    if 'pair' in disable:
        configure(model, {'pair'})

    # Zero-initialized readouts and removed branches must preserve the source
    # function up to floating-point reduction noise.
    audit = AblatedBindingInterfaceRefiner(
        **config, feature_groups=spec.get('feature_groups', 'joint'),
        neighborhood=spec.get('neighborhood', 'dynamic')).to(device)
    if 'binding' in disable:
        configure(audit, {'binding'})
    audit.load_state_dict(state['model'])
    audit.eval()
    model.eval()
    _, x, _, _ = rows['valid'][0]
    x = move(x, device)
    with torch.no_grad():
        a, b = audit(x), model(x)
        delta = {k: float((a[k] - b[k]).abs().max()) for k in ('backbone', 'sequence_logits')}
    assert max(delta.values()) < 1e-4, delta
    del audit
    pred = model(x)
    (pred['backbone'].square().mean() + pred['sequence_logits'].square().mean()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    model.zero_grad(set_to_none=True)
    dump(out / 'initialization_audit.json', dict(source_output_difference=delta, finite_gradient=True,
                                                 audit_model='masked/disabled source reconstruction'))

    selected = train_later(model, config, rows, out, 'pair', epochs['pair'],
                           args.seed, device, plan_sha)
    dump(out / 'status.json', dict(phase='smoke_complete' if args.smoke else 'complete',
                                   variant=args.variant, selected_epoch=selected['epoch'],
                                   test_access=False))
    print(json.dumps(dict(variant=args.variant, seed=args.seed, output=str(out),
                          selected_epoch=selected['epoch'], epochs=epochs)))


if __name__ == '__main__':
    main()
