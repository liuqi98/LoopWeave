"""Stage 1: bridge-attention pair refiner (zero-init = exact baseline at start)."""
import argparse, json, sys
from pathlib import Path
import torch

WS = Path('/media/xd/code1/dyGF')
SNAP = WS / 'deliverables/dyGF_scoped_research_snapshot_20261015_v1'
sys.path.insert(0, str(SNAP / 'code'))
sys.path.insert(0, str(WS))
import train.train_reconstructed_two_stage as trs
from train.compare_two_stage_binding import FeatureRows
from two_scale_pair_refiner import TwoScalePairRefiner
from adaptive_pair_refiner import build_bridge_model
from adaptive_pair_refiner_v2 import build_v2_model

FEATURES = WS / 'data/abgaze_binding_features_finalized_v1'
RC = dict(width=64, layers=4, use_orientation=True)


def load_rows(split, limit=0):
    rows = torch.load(WS / f'outputs/full_pipeline_seed2026_v1/{split}_cache.pt',
                      map_location='cpu', weights_only=False)
    plan = json.loads((WS / 'outputs/two_stage_binding60_gpu_v1/plan.json').read_text())
    if limit:
        rows = rows[:limit]
    return FeatureRows(rows, FEATURES, split, plan['feature_sha256'])


def make_model(device, binding_state, variant='base'):
    model = TwoScalePairRefiner(**RC, coordinated_bounds=[0.8] * 4, local_bounds=[0.2] * 4)
    if variant == 'bridge':
        build_bridge_model(model, RC['width'], 0.8)
    elif variant == 'gate':
        build_v2_model(model, RC['width'], 0.8, geo_feedback=False)
    elif variant == 'full':
        build_v2_model(model, RC['width'], 0.8, geo_feedback=True)
    model = model.to(device)
    model.load_binding(binding_state['model'])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--device', default='cuda:1')
    ap.add_argument('--output', type=Path, default=WS / 'outputs/pipeline_pair_bridge_a_v1')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--variant', choices=['bridge', 'gate', 'full'], default='bridge')
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(1)
    data = {'train': load_rows('train', 16 if args.smoke else 0),
            'valid': load_rows('valid', 8 if args.smoke else 0)}
    print('rows', {k: len(v) for k, v in data.items()}, flush=True)
    binding = torch.load(WS / 'outputs/full_pipeline_seed2026_v1/binding/best.pt',
                         map_location='cpu', weights_only=False)
    torch.manual_seed(2026)
    base = make_model(device, binding, variant='base')
    torch.manual_seed(2026)
    variant_model = make_model(device, binding, variant=args.variant)
    variant_model.load_state_dict(base.state_dict(), strict=False)
    # stage-0 sanity check: identical outputs at initialisation
    sid, x0, y0, g0 = data['valid'][0]
    with torch.no_grad():
        xb = trs.move(x0, device)
        out_base = base(dict(xb))['backbone']
        out_bridge = variant_model(dict(xb))['backbone']
    diff = float((out_base - out_bridge).abs().max())
    print(f'sanity: max|Δbackbone| = {diff:.3e}', flush=True)
    if diff > 1e-4:
        raise SystemExit('bridge init is not equivalent to baseline')
    if args.smoke:
        print('smoke ok', flush=True)
        return
    pass
    state = trs.fit(variant_model, RC, data, args.output, 'pair', args.epochs, device, args.variant, 2026)
    print('best epoch', state['best_epoch'], state['validation'], flush=True)


if __name__ == '__main__':
    main()
