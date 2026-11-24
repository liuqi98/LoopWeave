"""Sequence diagnostics for the sequence-prior pilot (validation set only, no training).

Step 1 of the agreed protocol. Four predictors are compared under one convention: every
H3 residue position, greedy decoding, positions pooled across all 340 validation
complexes per seed and then averaged over seeds.

    S^0               stage-1 masked readout (trained head, geometry frozen)
    baseline          refiner with zero input prior and zero residual
    probability_init  refiner with B on the input path only
    full              refiner with B on the input path and in the residual decode

Readouts
--------
* AAR and sequence NLL — separately answers "is S^0 more accurate" and "are S^0's
  probabilities reasonable";
* correctness contingency against S^0 — S^0 right -> final wrong (did the refiner destroy
  a correct prior?) and S^0 wrong -> final right (can it fix a wrong prior?);
* error overlap — are the two predictors' mistakes complementary?
* de-meaned logit standard deviation and confidence at error positions — is the prior too
  sharp, or clearly larger in scale than the residual?

``--mode sweep`` (step 2) reuses the fixed probability_init checkpoints and only changes the
prior weight / temperature at inference on the validation set. That is a sensitivity
diagnostic, not a retrained model.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sequence_prior_refiner import SequencePriorRefiner
from train.compare_two_stage_binding import FeatureRows
from train.train_abgaze_reconstructed import dump, move

WORKSPACE = Path('/media/xd/code1/dyGF')
CONDITIONS = ('baseline', 'probability_init', 'full')
SEEDS = (2026, 2027, 2028)
REPORT_KEYS = ('aar', 'nll', 'logit_std_mean', 'logit_std_p90', 'confidence_mean',
               'confidence_at_errors', 'nll_at_errors')


def load_cache(seed):
    """Validation rows (sid, x, y, graph) plus the features and the targets."""
    parent = WORKSPACE / f'outputs/full_pipeline_seed{seed}_v1'
    plan = json.loads((parent / 'plan.json').read_text())
    binding_plan = json.loads((WORKSPACE / 'outputs/two_stage_binding60_gpu_v1/plan.json')
                              .read_text())
    features = WORKSPACE / 'data/abgaze_binding_features_finalized_v1'
    raw = torch.load(parent / 'valid_cache.pt', map_location='cpu', weights_only=False)
    if [r[0] for r in raw] != plan['ids']['valid']:
        raise ValueError('validation cache order does not match the parent plan')
    rows = FeatureRows(raw, features, 'valid', binding_plan['feature_sha256'])
    targets = {sid: y['sequence'] for sid, _, y, _ in raw}
    return rows, targets


def load_prior(path):
    document = torch.load(path, map_location='cpu', weights_only=False)
    return {sid: entry['logits'] for sid, entry in document['logits'].items()
            if entry['split'] == 'valid'}


def position_stats(logits, target):
    """Accuracy, calibration and scale for one predictor, over pooled positions."""
    logits = logits.float()
    correct = logits.argmax(-1) == target
    centred = logits - logits.mean(-1, keepdim=True)
    probability = logits.softmax(-1)
    confidence = probability.max(-1).values
    losses = F.cross_entropy(logits, target, reduction='none')
    return dict(
        aar=float(correct.float().mean()),
        nll=float(losses.mean()),
        logit_std_mean=float(centred.std(-1).mean()),
        logit_std_p90=float(centred.std(-1).quantile(.9)),
        confidence_mean=float(confidence.mean()),
        confidence_at_errors=(float(confidence[~correct].mean())
                              if bool((~correct).any()) else None),
        nll_at_errors=(float(losses[~correct].mean()) if bool((~correct).any()) else None),
        correct=correct)


def collect(model, rows, prior, condition, device, residual_scale=1.0):
    """Final logits for one condition, plus the prior that was wired in.

    ``residual_scale`` multiplies only the residual-side prior, which lets the
    multiplicative-constraint hypothesis be probed on the trained ``full`` checkpoint
    without changing the input path.
    """
    entries = {}
    with torch.no_grad():
        for sid, x, y, _ in rows:
            inputs = move(x, device)
            prior_tensor = prior[sid].to(device)
            if condition != 'baseline':
                inputs['base_logits'] = prior_tensor
                inputs['base_logits_output'] = (prior_tensor * residual_scale
                                                if condition == 'full'
                                                else torch.zeros_like(prior_tensor))
            out = model(inputs)
            entries[sid] = dict(final=out['sequence_logits'].detach().cpu(),
                                prior=prior_tensor.detach().cpu(),
                                target=y['sequence'].cpu())
    return entries


def summarise(entries, prior_logits):
    """Pool positions and build the contingency / overlap / scale block."""
    final = torch.cat([entry['final'] for entry in entries.values()])
    target = torch.cat([entry['target'] for entry in entries.values()])
    reference = torch.cat([prior_logits[sid] for sid in entries])

    prior = position_stats(reference, target)
    result = dict(n_positions=int(len(target)))
    for name, logits in (('prior', reference), ('final', final)):
        stats = position_stats(logits, target)
        result[name] = {key: value for key, value in stats.items() if key != 'correct'}
        if name == 'prior':
            prior_correct = stats['correct']
        else:
            final_correct = stats['correct']
    prior_error, final_error = ~prior_correct, ~final_correct
    shared = prior_error & final_error
    union = (prior_error | final_error).sum().clamp_min(1)
    result['contingency'] = dict(
        both_right=int((prior_correct & final_correct).sum()),
        prior_right_final_wrong=int((prior_correct & final_error).sum()),
        prior_wrong_final_right=int((prior_error & final_correct).sum()),
        both_wrong=int(shared.sum()),
        fraction_prior_right_lost=float((prior_correct & final_error).sum() /
                                        max(int(prior_correct.sum()), 1)),
        fraction_prior_wrong_fixed=float((prior_error & final_correct).sum() /
                                         max(int(prior_error.sum()), 1)),
        error_jaccard=float(shared.sum() / union),
        fraction_final_errors_shared=float(shared.sum() / max(int(final_error.sum()), 1)),
        fraction_prior_errors_shared=float(shared.sum() / max(int(prior_error.sum()), 1)))
    residual = final - reference
    residual_centred = residual - residual.mean(-1, keepdim=True)
    result['residual'] = dict(
        logit_std_mean=float(residual_centred.std(-1).mean()),
        logit_std_p90=float(residual_centred.std(-1).quantile(.9)),
        ratio_prior_to_residual=float(result['prior']['logit_std_mean'] /
                                      max(float(residual_centred.std(-1).mean()), 1e-8)))
    return result


def print_table(results, label):
    print(f'\n=== {label} (validation, mean over available seeds) ===')
    print('%-18s %8s %8s %9s %9s %9s' % ('predictor', 'AAR', 'NLL', 'logit_sd',
                                         'conf@err', 'nll@err'))
    for condition in ('S0',) + CONDITIONS:
        rows = [results[seed][condition] for seed in SEEDS
                if condition in results.get(seed, {})]
        if not rows:
            continue
        side = 'prior' if condition == 'S0' else 'final'
        values = {key: statistics.mean(row[side][key] for row in rows)
                  for key in ('aar', 'nll')}
        scale = statistics.mean(row[side]['logit_std_mean'] for row in rows)
        conf = statistics.mean(row[side]['confidence_at_errors'] for row in rows)
        nll_err = statistics.mean(row[side]['nll_at_errors'] for row in rows)
        print('%-18s %8.4f %8.4f %9.3f %9.4f %9.3f'
              % (condition, values['aar'], values['nll'], scale, conf, nll_err))
    print('%-18s %8s %8s %9s %9s %9s' % ('vs S0', 'lost', 'fixed', 'jaccard',
                                          'fin-share', 'resid_sd'))
    for condition in CONDITIONS:
        rows = [results[seed][condition] for seed in SEEDS
                if condition in results.get(seed, {})]
        if not rows:
            continue
        print('%-18s %8.4f %8.4f %9.4f %9.4f %9.3f' % (
            f'vs S0 [{condition[:9]}]',
            statistics.mean(row['contingency']['fraction_prior_right_lost'] for row in rows),
            statistics.mean(row['contingency']['fraction_prior_wrong_fixed'] for row in rows),
            statistics.mean(row['contingency']['error_jaccard'] for row in rows),
            statistics.mean(row['contingency']['fraction_final_errors_shared'] for row in rows),
            statistics.mean(row['residual']['logit_std_mean'] for row in rows)))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('diag', 'sweep', 'sweep_full'), default='diag')
    parser.add_argument('--prior-root', type=Path, required=True)
    parser.add_argument('--runs', type=Path,
                        default=WORKSPACE / 'outputs/sequence_prior_arms_v1')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    if args.mode == 'diag':
        out = args.output.resolve()
        out.mkdir(parents=True, exist_ok=False)
        results = {}
        for seed in SEEDS:
            rows, targets = load_cache(seed)
            prior = load_prior(args.prior_root / f'stage1_head_seed{seed}.pt')
            ids = [sid for sid, _, _, _ in torch.load(
                WORKSPACE / f'outputs/full_pipeline_seed{seed}_v1/valid_cache.pt',
                map_location='cpu', weights_only=False)]
            if set(ids) != set(prior):
                raise ValueError(f'seed {seed}: prior ids do not match the validation cache')
            results[seed] = {}
            s0_entries = {sid: dict(prior=prior[sid], final=prior[sid], target=targets[sid])
                          for sid in ids}
            results[seed]['S0'] = summarise(s0_entries, prior)
            for condition in CONDITIONS:
                run = args.runs / f'{condition}_seed{seed}'
                checkpoint = run / 'pair/best.pt'
                status_path = run / 'status.json'
                phase = (json.loads(status_path.read_text()).get('phase')
                         if status_path.is_file() else None)
                if not checkpoint.is_file() or phase != 'complete':
                    # A run still training only has its epoch-0 initialisation as best.pt,
                    # which would make the comparison meaningless.
                    print(json.dumps(dict(skip=f'{condition}_{seed}',
                                          reason='checkpoint not final', phase=phase)),
                          flush=True)
                    continue
                state = torch.load(checkpoint, map_location='cpu', weights_only=False)
                model = SequencePriorRefiner(**state['model_config']).to(device)
                model.load_state_dict(state['model'])
                model.eval()
                results[seed][condition] = summarise(
                    collect(model, rows, prior, condition, device), prior)
                print(json.dumps(dict(seed=seed, condition=condition,
                                      epoch=state['epoch'],
                                      aar=round(results[seed][condition]['final']['aar'], 4),
                                      nll=round(results[seed][condition]['final']['nll'], 4))),
                      flush=True)
                del model
                torch.cuda.empty_cache()
        dump(out / 'sequence_diagnostics.json', dict(
            cohort='validation split (340 complexes)', seeds=list(SEEDS),
            convention='every H3 position pooled, greedy decoding, equal position weights',
            readouts='AAR, NLL, contingency vs S0, error overlap, de-meaned logit std, '
                     'confidence at errors',
            results=results))
        print_table(results, 'sequence diagnostics')
        return

    if args.mode == 'sweep_full':
        # Hypothesis check for the multiplicative constraint: on the trained ``full``
        # checkpoint, scale only the residual-side prior and watch AAR / NLL.
        out = args.output.resolve()
        out.mkdir(parents=True, exist_ok=True)
        scales = (0.0, 0.25, 0.5, 1.0, 2.0)
        result = {}
        for seed in SEEDS:
            run = args.runs / f'full_seed{seed}'
            status_path = run / 'status.json'
            phase = (json.loads(status_path.read_text()).get('phase')
                     if status_path.is_file() else None)
            if not (run / 'pair/best.pt').is_file() or phase != 'complete':
                print(json.dumps(dict(skip=f'full_{seed}', phase=phase)), flush=True)
                continue
            rows, _ = load_cache(seed)
            prior = load_prior(args.prior_root / f'stage1_head_seed{seed}.pt')
            state = torch.load(run / 'pair/best.pt', map_location='cpu', weights_only=False)
            model = SequencePriorRefiner(**state['model_config']).to(device)
            model.load_state_dict(state['model'])
            model.eval()
            result[seed] = {}
            for scale in scales:
                values = summarise(collect(model, rows, prior, 'full', device, scale), prior)
                result[seed][f'residual_scale{scale}'] = dict(
                    aar=values['final']['aar'], nll=values['final']['nll'],
                    logit_std=values['final']['logit_std_mean'],
                    residual_std=values['residual']['logit_std_mean'],
                    confidence_at_errors=values['final']['confidence_at_errors'])
                print(json.dumps(dict(seed=seed, residual_scale=scale,
                                      aar=round(values['final']['aar'], 5),
                                      nll=round(values['final']['nll'], 5))), flush=True)
            del model
            torch.cuda.empty_cache()
        dump(out / 'full_residual_sweep.json', dict(
            note='inference-time diagnostic on the trained full checkpoints: only the '
                 'residual-side prior is scaled, the input prior stays at its trained scale',
            scales=list(scales), results=result))
        return

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    alphas = (0.0, 0.25, 0.5, 1.0, 2.0)
    temperatures = (0.5, 1.0, 2.0)
    sweep = {}
    for seed in SEEDS:
        run = args.runs / f'probability_init_seed{seed}'
        checkpoint = run / 'pair/best.pt'
        if not checkpoint.is_file():
            continue
        rows, _ = load_cache(seed)
        prior = load_prior(args.prior_root / f'stage1_head_seed{seed}.pt')
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model = SequencePriorRefiner(**state['model_config']).to(device)
        model.load_state_dict(state['model'])
        model.eval()
        sweep[seed] = {}
        for alpha in alphas:
            for temperature in temperatures:
                scaled = {sid: prior[sid] * (alpha / temperature) for sid in prior}
                values = summarise(collect(model, rows, scaled, 'probability_init', device),
                                   scaled)
                sweep[seed][f'alpha{alpha}_T{temperature}'] = dict(
                    aar=values['final']['aar'], nll=values['final']['nll'],
                    prior_aar=values['prior']['aar'], prior_nll=values['prior']['nll'])
                print(json.dumps(dict(seed=seed, alpha=alpha, temperature=temperature,
                                      aar=round(values['final']['aar'], 5),
                                      nll=round(values['final']['nll'], 5))), flush=True)
        del model
        torch.cuda.empty_cache()
    dump(out / 'prior_weight_sweep.json', dict(
        note='inference-time sensitivity diagnostic on the fixed probability_init '
             'checkpoints; these are not retrained models',
        cohort='validation split', alphas=list(alphas), temperatures=list(temperatures),
        results=sweep))


if __name__ == '__main__':
    main()
