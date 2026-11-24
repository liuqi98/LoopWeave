"""Offline diagnostics for already saved predictions and measurements."""
import itertools
import numpy as np
from .common import finite, metric_spec
from .statistics import interval


def export_distances(targets, manifest, directory):
    """Convert existing backbone JSON and evaluation tensors to distance records.

    No model is imported or executed. IDs are chosen explicitly by the locked
    evaluation manifest, never by score, and prediction coverage must be exact.
    """
    from pathlib import Path
    from .common import read_json, digest
    expected = manifest['ids']
    if not expected or len(expected) != len(set(expected)):
        raise ValueError('Need nonempty unique locked evaluation IDs')
    lookup = {}
    for sid, x, y in targets:
        if sid in lookup:
            raise ValueError(f'Duplicate target: {sid}')
        lookup[sid] = (x, y)
    if set(expected) - set(lookup):
        raise ValueError('Missing locked evaluation target IDs')
    result, seen = [], set()
    for run in manifest['runs']:
        path = Path(run['predictions_path'])
        if not path.is_absolute():
            path = Path(directory) / path
        if run.get('sha256') != digest(path):
            raise ValueError(f'Missing/mismatched prediction hash: {path}')
        records = read_json(path)
        if len(records) != len(expected) or {r['sample_id'] for r in records} != set(expected):
            raise ValueError(f'Prediction coverage differs from locked cohort: {path}')
        predicted = {r['sample_id']: r for r in records}
        for sid in expected:
            key = (run['arm'], int(run['seed']), sid, int(run.get('stage_index', 0)))
            if key in seen:
                raise ValueError(f'Duplicate run/stage/sample: {key}')
            seen.add(key)
            x, y = lookup[sid]
            native, bb = np.asarray(y['bb']), np.asarray(predicted[sid]['backbone'])
            observed = np.asarray(y['mask'], dtype=bool)[:, 1]
            context = np.asarray(x['context_bb'])[:, 1]
            valid_context = np.asarray(x['antigen_mask'], dtype=bool) & np.asarray(x['context_mask'], dtype=bool)[:, 1]
            if bb.shape != native.shape or native.ndim != 3 or native.shape[1:] != (4, 3):
                raise ValueError(f'Unexpected backbone shape: {sid}')
            if not np.isfinite(bb).all():
                raise ValueError(f'Nonfinite prediction: {sid}')
            ref_ca, pred_ca, ag_ca = native[observed, 1], bb[observed, 1], context[valid_context]
            if not len(ref_ca) or not len(ag_ca) or not np.isfinite(ref_ca).all() or not np.isfinite(ag_ca).all():
                raise ValueError(f'Missing/nonfinite observed CA support: {sid}')
            result.append(dict(arm=key[0], seed=key[1], sample_id=sid, stage_index=key[3],
                stage_name=run.get('stage_name'),
                reference_distances=np.linalg.norm(ref_ca[:, None] - ag_ca[None], axis=-1).tolist(),
                predicted_distances=np.linalg.norm(pred_ca[:, None] - ag_ca[None], axis=-1).tolist()))
    if not result:
        raise ValueError('No prediction runs')
    return result


def distance_arrays(record):
    truth = np.asarray(record['reference_distances'], dtype=float)
    pred = np.asarray(record['predicted_distances'], dtype=float)
    mask = np.asarray(record.get('pair_mask', np.ones(truth.shape, dtype=bool)))
    if truth.ndim != 2 or pred.shape != truth.shape or mask.shape != truth.shape or mask.dtype != bool:
        raise ValueError('Expected equal 2D distance matrices and a boolean pair mask')
    if not np.isfinite(truth[mask]).all() or not np.isfinite(pred[mask]).all():
        raise ValueError('Nonfinite observed distance')
    if (truth[mask] < 0).any() or (pred[mask] < 0).any():
        raise ValueError('Negative distance')
    return truth, pred, mask


def contact_counts(truth, pred, mask, threshold):
    ref, gen = (truth <= threshold) & mask, (pred <= threshold) & mask
    tp, fp, fn = int((ref & gen).sum()), int((~ref & gen).sum()), int((ref & ~gen).sum())
    return dict(tp=tp, fp=fp, fn=fn, n_native=int(ref.sum()), n_predicted=int(gen.sum()),
                f1=2 * tp / max(2 * tp + fp + fn, 1),
                precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                both_empty=not bool((ref | gen).any()), n_valid_pairs=int(mask.sum()))


def contact_changes(truth, before, after, mask, threshold):
    ref = (truth <= threshold) & mask
    old, new = (before <= threshold) & mask, (after <= threshold) & mask
    return dict(gained_correct=int((~old & new & ref).sum()),
                lost_correct=int((old & ~new & ref).sum()),
                gained_false=int((~old & new & ~ref).sum()),
                removed_false=int((old & ~new & ~ref).sum()))


def contacts(records, thresholds=(6., 7., 8., 9., 10.)):
    if not records or not thresholds or len(set(thresholds)) != len(thresholds):
        raise ValueError('Nonempty records and unique thresholds required')
    thresholds = [finite(t, 'threshold') for t in thresholds]
    if min(thresholds) <= 0:
        raise ValueError('Thresholds must be positive')
    histories = {}
    for r in records:
        key = (str(r['arm']), int(r['seed']), str(r['sample_id']))
        stage = int(r.get('stage_index', 0))
        if stage < 0:
            raise ValueError('Stage indices must start at zero')
        histories.setdefault(key, {})
        if stage in histories[key]:
            raise ValueError(f'Duplicate stage: {key}/{stage}')
        histories[key][stage] = r
    result = []
    for key, stages in sorted(histories.items()):
        order = sorted(stages)
        if order != list(range(len(order))):
            raise ValueError(f'Missing stage in {key}; do not label a skipped transition as adjacent')
        previous = None
        for stage in order:
            r = stages[stage]
            truth, pred, mask = distance_arrays(r)
            if previous is not None:
                old_truth, old_pred, old_mask = previous
                if not np.array_equal(mask, old_mask) or not np.array_equal(truth[mask], old_truth[old_mask]):
                    raise ValueError(f'Reference correspondence changed between stages: {key}')
            absolute = np.abs(pred[mask] - truth[mask])
            near = mask & (truth <= 12.)
            summary = dict(arm=key[0], seed=key[1], sample_id=key[2], stage_index=stage,
                           stage_name=r.get('stage_name'),
                           distance_mae=float(absolute.mean()) if absolute.size else None,
                           distance_p95=float(np.quantile(absolute, .95)) if absolute.size else None,
                           native_near12_distance_mae=float(np.abs(pred[near]-truth[near]).mean()) if near.any() else None,
                           n_valid_pairs=int(mask.sum()), thresholds={})
            for threshold in thresholds:
                entry = contact_counts(truth, pred, mask, threshold)
                entry['changes_from_previous'] = (contact_changes(truth, old_pred, pred, mask, threshold)
                                                  if previous is not None else None)
                summary['thresholds'][str(threshold)] = entry
            result.append(summary)
            previous = truth, pred, mask
    return dict(per_stage=result,
                metric_definition='CA distance pairs in Angstrom; <= threshold. Same masks and pair correspondence at every stage.',
                empty_contact_policy='F1=0 when both sets empty, matching archived scoring; both_empty is reported explicitly.',
                interpretation='Descriptive threshold and within-forward diagnostics, not an ablation or binding assay.',
                coordinate_requirement='Stage contact turnover requires saved stage coordinates/distances; scalar stage F1 cannot reconstruct it.')


def baseline_comparison(rows, specs, reference, comparator, replicates=10000):
    """Different methods need not have the same number of training seeds.

    Average each method's available seeds within each complex, then pair
    complexes. The interval is conditional on these checkpoints, not a paired
    seed interval or an estimate of single-seed baseline training variability.
    """
    from .statistics import rectangular
    if reference == comparator:
        raise ValueError('Reference and comparator must differ')
    names = [metric_spec(s) for s in specs]
    views = {}
    for arm in (reference, comparator):
        selected = [r for r in rows if r['arm'] == arm]
        views[arm] = rectangular(selected, (arm,), [n for n, _ in names])
    if views[reference][2:4] != views[comparator][2:4]:
        raise ValueError('Models must have identical sample IDs and dependence groups')
    groups = views[reference][3]
    report = {}
    for metric, direction in names:
        a = views[reference][0][metric][:, :, 0]
        b = views[comparator][0][metric][:, :, 0]
        summary = interval((direction * (a.mean(0) - b.mean(0)))[None], groups, replicates)
        summary['method_raw_means'] = {reference: float(a.mean()), comparator: float(b.mean())}
        summary['method_seed_means'] = {reference: a.mean(1).tolist(), comparator: b.mean(1).tolist()}
        report[metric] = summary
    return dict(reference=reference, comparator=comparator,
                seeds={a: views[a][1] for a in views}, metrics=report,
                inference='Complex-paired group intervals conditional on all supplied checkpoints; training seeds are not paired across methods.',
                limitation='One-seed MEAN or DiffAb does not establish training-seed stability.',
                positive_means=f'{reference} better than {comparator}')


def drift(records, metrics):
    """Numerical repeats require identical checkpoint, input, RNG and execution metadata."""
    required = ('checkpoint_sha256', 'input_sha256', 'inference_seed', 'device', 'precision', 'runtime_id')
    cases = {}
    for row in records:
        key = (str(row['arm']), str(row['sample_id']))
        if any(k not in row or row[k] in ('', None) for k in required):
            raise ValueError(f'Missing repeat provenance: {key}')
        cases.setdefault(key, []).append(row)
    if not cases:
        raise ValueError('No repeat observations')
    output = []
    for key, repeats in sorted(cases.items()):
        if len(repeats) < 2 or len({str(r['repeat_id']) for r in repeats}) != len(repeats):
            raise ValueError(f'Need at least two distinct repeats: {key}')
        if any(any(r[k] != repeats[0][k] for k in required) for r in repeats):
            raise ValueError(f'Checkpoint/input/RNG/runtime differs; not numerical-repeat drift: {key}')
        entry = dict(arm=key[0], sample_id=key[1], repeats=len(repeats),
                     provenance={k: repeats[0][k] for k in required}, metrics={})
        for metric in metrics:
            values = np.array([finite(r[metric], metric) for r in repeats])
            entry['metrics'][metric] = dict(min=float(values.min()), max=float(values.max()),
                spread=float(np.ptp(values)), repeat_sd=float(values.std(ddof=1)),
                max_abs_difference_from_first=float(np.abs(values-values[0]).max()))
        for field in ('backbone', 'sequence_logits'):
            if any(field in r for r in repeats):
                if not all(field in r for r in repeats):
                    raise ValueError(f'Partial {field} repeat coverage')
                arrays = [np.asarray(r[field], dtype=float) for r in repeats]
                if any(a.shape != arrays[0].shape or not np.isfinite(a).all() or not a.size for a in arrays):
                    raise ValueError(f'Invalid {field} arrays')
                spread = [np.abs(a-b) for a, b in itertools.combinations(arrays, 2)]
                entry[field] = dict(max_abs_pairwise=float(max(d.max() for d in spread)),
                                    max_pairwise_rmse=float(max(np.sqrt((d*d).mean()) for d in spread)))
        output.append(entry)
    return dict(per_complex=output,
                interpretation='Repeat variability under identical declared execution conditions; distinct from training-seed SD or cross-device differences.')


def cost_summary(rows):
    """Preserve missing measurements; checkpoint file size is not peak memory."""
    fields = ('parameter_count', 'trainable_parameter_count', 'training_seconds',
              'inference_seconds', 'peak_memory_bytes', 'n_inference_examples')
    result, seen = [], set()
    for row in rows:
        key = (row['run'], row['seed'])
        if key in seen:
            raise ValueError(f'Duplicate cost record: {key}')
        seen.add(key)
        record = {k: row.get(k) for k in ('run', 'seed', 'device', 'precision', 'timing_scope')}
        if not all(record[k] for k in ('device', 'precision', 'timing_scope')):
            raise ValueError('Cost records require device, precision and timing_scope')
        for field in fields:
            value = row.get(field)
            record[field] = None if value in ('', None) else finite(value, field)
            if record[field] is not None and record[field] < 0:
                raise ValueError(f'Negative cost: {field}')
        count = record['n_inference_examples']
        record['inference_seconds_per_example'] = (record['inference_seconds'] / count
            if count and record['inference_seconds'] is not None else None)
        result.append(record)
    return dict(runs=result, limitations=['Do not compare timings across different device/precision/scope as matched compute.',
                                         'Missing measurements remain null; training logs cannot recover peak GPU memory.'])
