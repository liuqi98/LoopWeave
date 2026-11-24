"""Paired factorial contrasts with dependence-group resampling."""
import numpy as np
from .common import finite, metric_spec


ARMS = ('base', 'only_pair', 'only_binding', 'full')
CONTRASTS = {
    'interaction': (1, -1, -1, 1),
    'pair_when_binding_absent': (-1, 1, 0, 0),
    'pair_when_binding_present': (0, 0, -1, 1),
    'binding_when_pair_absent': (-1, 0, 1, 0),
    'binding_when_pair_present': (0, -1, 0, 1),
    'pair_main_effect': (-.5, .5, -.5, .5),
    'binding_main_effect': (-.5, -.5, .5, .5),
    'full_minus_base': (-1, 0, 0, 1),
}


def rectangular(rows, arms, metrics, expected_seeds=None):
    """No silent row intersection, pooling of training seeds, or imputation."""
    if not rows or not arms or len(set(arms)) != len(arms) or not metrics:
        raise ValueError('Nonempty observations/metrics and distinct arms required')
    lookup, groups, lengths = {}, {}, {}
    for row in rows:
        sid, arm, seed = str(row['sample_id']), str(row['arm']), int(row['seed'])
        group = str(row['group'])
        if not sid or not group or arm not in arms:
            raise ValueError(f'Invalid sample/group/arm: {sid}/{group}/{arm}')
        key = (seed, sid, arm)
        if key in lookup:
            raise ValueError(f'Duplicate observation: {key}')
        if sid in groups and groups[sid] != group:
            raise ValueError(f'Inconsistent group: {sid}')
        groups[sid] = group
        if 'length' in row and row['length'] != '':
            length = int(row['length'])
            if length <= 0 or (sid in lengths and lengths[sid] != length):
                raise ValueError(f'Inconsistent/nonpositive length: {sid}')
            lengths[sid] = length
        lookup[key] = {m: finite(row[m], f'{key}/{m}') for m in metrics}
    seeds = sorted({k[0] for k in lookup})
    if expected_seeds is not None and seeds != sorted(expected_seeds):
        raise ValueError(f'Seed mismatch: observed {seeds}, expected {expected_seeds}')
    ids = sorted(groups)
    missing = [(s, sid, a) for s in seeds for sid in ids for a in arms
               if (s, sid, a) not in lookup]
    if missing:
        raise ValueError(f'Incomplete paired grid: {len(missing)} missing cells; first={missing[:5]}')
    values = {m: np.array([[[lookup[s, sid, a][m] for a in arms] for sid in ids]
                           for s in seeds], dtype=float) for m in metrics}
    return values, seeds, ids, [groups[s] for s in ids], lengths


def interval(delta, groups, replicates=10000, random_seed=2026):
    """delta[training_seed, sample]; positive consistently means improvement.

    Primary interval averages paired seeds before resampling entire groups.
    A crossed seed/group bootstrap is a separate small-seed sensitivity check.
    Both estimators weight samples equally, retaining unequal group sizes.
    """
    delta = np.asarray(delta, dtype=float)
    if delta.ndim != 2 or delta.shape[1] != len(groups) or not np.isfinite(delta).all():
        raise ValueError('Expected a finite seed by sample matrix')
    if delta.shape[0] == 0 or delta.shape[1] == 0 or replicates < 100:
        raise ValueError('Empty contrast or fewer than 100 bootstrap draws')
    unique = sorted(set(groups))
    blocks = [np.flatnonzero(np.asarray(groups) == g) for g in unique]
    sizes = np.array([len(b) for b in blocks])
    sums = np.stack([delta[:, b].sum(axis=1) for b in blocks], axis=1)
    result = dict(mean=float(delta.mean()), n_samples=delta.shape[1], n_groups=len(blocks),
                  per_seed_mean=delta.mean(axis=1).tolist(),
                  training_seed_sd=float(delta.mean(axis=1).std(ddof=1)) if len(delta) > 1 else None,
                  group_balanced_mean=float((sums.mean(axis=0) / sizes).mean()),
                  conditional_group_ci95=None, crossed_seed_group_ci95=None)
    if len(blocks) < 2:
        result['interval_note'] = 'Only one dependence group; sampling interval not estimable.'
        return result
    rng = np.random.default_rng(random_seed)
    draw = rng.integers(len(blocks), size=(replicates, len(blocks)))
    denom = sizes[draw].sum(axis=1)
    estimates = sums.mean(axis=0)[draw].sum(axis=1) / denom
    result['conditional_group_ci95'] = np.quantile(estimates, [.025, .975]).tolist()
    if len(delta) > 1:
        seed_draw = rng.integers(len(delta), size=(replicates, len(delta)))
        # Group and seed draws are shared across arms because contrasts were
        # formed before resampling. A seed is never counted as a new complex.
        estimates = sums[seed_draw[:, :, None], draw[:, None, :]].sum(axis=2).mean(axis=1) / denom
        result['crossed_seed_group_ci95'] = np.quantile(estimates, [.025, .975]).tolist()
    return result


def factorial(rows, specs, expected_seeds=(2026, 2027, 2028), replicates=10000):
    parsed = [metric_spec(s) for s in specs]
    if len({m for m, _ in parsed}) != len(parsed):
        raise ValueError('Repeated metric specification')
    values, seeds, ids, groups, _ = rectangular(rows, ARMS, [m for m, _ in parsed], expected_seeds)
    results = {}
    for metric, direction in parsed:
        v = values[metric]
        results[metric] = dict(
            direction='higher' if direction == 1 else 'lower',
            raw_arm_means={a: float(v[:, :, i].mean()) for i, a in enumerate(ARMS)},
            contrasts={name: interval(direction * (v @ np.asarray(weights)), groups, replicates)
                       for name, weights in CONTRASTS.items()})
    return dict(n=len(ids), seeds=seeds, sample_ids=ids, metrics=results,
                formula='direction * (full - only_pair - only_binding + base)',
                positive_means='Benefit; positive interaction is super-additivity on the specified metric scale.',
                inference='Exploratory; intervals conditional on fitted seeds. Crossed intervals with three seeds are sensitivity analyses.',
                multiplicity='Unadjusted intervals; no automatic confirmatory significance claim.',
                protocol_verified=False)


def paired_comparison(rows, specs, reference, comparator, replicates=10000, long_min=16):
    """For paired training seeds only; never pairs unrelated baseline seeds."""
    parsed = [metric_spec(s) for s in specs]
    values, seeds, ids, groups, lengths = rectangular(rows, (reference, comparator), [m for m, _ in parsed])
    result = dict(reference=reference, comparator=comparator, seeds=seeds, groups={})
    selections = {'all': list(range(len(ids)))}
    if lengths:
        if set(lengths) != set(ids):
            raise ValueError('Partial length annotation')
        selections.update(short=[i for i, s in enumerate(ids) if lengths[s] < long_min],
                          long=[i for i, s in enumerate(ids) if lengths[s] >= long_min])
    for name, ix in selections.items():
        if not ix:
            result['groups'][name] = {'n': 0, 'metrics': {}}
            continue
        gs = [groups[i] for i in ix]
        metrics = {}
        for metric, direction in parsed:
            delta = direction * (values[metric][:, ix, 0] - values[metric][:, ix, 1])
            stats = interval(delta, gs, replicates)
            loo = {g: float(delta[:, np.asarray(gs) != g].mean()) for g in sorted(set(gs))
                   if any(v != g for v in gs)}
            stats['leave_one_group_out'] = loo
            stats['leave_one_group_out_range'] = [min(loo.values()), max(loo.values())] if loo else None
            metrics[metric] = stats
        result['groups'][name] = dict(n=len(ix), n_groups=len(set(gs)), metrics=metrics)
    result.update(long_min=long_min, interpretation='Post-hoc length strata; leave-one-group-out range is not a confidence interval.',
                  positive_means=f'{reference} better than {comparator}')
    return result
