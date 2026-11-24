"""Audit historical full-data runs without importing models or loading weights."""
from pathlib import Path
import json
import math
from .common import digest, read_json
from .statistics import ARMS


VARIANTS = {'only_pair': 'no_binding', 'only_binding': 'no_pair',
            'base': 'no_pair_no_binding', 'full': 'full'}
SHARED_TRAINING_CODE = ('train/train_full_pipeline_replica_v1.py',
                        'train/train_reconstructed_two_stage.py',
                        'train/train_loop_pair_multiseed.py',
                        'abgaze_reconstructed_objective.py')


def split_index(workspace):
    source = Path(workspace) / 'data/abgaze_reconstructed_v1/splits'
    index = {}
    for name in ('train', 'valid', 'test'):
        path = source / f'{name}.json'
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        ids = [r['pdb'] for r in records]
        if len(ids) != len(set(ids)):
            raise ValueError(f'Duplicate IDs in {name}')
        index[name] = dict(n=len(ids), ids=ids, sha256=digest(path))
    for a, b in (('train', 'valid'), ('train', 'test'), ('valid', 'test')):
        if set(index[a]['ids']) & set(index[b]['ids']):
            raise ValueError(f'Split ID overlap: {a}/{b}')
    return index


def stage_cost(run, stage, expected_epochs):
    """Logged time includes training, validation and checkpoint serialization."""
    rows = []
    for path in sorted((run / stage).glob('epoch_*.json')):
        d = read_json(path)
        if d['epoch'] > 0:
            rows.append(d)
    epochs = [r['epoch'] for r in rows]
    complete = epochs == list(range(1, expected_epochs + 1))
    times = [r.get('seconds') for r in rows]
    known = all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in times)
    return dict(expected_epochs=expected_epochs, observed_epochs=epochs, complete=complete,
                logged_seconds=sum(times) if known and times else None,
                wall_time_definition='Sum of logged epoch seconds; includes validation and saving.',
                peak_memory_bytes=None, parameter_count=None)


def audit_history(workspace, seeds=(2026, 2027, 2028), verify_artifacts=False):
    root = Path(workspace).resolve()
    splits = split_index(root)
    runs, sources, common = [], {}, {}

    def read(path):
        path = Path(path)
        sources[str(path)] = digest(path)
        return read_json(path)

    for seed in seeds:
        parent = root / f'outputs/full_pipeline_seed{seed}_v1'
        pp = read(parent / 'plan.json')
        common[seed] = dict(parent_plan_sha256=digest(parent / 'plan.json'),
                            ancestor_path=str(parent / 'refiner/best.pt'),
                            ancestor_sha256=digest(parent / 'refiner/best.pt') if verify_artifacts else None)
        for arm in ARMS:
            variant = VARIANTS[arm]
            if arm == 'full':
                run = parent
            else:
                # v3 runs use the same protocol; accept either directory generation.
                candidates = [root / f'outputs/matched_ablation_v3_{variant}_seed{seed}',
                              root / f'outputs/matched_ablation_v2_{variant}_seed{seed}']
                run = next((c for c in candidates if (c / 'plan.json').is_file()), candidates[-1])
            row = dict(seed=seed, arm=arm, run=str(run), issues=[], present=(run / 'plan.json').is_file())
            runs.append(row)
            if not row['present']:
                row['issues'].append('Missing run/plan; no substitute checkpoint is permitted.')
                continue
            plan = read(run / 'plan.json')
            issues = row['issues']
            if plan.get('seed') != seed or plan.get('smoke') is not False:
                issues.append('Seed mismatch or missing/non-false smoke flag.')
            if any(pp.get('ids', {}).get(s) != splits[s]['ids'] for s in ('train', 'valid')):
                issues.append('Parent split order/content differs from full-data split files.')
            status_path = run / 'status.json'
            status = read(status_path) if status_path.exists() else {}
            if status.get('phase') != 'complete':
                issues.append('Run is not complete.')
            row['selected_epoch'] = status.get('selected_epoch')
            row['plan_sha256'] = digest(run / 'plan.json')
            row['common_ancestor'] = common[seed]
            row['stages'] = {}
            if pp.get('epochs') != dict(generator=100, refiner=30, binding=60, pair=30):
                issues.append('Unexpected historical full-pipeline epoch schedule.')
            if arm != 'full':
                if plan.get('variant') != variant:
                    issues.append('Ablation variant mismatch.')
                if Path(plan.get('parent', '')).resolve() != parent:
                    issues.append('Wrong parent run.')
                if plan.get('parent_plan_sha256') != digest(parent / 'plan.json'):
                    issues.append('Parent plan hash mismatch.')
                if plan.get('feature_plan_sha256') != pp.get('feature_plan_sha256'):
                    issues.append('Feature provenance mismatch.')
                expected_source = parent / ('refiner/best.pt' if arm in ('only_pair', 'base') else 'binding/last.pt')
                if Path(plan.get('source', '')).resolve() != expected_source:
                    issues.append('Wrong stage initialization source.')
                expected_epochs = dict(binding=60 if arm in ('only_pair', 'base') else 0, pair=30)
                if plan.get('epochs') != expected_epochs:
                    issues.append('Ablation epoch budget mismatch.')
                for name in SHARED_TRAINING_CODE:
                    a = plan.get('code_sha256', {}).get(name)
                    b = pp.get('code_sha256', {}).get(name)
                    if not a or not b or a != b:
                        issues.append(f'Training/loss/selection code mismatch: {name}')
                if verify_artifacts:
                    for path, expected in [(expected_source, plan.get('source_sha256')),
                                           (parent / 'generator/best.pt', plan.get('generator_sha256'))]:
                        if not path.is_file() or digest(path) != expected:
                            issues.append(f'Artifact hash mismatch: {path}')
                    for split in ('train', 'valid'):
                        path = parent / f'{split}_cache.pt'
                        if not path.is_file() or digest(path) != plan.get('cache_sha256', {}).get(split):
                            issues.append(f'Generator cache hash mismatch: {split}')
                row['cache_sha256'] = plan.get('cache_sha256')
                row['generator_sha256'] = plan.get('generator_sha256')
                row['source_sha256'] = plan.get('source_sha256')
                row['selection'] = plan.get('selection')
                row['objective'] = plan.get('objective')
            binding_run = run if arm in ('only_pair', 'base') else parent
            row['stages']['binding'] = stage_cost(binding_run, 'binding', 60)
            row['stages']['pair'] = stage_cost(run, 'pair', 30)
            if any(not v['complete'] for v in row['stages'].values()):
                issues.append('Missing/duplicate/extra epoch logs; completed budget not established.')
            if row['selected_epoch'] not in range(31):
                issues.append('Invalid validation-selected pair epoch.')
            if verify_artifacts:
                for name, expected in plan.get('code_sha256', {}).items():
                    path = root / name
                    if not path.is_file() or digest(path) != expected:
                        issues.append(f'Current code differs from training provenance: {name}')
            if arm == 'full':
                metric_path = parent / 'test/metrics.json'
            else:
                legacy = root / f'outputs/matched_ablations_test_v2/{variant}_seed{seed}/test/metrics.json'
                current = root / f'outputs/matched_ablations_test_v3_2x2/metrics_{arm}_seed{seed}.json'
                metric_path = legacy if legacy.is_file() else current
            row['metrics_path'] = str(metric_path)
            row['metrics_present'] = metric_path.is_file()
            if metric_path.is_file():
                metrics = read(metric_path)
                row['evaluation_sha256'] = metrics.get('test_sha256')
                if [r['sample_id'] for r in metrics['per_complex']] != splits['test']['ids']:
                    issues.append('Evaluation IDs/order differ from full-data development split.')
        cells = [r for r in runs if r['seed'] == seed and r['present'] and r['arm'] != 'full']
        for key in ('cache_sha256', 'generator_sha256', 'selection', 'objective'):
            if cells and any(c.get(key) is None or c.get(key) != cells[0].get(key) for c in cells):
                for cell in cells:
                    cell['issues'].append(f'Within-seed mismatch/missing metadata: {key}')
    existing = [r for r in runs if r.get('metrics_present')]
    if existing and any(not r.get('evaluation_sha256') or r['evaluation_sha256'] != existing[0].get('evaluation_sha256') for r in existing):
        for r in existing:
            r['issues'].append('Evaluation tensor hash mismatch or missing hash.')
    complete = all(r['present'] and r.get('metrics_present', False) and not r['issues'] for r in runs)
    return dict(schema_version=1, source_workspace=str(root), splits=splits, runs=runs,
                sources_sha256=sources, complete=complete, artifact_hashes_checked=verify_artifacts,
                metadata_consistent_for_existing_runs=all(not r['issues'] for r in runs if r['present']),
                protocol_verified=False,
                interpretation='Metadata/log audit only. Runtime optimizer-state and retained-weight equality are not proved by plan strings.',
                limitations=['Stage-wise treatment histories differ by design; compare their common pre-treatment ancestor, not only the immediate checkpoint.',
                             'Equal epoch budgets do not establish equal FLOPs, memory, parameters or elapsed time.',
                             'A missing base arm prevents an interaction estimate.',
                             'Historical development evaluation is not an untouched confirmatory test.'])


def collect_metrics(manifest_path):
    """Import existing score files, preserving every sample and each seed."""
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    from .common import membership, finite
    member = membership(manifest['groups'])
    rows, seen = [], set()
    if not manifest.get('metrics') or not manifest.get('runs'):
        raise ValueError('Manifest requires metrics and runs')
    for run in manifest['runs']:
        path = Path(run['metrics_path'])
        if not path.is_absolute():
            path = manifest_path.parent / path
        document = read_json(path)
        if run.get('sha256') != digest(path):
            raise ValueError(f'Missing/mismatched score file hash: {path}')
        cases = document['per_complex']
        ids = [r['sample_id'] for r in cases]
        if len(ids) != len(set(ids)) or set(ids) != set(member):
            raise ValueError(f'Incomplete/duplicate evaluation cohort: {path}')
        for case in cases:
            sid = case['sample_id']
            key = (run['arm'], int(run['seed']), sid)
            if key in seen:
                raise ValueError(f'Duplicate run/sample: {key}')
            seen.add(key)
            row = dict(sample_id=sid, group=member[sid], seed=int(run['seed']), arm=run['arm'])
            if 'length' in case:
                row['length'] = int(case['length'])
            row.update({k: finite(case[k], f'{key}/{k}') for k in manifest['metrics']})
            rows.append(row)
    return rows
