"""Synthetic invariants for offline statistics; no research model execution."""
import csv
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from research_audit.common import digest, write_json
from research_audit.statistics import factorial, interval, paired_comparison
from research_audit.diagnostics import (contacts, drift, baseline_comparison,
                                       cost_summary, export_distances)
from research_audit.protocols import split_index, collect_metrics, stage_cost


def grid(interaction=0):
    rows = []
    for seed in (2026, 2027, 2028):
        for sample in range(5):
            for arm, p, b in [('base', 0, 0), ('only_pair', 1, 0), ('only_binding', 0, 1), ('full', 1, 1)]:
                gain = 2*p + 3*b + interaction*p*b
                rows.append(dict(seed=seed, sample_id=f's{sample}', group=f'g{sample//2}', arm=arm,
                                 length=12+sample, quality=sample+(seed-2026)+gain,
                                 error=20+sample+(seed-2026)-gain))
    return rows


@pytest.mark.parametrize('interaction', [0, 1.5, -2])
def test_interaction_direction_and_exact_constant_interval(interaction):
    out = factorial(grid(interaction), ['quality:max', 'error:min'], replicates=500)
    for metric in ('quality', 'error'):
        effects = out['metrics'][metric]['contrasts']
        assert effects['interaction']['mean'] == pytest.approx(interaction)
        assert effects['interaction']['conditional_group_ci95'] == pytest.approx([interaction]*2)
        assert effects['interaction']['crossed_seed_group_ci95'] == pytest.approx([interaction]*2)
        assert effects['pair_when_binding_absent']['mean'] == 2
        assert effects['binding_when_pair_present']['mean'] == pytest.approx(3+interaction)
    assert out['protocol_verified'] is False


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'group', 'nan', 'seed'])
def test_invalid_pairing_is_rejected(change):
    rows = grid()
    if change == 'missing': rows.pop()
    elif change == 'duplicate': rows.append(dict(rows[0]))
    elif change == 'group': rows[0]['group'] = 'wrong'
    elif change == 'nan': rows[0]['quality'] = float('nan')
    else: rows[0]['seed'] = 999
    with pytest.raises(ValueError):
        factorial(rows, ['quality:max'], replicates=100)


def test_group_resampling_keeps_sample_weights_and_is_reproducible():
    delta = np.array([[0., 0., 9.], [2., 2., 11.]])
    a = interval(delta, ['a', 'a', 'b'], 2000)
    assert a == interval(delta, ['a', 'a', 'b'], 2000)
    assert a['mean'] == 4
    assert a['group_balanced_mean'] == 5.5
    assert a['n_samples'] == 3
    assert interval(delta, ['a']*3, 100)['conditional_group_ci95'] is None


def test_length_strata_leave_groups_out_not_individual_rows():
    rows = [r for r in grid(1) if r['arm'] in ('full', 'base')]
    out = paired_comparison(rows, ['error:min'], 'full', 'base', 100, long_min=14)
    assert out['groups']['long']['n'] == 3
    assert out['groups']['long']['n_groups'] == 2
    assert out['groups']['long']['metrics']['error']['leave_one_group_out_range'] == [6, 6]


def test_baseline_single_seed_is_not_paired_to_three_seeds():
    rows = [r for r in grid() if r['arm'] == 'full' or (r['arm'] == 'base' and r['seed'] == 2026)]
    out = baseline_comparison(rows, ['quality:max'], 'full', 'base', 100)
    assert out['seeds']['base'] == [2026]
    assert out['metrics']['quality']['mean'] == 6
    assert out['metrics']['quality']['crossed_seed_group_ci95'] is None


def contact_record(stage, pred):
    return dict(arm='full', seed=2026, sample_id='s', stage_index=stage,
                reference_distances=[[7., 7., 9., 9., 7.]], predicted_distances=[pred],
                pair_mask=[[True, True, True, True, False]])


def test_contact_turnover_continuous_error_and_missing_mask():
    out = contacts([contact_record(0, [7, 9, 9, 7, 100]),
                    contact_record(1, [9, 7, 7, 9, 200])], [6, 8, 10])
    last = out['per_stage'][1]
    at8 = last['thresholds']['8.0']
    assert (at8['tp'], at8['fp'], at8['fn']) == (1, 1, 1)
    assert at8['changes_from_previous'] == dict(gained_correct=1, lost_correct=1, gained_false=1, removed_false=1)
    assert last['distance_mae'] == 1
    assert last['thresholds']['6.0']['both_empty'] is True
    assert out['per_stage'][0]['thresholds']['8.0']['changes_from_previous'] is None
    with pytest.raises(ValueError, match='Missing stage'):
        contacts([contact_record(1, [7]*5)])
    changed = contact_record(1, [7]*5)
    changed['reference_distances'][0][0] = 6
    with pytest.raises(ValueError, match='correspondence'):
        contacts([contact_record(0, [7]*5), changed])


def test_drift_rejects_rng_changes_and_reports_pairwise_spread():
    base = dict(arm='model', sample_id='s', checkpoint_sha256='a', input_sha256='b',
                inference_seed=2026, device='cpu', precision='float32', runtime_id='same')
    rows = [dict(base, repeat_id=i, rmsd=v, backbone=[[v, 0, 0]]) for i, v in enumerate([1., .9, 1.1])]
    out = drift(rows, ['rmsd'])['per_complex'][0]
    assert out['metrics']['rmsd']['spread'] == pytest.approx(.2)
    assert out['backbone']['max_abs_pairwise'] == pytest.approx(.2)
    rows[-1]['inference_seed'] = 2027
    with pytest.raises(ValueError, match='not numerical'):
        drift(rows, ['rmsd'])


def test_cost_missing_memory_stays_unknown():
    out = cost_summary([dict(run='a', seed=2026, device='cpu', precision='float32', timing_scope='forward only',
                             inference_seconds=2, n_inference_examples=4)])['runs'][0]
    assert out['peak_memory_bytes'] is None
    assert out['inference_seconds_per_example'] == .5


def test_full_split_ids_and_overlap(tmp_path):
    directory = tmp_path / 'data/abgaze_reconstructed_v1/splits'
    directory.mkdir(parents=True)
    for name, count in [('train', 3269), ('valid', 340), ('test', 60)]:
        (directory / f'{name}.json').write_text('\n'.join(json.dumps({'pdb':f'{name}_{i}'}) for i in range(count)))
    assert split_index(tmp_path)['train']['n'] == 3269
    (directory / 'test.json').write_text(json.dumps({'pdb':'train_0'}))
    with pytest.raises(ValueError, match='overlap'):
        split_index(tmp_path)


def test_missing_epoch_is_not_a_matched_budget(tmp_path):
    (tmp_path / 'pair').mkdir()
    (tmp_path / 'pair/epoch_001.json').write_text(json.dumps(dict(epoch=1, seconds=2)))
    assert not stage_cost(tmp_path, 'pair', 30)['complete']


def test_import_existing_scores_checks_hash_and_coverage(tmp_path):
    scores = tmp_path / 'scores.json'
    scores.write_text(json.dumps({'per_complex':[dict(sample_id='s', rmsd=2)]}))
    manifest = tmp_path / 'manifest.json'
    obj = dict(metrics=['rmsd'], groups={'g':['s']},
               runs=[dict(arm='full', seed=2026, metrics_path='scores.json', sha256=digest(scores))])
    manifest.write_text(json.dumps(obj))
    assert collect_metrics(manifest)[0]['rmsd'] == 2
    scores.write_text('{}')
    with pytest.raises(ValueError, match='hash'):
        collect_metrics(manifest)


def test_export_saved_backbones_masks_missing_atoms(tmp_path):
    bb = np.zeros((2, 4, 3))
    bb[0, 1] = [1, 0, 0]
    mask = np.ones((2, 4), dtype=bool)
    mask[1, 1] = False
    context = np.zeros((1, 4, 3))
    context[0, 1] = [8, 0, 0]
    targets = [('s', dict(context_bb=context, context_mask=np.ones((1,4), bool), antigen_mask=np.array([True])),
                dict(bb=bb, mask=mask))]
    path = tmp_path / 'pred.json'
    path.write_text(json.dumps([dict(sample_id='s', backbone=bb.tolist())]))
    manifest = dict(ids=['s'], runs=[dict(arm='full', seed=2026, predictions_path=path.name, sha256=digest(path))])
    records = export_distances(targets, manifest, tmp_path)
    assert records[0]['reference_distances'] == [[7.]]
    assert contacts(records, [8])['per_stage'][0]['thresholds']['8.0']['f1'] == 1


def test_cli_factorial_and_no_overwrite(tmp_path):
    path = tmp_path / 'input.csv'
    rows = grid(1)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    output = tmp_path / 'report.json'
    command = [sys.executable, '-m', 'research_audit', 'factorial', '--input', str(path),
               '--output', str(output), '--metric', 'error:min', '--replicates', '100']
    run = subprocess.run(command, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert json.loads(output.read_text())['metrics']['error']['contrasts']['interaction']['mean'] == 1
    assert subprocess.run(command, capture_output=True).returncode == 2
    with pytest.raises(FileExistsError):
        write_json(output, {})
