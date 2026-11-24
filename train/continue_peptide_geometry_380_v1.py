"""Resume the scoped geometry candidate to epoch30, structure-first selection.

Only the audited 380 original-training IDs and fixed18 validation IDs are used.
The architecture, projection settings, and training loss stay unchanged.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import statistics as st
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train.train_peptide_geometry_human_pilot_v1 import evaluate
from train.train_abgaze_reconstructed import sha, dump, save, move
from eval.validate_peptide_geometry_v1 import read, FeatureRows
from abgaze_reconstructed_objective import objective
from peptide_geometry_refiner import PeptideConstrainedLoopPairRefiner

OUT = ROOT/'outputs/peptide_geometry_380_continue_v1'
STRUCTURAL = ('aligned_rmsd', 'rmsd', 'interface_fape', 'contact_f1')


def structure_score(m, reference):
    return sum((1-m[k])/max(1-reference[k], 1e-8) if k == 'contact_f1'
               else m[k]/max(reference[k], 1e-8) for k in STRUCTURAL)/len(STRUCTURAL)


def geometry_guards(m, reference):
    # No AAR gate. Keep the previously declared peptide/clash tolerances.
    return dict(cn=m['cn_mae'] <= reference['cn_mae'],
                plane=m['omega_planarity_deviation_deg'] <= reference['omega_planarity_deviation_deg'],
                clash=m['clash'] <= reference['clash']*1.1 + 1e-5)


def pareto_epochs(records):
    def vector(r):
        m = r['validation']['summary']
        return [m[k] if k != 'contact_f1' else -m[k] for k in STRUCTURAL]
    return [r['epoch'] for r in records if not any(
        all(a <= b for a,b in zip(vector(s), vector(r))) and
        any(a < b for a,b in zip(vector(s), vector(r))) for s in records)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1); torch.manual_seed(2026)
    device = torch.device('cuda:0'); torch.cuda.set_device(device)
    old = ROOT/'outputs/peptide_geometry_human_train_pilot_v1'
    parent = ROOT/'outputs/full_pipeline_seed2026_v1'
    scope_dir = ROOT/'outputs/peptide_human_expanded_scope_v1'
    pp, oldplan = read(parent/'plan.json'), read(old/'plan.json')
    scope, scopeplan = read(scope_dir/'scope_audit.json'), read(scope_dir/'plan.json')
    assert read(scope_dir/'status.json')['phase'] == 'complete'
    assert read(old/'status.json')['phase'] == 'complete'
    assert sha(ROOT/'scripts/prepare_peptide_human_expanded_v1.py') == scopeplan['script_sha256']
    assert sha(ROOT/'data/abgaze_reconstructed_v1/splits/train.json') == scopeplan['original_train_sha256']
    ids = dict(train=scope['selected_ids'], valid=oldplan['ids']['valid'])
    assert len(ids['train']) == len(set(ids['train'])) == 380
    assert len(ids['valid']) == 18 and ids['valid'] == scopeplan['unchanged_valid_ids']
    assert set(ids['train']) <= set(pp['ids']['train'])
    assert set(ids['valid']) <= set(pp['ids']['valid'])
    assert not set(ids['train']) & set(ids['valid'])
    for decision in scope['decisions']:
        if decision['retained']:
            assert decision['sample_id'] in ids['train']
            for eid in decision['entities']:
                t = scope['taxonomy'][eid]
                assert t['passed'] and t['tax_ids'] == [9606]
                assert t['description'] in scopeplan['allowed_descriptions']
                assert sha(scope_dir/'metadata'/f'{eid}.json') == t['sha256']
    for name,digest in oldplan['code_sha256'].items():
        assert sha(ROOT/name) == digest, name
    source = old/'last.pt'
    initial_state = torch.load(source, map_location='cpu', weights_only=False)
    assert initial_state['epoch'] == 3 and initial_state['plan_sha256'] == sha(old/'plan.json')
    assert sha(parent/'pair/best.pt') == oldplan['checkpoint_sha256']
    features = ROOT/'data/abgaze_binding_features_finalized_v1'
    fp = read(ROOT/'outputs/two_stage_binding60_gpu_v1/plan.json')
    assert sha(features/'plan.json') == pp['feature_plan_sha256']
    for name,digest in read(features/'plan.json')['code_sha256'].items():
        assert sha(ROOT/name) == digest, name
    data, feature_hashes, cache_hashes = {}, {}, {}
    for split in ('train','valid'):
        path = parent/f'{split}_cache.pt'; cache_hashes[split] = sha(path)
        assert cache_hashes[split] == oldplan['cache_sha256'][split]
        rows = torch.load(path, map_location='cpu', weights_only=False)
        assert [r[0] for r in rows] == pp['ids'][split]
        lookup = {r[0]:r for r in rows}
        for sid in ids[split]:
            path = features/split/sid/'features.pt'; digest = sha(path)
            assert digest == fp['feature_sha256'][f'{split}/{sid}']
            meta = torch.load(path, map_location='cpu', weights_only=False)['metadata']
            assert meta['antigen_only'] and not meta['native_h3_used'] and not meta['native_contact_crop']
            assert meta['pdb_sha256'] == sha(ROOT/'data/abgaze_reconstructed_v1/pdb'/f'{sid}.pdb')
            feature_hashes[f'{split}/{sid}'] = digest
        data[split] = FeatureRows([lookup[sid] for sid in ids[split]], features, split, feature_hashes)
    reference = read(old/'comparison.json')['baseline']
    dependencies = set(oldplan['code_sha256']) | {'train/continue_peptide_geometry_380_v1.py'}
    plan = dict(seed=2026, start_epoch=3, final_epoch=30, additional_epochs=27,
        training_history='Epochs1-3 used36 samples; epochs4-30 use380. Not30 epochs on380 samples.',
        ids=ids, counts={s:len(v) for s,v in ids.items()},
        source_checkpoint=str(source), source_sha256=sha(source), prior_plan_sha256=sha(old/'plan.json'),
        model_class='PeptideConstrainedLoopPairRefiner', model_config=initial_state['model_config'],
        scope_sha256=sha(scope_dir/'scope_audit.json'), scope_plan_sha256=sha(scope_dir/'plan.json'),
        cache_sha256=cache_hashes, feature_sha256=feature_hashes,
        code_sha256={f:sha(ROOT/f) for f in sorted(dependencies)},
        optimizer='Resume Adam state; constant lr1e-5; accumulate16; gradient clip1',
        objective='UNCHANGED geometry +0.5 interface FAPE +1 sequence CE',
        selection='Mean normalized aligned RMSD, raw RMSD, interface FAPE, and1-contactF1; lower better. No AAR gating. Save every epoch and Pareto front.',
        geometry_guards=dict(cn_max=reference['cn_mae'], plane_max=reference['omega_planarity_deviation_deg'],
                             clash_max=reference['clash']*1.1+1e-5),
        reference_metrics=reference, early_stopping=False, test_access=False,
        limitations=['Single-seed scoped continuation; no independent generalization claim.',
                     'No matched continued-training control; effects cannot be attributed solely to projection.',
                     'Validation18 has been inspected before; extra4 and external test are not used for selection.',
                     'No AAR hard gate, but unchanged CE remains in the loss.',
                     'Clash score is a backbone proxy; all checkpoints retained even if geometry gates fail.'])
    if args.resume:
        stored = read(OUT/'plan.json')
        assert stored == plan, 'Resume dependencies/configuration differ'
        assert read(OUT/'status.json')['phase'] != 'complete'
        state = torch.load(OUT/'last.pt', map_location='cpu', weights_only=False)
        assert state['plan_sha256'] == sha(OUT/'plan.json')
        records = [read(p) for p in sorted(OUT.glob('epoch_*.json')) if int(p.stem.split('_')[1]) <= state['epoch']]
        assert records[-1]['epoch'] == state['epoch']
    else:
        OUT.mkdir(exist_ok=False); dump(OUT/'plan.json', plan)
        state = initial_state; records = []
    ph = sha(OUT/'plan.json')
    model = PeptideConstrainedLoopPairRefiner(**plan['model_config']).to(device)
    model.load_state_dict(state['model'], strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    optimizer.load_state_dict(state['optimizer'])
    for group in optimizer.param_groups: group['lr'] = 1e-5
    optimizer_steps = state.get('optimizer_steps', read(old/'status.json')['optimizer_steps'])

    def record(epoch, validation, training_metrics=None, seconds=0):
        m = validation['summary']; checks = geometry_guards(m, reference)
        r = dict(epoch=epoch, validation=validation, train=training_metrics,
                 score=structure_score(m, reference), geometry_guards=checks,
                 eligible=all(checks.values()), seconds=seconds, optimizer_steps=optimizer_steps)
        records.append(r)
        eligible = [v for v in records if v['eligible']]
        best = min(eligible,key=lambda v:v['score']) if eligible else None
        best_structure = min(records,key=lambda v:v['score'])
        ckpt = dict(model=model.state_dict(), model_class=plan['model_class'], model_config=plan['model_config'],
                    optimizer=optimizer.state_dict(), epoch=epoch, plan_sha256=ph, validation=m,
                    optimizer_steps=optimizer_steps)
        save(OUT/f'checkpoint_{epoch:03d}.pt', ckpt)
        dump(OUT/f'epoch_{epoch:03d}.json', r)
        save(OUT/'last.pt', ckpt)
        if best is not None and best['epoch'] == epoch: save(OUT/'best_guarded.pt', ckpt)
        if best_structure['epoch'] == epoch: save(OUT/'best_structure.pt', ckpt)
        best_by_metric = {k:min(records,key=lambda v:-v['validation']['summary'][k] if k=='contact_f1' else v['validation']['summary'][k])['epoch'] for k in STRUCTURAL}
        dump(OUT/'selection.json', dict(best_guarded_epoch=None if best is None else best['epoch'],
             best_structure_epoch=best_structure['epoch'], best_by_metric=best_by_metric,
             pareto_epochs=pareto_epochs(records), baseline_replaced=False))
        print(json.dumps(dict(epoch=epoch, validation=m, score=r['score'], guards=checks, seconds=seconds)),flush=True)

    dump(OUT/'status.json',dict(phase='initial_validation',epoch=state['epoch'],final_epoch=30,pid=os.getpid(),test_access=False))
    if not args.resume:
        initial = evaluate(model,data['valid'],device)
        for k,v in state['validation'].items(): assert math.isclose(initial['summary'][k],v,abs_tol=1e-5),k
        record(3, initial)
    for epoch in range(state['epoch']+1,31):
        tick=time.monotonic(); model.train(); torch.manual_seed(2026+epoch)
        order=list(range(len(data['train']))); random.Random(2026+epoch).shuffle(order)
        metrics=[]
        for start in range(0,len(order),16):
            chunk=order[start:start+16]; optimizer.zero_grad(set_to_none=True)
            for index in chunk:
                sid,xc,yc,_=data['train'][index]; x,y=move(xc,device),move(yc,device)
                loss,m=objective(model(x),x,y)
                assert torch.isfinite(loss) and all(math.isfinite(v) for v in m.values()),sid
                (loss/len(chunk)).backward();metrics.append(m)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            assert torch.isfinite(norm),'Nonfinite gradient'
            optimizer.step();optimizer_steps+=1
            dump(OUT/'status.json',dict(phase='training',epoch=epoch,final_epoch=30,seen=len(metrics),total=380,
                optimizer_steps=optimizer_steps,pid=os.getpid(),test_access=False))
        validation=evaluate(model,data['valid'],device)
        train_metrics={k:st.mean(m[k] for m in metrics) for k in metrics[0]}
        record(epoch,validation,train_metrics,time.monotonic()-tick)
    assert sha(source)==plan['source_sha256']
    last5=records[-5:]
    dump(OUT/'comparison.json',dict(reference=reference,continuation_start=records[0]['validation']['summary'],
        epochs=records,selection=read(OUT/'selection.json'),
        last5_validation={k:dict(mean=st.mean(r['validation']['summary'][k] for r in last5),
             min=min(r['validation']['summary'][k] for r in last5),max=max(r['validation']['summary'][k] for r in last5)) for k in reference if k in last5[0]['validation']['summary']},
        limitations=plan['limitations']))
    dump(OUT/'status.json',dict(phase='complete',epoch=30,final_epoch=30,counts=plan['counts'],
        optimizer_steps=optimizer_steps,test_access=False,baseline_replaced=False))


if __name__=='__main__':
    try: main()
    except Exception as exc:
        if OUT.exists():
            dump(OUT/'failure.json',dict(error=repr(exc)))
            dump(OUT/'status.json',dict(phase='failed',error=repr(exc),test_access=False))
        raise
