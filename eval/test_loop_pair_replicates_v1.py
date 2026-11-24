"""Evaluate all validation-locked continuation replicas, never select by test.

These are NOT independent full-pipeline training seeds.
"""
import json
import shutil
import statistics
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from eval.test_reconstructed_two_stage import FrameRelationGraph,FrameRelationRefiner,sha,dump,move,objective,aligned_rmsd,select_by_aar
from persistent_loop_pair_refiner import PersistentLoopPairRefiner
from binding_interface_refiner import BindingInterfaceRefiner


def score(rows,generated):
    alphabet='ACDEFGHIKLMNPQRSTVWY';metrics=[]
    with torch.no_grad():
        for (sid,x,y),g in zip(rows,generated):
            assert sid==g['sample_id']
            pred=dict(backbone=torch.tensor(g['backbone']),sequence_logits=torch.tensor(g['sequence_logits']))
            _,m=objective(pred,x,y);valid=y['mask'][:,1]
            m['aligned_rmsd']=aligned_rmsd(pred['backbone'][valid,1],y['bb'][valid,1])
            native=''.join(alphabet[i] for i in y['sequence'].tolist())
            selected,aars=select_by_aar(g['candidates'],native)
            metrics.append(dict(sample_id=sid,length=len(native),**m,oracle_aar=aars[selected],
                first_sample_aar=aars[0],mean_sample_aar=sum(aars)/5,selected_candidate=selected,candidate_aars=aars))
    keys=('aar','first_sample_aar','mean_sample_aar','oracle_aar','rmsd','aligned_rmsd','contact_f1','cn_mae','clash','interface_fape')
    return dict(n=len(metrics),summary={k:statistics.mean(m[k] for m in metrics) for k in keys},per_complex=metrics)


def aggregate(summaries):
    return {k:dict(mean=statistics.mean(s[k] for s in summaries),sample_std=statistics.stdev(s[k] for s in summaries))
            for k in summaries[0]}


def main():
    torch.set_num_threads(1);device=torch.device('cuda:0');torch.cuda.set_device(device)
    out=ROOT/'outputs/loop_pair_replicates_test_v1';out.mkdir(exist_ok=False)
    reference=ROOT/'outputs/persistent_loop_pair_test_v1'
    refplan=json.loads((reference/'plan.json').read_text())
    sourceplan=json.loads((ROOT/'outputs/persistent_loop_pair_v1/plan.json').read_text())
    queue=json.loads((ROOT/'outputs/loop_pair_multiseed_queue_v1/paired_validation_summary.json').read_text())
    models={'full':PersistentLoopPairRefiner,'control':BindingInterfaceRefiner}
    locks={}
    # Lock every candidate before accessing test tensors.
    for pair in queue['pairs']:
        seed=pair['seed']
        for arm,selection in pair['arms'].items():
            run=ROOT/f'outputs/loop_pair_{arm}_seed{seed}_v1'
            plan=json.loads((run/'plan.json').read_text())
            status=json.loads((run/'status.json').read_text())
            assert status['phase']=='complete' and not plan['smoke'] and plan['seed']==seed
            for split in ('train','valid'):assert plan['ids'][split]==sourceplan['ids'][split]
            for name,digest in plan['code_sha256'].items():assert sha(ROOT/name)==digest,name
            assert plan['generator_sha256']==refplan['checkpoints']['generator']
            assert plan['feature_plan_sha256']==refplan['feature_plan_sha256']
            assert sha(run/'best.pt')==selection['checkpoint_sha256']
            target=out/f'{arm}_{seed}';target.mkdir()
            shutil.copyfile(run/'best.pt',target/'locked.pt')
            state=torch.load(target/'locked.pt',map_location='cpu',weights_only=False)
            assert state['plan_sha256']==sha(run/'plan.json') and state['epoch']==selection['epoch']
            locks[f'{arm}_{seed}']=dict(arm=arm,seed=seed,epoch=state['epoch'],
                checkpoint_sha256=sha(target/'locked.pt'),training_plan_sha256=sha(run/'plan.json'))
    generator_path=ROOT/'outputs/reconstructed_two_stage_v1/generator/best.pt'
    assert sha(generator_path)==refplan['checkpoints']['generator']
    dataset=ROOT/'data/abgaze_reconstructed_v1'
    test=dataset/'tensors/test.pt';graphpath=ROOT/'data/raad_abgaze_reconstructed_v1/test.pt'
    assert sha(test)==refplan['test_sha256'] and sha(graphpath)==refplan['graph_sha256']
    dump(out/'plan.json',dict(locks=locks,test_sha256=sha(test),graph_sha256=sha(graphpath),
        generator_sha256=sha(generator_path),code_sha256=sha(Path(__file__)),K=5,temperature=.5,sampling_seed=2026,
        selection='all pre-existing validation-selected checkpoints; no test checkpoint sweep',
        limitations=['Two prospective continuation seeds sharing a generator and a pretrained refiner source.',
                    'Not independent full-pipeline repeats; not symmetric to RAAD full-training seeds.',
                    'Original seed2026 exploratory and reported separately.',
                    'Repeatedly inspected development test; not an independent holdout.',
                    'Five sampled sequences share one deterministic backbone.']))
    gs=torch.load(generator_path,map_location='cpu',weights_only=False)
    generator=FrameRelationRefiner(**gs['model_config']).to(device)
    generator.load_state_dict(gs['model']);generator.eval()
    rows=torch.load(test,map_location='cpu',weights_only=False)
    graphs={item['sample_id']:g for item,g in torch.load(graphpath,map_location='cpu',weights_only=False)}
    assert len(rows)==60 and {r[0] for r in rows}==set(graphs)
    inputs_by_id={};feature_hashes={}
    with torch.no_grad():
        for sid,x,_ in rows:
            directory=ROOT/'outputs/binding_interface_test_v1/features'/sid
            path=directory/'features.pt';digest=sha(path)
            assert digest==json.loads((directory/'ready.json').read_text())['sha256']
            feature=torch.load(path,map_location='cpu',weights_only=False)
            meta=feature['metadata']
            assert meta['antigen_only'] and not meta['native_h3_used']
            assert meta['pdb_sha256']==sha(dataset/'pdb'/f'{sid}.pdb')
            assert not set(x)&set(feature['features'])
            raw=generator(FrameRelationGraph(**graphs[sid]).to(device))
            inputs=move({**x,**feature['features']},device)
            inputs.update(base_bb=raw['h3_backbone'],base_logits=torch.zeros_like(inputs['base_logits']))
            inputs_by_id[sid]=inputs;feature_hashes[sid]=digest
    del generator
    summaries={};alphabet='ACDEFGHIKLMNPQRSTVWY'
    for label,lock in locks.items():
        target=out/label
        state=torch.load(target/'locked.pt',map_location='cpu',weights_only=False)
        model=models[lock['arm']](**state['model_config']).to(device)
        model.load_state_dict(state['model']);model.eval()
        rng=torch.Generator().manual_seed(2026);generated=[]
        with torch.no_grad():
            for sid,_,_ in rows:
                result=model(dict(inputs_by_id[sid]));bb=result['backbone'].cpu();logits=result['sequence_logits'].cpu()
                assert torch.isfinite(bb).all() and torch.isfinite(logits).all(),sid
                samples=torch.multinomial((logits/.5).softmax(-1),5,replacement=True,generator=rng).T
                generated.append(dict(sample_id=sid,backbone=bb.tolist(),sequence_logits=logits.tolist(),
                    greedy=''.join(alphabet[i] for i in logits.argmax(-1).tolist()),
                    candidates=[''.join(alphabet[i] for i in seq) for seq in samples.tolist()]))
                dump(out/'status.json',dict(phase='generating',run=label,completed=len(generated),total=60))
        dump(target/'generated.json',generated)
        dump(target/'generation_audit.json',dict(sha256=sha(target/'generated.json'),feature_sha256=feature_hashes,
            labels_not_passed_to_models=True,all_candidates_committed_before_scoring=True))
        result=score(rows,generated);result['test_sha256']=sha(test)
        dump(target/'metrics.json',result);summaries[label]=result['summary']
        print(label,json.dumps(result['summary']),flush=True)
        del model
    delta={str(seed):{k:summaries[f'full_{seed}'][k]-summaries[f'control_{seed}'][k]
                     for k in summaries[f'full_{seed}']} for seed in (2027,2028)}
    report=dict(n=60,by_run=summaries,
        prospective_continuation_mean_std={arm:aggregate([summaries[f'{arm}_{s}'] for s in (2027,2028)]) for arm in models},
        paired_full_minus_control=delta,
        original_exploratory_2026=json.loads((reference/'metrics.json').read_text())['summary'],
        limitations=json.loads((out/'plan.json').read_text())['limitations'])
    dump(out/'comparison.json',report);dump(out/'status.json',dict(phase='complete',runs=4))


if __name__=='__main__':main()
