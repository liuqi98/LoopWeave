"""One validation-locked two-stage test against the existing matched RAAD run."""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from frame_relation_refiner import FrameRelationGraph,FrameRelationRefiner
from orientation_interface_refiner import OrientationInterfaceRefiner
from train.train_abgaze_reconstructed import sha,dump,move
from abgaze_reconstructed_objective import objective
from eval.test_abgaze_reconstructed_ours import aligned_rmsd
from eval.exploratory_abgaze_best5 import select_by_aar


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--wait',action='store_true')
    args=p.parse_args();torch.set_num_threads(1)
    run=ROOT/'outputs/reconstructed_two_stage_v1'
    while json.loads((run/'status.json').read_text()).get('phase')!='complete':
        if not args.wait:raise RuntimeError('Training incomplete')
        print('Waiting for all 30 refiner epochs; test data not opened.',flush=True)
        time.sleep(30)
    training_plan=json.loads((run/'plan.json').read_text())
    assert not training_plan['smoke']
    for name,digest in training_plan['code_sha256'].items():assert sha(ROOT/name)==digest,name
    dataset=ROOT/'data/abgaze_reconstructed_v1'
    assert sha(dataset/'tensors/manifest.json')==training_plan['manifest_sha256']
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    states={}
    for stage in ('generator','refiner'):
        shutil.copyfile(run/stage/'best.pt',out/f'{stage}.pt')
        state=torch.load(out/f'{stage}.pt',map_location='cpu',weights_only=False)
        assert state['plan_sha256']==sha(run/'plan.json')
        states[stage]=state
    audit=json.loads((run/'cache_audit.json').read_text())
    assert sha(out/'generator.pt')==audit['generator_sha256']
    raad=ROOT/'outputs/raad_abgaze_reconstructed_test_v1'
    raadplan=json.loads((raad/'plan.json').read_text())
    graphpath=ROOT/'data/raad_abgaze_reconstructed_v1/test.pt'
    assert sha(graphpath)==raadplan['inputs_sha256']
    oldplan=json.loads((ROOT/'outputs/binding_interface_test_v1/plan.json').read_text())
    assert sha(dataset/'tensors/test.pt')==oldplan['test_sha256']
    raadtrain=json.loads((ROOT/'outputs/raad_abgaze_reconstructed_seed2026_v1/plan.json').read_text())
    for split in ('train','valid'):assert training_plan['ids'][split]==raadtrain['ids'][split]
    plan=dict(epochs={k:s['epoch'] for k,s in states.items()},K=5,temperature=.5,seed=2026,
        checkpoints={k:sha(out/f'{k}.pt') for k in states},test_sha256=oldplan['test_sha256'],
        graph_sha256=sha(graphpath),raad_metrics_sha256=sha(raad/'metrics.json'),
        training_plan_sha256=sha(run/'plan.json'),code_sha256=sha(__file__),
        scoring_dependencies={str(f.relative_to(ROOT)):sha(f) for f in
            [ROOT/'abgaze_reconstructed_objective.py',ROOT/'eval/test_abgaze_reconstructed_ours.py',ROOT/'eval/exploratory_abgaze_best5.py']},
        selection='validation-only best generator and refiner; no test checkpoint sweep',
        protocol='same train/valid/test IDs and retained context as matched RAAD; bound framework; no LM or surface features',
        limitations=['Reconstructed, not author-certified','Single seed; previously inspected test',
                    'Five sequence samples share one backbone; oracle native-AAR selection',
                    'CA contacts <=8A on retained antigen context, not DockQ','Training budgets and architectures differ'])
    dump(out/'plan.json',plan)
    device=torch.device(args.device)
    if device.type=='cuda':torch.cuda.set_device(device)
    torch.manual_seed(2026)
    generator=FrameRelationRefiner(**states['generator']['model_config']).to(device)
    refiner=OrientationInterfaceRefiner(**states['refiner']['model_config']).to(device)
    generator.load_state_dict(states['generator']['model']);refiner.load_state_dict(states['refiner']['model'])
    generator.eval();refiner.eval()
    graphs={item['sample_id']:g for item,g in torch.load(graphpath,map_location='cpu',weights_only=False)}
    rows=torch.load(dataset/'tensors/test.pt',map_location='cpu',weights_only=False)
    assert len(rows)==60 and {r[0] for r in rows}==set(graphs)
    alphabet='ACDEFGHIKLMNPQRSTVWY';rng=torch.Generator().manual_seed(2026);generated=[]
    dump(out/'status.json',dict(phase='generating',n=60))
    with torch.no_grad():
        for sid,x,_ in rows:
            raw=generator(FrameRelationGraph(**graphs[sid]).to(device))
            inputs=move(x,device)
            inputs.update(base_bb=raw['h3_backbone'],base_logits=torch.zeros_like(inputs['base_logits']))
            result=refiner(inputs)
            bb=result['backbone'].cpu();logits=result['sequence_logits'].cpu()
            assert torch.isfinite(bb).all() and torch.isfinite(logits).all(),sid
            samples=torch.multinomial((logits/.5).softmax(-1),5,replacement=True,generator=rng).T
            generated.append(dict(sample_id=sid,backbone=bb.tolist(),sequence_logits=logits.tolist(),
                greedy=''.join(alphabet[i] for i in logits.argmax(-1).tolist()),
                candidates=[''.join(alphabet[i] for i in seq) for seq in samples.tolist()]))
    dump(out/'generated.json',generated)
    dump(out/'generation_audit.json',dict(sha256=sha(out/'generated.json'),native_labels_packaged=True,
        native_labels_not_passed_to_models=True,all_candidates_committed_before_scoring=True))
    metrics=[]
    with torch.no_grad():
        for (sid,x,y),g in zip(rows,generated):
            pred=dict(backbone=torch.tensor(g['backbone']),sequence_logits=torch.tensor(g['sequence_logits']))
            native=''.join(alphabet[i] for i in y['sequence'].tolist())
            selected,aars=select_by_aar(g['candidates'],native)
            _,m=objective(pred,x,y);v=y['mask'][:,1]
            m['aligned_rmsd']=aligned_rmsd(pred['backbone'][v,1],y['bb'][v,1])
            context=x['context_bb'][x['antigen_mask']&x['context_mask'][:,1],1]
            ref=torch.cdist(y['bb'][v,1],context)<=8;gen=torch.cdist(pred['backbone'][v,1],context)<=8
            tp=int((ref&gen).sum())
            m.update(contact_precision=tp/max(int(gen.sum()),1),contact_recall=tp/max(int(ref.sum()),1))
            metrics.append(dict(sample_id=sid,length=len(native),selected_candidate=selected,oracle_aar=aars[selected],
                candidate_aars=aars,mean_sample_aar=sum(aars)/5,**m))
    keys=['aar','oracle_aar','mean_sample_aar','rmsd','aligned_rmsd','contact_f1','contact_precision','contact_recall','cn_mae','clash','interface_fape']
    summary={k:sum(m[k] for m in metrics)/len(metrics) for k in keys}
    for k in ('aar','oracle_aar'):summary['pooled_'+k]=sum(m[k]*m['length'] for m in metrics)/sum(m['length'] for m in metrics)
    dump(out/'metrics.json',dict(n=60,summary=summary,per_complex=metrics,test_sha256=plan['test_sha256']))
    baseline=json.loads((raad/'metrics.json').read_text())
    assert baseline['test_sha256']==plan['test_sha256']
    assert [m['sample_id'] for m in baseline['per_complex']]==[m['sample_id'] for m in metrics]
    prior=json.loads((ROOT/'outputs/binding_interface_test_v1/comparison.json').read_text())['summaries']['binding']
    result=dict(n=60,summaries=dict(two_stage=summary,raad=baseline['summary'],previous_binding=prior),
        two_stage_minus_raad={k:summary[k]-baseline['summary'][k] for k in summary},limitations=plan['limitations'])
    dump(out/'comparison.json',result);dump(out/'status.json',dict(phase='complete',n=60))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
