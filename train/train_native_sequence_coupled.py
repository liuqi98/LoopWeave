"""Historical-protocol sequence-head ablation and validation-locked final test.

No claim of clean shared-benchmark retraining: ancestors and caches are old.
Smoke runs never load test inputs/targets. Production tests exactly one selected
checkpoint after both fixed-budget arms finish, with baseline fallback.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from native_sequence_coupled_refiner import NativeSequenceCoupledRefiner,objective,geometry_guard
from relation_interface_refiner import AA_TO_INDEX
from train.train_v21_atomic import move


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def dump(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def save(path,value):
    tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)


def load_data(smoke):
    data={}
    for split in ('train','valid'):
        rows=torch.load(ROOT/f'outputs/structure_refinement_comparison/{split}_cache.pt',map_location='cpu',weights_only=False)
        references=torch.load(ROOT/f'data/raad_protocol_frame_v1/{split}_targets.pt',map_location='cpu',weights_only=False)
        lookup={r['sample_id']:r['native_h3_sequence'] for r in references}
        assert len(lookup)==len(references)
        if smoke:rows=rows[:4]
        data[split]=[]
        for sid,x,y in rows:
            sequence=lookup[sid]
            assert len(sequence)==len(x['base_bb'])==len(y['bb'])
            data[split].append((sid,x,{**y,'sequence':torch.tensor([AA_TO_INDEX[a] for a in sequence],dtype=torch.long)}))
    assert not ({r[0] for r in data['train']}&{r[0] for r in data['valid']})
    return data


def epoch(model,data,device,optimizer=None,seed=0):
    model.train(optimizer is not None);indices=list(range(len(data)))
    if optimizer:random.Random(seed).shuffle(indices)
    rows=[]
    with torch.set_grad_enabled(optimizer is not None):
        for i in indices:
            sid,x,y=data[i];x,y=move(x,device),move(y,device)
            loss,m=objective(model(x),x,y)
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss {sid}')
            if optimizer:
                optimizer.zero_grad(set_to_none=True);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise RuntimeError(f'Nonfinite gradients {sid}')
                optimizer.step()
            rows.append(dict(sample_id=sid,**m))
    return dict(mean={k:sum(r[k] for r in rows)/len(rows) for k in rows[0] if k!='sample_id'},rows=rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu')
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--run-name',default='native_sequence_coupled_v1');args=parser.parse_args()
    if Path(args.run_name).name!=args.run_name:parser.error('Invalid run name')
    torch.set_num_threads(1);device=torch.device(args.device)
    out=ROOT/'outputs'/args.run_name;out.mkdir(exist_ok=args.resume)
    source=ROOT/'outputs/relation_coupling_comparison/coupled/best.pt'
    state=torch.load(source,map_location='cpu',weights_only=False)
    config={**state['model_config'],'final_native_sequence':True}
    data=load_data(args.smoke)
    code=[Path(__file__),ROOT/'native_sequence_coupled_refiner.py',ROOT/'relation_coupled_refiner.py',
          ROOT/'structure_refinement_ablation.py',ROOT/'v21_atomic_refiner.py',ROOT/'v22_joint_refiner.py',
          ROOT/'train/train_v21_atomic.py',ROOT/'eval/test_selected_structure.py',ROOT/'eval/evaluate_rabd_generated.py',
          ROOT/'external/chimera-bench/chimera_bench/metrics.py']
    artifacts=[source,Path(state['base_checkpoint'])]
    for split in ('train','valid'):
        artifacts.extend([ROOT/f'outputs/structure_refinement_comparison/{split}_cache.pt',
                          ROOT/f'data/raad_protocol_frame_v1/{split}_targets.pt'])
    plan=dict(source_checkpoint=str(source),model_config=config,seed=2026,device=str(device),smoke=args.smoke,
              epochs=1 if args.smoke else 30,arms=['head_only','joint'],joint_head_warmup_epochs=0 if args.smoke else 3,
              sequence='native v18 logits plus trainable residual from final coupled H3 hidden state; no AntiBERTy',
              loss='existing constrained geometry objective + 1.0 per-residue CE; equal weighting per complex',
              optimizer='AdamW weight_decay1e-4, clip1; head1e-4/shared1e-5, cosine to 10% over30epochs',
              selection='maximum validation macro AAR among geometry-guard eligible epochs/arms; lower CE breaks ties; original checkpoint fallback',
              guards='RMSD/aligned/contact-position <= source+.02 A; F1/precision/recall >= source-.002; CN/endpoint <= source+.01 A; clash <=1.1*source+1e-5',
              counts={s:len(v) for s,v in data.items()},training_ids={s:[r[0] for r in v] for s,v in data.items()},
              protocol='historical old-split continuation, NOT current purged shared-baseline track; no claim of blind test or RAAD superiority',
              test='one validation-selected checkpoint only after both arms complete; K1 localRAbD60; smoke forbids test',
              code_hashes={str(p.relative_to(ROOT)):sha(p) for p in code},
              artifact_hashes={str(p):sha(p) for p in artifacts})
    if args.resume:assert json.loads((out/'plan.json').read_text())==plan,'Code/data/config changed'
    else:dump(out/'plan.json',plan)
    torch.manual_seed(2026);model=NativeSequenceCoupledRefiner(**config).to(device);model.load_coupled(state['model'])
    baseline=epoch(model,data['valid'],device)
    dump(out/'baseline.json',baseline)
    for arm in plan['arms']:
        directory=out/arm;directory.mkdir(exist_ok=True)
        if (directory/'completed.json').exists():continue
        torch.manual_seed(2026);model=NativeSequenceCoupledRefiner(**config).to(device);model.load_coupled(state['model'])
        new=[p for n,p in model.named_parameters() if n.startswith('native_head.')]
        shared=[p for n,p in model.named_parameters() if not n.startswith('native_head.')]
        optimizer=torch.optim.AdamW([dict(params=new,lr=1e-4,initial_lr=1e-4),
                                     dict(params=shared,lr=1e-5,initial_lr=1e-5)],weight_decay=1e-4)
        start=0;best_key=(baseline['mean']['aar'],-baseline['mean']['sequence_ce']);best_epoch=None
        if (directory/'last.pt').exists():
            if not args.resume:raise RuntimeError('Existing checkpoint')
            saved=torch.load(directory/'last.pt',map_location=device,weights_only=False)
            model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);start=saved['epoch']
            best_key=tuple(saved['best_key']);best_epoch=saved['best_epoch']
            torch.set_rng_state(saved['torch_rng'].cpu())
            if device.type=='cuda':torch.cuda.set_rng_state(saved['cuda_rng'].cpu(),device)
        for step in range(start+1,plan['epochs']+1):
            tick=time.monotonic();joint=arm=='joint' and step>plan['joint_head_warmup_epochs']
            for p in shared:p.requires_grad_(joint)
            for g in optimizer.param_groups:g['lr']=g['initial_lr']*(.1+.45*(1+math.cos(math.pi*(step-1)/max(plan['epochs']-1,1))))
            dump(out/'status.json',dict(phase='training',arm=arm,epoch=step,total=plan['epochs'],shared_trainable=joint,test_access=False))
            train=epoch(model,data['train'],device,optimizer,2026+step)['mean']
            validation=epoch(model,data['valid'],device);m=validation['mean']
            allowed=geometry_guard(m,baseline['mean']);key=(m['aar'],-m['sequence_ce'])
            improved=allowed and key>best_key
            if improved:best_key=key;best_epoch=step
            saved=dict(model=model.state_dict(),model_config=config,epoch=step,validation=m,optimizer=optimizer.state_dict(),
                       base_checkpoint=state['base_checkpoint'],source_checkpoint=str(source),arm=arm,best_key=best_key,best_epoch=best_epoch,
                       torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device) if device.type=='cuda' else None)
            if improved:
                save(directory/'best.pt',saved);dump(directory/'best_validation.json',dict(epoch=step,**validation))
            save(directory/'last.pt',saved)
            record=dict(epoch=step,train=train,validation=m,guard_passed=allowed,best_epoch=best_epoch,seconds=time.monotonic()-tick)
            with (directory/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps(dict(arm=arm,**record)),flush=True)
        dump(directory/'completed.json',dict(complete=True,best_epoch=best_epoch))
    selected=dict(mode='baseline_native',checkpoint=str(source),epoch=state['epoch'],validation=baseline['mean'])
    best_key=(baseline['mean']['aar'],-baseline['mean']['sequence_ce']);comparison={}
    for arm in plan['arms']:
        path=out/arm/'best_validation.json'
        if not path.exists():comparison[arm]={'eligible_improvement':False};continue
        value=json.loads(path.read_text());m=value['mean'];comparison[arm]=dict(epoch=value['epoch'],validation=m)
        key=(m['aar'],-m['sequence_ce'])
        if key>best_key:
            best_key=key;selected=dict(mode=arm,checkpoint=str(out/arm/'best.pt'),epoch=value['epoch'],validation=m)
    selected['checkpoint_sha256']=sha(selected['checkpoint'])
    dump(out/'selection.json',dict(selected=selected,comparison=comparison,test_access=False))
    if args.smoke:
        dump(out/'status.json',dict(phase='smoke_complete',test_access=False));return
    dump(out/'status.json',dict(phase='testing_locked_selection',selected=selected['mode']))
    test_name=args.run_name+'_test'
    if not (ROOT/'outputs'/test_name/'completed.json').exists():
        subprocess.run([sys.executable,str(ROOT/'eval/test_selected_structure.py'),'--selection-run',args.run_name,
                        '--run-name',test_name,'--device',str(device)],cwd=ROOT,check=True)
    dump(out/'completed.json',dict(complete=True,selected=selected,test_run=test_name,historical_protocol=True))
    dump(out/'status.json',dict(phase='complete',test_run=test_name,selected=selected['mode']))


if __name__=='__main__':main()
