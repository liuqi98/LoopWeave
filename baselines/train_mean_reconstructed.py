"""Train MEAN on current reconstructed 3269/340 split; never test.

Native architecture/loss, standardized context and masked global centroids.
This is a shared-data reproduction, not a claim of paper-exact reproduction.
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'external/chimera-bench/baselines/mean'
sys.path.insert(0,str(ROOT/'.raad_dependencies'))
sys.path.insert(0,str(CODE))
from data import VOCAB
from models.MCAttGNN import EfficientMCAttModel


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def batch(items,device):
    lengths=[len(r['S']) for r in items]
    return dict(X=torch.cat([r['X'] for r in items]).to(device),
                S=torch.cat([r['S'] for r in items]).to(device),L=[r['L'] for r in items],
                offsets=torch.tensor([0,*np.cumsum(lengths).tolist()],device=device))


def prediction_audit(model,item,device):
    altered={**item,'X':item['X'].clone(),'S':item['S'].clone()}
    start,end=item['L'].index('3'),item['L'].rindex('3')
    altered['X'][start:end+1]+=500
    altered['S'][start:end+1]=VOCAB.symbol_to_idx('A')
    model.eval()
    with torch.no_grad():
        torch.manual_seed(73);a=model.infer(batch([item],device),device)
        torch.manual_seed(73);b=model.infer(batch([altered],device),device)
    assert a[1]==b[1] and np.allclose(a[2][0],b[2][0]), 'Native H3 leakage'
    assert np.isfinite(a[2][0]).all()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cpu');parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--epochs',type=int,default=120);parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--run-name',required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device=torch.device(args.device)
    assert device.type=='cuda', 'GPU required'
    cache=ROOT/'data/mean_reconstructed_v1'
    manifest=json.loads((cache/'plan.json').read_text());data={}
    original=json.loads((ROOT/'outputs/persistent_loop_pair_v1/plan.json').read_text())
    for split in ('train','valid'):
        assert sha(cache/f'{split}.pt')==manifest['splits'][split]['sha256']
        rows=torch.load(cache/f'{split}.pt',map_location='cpu',weights_only=False)
        assert [r['sample_id'] for r in rows]==original['ids'][split]
        for row in rows:row['S']=torch.tensor([VOCAB.symbol_to_idx(s) for s in row['symbols']])
        data[split]=rows[:8] if args.smoke else rows
    out=ROOT/'outputs'/args.run_name;out.mkdir(exist_ok=False)
    config=dict(embed_size=64,hidden_size=128,n_channel=4,n_edge_feats=1,n_layers=3,
                cdr_type='3',alpha=.8,n_iter=3)
    plan=dict(method='MEAN',config=config,seed=args.seed,epochs=1 if args.smoke else args.epochs,
              smoke=args.smoke,test_access=False,benchmark_lock_sha256=sha(cache/'plan.json'),
              training_ids={s:[r['sample_id'] for r in rows] for s,rows in data.items()},
              selection='minimum validation native MEAN CE + 0.8 Huber',
              optimizer='Adam lr1e-3; ExponentialLR gamma.95; clip1; batch8; no early stop',
              task='matched-context retrospective benchmark; not paper-exact split or full-antigen native pipeline',
              source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),
                    CODE/'models/MCAttGNN/mc_att_model.py',CODE/'models/MCAttGNN/mc_egnn.py',CODE/'data/pdb_utils.py']})
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    model=EfficientMCAttModel(**config).to(device)
    prediction_audit(model,data['valid'][0],device)
    print('native_H3_perturbation_audit: passed',flush=True)
    torch.manual_seed(args.seed)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=.95)
    best=float('inf');best_epoch=0
    for epoch in range(1,plan['epochs']+1):
        (out/'status.json').write_text(json.dumps(dict(phase='training',epoch=epoch,epochs=plan['epochs'],test_access=False)))
        tick=time.monotonic();model.train();order=list(range(len(data['train'])));random.shuffle(order)
        total=0.
        for step in range(0,len(order),8):
            rows=[data['train'][i] for i in order[step:step+8]];x=batch(rows,device)
            loss,_,_=model(**x)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradients')
            optimizer.step();total+=float(loss.detach())*len(rows)
        model.eval();valid=0.
        with torch.no_grad():
            for step in range(0,len(data['valid']),8):
                rows=data['valid'][step:step+8];loss,_,_=model(**batch(rows,device))
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite validation loss')
                valid+=float(loss)*len(rows)
        valid/=len(data['valid']);scheduler.step()
        improved=valid<best
        if improved:best=valid;best_epoch=epoch
        state=dict(model=model.state_dict(),config=config,epoch=epoch,seed=args.seed,best_epoch=best_epoch,
                   best_loss=best,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                   benchmark_lock_sha256=sha(cache/'plan.json'),plan_sha256=sha(out/'plan.json'))
        for name in (['best','last'] if improved else ['last']):
            tmp=out/f'{name}.tmp';torch.save(state,tmp);tmp.replace(out/f'{name}.pt')
        record=dict(epoch=epoch,best_epoch=best_epoch,train_loss=total/len(data['train']),
                    val_loss=valid,seconds=time.monotonic()-tick,test_access=False)
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    (out/'status.json').write_text(json.dumps(dict(phase='smoke_complete' if args.smoke else 'complete',best_epoch=best_epoch,test_access=False)))
    (out/'completed.json').write_text(json.dumps(dict(training='complete',smoke_only=args.smoke,
         test_access=False,seed=args.seed,best_epoch=best_epoch,benchmark_lock_sha256=sha(cache/'plan.json'))))


if __name__=='__main__':main()

