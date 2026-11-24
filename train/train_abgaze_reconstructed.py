"""Scratch training on reconstructed data; no historical checkpoints or test reads."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from orientation_interface_refiner import OrientationInterfaceRefiner
from abgaze_reconstructed_objective import objective


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def save(path,value):
    temp=path.with_suffix('.tmp');torch.save(value,temp);temp.replace(path)


def move(x,device):
    return {k:v.to(device) for k,v in x.items()}


def epoch(model,rows,device,fape,sequence_weight,optimizer=None,batch_size=16,seed=0):
    model.train(optimizer is not None)
    order=list(range(len(rows)))
    if optimizer is not None:
        random.Random(seed).shuffle(order)
    total={}
    with torch.set_grad_enabled(optimizer is not None):
        for start in range(0,len(order),batch_size):
            chunk=order[start:start+batch_size]
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            for index in chunk:
                sid,x,y=rows[index]; x,y=move(x,device),move(y,device)
                loss,metrics=objective(model(x),x,y,fape,sequence_weight)
                if not torch.isfinite(loss) or not all(math.isfinite(v) for v in metrics.values()):
                    raise RuntimeError(f'Nonfinite objective: {sid}')
                if optimizer is not None:
                    (loss/len(chunk)).backward()
                for key,value in metrics.items():
                    total[key]=total.get(key,0.)+value
            if optimizer is not None:
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):
                    raise RuntimeError('Nonfinite gradient')
                optimizer.step()
    return {k:v/len(rows) for k,v in total.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--arm',choices=['pose','control'],default='pose')
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=16,help='gradient accumulation over variable-size complexes')
    p.add_argument('--width',type=int,default=64)
    p.add_argument('--layers',type=int,default=4)
    p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--final-lr',type=float,default=1e-4)
    p.add_argument('--fape-weight',type=float,default=.5)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if min(args.epochs,args.batch_size,args.width,args.layers)<=0 or not 0<args.final_lr<=args.lr:
        p.error('Invalid training sizes or learning rates')
    if args.fape_weight<0:
        p.error('FAPE weight must be nonnegative')
    torch.set_num_threads(1);torch.manual_seed(args.seed);random.seed(args.seed)
    root=args.dataset.resolve(); tensor_dir=root/'tensors'; out=args.output.resolve()
    manifest=json.loads((tensor_dir/'manifest.json').read_text())
    assert sha(root/'lock.json')==manifest['source_lock_sha256'], 'Dataset lock changed'
    for name,digest in json.loads((root/'lock.json').read_text()).items():
        assert sha(root/name)==digest, f'Changed split metadata: {name}'
    if any(r['failed'] for r in manifest['reports'].values()):
        raise RuntimeError('Conversion coverage is incomplete; fix data instead of silently dropping rows')
    data={}
    for split in ('train','valid'):
        path=tensor_dir/f'{split}.pt'
        assert sha(path)==manifest['files'][path.name], f'Changed {split} tensors'
        rows=torch.load(path,map_location='cpu',weights_only=False)
        ids=[r[0] for r in rows]
        expected=[json.loads(l)['pdb'] for l in (root/'splits'/f'{split}.json').read_text().splitlines()]
        assert ids==expected and len(ids)==len(set(ids)), 'Missing, duplicate, or reordered samples'
        data[split]=rows[:(4 if split=='train' else 2)] if args.smoke else rows
        if not data[split]:
            raise RuntimeError('Empty split')
    config=dict(width=args.width,layers=args.layers,use_orientation=args.arm=='pose')
    dependencies=['abgaze_reconstructed_objective.py','orientation_interface_refiner.py','native_sequence_coupled_refiner.py',
        'relation_coupled_refiner.py','structure_refinement_ablation.py','v21_atomic_refiner.py',
        'v22_joint_refiner.py','link_interface_refiner.py','train/train_v21_atomic.py',
        'train/train_frame_relation_refiner.py','dygf.py']
    plan=dict(protocol=manifest['protocol'],task=manifest['task'],model_config=config,
        epochs=1 if args.smoke else args.epochs,batch_size=args.batch_size,seed=args.seed,
        lr=args.lr,final_lr=args.final_lr,fape_weight=args.fape_weight,arm=args.arm,
        sequence_warmup_epochs=10,optimizer='Adam; exponential LR; global gradient clip 1',
        selection='minimum validation total loss, sequence weight fixed at 1',
        initialization='random weights; anchor-line input; no supervised ancestors',test_inference=False,
        smoke=args.smoke,counts={k:len(v) for k,v in data.items()},
        manifest_sha256=sha(tensor_dir/'manifest.json'),
        code_sha256={str(f.relative_to(ROOT)):sha(f) for f in [Path(__file__),*[ROOT/d for d in dependencies]]})
    if args.resume:
        assert json.loads((out/'plan.json').read_text())==plan,'Resume code/data/config mismatch'
    else:
        out.mkdir(parents=True,exist_ok=False);dump(out/'plan.json',plan)
    device=torch.device(args.device)
    model=OrientationInterfaceRefiner(**config).to(device)
    if args.arm=='control':
        for param in model.orientation_layers.parameters():
            param.requires_grad_(False)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=args.lr)
    start,best=0,float('inf')
    if args.resume:
        state=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        assert state['manifest_sha256']==plan['manifest_sha256']
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        start,best=state['epoch'],state['best_loss']
        torch.set_rng_state(state['rng'])
        if device.type=='cuda' and state['cuda_rng'] is not None:
            torch.cuda.set_rng_state(state['cuda_rng'],device)
    for e in range(start+1,plan['epochs']+1):
        tick=time.monotonic()
        lr=args.lr*(args.final_lr/args.lr)**((e-1)/max(plan['epochs']-1,1))
        for group in optimizer.param_groups:
            group['lr']=lr
        dump(out/'status.json',dict(phase='training',epoch=e,test_inference=False))
        training=epoch(model,data['train'],device,args.fape_weight,min(e/10,1),optimizer,args.batch_size,args.seed+e)
        validation=epoch(model,data['valid'],device,args.fape_weight,1.)
        improved=validation['loss']<best;best=min(best,validation['loss'])
        state=dict(model=model.state_dict(),model_config=config,optimizer=optimizer.state_dict(),epoch=e,
            best_loss=best,validation=validation,manifest_sha256=plan['manifest_sha256'],rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(device) if device.type=='cuda' else None)
        save(out/'last.pt',state)
        if improved:
            save(out/'best.pt',state)
        record=dict(epoch=e,lr=lr,train=training,validation=validation,seconds=time.monotonic()-tick)
        dump(out/f'epoch_{e:03d}.json',record);print(json.dumps(record),flush=True)
    dump(out/'status.json',dict(phase='smoke_complete' if args.smoke else 'complete',epoch=plan['epochs'],
                               test_inference=False,best_validation_loss=best))


if __name__=='__main__':
    main()
