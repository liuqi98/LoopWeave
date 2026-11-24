#!/usr/bin/env python3
"""Train v21 on frozen v18 predictions; validation-only model selection."""
import argparse
import csv
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from frame_relation_refiner import FrameRelationGraph,FrameRelationRefiner,build_frame_relation_graph
from v21_atomic_refiner import AtomicPairRefiner,context_inputs
from train.train_frame_relation_refiner import kabsch_align


def move(x,device):
    return {k:v.to(device) for k,v in x.items()}


def cache(split,base,device,smoke):
    # Labels stay outside the model input dictionary.
    refs=torch.load(ROOT/f'data/raad_protocol_frame_v1/{split}_targets.pt',weights_only=False)
    if split=='test':
        inputs=torch.load(ROOT/'data/raad_protocol_frame_v1/test_inputs.pt',weights_only=False)
        with (ROOT/'data/raad_protocol_frame_v1/test_initial.csv').open() as f:
            rows={r['sample_id']:r for r in csv.DictReader(f)}
        graphs={x['sample_id']:build_frame_relation_graph(x,rows[x['sample_id']]) for x in inputs}
    else:
        graphs={x['sample_id']:FrameRelationGraph(**x['graph']) for x in
                torch.load(ROOT/f'data/raad_protocol_frame_v1/{split}_graphs.pt',weights_only=False)}
    if smoke: refs=refs[:8]
    result=[]
    with torch.no_grad():
        for i,item in enumerate(refs):
            graph=graphs[item['sample_id']].to(device)
            pred=base(graph)
            x=context_inputs(item,graph,pred)
            if i==0:
                altered=dict(item)
                for key in ('cdr3_backbone','native_h3_ca'):
                    altered[key]=torch.as_tensor(item[key])+999
                altered['native_h3_sequence']='A'*len(item['native_h3_sequence'])
                check=context_inputs(altered,graph,pred)
                assert all(torch.equal(x[k],check[k]) for k in x)
            labels=dict(bb=torch.as_tensor(item['cdr3_backbone']).float(),
                        mask=torch.as_tensor(item['cdr3_backbone_mask']).bool())
            result.append((item['sample_id'],x,labels))
            if (i+1)%500==0:print(f'cached {split} {i+1}',flush=True)
    print(f'cached {split}: {len(result)}; H3-label perturbation audit passed',flush=True)
    return result


def objective(output,x,y):
    bb=output['backbone']; ca=bb[:,1]; target=y['bb'][:,1]
    atomic=F.smooth_l1_loss(bb[y['mask']],y['bb'][y['mask']])
    distances=torch.cdist(ca,x['context_bb'][x['antigen_mask'],1])
    truth=torch.cdist(target,x['context_bb'][x['antigen_mask'],1])
    contacts=truth<=8
    near=truth<=12
    interface=F.smooth_l1_loss(distances[near],truth[near]) if near.any() else atomic*0
    all_distance=F.smooth_l1_loss(distances.clamp_max(16),truth.clamp_max(16))
    soft=torch.sigmoid((8-distances)/.75)
    f1soft=(2*(soft*contacts).sum()+1)/(soft.sum()+contacts.sum()+1)
    chain=torch.cat([x['anchors'][:1,1],ca,x['anchors'][1:,1]])
    native_chain=torch.cat([x['anchors'][:1,1],target,x['anchors'][1:,1]])
    bond=F.smooth_l1_loss(torch.diff(chain,dim=0).norm(dim=-1),
                         torch.diff(native_chain,dim=0).norm(dim=-1))
    shape=F.smooth_l1_loss(torch.cdist(ca,ca),torch.cdist(target,target))
    # Retain chemically sensible within-residue distances while allowing all
    # four backbone atoms to move independently.
    intra=F.smooth_l1_loss(torch.cdist(bb,bb)[y['mask'][:,:,None]&y['mask'][:,None,:]],
                          torch.cdist(y['bb'],y['bb'])[y['mask'][:,:,None]&y['mask'][:,None,:]])
    loss=atomic+.5*interface+.25*all_distance+.5*(1-f1soft)+.2*bond+.1*shape+.2*intra
    with torch.no_grad():
        aligned=kabsch_align(ca,target)
        rmsd=(ca-target).square().sum(-1).mean().sqrt()
        ar=(aligned-target).square().sum(-1).mean().sqrt()
        pred=distances<=8
        f1=2*(pred&contacts).sum()/(pred.sum()+contacts.sum()).clamp_min(1)
    return loss,dict(loss=float(loss.detach()),rmsd=float(rmsd),aligned_rmsd=float(ar),contact_f1=float(f1))


def epoch(model,data,device,optimizer=None):
    model.train(optimizer is not None)
    order=list(range(len(data)))
    if optimizer:random.shuffle(order)
    totals={}
    with torch.set_grad_enabled(optimizer is not None):
        for idx in order:
            _,x,y=data[idx];x=move(x,device);y=move(y,device)
            loss,metrics=objective(model(x),x,y)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite v21 loss')
            if optimizer:
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            for k,v in metrics.items():totals[k]=totals.get(k,0)+v
    return {k:v/len(data) for k,v in totals.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true')
    p.add_argument('--run-name',default=None)
    p.add_argument('--resume-dir',type=Path,default=None)
    p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--no-early-stop',action='store_true')
    p.add_argument('--final-lr',type=float,default=None)
    args=p.parse_args()
    if args.final_lr is not None and not 0 < args.final_lr <= 1e-4:
        p.error('--final-lr must be in (0, 1e-4]')
    resume=None; previous_best=None
    if args.resume_dir:
        if not (args.resume_dir/'completed.json').exists():
            raise RuntimeError('Source run must finish before continuation')
        resume=torch.load(args.resume_dir/'last.pt',map_location='cpu',weights_only=False)
        previous_best=torch.load(args.resume_dir/'best.pt',map_location='cpu',weights_only=False)
        if args.epochs <= resume['epoch']:p.error('--epochs must exceed resumed epoch')
    torch.set_num_threads(1);torch.manual_seed(2026);random.seed(2026)
    device=torch.device('cuda')
    out=ROOT/'outputs'/(args.run_name or ('raad_protocol_v21_atomic'+('_smoke' if args.smoke else '')))
    out.mkdir(parents=True,exist_ok=True)
    if (out/'plan.json').exists():raise RuntimeError('Refusing existing experiment overwrite')
    checkpoint=ROOT/'checkpoints/raad_protocol_v18_staged/geometry.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    base=FrameRelationRefiner(**state['model_config']).to(device)
    base.load_state_dict(state['model']);base.eval()
    for param in base.parameters():param.requires_grad_(False)
    config=dict(width=64,layers=4)
    (out/'plan.json').write_text(json.dumps(dict(base=str(checkpoint),model_config=config,
        seed=2026,max_epochs=args.epochs,min_epochs=40,patience=25,lr=1e-4,
        resume_dir=str(args.resume_dir) if args.resume_dir else None,
        resume_epoch=resume['epoch'] if resume else None,
        no_early_stop=args.no_early_stop,final_lr=args.final_lr,
        schedule='cosine from restored learning rate to final_lr over continuation epochs',
        selection='validation contact F1 - 0.2 absolute RMSD - 0.2 aligned RMSD; epoch0 eligible',
        input='v18 predicted H3 backbone/logits; known context N CA C O masks; both anchor backbones',
        architecture='four-atom equivariant updates; persistent H3-context pair states; frozen sequence',
        protocol='same native RAAD split and retained context; K1; no native H3 inputs'),indent=2))
    train=cache('train',base,device,args.smoke);valid=cache('valid',base,device,args.smoke)
    model=AtomicPairRefiner(**config).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    best=-float('inf');best_epoch=0
    start_epoch=0
    if resume:
        if resume['model_config'] != config or resume['base_checkpoint'] != str(checkpoint):
            raise ValueError('Resume architecture/base mismatch')
        model.load_state_dict(resume['model']);optimizer.load_state_dict(resume['optimizer'])
        best=float(previous_best['score']);best_epoch=int(previous_best['epoch'])
        start_epoch=int(resume['epoch'])+1
        torch.save(previous_best,out/'best.pt')
        # Old checkpoints lack RNG state. Replay Python's deterministic sample
        # permutations to continue the original ordering, rather than repeat it.
        if 'python_rng' in resume:
            random.setstate(resume['python_rng'])
        else:
            for _ in range(resume['epoch']):
                order=list(range(len(train)));random.shuffle(order)
        print(json.dumps(dict(resumed_epoch=resume['epoch'],best_epoch=best_epoch,
                              optimizer_restored=True)),flush=True)
    end_epoch=(start_epoch if args.smoke and resume else 1) if args.smoke else args.epochs
    initial_lr=optimizer.param_groups[0]['lr']
    for e in range(start_epoch,end_epoch+1):
        if args.final_lr is not None:
            progress=(e-start_epoch)/max(1,end_epoch-start_epoch)
            lr=args.final_lr+.5*(initial_lr-args.final_lr)*(1+math.cos(math.pi*progress))
            for group in optimizer.param_groups:group['lr']=lr
        start=time.monotonic()
        training=epoch(model,train,device,optimizer) if e else None
        validation=epoch(model,valid,device)
        score=validation['contact_f1']-.2*validation['rmsd']-.2*validation['aligned_rmsd']
        state=dict(model=model.state_dict(),model_config=config,base_checkpoint=str(checkpoint),
                   epoch=e,validation=validation,score=score,optimizer=optimizer.state_dict(),
                   python_rng=random.getstate())
        if score>best:best=score;best_epoch=e;torch.save(state,out/'best.pt')
        torch.save(state,out/'last.pt')
        record=dict(epoch=e,best_epoch=best_epoch,train=training,validation=validation,
                    seconds=time.monotonic()-start,lr=optimizer.param_groups[0]['lr'])
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        if not args.no_early_stop and e>=40 and e-best_epoch>=25:break
    if not args.smoke:
        state=torch.load(out/'best.pt',map_location=device,weights_only=False)
        model.load_state_dict(state['model']);model.eval()
        test=cache('test',base,device,False)
        from relation_interface_refiner import AMINO_ACIDS
        rows=[]
        with torch.no_grad():
            for sid,x,y in test:
                pred=model(move(x,device))
                rows.append(dict(sample_id=sid,candidate=0,
                    generated_cdrh3=''.join(AMINO_ACIDS[i] for i in pred['sequence_logits'].argmax(-1).tolist()),
                    ca_coords=json.dumps(pred['backbone'][:,1].cpu().tolist())))
        generated=out/'native_test_k1.csv'
        with generated.open('w') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
        subprocess.run([sys.executable,str(ROOT/'eval/evaluate_rabd_generated.py'),
            '--targets',str(ROOT/'data/raad_protocol_frame_v1/test_targets.pt'),
            '--generated',str(generated),'--expected-k','1','--bootstrap','10000',
            '--output',str(out/'native_test_metrics.json')],cwd=ROOT,check=True)
    (out/'completed.json').write_text(json.dumps(dict(complete=True,smoke=args.smoke,
        best_epoch=best_epoch,last_epoch=e)))


if __name__=='__main__':main()
