"""Validation-only scratch generator -> refiner on the current locked split.

Historical generator architecture, current missing-atom-aware objective.
Not an exact historical recipe reproduction or a causal single-factor ablation.
No historical weights, external LM, surface features, or test tensors are used.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from frame_relation_refiner import FrameRelationGraph, FrameRelationRefiner
from orientation_interface_refiner import OrientationInterfaceRefiner
from abgaze_reconstructed_objective import objective
from train.train_abgaze_reconstructed import sha, dump, save, move
from eval.test_abgaze_reconstructed_ours import aligned_rmsd


def predict(model,x,graph,device,stage):
    if stage=='generator':
        raw=model(FrameRelationGraph(**graph).to(device))
        # Geometry-only generator: its untrained sequence head is not a prior.
        return dict(backbone=raw['h3_backbone'],sequence_logits=torch.zeros_like(raw['sequence_logits']))
    return model(x)


def epoch(model,rows,device,stage,optimizer=None,seed=0,status=None):
    model.train(optimizer is not None)
    order=list(range(len(rows)))
    if optimizer is not None:random.Random(seed).shuffle(order)
    totals={};seen=0
    with torch.set_grad_enabled(optimizer is not None):
        for start in range(0,len(order),16):
            chunk=order[start:start+16]
            if optimizer is not None:optimizer.zero_grad(set_to_none=True)
            for index in chunk:
                sid,xcpu,ycpu,graph=rows[index]
                x,y=move(xcpu,device),move(ycpu,device)
                out=predict(model,x,graph,device,stage)
                loss,m=objective(out,x,y,.5,0. if stage=='generator' else 1.)
                if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss {stage} {sid}')
                if optimizer is not None:(loss/len(chunk)).backward()
                else:
                    mask=y['mask'][:,1]
                    m['aligned_rmsd']=aligned_rmsd(out['backbone'][mask,1].detach().cpu(),y['bb'][mask,1].cpu())
                if not all(math.isfinite(v) for v in m.values()):raise RuntimeError(f'Nonfinite metric {sid}')
                for k,v in m.items():totals[k]=totals.get(k,0.)+v
                seen+=1
            if optimizer is not None:
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient')
                optimizer.step()
            if status and (seen%400==0 or seen==len(rows)):
                path,record=status
                dump(path,dict(**record,seen=seen,total=len(rows)))
                print(json.dumps(dict(**record,seen=seen,loss=totals['loss']/seen)),flush=True)
    return {k:v/len(rows) for k,v in totals.items()}


def fit(model,config,data,out,stage,epochs,device,plan_sha,seed):
    out.mkdir(exist_ok=False)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    best=float('inf');best_epoch=0
    initial=epoch(model,data['valid'],device,stage)
    dump(out/'initial_validation.json',initial)
    for e in range(1,epochs+1):
        tick=time.monotonic()
        lr=1e-4*(.1**((e-1)/max(epochs-1,1)))
        for group in optimizer.param_groups:group['lr']=lr
        status=dict(stage=stage,phase='training',epoch=e,epochs=epochs,test_access=False)
        dump(out.parent/'status.json',status)
        tr=epoch(model,data['train'],device,stage,optimizer,seed+e,(out.parent/'status.json',status))
        dump(out.parent/'status.json',dict(stage=stage,phase='validation',epoch=e,epochs=epochs,test_access=False))
        val=epoch(model,data['valid'],device,stage)
        # Same structural selection metric in both stages, not test metrics or AAR.
        value=val['geometry_loss']+.5*val['interface_fape']
        improved=value<best
        if improved:best,best_epoch=value,e
        state=dict(model=model.state_dict(),model_config=config,epoch=e,stage=stage,
            validation=val,best_score=best,best_epoch=best_epoch,optimizer=optimizer.state_dict(),
            plan_sha256=plan_sha,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        save(out/'last.pt',state)
        if improved:save(out/'best.pt',state)
        record=dict(epoch=e,best_epoch=best_epoch,train=tr,validation=val,selection_score=value,
            lr=lr,seconds=time.monotonic()-tick)
        dump(out/f'epoch_{e:03d}.json',record)
        print(json.dumps(dict(stage=stage,**record)),flush=True)
    return torch.load(out/'best.pt',map_location='cpu',weights_only=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--generator-epochs',type=int,default=100)
    p.add_argument('--refiner-epochs',type=int,default=30)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args();torch.set_num_threads(1)
    if min(args.generator_epochs,args.refiner_epochs)<1:raise ValueError('Positive epochs required')
    device=torch.device(args.device)
    if device.type=='cuda':torch.cuda.set_device(device)
    torch.manual_seed(2026);random.seed(2026)
    dataset=ROOT/'data/abgaze_reconstructed_v1'
    manifest=json.loads((dataset/'tensors/manifest.json').read_text())
    assert sha(dataset/'lock.json')==manifest['source_lock_sha256']
    for name,digest in json.loads((dataset/'lock.json').read_text()).items():assert sha(dataset/name)==digest
    graphdir=ROOT/'data/raad_abgaze_reconstructed_v1'
    graphmanifest=json.loads((graphdir/'manifest.json').read_text())
    assert graphmanifest['manifest_sha256']==sha(dataset/'tensors/manifest.json')
    data={}
    for split in ('train','valid'):
        path=dataset/f'tensors/{split}.pt';assert sha(path)==manifest['files'][path.name]
        assert sha(graphdir/f'{split}.pt')==graphmanifest[split]['sha256']
        source=torch.load(graphdir/f'{split}.pt',map_location='cpu',weights_only=False)
        graphs={item['sample_id']:g for item,g in source}
        rows=torch.load(path,map_location='cpu',weights_only=False)
        expected=[json.loads(l)['pdb'] for l in (dataset/f'splits/{split}.json').read_text().splitlines()]
        assert [r[0] for r in rows]==expected and set(graphs)==set(expected)
        if args.smoke:rows=rows[:4 if split=='train' else 2]
        data[split]=[(sid,x,y,graphs[sid]) for sid,x,y in rows]
    genconfig=dict(hidden_dim=128,layers=6,coordinate_step_scale=4.,sequence_adapter_dim=0,
        dynamic_relation_layers=2,dynamic_relation_dim=64,dynamic_max_displacement=1.5,
        dynamic_prior_adjustment=2.,dynamic_candidate_base_strength=0.,dynamic_dense_interface=True)
    refconfig=dict(width=64,layers=4,use_orientation=True)
    epochs=dict(generator=1 if args.smoke else args.generator_epochs,refiner=1 if args.smoke else args.refiner_epochs)
    deps=[Path(__file__),ROOT/'frame_relation_refiner.py',ROOT/'relation_interface_refiner.py',
        ROOT/'abgaze_reconstructed_objective.py',ROOT/'orientation_interface_refiner.py',
        ROOT/'native_sequence_coupled_refiner.py',ROOT/'relation_coupled_refiner.py',
        ROOT/'structure_refinement_ablation.py',ROOT/'v21_atomic_refiner.py',ROOT/'v22_joint_refiner.py',
        ROOT/'link_interface_refiner.py',ROOT/'dygf.py',ROOT/'train/train_abgaze_reconstructed.py',
        ROOT/'eval/test_abgaze_reconstructed_ours.py']
    plan=dict(protocol='current reconstructed split; validation-only two-stage diagnostic',seed=2026,
        smoke=args.smoke,epochs=epochs,early_stopping=False,device=str(device),
        counts={s:len(rows) for s,rows in data.items()},ids={s:[r[0] for r in rows] for s,rows in data.items()},
        generator_config=genconfig,refiner_config=refconfig,
        manifest_sha256=sha(dataset/'tensors/manifest.json'),graph_manifest_sha256=sha(graphdir/'manifest.json'),
        initialization='both models random; no historical checkpoint/LM/prior; no surface branch',
        objective='current missing-atom-aware geometry +0.5 FAPE; stage1 CE0; stage2 CE1',
        selection='minimum validation geometry_loss +0.5 interface_fape in both stages',
        optimizer='Adam 1e-4 ->1e-5 exponential; accumulate16; clip1',
        cache='frozen validation-selected generator; training predictions in-sample, validation held out; zero sequence logits',
        limitations=['Not exact historical recipe','Architecture, objective and compute differ from single-stage run; not causal attribution',
                    'Training-cache in-sample prediction quality may differ from validation; report both','No test evaluation scheduled'],
        code_sha256={str(f.relative_to(ROOT)):sha(f) for f in deps})
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);dump(out/'plan.json',plan)
    model=FrameRelationRefiner(**genconfig).to(device)
    # Labels are never passed to forward. CUDA graph reductions may vary slightly.
    model.eval()
    with torch.no_grad():
        sid,x,y,g=data['valid'][0]
        a=predict(model,move(x,device),g,device,'generator')
        b=predict(model,move(x,device),g,device,'generator')
        assert torch.isfinite(a['backbone']).all() and torch.isfinite(b['backbone']).all()
        difference=float((a['backbone']-b['backbone']).abs().max())
        assert difference<1e-3, f'Repeated forward differs by {difference} A'
        dump(out/'forward_audit.json',dict(repeated_forward_max_difference_A=difference,
             tolerance_A=1e-3,labels_passed_to_forward=False))
    bestgen=fit(model,genconfig,data,out/'generator','generator',epochs['generator'],device,sha(out/'plan.json'),2026)
    model.load_state_dict(bestgen['model']);model.eval()
    refined={};cache_metrics={}
    dump(out/'status.json',dict(phase='caching',stage='generator',test_access=False))
    with torch.no_grad():
        for split,rows in data.items():
            refined[split]=[];initial_errors=[]
            for sid,x,y,g in rows:
                pred=predict(model,move(x,device),g,device,'generator')
                newx={**x,'base_bb':pred['backbone'].cpu(),'base_logits':torch.zeros_like(x['base_logits'])}
                v=y['mask'][:,1]
                initial_errors.append(float((newx['base_bb'][v,1]-y['bb'][v,1]).square().sum(-1).mean().sqrt()))
                refined[split].append((sid,newx,y,None))
            save(out/f'{split}_cache.pt',refined[split])
            cache_metrics[split]=dict(n=len(rows),input_frame_ca_rmsd=sum(initial_errors)/len(rows))
    dump(out/'cache_audit.json',dict(generator_epoch=bestgen['epoch'],generator_sha256=sha(out/'generator/best.pt'),
        metrics=cache_metrics,files={s:sha(out/f'{s}_cache.pt') for s in refined},test_access=False))
    del model,bestgen,data
    if device.type=='cuda':torch.cuda.empty_cache()
    torch.manual_seed(2026)
    model=OrientationInterfaceRefiner(**refconfig).to(device)
    bestref=fit(model,refconfig,refined,out/'refiner','refiner',epochs['refiner'],device,sha(out/'plan.json'),2026)
    dump(out/'status.json',dict(phase='smoke_complete' if args.smoke else 'complete',test_access=False,
        refiner_best_epoch=bestref['epoch'],validation=bestref['validation']))


if __name__=='__main__':main()
