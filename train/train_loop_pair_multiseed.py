"""Multi-seed continuation of the fixed full model; structure-priority validation selection."""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from persistent_loop_pair_refiner import PersistentLoopPairRefiner
from binding_interface_refiner import BindingInterfaceRefiner
from train.compare_two_stage_binding import FeatureRows
from train.train_reconstructed_two_stage import epoch
from train.train_abgaze_reconstructed import sha,dump,save,move


def eligible(m,b):
    return (m['rmsd']<=b['rmsd']+.03 and m['contact_f1']>=b['contact_f1']-.01
            and m['cn_mae']<=b['cn_mae']+.01 and m['clash']<=b['clash']*1.1+1e-5
            and m['aar']>=b['aar']-.015)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:1')
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--smoke',action='store_true')
    args=p.parse_args();torch.set_num_threads(1);torch.manual_seed(args.seed)
    device=torch.device(args.device)
    assert device.type=='cuda', 'GPU required'
    if device.type=='cuda':torch.cuda.set_device(device)
    parent=ROOT/'outputs/two_stage_binding60_gpu_v1';pp=json.loads((parent/'plan.json').read_text())
    assert json.loads((parent/'status.json').read_text())['phase']=='complete'
    for name,digest in pp['code_sha256'].items():assert sha(ROOT/name)==digest,name
    assert sha(ROOT/'train/continue_two_stage_binding60_gpu.py')==pp['continuation_script_sha256']
    features=ROOT/'data/abgaze_binding_features_finalized_v1'
    assert sha(features/'plan.json')==pp['feature_plan_sha256']
    for name,digest in json.loads((features/'plan.json').read_text())['code_sha256'].items():assert sha(ROOT/name)==digest,name
    state=torch.load(parent/'last.pt',map_location='cpu',weights_only=False)
    assert state['epoch']==60 and state['plan_sha256']==sha(parent/'plan.json')
    data={};ids={}
    for split in ('train','valid'):
        path=ROOT/'outputs/reconstructed_two_stage_v1'/f'{split}_cache.pt';assert sha(path)==pp['cache_sha256'][split]
        rows=torch.load(path,map_location='cpu',weights_only=False)
        assert [r[0] for r in rows]==pp['ids'][split]
        if args.smoke:rows=rows[:2]
        ids[split]=[r[0] for r in rows]
        data[split]=FeatureRows(rows,features,split,pp['feature_sha256'])
    config=state['model_config'];model=PersistentLoopPairRefiner(**config).to(device);model.load_binding(state['model'])
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);shutil.copyfile(parent/'last.pt',out/'source.pt')
    dependencies=set(pp['code_sha256'])|{'persistent_loop_pair_refiner.py','train/train_loop_pair_multiseed.py'}
    plan=dict(model_class='PersistentLoopPairRefiner',model_config=config,source_sha256=sha(out/'source.pt'),
        source_epoch=60,parent_plan_sha256=sha(parent/'plan.json'),seed=args.seed,epochs=1 if args.smoke else args.epochs,
        smoke=args.smoke,device=str(device),ids=ids,counts={s:len(r) for s,r in data.items()},
        cache_sha256=pp['cache_sha256'],feature_plan_sha256=pp['feature_plan_sha256'],
        feature_sha256=pp['feature_sha256'],generator_sha256=pp['generator_sha256'],
        architecture='persistent H3-H3 pair state, relative geometry and predicted sequence probabilities, triangle multiplicative update; interleaved at every existing layer',
        initialization='zero hidden/rigid readouts preserve original model; all refiner parameters trainable; generator frozen',
        objective='UNCHANGED geometry+0.5 interface FAPE+1 CE',
        optimizer='fresh Adam; old parameters1e-5->1e-6; new module1e-4->1e-5; accumulate16; clip1',
        selection='lowest validation aligned CA RMSD under guards, epoch0 eligible; no early stop',
        guards=dict(raw_rmsd_max_increase=.03,contact_f1_max_drop=.01,cn_mae_max_increase=.01,
            clash_multiplier=1.1,clash_allowance=1e-5,aar_max_drop=.015),
        limitations=['continuation seeds share a pretrained source, not independent full retraining','No new torsion loss, denoising, or sampling'],
        test_access=False,code_sha256={f:sha(ROOT/f) for f in sorted(dependencies)})
    dump(out/'plan.json',plan)
    base=BindingInterfaceRefiner(**config).to(device);base.load_state_dict(state['model']);base.eval();model.eval()
    with torch.no_grad():
        _,x,_,_=data['valid'][0];x=move(x,device)
        a=base(x);b=model(x);c=model(x)
        diffs={k:float((a[k]-b[k]).abs().max()) for k in ('backbone','sequence_logits')}
        assert all(v<1e-4 for v in diffs.values()),diffs
        assert '_loop_pair_state' not in x
        assert torch.allclose(b['backbone'],c['backbone'],atol=1e-4,rtol=1e-5)
        # Backpropagate once with readouts active to test all pair-state paths.
        for layer in model.loop_pair_layers:torch.nn.init.normal_(layer.node.weight,std=1e-3)
    testout=model(x);testout['backbone'].square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    for layer in model.loop_pair_layers:torch.nn.init.zeros_(layer.node.weight)
    model.zero_grad(set_to_none=True)
    dump(out/'initialization_audit.json',dict(max_difference=diffs,input_state_unmodified=True,repeated_forward_checked=True,finite_gradient=True))
    del base,state
    if device.type=='cuda':torch.cuda.empty_cache()
    old=[];new=[]
    for name,param in model.named_parameters():(new if name.startswith('loop_pair_layers.') else old).append(param)
    optimizer=torch.optim.Adam([dict(params=old,lr=1e-5,initial_lr=1e-5),dict(params=new,lr=1e-4,initial_lr=1e-4)])
    dump(out/'status.json',dict(phase='initial_validation',epoch=0,test_access=False))
    baseline=epoch(model,data['valid'],device,'refiner');dump(out/'initial_validation.json',baseline)
    best=baseline['aligned_rmsd'];best_epoch=0
    def checkpoint(e,m):
        return dict(model=model.state_dict(),model_config=config,epoch=e,validation=m,source_sha256=plan['source_sha256'],
            best_epoch=best_epoch,best_aligned_rmsd=best,optimizer=optimizer.state_dict(),plan_sha256=sha(out/'plan.json'))
    save(out/'best.pt',checkpoint(0,baseline))
    for e in range(1,plan['epochs']+1):
        tick=time.monotonic();torch.manual_seed(args.seed+e)
        for group in optimizer.param_groups:group['lr']=group['initial_lr']*.1**((e-1)/max(plan['epochs']-1,1))
        status=dict(phase='training',epoch=e,epochs=plan['epochs'],test_access=False);dump(out/'status.json',status)
        tr=epoch(model,data['train'],device,'refiner',optimizer,args.seed+e,(out/'status.json',status))
        val=epoch(model,data['valid'],device,'refiner');allowed=eligible(val,baseline)
        improved=allowed and val['aligned_rmsd']<best
        if improved:best,best_epoch=val['aligned_rmsd'],e
        save(out/'last.pt',checkpoint(e,val))
        if improved:save(out/'best.pt',checkpoint(e,val))
        record=dict(epoch=e,train=tr,validation=val,guard_passed=allowed,best_epoch=best_epoch,seconds=time.monotonic()-tick)
        dump(out/f'epoch_{e:03d}.json',record);print(json.dumps(record),flush=True)
    dump(out/'status.json',dict(phase='smoke_complete' if args.smoke else 'complete',best_epoch=best_epoch,
        best_aligned_rmsd=best,test_access=False))


if __name__=='__main__':main()

