"""Equal-budget continuation with/without binding features; validation only."""
import argparse
import json
import math
import sys
import time
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from binding_interface_refiner import BindingInterfaceRefiner
from orientation_interface_refiner import OrientationInterfaceRefiner
from train.train_reconstructed_two_stage import epoch
from train.train_abgaze_reconstructed import sha,dump,save,move


class FeatureRows:
    """Load one full-antigen feature tensor at a time, keeping RAM bounded."""
    def __init__(self,rows,directory,split,hashes):
        self.rows,self.directory,self.split,self.hashes=rows,directory,split,hashes
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        sid,x,y,g=self.rows[index]
        path=self.directory/self.split/sid/'features.pt'
        assert sha(path)==self.hashes[f'{self.split}/{sid}'],sid
        features=torch.load(path,map_location='cpu',weights_only=False)['features']
        assert not set(x)&set(features)
        return sid,{**x,**features},y,g


def score(metrics):return metrics['geometry_loss']+.5*metrics['interface_fape']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args();torch.set_num_threads(1)
    if args.epochs<1:raise ValueError('Positive epoch count required')
    device=torch.device(args.device)
    if device.type=='cuda':torch.cuda.set_device(device)
    torch.manual_seed(2026)
    parent=ROOT/'outputs/reconstructed_two_stage_v1'
    dataset=ROOT/'data/abgaze_reconstructed_v1'
    features=ROOT/'data/abgaze_binding_features_finalized_v1'
    assert json.loads((parent/'status.json').read_text())['phase']=='complete'
    parentplan=json.loads((parent/'plan.json').read_text())
    for name,digest in parentplan['code_sha256'].items():assert sha(ROOT/name)==digest,name
    assert sha(dataset/'tensors/manifest.json')==parentplan['manifest_sha256']
    featureplan=json.loads((features/'plan.json').read_text())
    assert featureplan['source_lock_sha256']==sha(dataset/'lock.json')
    for name,digest in json.loads((dataset/'lock.json').read_text()).items():assert sha(dataset/name)==digest,name
    for name,digest in featureplan['code_sha256'].items():assert sha(ROOT/name)==digest,name
    cache=json.loads((parent/'cache_audit.json').read_text())
    assert sha(parent/'generator/best.pt')==cache['generator_sha256']
    raw={};hashes={}
    for split in ('train','valid'):
        path=parent/f'{split}_cache.pt';assert sha(path)==cache['files'][split]
        rows=torch.load(path,map_location='cpu',weights_only=False)
        assert [r[0] for r in rows]==parentplan['ids'][split]
        coverage=json.loads((features/f'{split}_coverage.json').read_text())
        assert not coverage['failures'] and set(coverage['success'])==set(parentplan['ids'][split])
        if args.smoke:rows=rows[:2]
        for index,(sid,x,y,g) in enumerate(rows):
            path=features/split/sid/'features.pt';digest=sha(path)
            assert digest==coverage['success'][sid]['sha256'],sid
            item=torch.load(path,map_location='cpu',weights_only=False);meta=item['metadata']
            assert meta['antigen_only'] and not meta['native_h3_used'],sid
            assert meta['pdb_sha256']==sha(dataset/'pdb'/f'{sid}.pdb'),sid
            assert not set(x)&set(item['features']),sid
            hashes[f'{split}/{sid}']=digest
            if (index+1)%400==0:print(json.dumps(dict(phase='feature_audit',split=split,seen=index+1)),flush=True)
        raw[split]=rows
    sourcepath=parent/'refiner/best.pt'
    source=torch.load(sourcepath,map_location='cpu',weights_only=False)
    assert source['plan_sha256']==sha(parent/'plan.json')
    config=source['model_config']
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    deps=set(parentplan['code_sha256'])|{'binding_interface_refiner.py',str(Path(__file__).resolve().relative_to(ROOT))}
    plan=dict(arms=['control','binding'],seed=2026,epochs=1 if args.smoke else args.epochs,smoke=args.smoke,
        source_epoch=source['epoch'],source_sha256=sha(sourcepath),source_config=config,
        parent_plan_sha256=sha(parent/'plan.json'),generator_sha256=cache['generator_sha256'],
        cache_sha256=cache['files'],feature_plan_sha256=sha(features/'plan.json'),feature_sha256=hashes,
        counts={s:len(r) for s,r in raw.items()},ids={s:[r[0] for r in rows] for s,rows in raw.items()},
        manifest_sha256=parentplan['manifest_sha256'],code_sha256={f:sha(ROOT/f) for f in sorted(deps)},
        optimizer='both fresh Adam; lr1e-4 ->1e-5 exponential; accumulate16; clip1; same epoch sample order',
        selection='minimum validation geometry_loss +0.5 interface_fape; epoch0 eligible; no early stop',
        objective='unchanged missing-atom-aware geometry +0.5 FAPE +1 sequence CE, both arms',
        trainable='all refiner parameters; generator frozen via identical caches',
        features='existing antigen-only SES/electrostatics/chemical heavy-atom branch; zero residual initialization',
        limitations=['Single seed','Additional branch capacity differs','Preprocessing approximations inherited and hashed',
                    'Previously inspected test set; no test reads or evaluations in this run'])
    dump(out/'plan.json',plan)
    models={};optimizers={};best={};best_epochs={};data={}
    for arm,cls in [('control',OrientationInterfaceRefiner),('binding',BindingInterfaceRefiner)]:
        torch.manual_seed(2026)
        model=cls(**config).to(device)
        if arm=='binding':model.load_source(source['model'])
        else:model.load_state_dict(source['model'])
        models[arm]=model;optimizers[arm]=torch.optim.Adam(model.parameters(),lr=1e-4)
        data[arm]=raw if arm=='control' else {s:FeatureRows(r,features,s,hashes) for s,r in raw.items()}
        (out/arm).mkdir()
    # Exact source preservation up to floating-point reduction noise.
    for model in models.values():model.eval()
    with torch.no_grad():
        _,x,_,_=data['control']['valid'][0];_,xf,_,_=data['binding']['valid'][0]
        a=models['control'](move(x,device));b=models['binding'](move(xf,device))
        diffs={k:float((a[k]-b[k]).abs().max()) for k in ('backbone','sequence_logits')}
        assert all(math.isfinite(v) and v<1e-4 for v in diffs.values()),diffs
    dump(out/'initialization_audit.json',dict(max_differences=diffs,tolerance=1e-4))
    def checkpoint(arm,e,metrics):
        return dict(model=models[arm].state_dict(),model_config=config,arm=arm,epoch=e,
            validation=metrics,optimizer=optimizers[arm].state_dict(),plan_sha256=sha(out/'plan.json'),
            best_epoch=best_epochs[arm],best_score=best[arm],torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [])
    for arm in models:
        dump(out/'status.json',dict(phase='initial_validation',arm=arm,epoch=0,test_access=False))
        metrics=epoch(models[arm],data[arm]['valid'],device,'refiner')
        best[arm]=score(metrics);best_epochs[arm]=0
        dump(out/arm/'initial_validation.json',metrics)
        save(out/arm/'best.pt',checkpoint(arm,0,metrics))
    for e in range(1,plan['epochs']+1):
        for arm in models:
            tick=time.monotonic();torch.manual_seed(2026+e)
            lr=1e-4*.1**((e-1)/max(plan['epochs']-1,1))
            for group in optimizers[arm].param_groups:group['lr']=lr
            state=dict(phase='training',arm=arm,epoch=e,epochs=plan['epochs'],test_access=False)
            dump(out/'status.json',state)
            tr=epoch(models[arm],data[arm]['train'],device,'refiner',optimizers[arm],2026+e,(out/'status.json',state))
            dump(out/'status.json',dict(**state,subphase='validation'))
            val=epoch(models[arm],data[arm]['valid'],device,'refiner')
            value=score(val);improved=value<best[arm]
            if improved:best[arm]=value;best_epochs[arm]=e
            ckpt=checkpoint(arm,e,val);save(out/arm/'last.pt',ckpt)
            if improved:save(out/arm/'best.pt',ckpt)
            record=dict(epoch=e,arm=arm,train=tr,validation=val,best_epoch=best_epochs[arm],
                selection_score=value,seconds=time.monotonic()-tick,lr=lr)
            dump(out/arm/f'epoch_{e:03d}.json',record);print(json.dumps(record),flush=True)
    dump(out/'status.json',dict(phase='smoke_complete' if args.smoke else 'complete',best_epochs=best_epochs,
        best_scores=best,test_access=False))


if __name__=='__main__':main()
