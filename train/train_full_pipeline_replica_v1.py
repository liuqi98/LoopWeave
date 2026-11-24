"""Independent initialization of every trainable stage on the locked split.

Frozen prospective recipe: 100 generator + 30 refiner + 60 binding + 30 pair.
No shared model weights; target-free antigen feature files may be shared.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from train.train_reconstructed_two_stage import fit,epoch,predict
from train.train_abgaze_reconstructed import sha,dump,save,move
from train.compare_two_stage_binding import FeatureRows,score
from train.train_loop_pair_multiseed import eligible
from frame_relation_refiner import FrameRelationGraph,FrameRelationRefiner
from orientation_interface_refiner import OrientationInterfaceRefiner
from binding_interface_refiner import BindingInterfaceRefiner
from persistent_loop_pair_refiner import PersistentLoopPairRefiner


def train_later(model,config,rows,out,stage,epochs,seed,device,plan_sha):
    directory=out/stage;directory.mkdir()
    initial=epoch(model,rows['valid'],device,'refiner')
    dump(directory/'initial_validation.json',initial)
    best=initial['aligned_rmsd'] if stage=='pair' else score(initial);best_epoch=0
    if stage=='pair':
        groups=[dict(params=[p for n,p in model.named_parameters() if not n.startswith('loop_pair_layers.')],initial_lr=1e-5,lr=1e-5),
                dict(params=[p for n,p in model.named_parameters() if n.startswith('loop_pair_layers.')],initial_lr=1e-4,lr=1e-4)]
        opt=torch.optim.Adam(groups)
    else:opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    def state(e,val):
        return dict(model=model.state_dict(),model_config=config,epoch=e,stage=stage,validation=val,
            best_epoch=best_epoch,best_score=best,optimizer=opt.state_dict(),plan_sha256=plan_sha,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
    save(directory/'best.pt',state(0,initial))
    for e in range(1,epochs+1):
        tick=time.monotonic();torch.manual_seed(seed+e)
        for group in opt.param_groups:
            if stage=='pair':group['lr']=group['initial_lr']*.1**((e-1)/max(epochs-1,1))
            else:group['lr']=1e-4*.1**((e-1)/29) if e<=30 else 1e-5*.3**((e-31)/29)
        status=dict(phase='training',stage=stage,epoch=e,epochs=epochs,test_access=False)
        dump(out/'status.json',status)
        tr=epoch(model,rows['train'],device,'refiner',opt,seed+e,(out/'status.json',status))
        val=epoch(model,rows['valid'],device,'refiner')
        value=val['aligned_rmsd'] if stage=='pair' else score(val)
        allowed=eligible(val,initial) if stage=='pair' else True
        improved=allowed and value<best
        if improved:best,best_epoch=value,e
        checkpoint=state(e,val);save(directory/'last.pt',checkpoint)
        if improved:save(directory/'best.pt',checkpoint)
        record=dict(epoch=e,train=tr,validation=val,guard_passed=allowed,best_epoch=best_epoch,
                    seconds=time.monotonic()-tick)
        dump(directory/f'epoch_{e:03d}.json',record);print(stage,json.dumps(record),flush=True)
    # Fixed binding epoch60 is the preregistered pair-stage initialization.
    return torch.load(directory/('last.pt' if stage=='binding' else 'best.pt'),map_location='cpu',weights_only=False)


def test(model,generator,out,device,plan):
    from eval.test_loop_pair_replicates_v1 import score as test_score
    dataset=ROOT/'data/abgaze_reconstructed_v1';reference=ROOT/'outputs/persistent_loop_pair_test_v1/plan.json'
    ref=json.loads(reference.read_text());testpath=dataset/'tensors/test.pt'
    graphpath=ROOT/'data/raad_abgaze_reconstructed_v1/test.pt'
    assert sha(testpath)==ref['test_sha256'] and sha(graphpath)==ref['graph_sha256']
    target=out/'test';target.mkdir()
    dump(target/'plan.json',dict(generator_sha256=sha(out/'generator/best.pt'),
        pair_sha256=sha(out/'pair/best.pt'),training_plan_sha256=sha(out/'plan.json'),
        test_sha256=sha(testpath),K=5,temperature=.5,sampling_seed=2026,
        selection='validation selected; first sampled sequence and native decoder reported separately; oracle auxiliary'))
    rows=torch.load(testpath,weights_only=False);graphs={i['sample_id']:g for i,g in torch.load(graphpath,weights_only=False)}
    assert len(rows)==60 and {r[0] for r in rows}==set(graphs)
    generator.eval();model.eval();rng=torch.Generator().manual_seed(2026)
    alphabet='ACDEFGHIKLMNPQRSTVWY';generated=[];hashes={}
    with torch.no_grad():
        for sid,x,_ in rows:
            directory=ROOT/'outputs/binding_interface_test_v1/features'/sid;path=directory/'features.pt'
            hashes[sid]=sha(path);assert hashes[sid]==json.loads((directory/'ready.json').read_text())['sha256']
            item=torch.load(path,weights_only=False);meta=item['metadata']
            assert meta['antigen_only'] and not meta['native_h3_used'] and meta['pdb_sha256']==sha(dataset/'pdb'/f'{sid}.pdb')
            assert not set(x)&set(item['features'])
            inputs=move({**x,**item['features']},device)
            raw=generator(FrameRelationGraph(**graphs[sid]).to(device))
            inputs.update(base_bb=raw['h3_backbone'],base_logits=torch.zeros_like(inputs['base_logits']))
            pred=model(inputs);bb=pred['backbone'].cpu();logits=pred['sequence_logits'].cpu()
            assert torch.isfinite(bb).all() and torch.isfinite(logits).all()
            samples=torch.multinomial((logits/.5).softmax(-1),5,replacement=True,generator=rng).T
            generated.append(dict(sample_id=sid,backbone=bb.tolist(),sequence_logits=logits.tolist(),
                greedy=''.join(alphabet[i] for i in logits.argmax(-1).tolist()),
                candidates=[''.join(alphabet[i] for i in seq) for seq in samples.tolist()]))
    dump(target/'generated.json',generated)
    dump(target/'generation_audit.json',dict(sha256=sha(target/'generated.json'),feature_sha256=hashes,
        labels_not_passed_to_models=True,all_candidates_committed_before_scoring=True))
    metrics=test_score(rows,generated);metrics['test_sha256']=sha(testpath);dump(target/'metrics.json',metrics)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    torch.set_num_threads(1);device=torch.device('cuda:0');torch.cuda.set_device(device)
    torch.manual_seed(args.seed);random.seed(args.seed)
    reference=json.loads((ROOT/'outputs/reconstructed_two_stage_v1/plan.json').read_text())
    binding=json.loads((ROOT/'outputs/two_stage_binding60_gpu_v1/plan.json').read_text())
    for name,digest in {**reference['code_sha256'],**binding['code_sha256']}.items():assert sha(ROOT/name)==digest,name
    dataset=ROOT/'data/abgaze_reconstructed_v1';graphdir=ROOT/'data/raad_abgaze_reconstructed_v1'
    manifest=json.loads((dataset/'tensors/manifest.json').read_text())
    gm=json.loads((graphdir/'manifest.json').read_text())
    assert sha(dataset/'tensors/manifest.json')==reference['manifest_sha256']
    assert sha(graphdir/'manifest.json')==reference['graph_manifest_sha256']
    features=ROOT/'data/abgaze_binding_features_finalized_v1'
    assert sha(features/'plan.json')==binding['feature_plan_sha256']
    data={}
    for split in ('train','valid'):
        path=dataset/f'tensors/{split}.pt';gpath=graphdir/f'{split}.pt'
        assert sha(path)==manifest['files'][path.name] and sha(gpath)==gm[split]['sha256']
        graphs={i['sample_id']:g for i,g in torch.load(gpath,weights_only=False)}
        rows=torch.load(path,weights_only=False)
        assert [r[0] for r in rows]==reference['ids'][split] and set(graphs)==set(reference['ids'][split])
        if args.smoke:rows=rows[:2]
        data[split]=[(sid,x,y,graphs[sid]) for sid,x,y in rows]
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    epochs=dict(generator=100,refiner=30,binding=60,pair=30)
    if args.smoke:epochs={k:1 for k in epochs}
    dependencies=set(reference['code_sha256'])|set(binding['code_sha256'])|{
        'persistent_loop_pair_refiner.py','train/train_loop_pair_multiseed.py',
        'train/compare_two_stage_binding.py','eval/test_loop_pair_replicates_v1.py',str(Path(__file__).relative_to(ROOT))}
    plan=dict(seed=args.seed,smoke=args.smoke,epochs=epochs,
        ids={s:[r[0] for r in rows] for s,rows in data.items()},
        generator_config=reference['generator_config'],refiner_config=reference['refiner_config'],
        independent_initialization=True,shared_model_weights=False,
        selection='generator/refiner: validation geometry+.5FAPE; binding fixed epoch60 initialization; pair validation aligned RMSD with original guards',
        early_stopping=False,feature_plan_sha256=sha(features/'plan.json'),
        source_reference_plan_sha256=sha(ROOT/'outputs/reconstructed_two_stage_v1/plan.json'),
        code_sha256={name:sha(ROOT/name) for name in sorted(dependencies)},
        caveats=['Frozen prospective recipe; historic seed2026 had interactive development and CPU-to-GPU migration.',
                 'Same downstream split, but different feature representation and compute from RAAD.',
                 'Current test is a development benchmark, not independent confirmation.'])
    dump(out/'plan.json',plan);ph=sha(out/'plan.json')
    gc,rc=reference['generator_config'],reference['refiner_config']
    generator=FrameRelationRefiner(**gc).to(device)
    gs=fit(generator,gc,data,out/'generator','generator',epochs['generator'],device,ph,args.seed)
    generator.load_state_dict(gs['model']);generator.eval();refined={}
    with torch.no_grad():
        for split,rows in data.items():
            refined[split]=[]
            for sid,x,y,g in rows:
                pred=predict(generator,move(x,device),g,device,'generator')
                refined[split].append((sid,{**x,'base_bb':pred['backbone'].cpu(),'base_logits':torch.zeros_like(x['base_logits'])},y,None))
            save(out/f'{split}_cache.pt',refined[split])
    del data,gs;generator.cpu();torch.cuda.empty_cache()
    torch.manual_seed(args.seed);model=OrientationInterfaceRefiner(**rc).to(device)
    state=fit(model,rc,refined,out/'refiner','refiner',epochs['refiner'],device,ph,args.seed)
    del model;torch.manual_seed(args.seed);model=BindingInterfaceRefiner(**rc).to(device);model.load_source(state['model'])
    rows={s:FeatureRows(r,features,s,binding['feature_sha256']) for s,r in refined.items()}
    state=train_later(model,rc,rows,out,'binding',epochs['binding'],args.seed,device,ph)
    del model;torch.manual_seed(args.seed);model=PersistentLoopPairRefiner(**rc).to(device);model.load_binding(state['model'])
    state=train_later(model,rc,rows,out,'pair',epochs['pair'],args.seed,device,ph)
    model.load_state_dict(state['model'])
    if not args.smoke:
        dump(out/'status.json',dict(phase='testing',selected_epoch=state['epoch']))
        test(model,generator.to(device),out,device,plan)
    dump(out/'status.json',dict(phase='smoke_complete' if args.smoke else 'complete',selected_epoch=state['epoch']))


if __name__=='__main__':main()
