#!/usr/bin/env python3
"""Train vendored RAAD on exactly the v18 split and retained context nodes.

Only adapter/training orchestration lives here; RAAD's network and CE+Huber
objective are imported unchanged. Global coordinates exclude native H3.
"""
import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT/'external/chimera-bench/baselines/raad/code'
sys.path.insert(0, str(ROOT/'.raad_dependencies'))
sys.path.insert(0, str(CODE))
from models import AntiDesigner, PosEmbedding, ProteinFeature
from data.pdb_utils import VOCAB


class DevicePosEmbedding(PosEmbedding):
    """Same parameter-free frequencies, allocated on the input device."""
    def forward(self, indices):
        frequency=torch.exp(torch.arange(0,self.num_embeddings,2,dtype=torch.float32,
                            device=indices.device)*-(np.log(10000.0)/self.num_embeddings))
        angles=indices.unsqueeze(-1)*frequency.view(1,1,1,-1)
        return torch.cat((torch.cos(angles),torch.sin(angles)),-1)


class DeviceProteinFeature(ProteinFeature):
    def __init__(self, interface_only):
        super().__init__(interface_only)
        self.node_pos_embedding=DevicePosEmbedding(16)
        self.edge_pos_embedding=DevicePosEmbedding(16)

    def _rbf(self, distance):
        centers=torch.linspace(0.,20.,16,device=distance.device).view(1,-1)
        return torch.exp(-((distance.unsqueeze(-1)-centers)/(20./16))**2)


def convert(item, graph):
    length = int(item['target_length'])
    framework = torch.as_tensor(item['framework_backbone']).float().clone()
    fw_mask = torch.as_tensor(item['framework_backbone_mask']).bool()
    # Same retained framework residues as v18; preserve their original order.
    selected_coords = graph['coords'][length:][~graph['antigen_mask'][length:]]
    matches = torch.cdist(selected_coords, torch.as_tensor(item['framework_ca']).float(),
                          compute_mode='donot_use_mm_for_euclid_dist')
    distance, indices = matches.min(-1)
    if distance.max() > 1e-3 or len(indices.unique()) != len(indices):
        raise ValueError('Cannot match v18 framework context without ambiguity')
    framework[~fw_mask] = framework[:,1:2].expand_as(framework)[~fw_mask]
    heavy_length = len(item['heavy_sequence_without_h3'])
    h3_indices = torch.as_tensor(item['heavy_h3_indices']).long()
    full_heavy_length = heavy_length + length
    keep = torch.ones(full_heavy_length,dtype=torch.bool)
    keep[h3_indices] = False
    fw_original = torch.arange(full_heavy_length)[keep]
    heavy_entries, light_entries = [], []
    for index in indices.tolist():
        entry = (int(fw_original[index]) if index < heavy_length else index-heavy_length,
                 item['framework_seq'][index], framework[index])
        (heavy_entries if index < heavy_length else light_entries).append(entry)
    h3_backbone = torch.as_tensor(item['cdr3_backbone']).float().clone()
    mask = torch.as_tensor(item['cdr3_backbone_mask']).bool()
    h3_backbone[~mask] = h3_backbone[:,1:2].expand_as(h3_backbone)[~mask]
    for i,pos in enumerate(h3_indices.tolist()):
        heavy_entries.append((pos,item['native_h3_sequence'][i],h3_backbone[i]))
    heavy_entries.sort(key=lambda x:x[0])
    positions = [x[0] for x in heavy_entries]
    if int(h3_indices[0])-1 not in positions or int(h3_indices[-1])+1 not in positions:
        raise ValueError('Missing H3 anchor in matched context')
    antigen_indices = graph['antigen_original_indices'].tolist()
    antigen = torch.as_tensor(item['antigen_backbone']).float().clone()
    ag_mask = torch.as_tensor(item['antigen_backbone_mask']).bool()
    antigen[~ag_mask] = antigen[:,1:2].expand_as(antigen)[~ag_mask]
    antigen_entries = [(i,item['antigen_seq'][i],antigen[i]) for i in antigen_indices]
    X,S,L = [],[],[]
    for entries,symbol,is_heavy in [(heavy_entries,VOCAB.BOH,True),
                                    (light_entries,VOCAB.BOL,False),
                                    (antigen_entries,VOCAB.BOA,False)]:
        if not entries:
            continue
        coordinates = torch.stack([x[2] for x in entries])
        proxy = coordinates.clone()
        labels = [('3' if is_heavy and x[0] in h3_indices.tolist() else '0') for x in entries]
        if is_heavy:
            begin,end = labels.index('3'),len(labels)-1-labels[::-1].index('3')
            fractions = torch.arange(1,end-begin+2).float()/(end-begin+2)
            proxy[begin:end+1] = proxy[begin-1] + fractions[:,None,None]*(proxy[end+1]-proxy[begin-1])
        # Critically, global centroids use masked H3 proxies, not ground truth.
        X.extend([proxy.mean(0),*coordinates])
        S.extend([VOCAB.symbol_to_idx(symbol),*[VOCAB.symbol_to_idx(x[1]) for x in entries]])
        L.extend(['0',*labels])
    labels = ''.join(L)
    assert labels.count('3') == length
    return dict(X=torch.stack(X),S=torch.tensor(S),L=labels,sample_id=item['sample_id'])


def batch(items,device):
    lengths = [len(x['S']) for x in items]
    return dict(X=torch.cat([x['X'] for x in items]).to(device),
                S=torch.cat([x['S'] for x in items]).to(device),
                L=[x['L'] for x in items],
                offsets=torch.tensor([0,*np.cumsum(lengths).tolist()],device=device))


def sanitize_input(item):
    """Inference never needs native H3 labels, even though API accepts them."""
    x = {**item, 'X':item['X'].clone(),'S':item['S'].clone()}
    begin,end = x['L'].index('3'),x['L'].rindex('3')
    length = end-begin+1
    fraction = torch.arange(1,length+1).float()/(length+1)
    x['X'][begin:end+1] = x['X'][begin-1] + fraction[:,None,None]*(x['X'][end+1]-x['X'][begin-1])
    x['S'][begin:end+1] = VOCAB.get_unk_idx()
    return x


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--run-name',default=None)
    parser.add_argument('--benchmark',type=Path,default=None)
    parser.add_argument('--train-only',action='store_true')
    parser.add_argument('--seed',type=int,default=2022)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--epochs',type=int,default=60)
    parser.add_argument('--no-early-stop',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device=torch.device(args.device)
    selected_ids=None
    benchmark_hash=None
    if args.benchmark:
        if not args.train_only:
            raise ValueError('Purged protocol requires separate locked-checkpoint evaluation; use --train-only')
        lock=json.loads((args.benchmark/'lock.json').read_text())
        for name,digest in lock.items():
            assert hashlib.sha256((args.benchmark/name).read_bytes()).hexdigest()==digest, name
        assert json.loads((args.benchmark/'summary.json').read_text())['feasibility_passed']
        sources=json.loads((args.benchmark/'source_artifacts.json').read_text())
        # Only train/validation artifacts are opened by the training job.
        for split in ('train','valid'):
            for name,digest in sources[split].items():
                h=hashlib.sha256()
                with (ROOT/name).open('rb') as handle:
                    for block in iter(lambda:handle.read(1048576),b''):h.update(block)
                assert h.hexdigest()==digest,name
        selected_ids={s:set(json.loads((args.benchmark/f'{s}_ids.json').read_text())) for s in ('train','valid')}
        benchmark_hash=hashlib.sha256((args.benchmark/'lock.json').read_bytes()).hexdigest()
    tag=args.run_name or ('raad_matched_v18'+('_smoke' if args.smoke else ''))
    out=ROOT/'outputs'/tag
    out.mkdir(parents=True,exist_ok=True)
    if (out/'plan.json').exists():raise RuntimeError('Refusing existing run overwrite')
    config=dict(alpha=.8,beta=.5,node_feats_mode='1111',edge_feats_mode='1111',interface_only=1)
    (out/'plan.json').write_text(json.dumps(dict(
        model='vendored RAAD AntiDesigner; 4 layers, hidden256, embed32',
        config=config,seed=args.seed,epochs=args.epochs,min_epochs=30,patience=20,
        no_early_stop=args.no_early_stop,train_only=args.train_only,
        smoke_only=args.smoke,device=str(device),
        device_compatibility='CPU-only parameter-free positional/RBF features use input device; no vendored source change',
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        benchmark=str(args.benchmark) if args.benchmark else None,benchmark_lock_sha256=benchmark_hash,
        training_ids={s:sorted(ids) for s,ids in selected_ids.items()} if selected_ids else None,
        selection='minimum validation native RAAD CE+0.8 Huber loss',
        inputs='same retained context nodes as v18: up to 96 framework / 128 antigen',
        modifications='adapter masks H3 contribution to global centroids; missing atoms filled with residue CA; degenerate RAAD edge min-max normalization uses probability 0.5',
        source_hashes={n:hashlib.sha256((CODE/n).read_bytes()).hexdigest()
                       for n in ['models.py','modules.py']},
        test='same native RAbD60; K=1 greedy; common evaluate_rabd_generated.py'),indent=2))
    data={}
    for split in ('train','valid'):
        refs=torch.load(ROOT/f'data/raad_protocol_frame_v1/{split}_targets.pt',weights_only=False)
        graphs=torch.load(ROOT/f'data/raad_protocol_frame_v1/{split}_graphs.pt',weights_only=False)
        assert len(refs)==len(graphs)
        by_id={g['sample_id']:g['graph'] for g in graphs}
        if selected_ids is not None:
            refs=[r for r in refs if r['sample_id'] in selected_ids[split]]
            assert {r['sample_id'] for r in refs}==selected_ids[split]
        if args.smoke:refs=refs[:8]
        data[split]=[convert(x,by_id[x['sample_id']]) for x in refs]
        print(json.dumps({'split':split,'n':len(data[split])}),flush=True)
    model=AntiDesigner(32,256,4,4,0.,'3',SimpleNamespace(**config)).to(device)
    if device.type=='cpu':model.protein_feature=DeviceProteinFeature(config['interface_only'])
    # Ground-truth perturbation audit: predictions must not depend on H3 labels.
    reference=data['valid'][0]
    altered={**reference,'X':reference['X'].clone(),'S':reference['S'].clone()}
    begin,end=reference['L'].index('3'),reference['L'].rindex('3')
    altered['X'][begin:end+1]+=500
    altered['S'][begin:end+1]=VOCAB.symbol_to_idx('A')
    model.eval()
    with torch.no_grad():
        torch.manual_seed(73)
        a=model.infer(batch([reference],device),device)
        torch.manual_seed(73)
        b=model.infer(batch([altered],device),device)
    assert a[1]==b[1] and np.allclose(a[2][0],b[2][0])
    print('native_H3_perturbation_audit: passed',flush=True)
    torch.manual_seed(args.seed)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=.9)
    best=float('inf');best_epoch=0
    for epoch in range(1,(1 if args.smoke else args.epochs)+1):
        start=time.monotonic();model.train()
        order=list(range(len(data['train'])));random.shuffle(order)
        total=0.;seen=0
        for step in range(0,len(order),8):
            subset=[data['train'][i] for i in order[step:step+8]]
            x=batch(subset,device)
            loss,_,_=model(x['X'],x['S'],x['L'],x['offsets'])
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step()
            total+=loss.item()*len(subset);seen+=len(subset)
            if seen % 400 == 0:
                print(json.dumps(dict(epoch=epoch,train_seen=seen,train_loss=total/seen)),flush=True)
        scheduler.step();model.eval();validation=0.
        with torch.no_grad():
            for step in range(0,len(data['valid']),8):
                subset=data['valid'][step:step+8];x=batch(subset,device)
                loss,_,_=model(x['X'],x['S'],x['L'],x['offsets'])
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite validation loss')
                validation+=loss.item()*len(subset)
        validation/=len(data['valid'])
        state=dict(model=model.state_dict(),epoch=epoch,config=config,optimizer=optimizer.state_dict(),
                   scheduler=scheduler.state_dict(),seed=args.seed,benchmark_lock_sha256=benchmark_hash,
                   python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
        if validation<best:
            best,best_epoch=validation,epoch;torch.save(state,out/'best.pt')
        torch.save(state,out/'last.pt')
        record=dict(epoch=epoch,best_epoch=best_epoch,train_loss=total/seen,
                    val_loss=validation,seconds=time.monotonic()-start)
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        if not args.no_early_stop and epoch>=30 and epoch-best_epoch>=20:break
    if args.train_only:
        (out/'completed.json').write_text(json.dumps(dict(training='complete',test_access=False,
            smoke_only=args.smoke,best_epoch=best_epoch,seed=args.seed,benchmark_lock_sha256=benchmark_hash)))
        return
    state=torch.load(out/'best.pt',map_location=device,weights_only=False)
    model.load_state_dict(state['model']);model.eval();torch.manual_seed(2022)
    refs=torch.load(ROOT/'data/raad_protocol_frame_v1/test_targets.pt',weights_only=False)
    # Build identical context selections with existing input-only graph builder
    # in a separate process to avoid RAAD's top-level data/utils import names.
    graph_path=out/'test_graphs.pt'
    subprocess.run([sys.executable,str(ROOT/'data/materialize_frame_relation_refiner.py'),
        '--references',str(ROOT/'data/raad_protocol_frame_v1/test_targets.pt'),
        '--generated',str(ROOT/'data/raad_protocol_frame_v1/test_initial.csv'),
        '--output',str(graph_path)],cwd=ROOT,check=True)
    graphs={x['sample_id']:x['graph'] for x in torch.load(graph_path,weights_only=False)}
    rows=[]
    with torch.no_grad():
        for item in refs:
            converted=sanitize_input(convert(item,graphs[item['sample_id']]))
            _,seq,coords,_,_=model.infer(batch([converted],device),device)
            rows.append(dict(sample_id=item['sample_id'],candidate=0,
                             generated_cdrh3=seq[0],ca_coords=json.dumps(coords[0][:,1].tolist())))
    generated=out/'native_test_k1.csv'
    with generated.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    subprocess.run([sys.executable,str(ROOT/'eval/evaluate_rabd_generated.py'),
        '--targets',str(ROOT/'data/raad_protocol_frame_v1/test_targets.pt'),
        '--generated',str(generated),'--expected-k','1','--bootstrap','10000',
        '--output',str(out/'native_test_metrics.json')],cwd=ROOT,check=True)
    (out/'completed.json').write_text(json.dumps({'training_and_testing':'complete',
        'smoke_only':args.smoke,'best_epoch':state['epoch']}))


if __name__=='__main__':main()
