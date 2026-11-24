"""Isolated RAAD namespace: fixed external inputs, fixed old checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from run_raad_matched import AntiDesigner,DeviceProteinFeature,VOCAB,batch,convert,sanitize_input,ROOT


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--lock',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    args=p.parse_args();torch.set_num_threads(1)
    lock=json.loads(args.lock.read_text());r=lock['runs'][str(args.seed)];run=Path(r['run'])
    assert sha(run/'best.pt')==r['checkpoint_sha256']
    plan=json.loads((run/'plan.json').read_text())
    for name,digest in plan['code_sha256'].items():assert sha(ROOT/name)==digest,name
    graphpath=ROOT/'data/independent_nonpathogen_inputs_v1/graphs.pt'
    assert sha(graphpath)==lock['graphs_sha256']
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.manual_seed(2026)
    state=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
    assert state['plan_sha256']==sha(run/'plan.json') and state['epoch']==r['epoch']
    model=AntiDesigner(32,256,4,4,0.,'3',SimpleNamespace(**state['config'])).to(device)
    model.protein_feature=DeviceProteinFeature(1).to(device);model.load_state_dict(state['model']);model.eval()
    records={i['sample_id']:(i,g) for i,g in torch.load(graphpath,map_location='cpu',weights_only=False)}
    alphabet='ACDEFGHIKLMNPQRSTVWY';indices=torch.tensor([VOCAB.symbol_to_idx(a) for a in alphabet])
    rng=torch.Generator().manual_seed(2026);generated=[]
    with torch.no_grad():
        for sid in lock['ids']:
            item,graph=records[sid];x=sanitize_input(convert(item,graph))
            altered=dict(item,cdr3_backbone=np.asarray(item['cdr3_backbone'])+500,native_h3_sequence='A'*item['target_length'])
            check=sanitize_input(convert(altered,graph))
            assert torch.equal(x['X'],check['X']) and torch.equal(x['S'],check['S']),sid
            _,seq,coords,_,raw_logits=model.infer(batch([x],device),device)
            logits=raw_logits[0].cpu()[:,indices];bb=torch.as_tensor(coords[0]).cpu()
            assert torch.isfinite(logits).all() and torch.isfinite(bb).all(),sid
            assert ''.join(alphabet[i] for i in logits.argmax(-1).tolist())==seq[0]
            samples=torch.multinomial((logits/.5).softmax(-1),5,replacement=True,generator=rng).T
            generated.append(dict(sample_id=sid,backbone=bb.tolist(),sequence_logits=logits.tolist(),greedy=seq[0],
                candidates=[''.join(alphabet[i] for i in s.tolist()) for s in samples]))
    target=args.lock.parent/f'generated_seed{args.seed}.json'
    with target.open('x') as f:json.dump(generated,f)
    print(json.dumps(dict(seed=args.seed,n=len(generated),sha256=sha(target),native_perturbation_passed=True)))


if __name__=='__main__':main()
