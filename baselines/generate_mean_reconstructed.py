"""Generate from a validation-locked MEAN checkpoint and pre-exported test inputs."""
import argparse
import json
import shutil
import torch
from types import SimpleNamespace
from pathlib import Path
import hashlib

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2))
    temporary.replace(path)
from train_mean_reconstructed import EfficientMCAttModel, VOCAB, batch, ROOT

def sanitize_input(item):
    x={**item,"X":item["X"].clone(),"S":torch.tensor([VOCAB.symbol_to_idx(s) for s in item["symbols"]])}
    start,end=x["L"].index("3"),x["L"].rindex("3")
    f=torch.arange(1,end-start+2).float()/(end-start+2)
    x["X"][start:end+1]=x["X"][start-1]+f[:,None,None]*(x["X"][end+1]-x["X"][start-1])
    x["S"][start:end+1]=VOCAB.get_unk_idx()
    return x



def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:1')
    args=p.parse_args();torch.set_num_threads(1)
    assert json.loads((args.run/'status.json').read_text())['phase']=='complete'
    plan=json.loads((args.run/'plan.json').read_text()); assert not plan['smoke']
    for name,digest in plan['source_hashes'].items(): assert sha(ROOT/name)==digest,name
    args.output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(args.run/'best.pt',args.output/'locked.pt')
    state=torch.load(args.output/'locked.pt',map_location='cpu',weights_only=False)
    assert state['plan_sha256']==sha(args.run/'plan.json')
    dump(args.output/'plan.json',dict(checkpoint_sha256=sha(args.output/'locked.pt'),epoch=state['epoch'],
        train_plan_sha256=sha(args.run/'plan.json'),inputs_sha256=sha(args.inputs),seed=2026,K=5,temperature=.5,
        sampling='one MEAN backbone draw; greedy plus five categorical sequences from same logits',
        generator_code_sha256=sha(__file__)))
    device=torch.device(args.device)
    if device.type=='cuda':torch.cuda.set_device(device)
    torch.manual_seed(2026)
    model=EfficientMCAttModel(**state['config']).to(device)
    model.load_state_dict(state['model']);model.eval()
    alphabet='ACDEFGHIKLMNPQRSTVWY'
    indices=torch.tensor([VOCAB.symbol_to_idx(a) for a in alphabet])
    generator=torch.Generator().manual_seed(2026);generated=[]
    rows=torch.load(args.inputs,weights_only=False)
    assert len(rows)==60 and len({r['sample_id'] for r in rows})==60
    with torch.no_grad():
        for item in rows:
            x=sanitize_input(item)
            _,seq,coords,_,raw_logits=model.infer(batch([x],device),device)
            logits=raw_logits[0][:,indices]
            assert torch.isfinite(logits).all()
            assert ''.join(alphabet[i] for i in logits.argmax(-1).tolist())==seq[0]
            sampled=torch.multinomial((logits/.5).softmax(-1),5,replacement=True,generator=generator).T
            generated.append(dict(sample_id=item['sample_id'],greedy=seq[0],
                candidates=[''.join(alphabet[i] for i in s) for s in sampled.tolist()],
                backbone=coords[0].tolist(),sequence_logits=logits.tolist()))
    dump(args.output/'generated.json',generated)
    dump(args.output/'generation_audit.json',dict(generated_sha256=sha(args.output/'generated.json'),
        native_labels_packaged=True,native_h3_sanitized_before_model=True,all_candidates_committed_before_scoring=True))


if __name__=='__main__':main()
