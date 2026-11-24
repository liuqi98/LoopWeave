"""Requested interim test on locked reconstructed data; no test-based selection."""
import argparse
import json
import shutil
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from orientation_interface_refiner import OrientationInterfaceRefiner
from abgaze_reconstructed_objective import objective
from train.train_native_sequence_coupled import dump,sha
from eval.exploratory_abgaze_best5 import select_by_aar


def aligned_rmsd(pred,true):
    a,b=pred-pred.mean(0),true-true.mean(0)
    u,_,vh=torch.linalg.svd(a.T@b)
    correction=torch.eye(3,dtype=a.dtype)
    correction[-1,-1]=torch.det(u@vh)
    return float(((a@u@correction@vh-b).square().sum(-1).mean()).sqrt())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-name',default='abgaze_reconstructed_ours_interim_test_v1')
    args=parser.parse_args()
    if Path(args.run_name).name!=args.run_name:parser.error('Invalid run name')
    torch.set_num_threads(1)
    out=ROOT/'outputs'/args.run_name;out.mkdir()
    root=ROOT/'data/abgaze_reconstructed_v1'
    training=ROOT/'outputs/abgaze_reconstructed_pose_seed2026'
    # Trainer atomically replaces best.pt, so this captures a whole checkpoint.
    frozen=out/'checkpoint.pt';shutil.copyfile(training/'best.pt',frozen)
    state=torch.load(frozen,map_location='cpu',weights_only=False)
    manifest_path=root/'tensors/manifest.json'
    manifest=json.loads(manifest_path.read_text())
    assert state['manifest_sha256']==sha(manifest_path)
    assert manifest['source_lock_sha256']==sha(root/'lock.json')
    for path,digest in json.loads((root/'lock.json').read_text()).items():
        assert sha(root/path)==digest,'Changed split metadata'
    testpath=root/'tensors/test.pt'
    assert sha(testpath)==manifest['files']['test.pt']
    plan=dict(checkpoint_sha256=sha(frozen),checkpoint_epoch=state['epoch'],
        checkpoint_selection='current minimum validation loss, snapshotted before test; no checkpoint sweep',
        training_run=str(training),training_still_in_progress=True,
        dataset=str(root),manifest_sha256=sha(manifest_path),test_sha256=sha(testpath),
        K=5,temperature=.5,seed=2026,aggregation='macro per complex, also report residue-pooled AAR',
        protocol='reconstructed, not author-certified; interim result, test no longer blind for future development',
        conditioning=manifest['conditioning'],
        sampling='categorical native logits; all 5 sequences share one backbone; not joint sequence-structure sampling',
        selection='one candidate by native AAR; earliest candidate wins ties; no per-metric oracle',
        contact_metric='H3/selected-context CA pairs <=8 A; not full-antigen/all-atom contacts or DockQ',
        no_antiberty=True,code_sha256=sha(Path(__file__)))
    dump(out/'plan.json',plan)
    model=OrientationInterfaceRefiner(**state['model_config']);model.load_state_dict(state['model']);model.eval()
    rows=torch.load(testpath,map_location='cpu',weights_only=False)
    expected=[json.loads(line)['pdb'] for line in (root/'splits/test.json').read_text().splitlines()]
    assert [sid for sid,_,_ in rows]==expected and len(rows)==60
    alphabet='ACDEFGHIKLMNPQRSTVWY';generator=torch.Generator().manual_seed(2026)
    generated=[];predictions=[]
    with torch.no_grad():
        for sid,x,_ in rows:
            pred=model(x);logits=pred['sequence_logits']
            assert torch.isfinite(logits).all() and torch.isfinite(pred['backbone']).all()
            sampled=torch.multinomial((logits/.5).softmax(-1),5,replacement=True,generator=generator).T
            generated.append(dict(sample_id=sid,greedy=''.join(alphabet[i] for i in logits.argmax(-1).tolist()),
                candidates=[''.join(alphabet[i] for i in seq) for seq in sampled.tolist()],
                backbone=pred['backbone'].tolist()))
            predictions.append(pred)
    dump(out/'generated.json',generated)
    dump(out/'generation_audit.json',dict(sha256=sha(out/'generated.json'),
        labels_packaged_with_inputs=True,labels_not_passed_to_forward_or_sampling=True,
        all_candidates_committed_before_scoring=True))
    metrics=[]
    with torch.no_grad():
        for (sid,x,y),pred,g in zip(rows,predictions,generated):
            native=''.join(alphabet[i] for i in y['sequence'].tolist())
            selected,aars=select_by_aar(g['candidates'],native)
            _,m=objective(pred,x,y)
            valid=y['mask'][:,1]
            m['aligned_rmsd']=aligned_rmsd(pred['backbone'][valid,1],y['bb'][valid,1])
            context=x['context_bb'][x['antigen_mask']&x['context_mask'][:,1],1]
            ref=torch.cdist(y['bb'][valid,1],context)<=8
            gen=torch.cdist(pred['backbone'][valid,1],context)<=8
            tp=int((ref&gen).sum());nr=int(ref.sum());ng=int(gen.sum())
            m.update(contact_precision=tp/max(ng,1),contact_recall=tp/max(nr,1))
            metrics.append(dict(sample_id=sid,length=len(native),selected_candidate=selected,
                oracle_aar=aars[selected],candidate_aars=aars,mean_sample_aar=sum(aars)/5,
                unique_sequences=len(set(g['candidates'])),**m))
    keys=['aar','oracle_aar','mean_sample_aar','rmsd','aligned_rmsd','contact_f1','contact_precision',
          'contact_recall','cn_mae','clash','interface_fape','unique_sequences']
    summary={k:sum(m[k] for m in metrics)/len(metrics) for k in keys}
    for key in ('aar','oracle_aar'):
        summary['pooled_'+key]=sum(m[key]*m['length'] for m in metrics)/sum(m['length'] for m in metrics)
    dump(out/'metrics.json',dict(checkpoint_epoch=state['epoch'],n=len(metrics),summary=summary,per_complex=metrics))
    dump(out/'status.json',dict(phase='complete',interim=True,training_selection_unchanged=True))
    print(json.dumps(dict(epoch=state['epoch'],**summary)),flush=True)


if __name__=='__main__':main()
