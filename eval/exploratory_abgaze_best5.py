"""Fixed-source categorical best-of-5 sequence diagnostic, not AbGaze reproduction."""
import csv
import json
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from relation_coupled_refiner import RelationCoupledRefiner
from frame_relation_refiner import FrameRelationRefiner, build_frame_relation_graph
from v21_atomic_refiner import context_inputs
from relation_interface_refiner import AA_TO_INDEX
from train.train_native_sequence_coupled import dump, sha


def select_by_aar(candidates, native):
    scores=[sum(a==b for a,b in zip(s,native))/len(native) for s in candidates]
    return max(range(len(scores)),key=lambda i:(scores[i],-i)),scores


def main():
    torch.set_num_threads(1)
    out=ROOT/'outputs/abgaze_exploratory_best5_v1'
    out.mkdir()
    source=ROOT/'outputs/relation_coupling_comparison/coupled/best.pt'
    state=torch.load(source,map_location='cpu',weights_only=False)
    basepath=Path(state['base_checkpoint'])
    base_state=torch.load(basepath,map_location='cpu',weights_only=False)
    directory=ROOT/'data/raad_protocol_frame_v1'
    plan=dict(checkpoint=str(source),checkpoint_sha256=sha(source),epoch=state['epoch'],
        base_checkpoint=str(basepath),base_sha256=sha(basepath),K=5,temperature=.5,seed=2026,
        candidate_sampling='5 independent categorical sequence samples from fixed logits; same deterministic structure',
        selection='one candidate per target by native H3 AAR; ties choose earliest index',
        purpose='Exploratory requested paper comparison, NOT fair AbGaze reproduction or model selection',
        limitations=['Historical split not AbGaze-matched','15 RAbD chain mappings unresolved',
                    'Previously examined test set; not blind','No joint structure diversity',
                    'No AbGaze model locally evaluated; paper numbers are reference only'],
        input_hashes={f:sha(directory/f) for f in ['test_inputs.pt','test_initial.csv']},
        code_hashes={f:sha(ROOT/f) for f in ['eval/exploratory_abgaze_best5.py','relation_coupled_refiner.py',
            'frame_relation_refiner.py','v21_atomic_refiner.py','structure_refinement_ablation.py']})
    dump(out/'plan.json',plan)
    base=FrameRelationRefiner(**base_state['model_config']);base.load_state_dict(base_state['model']);base.eval()
    model=RelationCoupledRefiner(**state['model_config']);model.load_state_dict(state['model']);model.eval()
    items=torch.load(directory/'test_inputs.pt',map_location='cpu',weights_only=False)
    assert len(items)==60 and len({r['sample_id'] for r in items})==60
    forbidden={'native_h3_sequence','native_h3_ca','cdr3_backbone','cdr3_seq','native_heavy_ca'}
    assert all(not(set(r)&forbidden) for r in items)
    with (directory/'test_initial.csv').open() as f:
        initial={r['sample_id']:r for r in csv.DictReader(f)}
    alphabet={i:a for a,i in AA_TO_INDEX.items()}
    generator=torch.Generator(device='cpu').manual_seed(2026)
    generated=[]
    with torch.no_grad():
        for item in sorted(items,key=lambda r:r['sample_id']):
            sid=item['sample_id'];graph=build_frame_relation_graph(item,initial[sid])
            x=context_inputs(item,graph,base(graph));prediction=model(x)
            logits=prediction['sequence_logits'].cpu()
            assert logits.shape==(item['target_length'],20) and torch.isfinite(logits).all()
            tokens=torch.multinomial((logits/.5).softmax(-1),5,replacement=True,generator=generator).T
            sequences=[''.join(alphabet[int(i)] for i in row) for row in tokens]
            generated.append(dict(sample_id=sid,sequences=sequences,
                greedy=''.join(alphabet[int(i)] for i in logits.argmax(-1)),
                ca_coords=prediction['backbone'][:,1].cpu().tolist()))
    dump(out/'generated.json',generated)
    dump(out/'generation_audit.json',dict(generated_sha256=sha(out/'generated.json'),
                                         n=60,K=5,target_labels_loaded_during_generation=False))
    # Open native labels ONLY after all candidates have been generated and hashed.
    targetpath=directory/'test_targets.pt'
    targets=torch.load(targetpath,map_location='cpu',weights_only=False)
    lookup={r['sample_id']:r['native_h3_sequence'] for r in targets}
    assert set(lookup)=={r['sample_id'] for r in generated}
    metrics=[]
    for row in generated:
        native=lookup[row['sample_id']]
        assert all(len(s)==len(native) for s in row['sequences'])
        selected,scores=select_by_aar(row['sequences'],native)
        metrics.append(dict(sample_id=row['sample_id'],length=len(native),selected_candidate=selected,
            selected_aar=scores[selected],candidate_aars=scores,mean_sample_aar=sum(scores)/5,
            greedy_aar=sum(a==b for a,b in zip(row['greedy'],native))/len(native),
            unique_sequences=len(set(row['sequences']))))
    summary={k:sum(r[k] for r in metrics)/60 for k in ['selected_aar','mean_sample_aar','greedy_aar','unique_sequences']}
    summary['pooled_selected_aar']=sum(r['selected_aar']*r['length'] for r in metrics)/sum(r['length'] for r in metrics)
    dump(out/'metrics.json',dict(summary=summary,per_complex=metrics,target_sha256=sha(targetpath),
        abgaze_paper_h3_aar=.456,abgaze_source='https://arxiv.org/html/2609.35296v1#S4.SS3',
        comparable=False,reason='Sampling budget/temperature aligned only; training, chain mapping and generation protocol differ'))
    dump(out/'status.json',dict(phase='complete',exploratory_test_access=True,training_selection_unchanged=True))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
