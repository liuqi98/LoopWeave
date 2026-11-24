"""Copy a scoped research snapshot without modifying live training or sources."""
import ast
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'deliverables/dyGF_scoped_research_snapshot_20261015_v1'


def read(p): return json.loads(p.read_text())


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    provenance={}
    def emit(rel,value):
        p=OUT/rel;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    def copy(src,rel):
        target=OUT/rel;target.parent.mkdir(parents=True,exist_ok=True)
        before=sha(src);shutil.copy2(src,target)
        assert sha(target)==before and sha(src)==before,src
        provenance[str(rel)]=dict(source=str(src),sha256=before)
    live=ROOT/'outputs/peptide_geometry_380_continue_v1'
    plan=read(live/'plan.json')
    for name,digest in plan['code_sha256'].items():assert sha(ROOT/name)==digest,name
    train_ids=plan['ids']['train'];valid_ids=plan['ids']['valid']
    external=read(ROOT/'outputs/external_temporal_evaluation_v1/evaluation_lock.json')
    test_ids=external['ids']
    assert len(train_ids)==380 and len(valid_ids)==18 and len(test_ids)==20
    assert not set(train_ids)&set(valid_ids)
    completed=sorted(live.glob('epoch_*.json'))
    epoch=max(int(p.stem.split('_')[1]) for p in completed)
    records=[read(p) for p in completed if int(p.stem.split('_')[1])<=epoch]
    for p in completed:
        e=int(p.stem.split('_')[1])
        if e<=epoch:
            copy(p,f'current_training/{p.name}')
            copy(live/f'checkpoint_{e:03d}.pt',f'checkpoints/continuation/checkpoint_{e:03d}.pt')
    copy(live/'plan.json','current_training/original_plan.json')
    eligible=[r for r in records if r['eligible']]
    selection=dict(snapshot_completed_epoch=epoch,
        best_guarded_epoch=min(eligible,key=lambda r:r['score'])['epoch'] if eligible else None,
        best_structure_epoch=min(records,key=lambda r:r['score'])['epoch'],
        live_training_was_running=read(live/'status.json')['phase']!='complete',
        snapshot_is_not_final_30_epoch_result=epoch<30)
    emit('current_training/snapshot.json',selection)

    # Snapshot code plus transitive local imports. Do not traverse unrelated projects.
    seeds=set(plan['code_sha256']) | {
        'eval/compare_peptide_candidates_scoped_v1.py','eval/replicate_peptide_geometry_v1.py',
        'eval/compare_external_raad_v1.py','eval/diagnose_external_comparison_v1.py',
        'eval/audit_fixed_model_geometry_v1.py','scripts/prepare_peptide_human_expanded_v1.py',
        'scripts/package_scoped_research_snapshot_v1.py',
        'baselines/generate_external_raad_v1.py','baselines/run_raad_matched.py',
        'baselines/generate_mean_reconstructed.py','baselines/train_mean_reconstructed.py',
        'baselines/diffab_reconstructed.py',
        'tests/test_peptide_geometry_refiner.py','tests/test_peptide_structure_selection.py',
        'tests/test_scoped_close_contacts.py','tests/test_persistent_loop_pair.py',
        'tests/test_loop_pair_ablations.py','tests/test_link_interface_refiner.py'}
    queue=[ROOT/s for s in seeds];seen=set()
    while queue:
        path=queue.pop()
        if path in seen or not path.is_file():continue
        seen.add(path);relative=path.relative_to(ROOT);copy(path,Path('code')/relative)
        tree=ast.parse(path.read_text())
        modules=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):modules.extend(a.name for a in node.names)
            elif isinstance(node,ast.ImportFrom) and node.module:modules.append(node.module)
        for module in modules:
            parts=Path(*module.split('.'))
            for base in [ROOT,path.parent,ROOT/'scripts',ROOT/'baselines']:
                p=base/parts.with_suffix('.py')
                if p.is_file():queue.append(p);break
                p=base/parts/'__init__.py'
                if p.is_file():queue.append(p);break
    for name in ['requirements.txt','requirements-experiment-lock.txt','requirements-physics-evolution.txt']:
        if (ROOT/name).exists():copy(ROOT/name,Path('environment')/name)
    packages=sorted({(d.metadata.get('Name') or 'unknown',d.version) for d in importlib.metadata.distributions()})
    emit('environment/installed_versions.json',dict(python=platform.python_version(),torch=torch.__version__,
        cuda=torch.version.cuda,packages=[dict(name=n,version=v) for n,v in packages]))

    # Dataset files are materialized copies, not links into the live workspace.
    input_root=ROOT/'data/abgaze_reconstructed_v1'
    for split,ids in [('train',train_ids),('valid',valid_ids)]:
        cache=ROOT/f'outputs/full_pipeline_seed2026_v1/{split}_cache.pt'
        assert sha(cache)==plan['cache_sha256'][split]
        lookup={r[0]:r for r in torch.load(cache,map_location='cpu',weights_only=False)}
        selected=[lookup[sid] for sid in ids]
        p=OUT/f'data/{split}/refiner_cache.pt';p.parent.mkdir(parents=True,exist_ok=True);torch.save(selected,p)
        base=torch.load(input_root/f'tensors/{split}.pt',map_location='cpu',weights_only=False)
        base={r[0]:r for r in base};torch.save([base[sid] for sid in ids],p.parent/'inputs.pt')
        rec={r['pdb']:r for r in map(json.loads,(input_root/f'splits/{split}.json').read_text().splitlines())}
        emit(f'data/{split}/records.json',[rec[sid] for sid in ids])
        for sid in ids:
            copy(input_root/'pdb'/f'{sid}.pdb',f'data/{split}/pdb/{sid}.pdb')
            feature=ROOT/f'data/abgaze_binding_features_finalized_v1/{split}/{sid}/features.pt'
            assert sha(feature)==plan['feature_sha256'][f'{split}/{sid}']
            copy(feature,f'data/{split}/features/{sid}/features.pt')
    independent=ROOT/'data/independent_nonpathogen_inputs_v1'
    lookup={r[0]:r for r in torch.load(independent/'test.pt',map_location='cpu',weights_only=False)}
    target=OUT/'data/external_test';target.mkdir(parents=True)
    torch.save([lookup[sid] for sid in test_ids],target/'inputs.pt')
    graphs={item['sample_id']:(item,g) for item,g in torch.load(independent/'graphs.pt',map_location='cpu',weights_only=False)}
    torch.save([graphs[sid] for sid in test_ids],target/'graphs.pt')
    rec={r['pdb']:r for r in read(independent/'records.json')}
    emit('data/external_test/records.json',[rec[sid] for sid in test_ids])
    for sid in test_ids:
        copy(independent/'pdb'/f'{sid}.pdb',f'data/external_test/pdb/{sid}.pdb')
        feature=ROOT/f'data/independent_nonpathogen_binding_v1/features/{sid}/features.pt'
        assert sha(feature)==external['feature_sha256'][sid]
        copy(feature,f'data/external_test/features/{sid}/features.pt')
    emit('data/dataset_index.json',dict(train=train_ids,valid=valid_ids,external_test=test_ids,
        original_full_counts=dict(train=3269,valid=340,development_test=60),
        original_full_raw_data_included=False,
        notes=['Training380 and validation18 retain original split membership.',
               'External20 was already evaluated and inspected; not an unseen final test for the new candidate.',
               'Cached base backbone predictions are from the fixed seed2026 generator.',
               'records.json retains original source metadata, including historical absolute paths; use bundled pdb directories.',
               'The extra4 cases used in the22-case diagnostic are not part of the locked18 validation dataset.']))

    # Preserve original evidence and sampling caveats; never relabel historical results.
    directories=['full_pipeline_replicates_queue_v1','raad_replicates_evaluation_v1',
        'mean_reconstructed_test_v1','diffab_reconstructed_seed2026_v1','matched_ablations_test_v2',
        'fixed_model_geometry_v1','external_temporal_evaluation_v1','external_raad_comparison_v1',
        'external_failure_diagnostics_v1','peptide_geometry_human_valid_replicates_v1',
        'peptide_geometry_human_train_pilot_v1','peptide_candidate_scoped_comparison_v1']
    for directory in directories:
        src=ROOT/'outputs'/directory
        for path in src.glob('*'):
            if path.is_file() and path.suffix in ('.json','.csv') and not path.name.startswith('generated'):
                copy(path,Path('results')/directory/path.name)
    for rel in ['outputs/peptide_human_expanded_scope_v1/plan.json','outputs/peptide_human_expanded_scope_v1/scope_audit.json',
                'outputs/peptide_geometry_human_valid_v1/plan.json','outputs/peptide_geometry_human_valid_v1/scope_audit.json',
                'outputs/peptide_geometry_human_valid_v1/comparison.json','outputs/external_temporal_evaluation_v1/evaluation_lock.json']:
        copy(ROOT/rel,Path('provenance')/rel)
    for seed in [2026,2027,2028]:
        for stage in ['generator','pair']:
            copy(ROOT/f'outputs/full_pipeline_seed{seed}_v1/{stage}/best.pt',f'checkpoints/original_seed{seed}/{stage}.pt')
    for name in ['best.pt','last.pt']:
        copy(ROOT/'outputs/peptide_geometry_human_train_pilot_v1'/name,f'checkpoints/pilot/{name}')
    baseline_runs={'raad2026':'raad_abgaze_reconstructed_seed2026_v1','raad2027':'raad_reconstructed_seed2027_v2',
                   'raad2028':'raad_reconstructed_seed2028_v2','mean':'mean_reconstructed_seed2026_v1',
                   'diffab':'diffab_reconstructed_seed2026_v1'}
    for label,run in baseline_runs.items():
        for name in ['best.pt','plan.json']:
            copy(ROOT/'outputs'/run/name,f'checkpoints/baselines/{label}/{name}')
    externals={'raad':ROOT/'external/chimera-bench/baselines/raad/code',
               'mean':ROOT/'external/chimera-bench/baselines/mean',
               'diffab':Path('/media/xd/code2/DiffCDRH3/external/diffab')}
    for label,src in externals.items():
        for path in src.rglob('*'):
            rel=path.relative_to(src)
            if any(part in {'.git','__pycache__','datasets','checkpoints','logs','outputs'} for part in rel.parts):continue
            if path.is_file() and (path.suffix in {'.py','.yaml','.yml','.md','.txt','.toml','.sh'} or path.name.startswith('LICENSE')):
                copy(path,Path('baseline_source')/label/rel)

    metrics=['aar','aligned_rmsd','rmsd','contact_f1','interface_fape','cn_mae']
    table=[]
    def add(dataset,model,n,seeds,values,notes):
        row=dict(dataset=dataset,model=model,n=n,seeds=seeds,notes=notes)
        for k in metrics:
            v=values.get(k)
            row[k]=v.get('mean') if isinstance(v,dict) else v
            row[k+'_seed_sd']=v.get('sample_std') if isinstance(v,dict) else None
        table.append(row)
    ours=read(ROOT/'outputs/full_pipeline_replicates_queue_v1/comparison.json')
    raad=read(ROOT/'outputs/raad_replicates_evaluation_v1/comparison.json')
    add('historical_development60','ours_original',60,3,ours['mean_std'],'Original full-split training; not current380 continuation')
    add('historical_development60','RAAD',60,3,raad['raad_mean_std'],'Locally retrained matched split; inputs and compute differ')
    add('historical_development60','MEAN',60,1,read(ROOT/'outputs/mean_reconstructed_test_v1/metrics.json')['summary'],'Single seed')
    diff=read(ROOT/'outputs/diffab_reconstructed_seed2026_v1/test_metrics.json')['summary']
    for label,summary in diff.items():add('historical_development60','DiffAb_'+label,60,1,summary,'Adapted protocol; oracle-selected row uses native AAR and is auxiliary')
    ext=read(ROOT/'outputs/external_raad_comparison_v1/comparison.json')
    for label,key in [('ours_original','ours_mean_std'),('RAAD','raad_mean_std')]:
        add('screened_external20',label,20,3,ext[key],'Original locked models; scoped temporal cohort, not universal family independence')
    scoped=read(ROOT/'outputs/peptide_candidate_scoped_comparison_v1/comparison.json')['subsets']['all']
    for label,summary in scoped['metrics'].items():add('human_validation22',label,22,1,summary,'Development diagnostic; only18 are used in current training validation')
    add('human_validation18',f'continuation_epoch{epoch}',18,1,records[-1]['validation']['summary'],'Live-training snapshot, not final result')
    p=OUT/'comparisons/summary.csv';p.parent.mkdir(parents=True)
    with p.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    emit('comparisons/summary.json',table)
    emit('provenance/copied_files.json',provenance)
    emit('PACKAGE.json',dict(created_local=time.strftime('%Y-%m-%d %H:%M:%S'),source_workspace=str(ROOT),
        scope='Current audited380-sample experiment plus historical comparison evidence',
        data_counts=dict(train=380,valid=18,external_test=20),snapshot=selection,
        original_files_modified=False,symlinks_used=False,live_training_stopped=False,
        source_code_files=len(seen),
        portability='Source archive preserves original paths. Dataset files and checkpoints are copied; original launchers need path adaptation. No clean-environment end-to-end rerun is claimed.'))
    manifest={str(p.relative_to(OUT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in OUT.rglob('*') if p.is_file()}
    emit('MANIFEST.json',manifest)
    print(json.dumps(dict(path=str(OUT),files=len(manifest),bytes=sum(v['bytes'] for v in manifest.values()),epoch=epoch)))


if __name__=='__main__':main()
