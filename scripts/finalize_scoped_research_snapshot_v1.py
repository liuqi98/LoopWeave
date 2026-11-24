"""Append final immutable training results to an earlier scoped snapshot."""
import csv
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from package_scoped_research_snapshot_v1 import OUT,sha,read


def main():
    live=ROOT/'outputs/peptide_geometry_380_continue_v1'
    status=read(live/'status.json');assert status['phase']=='complete' and status['epoch']==30
    final=OUT/'final_training';final.mkdir(exist_ok=False)
    for p in live.glob('*.json'):
        if p.name=='failure.json':continue
        shutil.copy2(p,final/p.name);assert sha(p)==sha(final/p.name)
    for p in live.glob('checkpoint_*.pt'):
        target=OUT/'checkpoints/continuation'/p.name
        if target.exists():assert sha(p)==sha(target)
        else:shutil.copy2(p,target);assert sha(p)==sha(target)
    for name in ['best_guarded.pt','best_structure.pt','last.pt']:
        p=live/name
        if p.exists():shutil.copy2(p,OUT/'checkpoints/continuation'/name)
    comp=read(live/'comparison.json')
    selected=[('original',comp['reference']),('start_epoch3',comp['continuation_start'])]
    selected.extend((f'epoch{r["epoch"]}',r['validation']['summary']) for r in comp['epochs'] if r['epoch']in [10,30])
    keys=['aligned_rmsd','rmsd','contact_f1','interface_fape','cn_mae','clash','aar','omega_planarity_deviation_deg']
    with (OUT/'comparisons/final_continuation.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['model','dataset','n']+keys);writer.writeheader()
        for name,m in selected:writer.writerow(dict(model=name,dataset='human_validation18',n=18,**{k:m[k] for k in keys}))
    description=dict(training_complete=True,final_epoch=30,
        selection=read(live/'selection.json'),
        conclusion='Longer continuation did not improve the best observed structural composite beyond epoch3. Epoch10 is best among geometry-eligible checkpoints. Do not replace baseline automatically.',
        prior_snapshot_retained=True,baseline_replaced=False)
    (final/'handoff.json').write_text(json.dumps(description,indent=2)+'\n')
    shutil.copy2(Path(__file__),OUT/'code/scripts'/Path(__file__).name)
    manifest={str(p.relative_to(OUT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in OUT.rglob('*') if p.is_file() and p.name!='MANIFEST.json'}
    (OUT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(path=str(OUT),files=len(manifest),bytes=sum(v['bytes'] for v in manifest.values()),final_epoch=30)))


if __name__=='__main__':main()
