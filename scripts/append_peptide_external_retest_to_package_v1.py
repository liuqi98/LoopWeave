"""Append completed fixed-checkpoint external re-evaluation to the scoped bundle."""
import csv
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from package_scoped_research_snapshot_v1 import OUT,sha,read


def main():
    source=ROOT/'outputs/peptide_continuation_external_test_v1'
    assert read(source/'status.json')['phase']=='complete'
    lock=read(source/'evaluation_lock.json')
    for name,digest in lock['code_sha256'].items():assert sha(ROOT/name)==digest,name
    for checkpoint in lock['checkpoints'].values():assert sha(Path(checkpoint['path']))==checkpoint['sha256']
    dest=OUT/'results'/source.name;dest.mkdir(exist_ok=False)
    for p in source.glob('*.json'):
        shutil.copy2(p,dest/p.name);assert sha(p)==sha(dest/p.name)
    script=ROOT/'eval/test_peptide_continuation_external_v1.py'
    shutil.copy2(script,OUT/'code/eval'/script.name)
    shutil.copy2(Path(__file__),OUT/'code/scripts'/Path(__file__).name)
    d=read(source/'comparison.json')
    with (OUT/'comparisons/peptide_external_retest.csv').open('x',newline='') as f:
        keys=['aligned_rmsd','rmsd','contact_f1','interface_fape','cn_mae','clash','aar']
        writer=csv.DictWriter(f,fieldnames=['model','n','seed','scope']+keys);writer.writeheader()
        for name,m in d['summary'].items():writer.writerow(dict(model=name,n=20,seed=2026,
            scope='Previously inspected external cohort; not independent confirmation',**{k:m[k] for k in keys}))
    package=read(OUT/'PACKAGE.json')
    package.update(final_training_complete=True,final_training_epoch=30,
        external_candidate_retest_complete=True,
        note='snapshot field preserves the initial epoch12 capture; final_training and latest external results are appended separately.')
    (OUT/'PACKAGE.json').write_text(json.dumps(package,indent=2)+'\n')
    manifest={str(p.relative_to(OUT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in OUT.rglob('*') if p.is_file() and p.name!='MANIFEST.json'}
    (OUT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(files=len(manifest),bytes=sum(v['bytes'] for v in manifest.values()))))


if __name__=='__main__':main()
