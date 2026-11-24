"""Extract GeoAB designs and score them with the manuscript's own metric functions.

GeoAB-Designer was run from the authors' released checkpoint (`cdrh3.zip`) on its own RAbD_H3
60-target test split, which is the same benchmark, by calling the model and `set_cdr` directly
because the authors' evaluation stage depends on an external lDDT binary and Docker TM-score.
This script converts each designed complex into the two-chain N/CA/C/O format used by the matched
comparison and scores it with the same functions, so the GeoAB row is directly comparable.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PACKAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE / 'code'))
sys.path.insert(0, '/media/xd/code1/dyGF')
from eval.three_method_diagnostics_v1 import (MEASURES, measure, read_saved_structure)
from eval.tmscore_caar_v1 import THREE as ONE_TO_THREE, tmscore, write_pdb

WORKSPACE = Path('/media/xd/code1/dyGF')
GEOAB = WORKSPACE / 'external/geoab_upstream'
ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
THREE = {three: one for one, three in ONE_TO_THREE.items()}  # three-letter -> one-letter


def residues(chain):
    return [r for r in chain if r.id[0] == ' ' and r.resname in THREE]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--designs', type=Path, nargs='+',
                        default=[GEOAB / 'results/cdrh3/H3_design/results/design_1'])
    parser.add_argument('--meta', type=Path, default=WORKSPACE / 'outputs/geoab_meta.json')
    parser.add_argument('--reference', type=Path,
                        default=PACKAGE / 'results/three_method_diagnostics_v1'
                                  / 'three_method_diagnostics.json')
    parser.add_argument('--replicates', type=int, default=10000)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    from Bio.PDB import PDBParser
    import warnings
    warnings.filterwarnings('ignore')

    meta = {row['pdb']: row for row in json.loads(args.meta.read_text())}
    dataset = WORKSPACE / 'data/abgaze_reconstructed_v1'
    pairs = torch.load(WORKSPACE / 'data/raad_abgaze_reconstructed_v1/test.pt',
                       map_location='cpu', weights_only=False)
    items = {item['sample_id']: item for item, _ in pairs}
    graphs = {item['sample_id']: graph for item, graph in pairs}
    rows = torch.load(dataset / 'tensors/test.pt', map_location='cpu', weights_only=False)
    ids = [row[0] for row in rows]
    clusters = {json.loads(line)['pdb']: f"cluster_{json.loads(line)['cluster']}"
                for line in (dataset / 'splits/test.json').read_text().splitlines() if line.strip()}

    native = {}
    for sid, x, y in rows:
        mask = x['antigen_mask']
        native[sid] = dict(x=x, y=y, antigen=x['context_bb'][mask],
                           antigen_atom_mask=x['context_mask'][mask],
                           native_ca=y['bb'][y['mask'][:, 1], 1])
        native[sid]['native_contacts'] = torch.cdist(native[sid]['native_ca'],
                                                     native[sid]['antigen'][:, 1]) <= 8.

    structure_dir = out / 'structures'
    structure_dir.mkdir()
    records = []
    with __import__('tempfile').TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for run_index, design_dir in enumerate(args.designs):
          for sid in ids:
            path = design_dir / f'{sid}.pdb'
            if not path.is_file():
                raise FileNotFoundError(path)
            structure = PDBParser(QUIET=True).get_structure('g', str(path))[0]
            info = meta[sid]
            heavy = residues(structure[info['heavy']])
            h3 = heavy[info['cdr_start']:info['cdr_end'] + 1]
            antigen_residues = []
            for chain_id in info['antigen']:
                antigen_residues.extend(residues(structure[chain_id]))
            h3_sequence = ''.join(THREE[r.resname] for r in h3)
            # Two-chain file, N/CA/C/O only, in the native complex frame.
            lines, serial = [], 1
            for chain_label, group in (('A', h3), ('B', antigen_residues)):
                for index, residue in enumerate(group, 1):
                    for atom in ('N', 'CA', 'C', 'O'):
                        if atom not in residue:
                            continue
                        xyz = residue[atom].coord
                        lines.append('ATOM  %5d  %-3s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s\n'
                                     % (serial, atom, residue.resname, chain_label, index,
                                        xyz[0], xyz[1], xyz[2], 1., 0., atom[0]))
                        serial += 1
                lines.append('TER\n')
            converted = structure_dir / (f'{sid}_geoab.pdb' if run_index == 0
                                          else f'{sid}_geoab_run{run_index}.pdb')
            converted.write_text(''.join(lines) + 'END\n')

            parsed = read_saved_structure(converted)
            retained = torch.as_tensor(graphs[sid]['antigen_original_indices'])
            if len(parsed['antigen_bb']) != len(items[sid]['antigen_seq']):
                raise ValueError(f'{sid}: antigen length mismatch')
            antigen = parsed['antigen_bb'][retained]
            antigen_mask = parsed['antigen_mask'][retained]
            n = native[sid]
            entry = measure(parsed['loop_bb'], antigen, antigen_mask, n['y']['bb'],
                            n['antigen'], n['antigen_atom_mask'], n['native_ca'],
                            n['native_contacts'])
            native_sequence = ''.join(ALPHABET[i] for i in n['y']['sequence'].tolist())
            correct = np.array([a == b for a, b in zip(h3_sequence, native_sequence)], float)
            contacts = (torch.cdist(n['native_ca'], n['antigen'][:, 1]) <= 8.).any(1).numpy()
            entry.update(sample_id=sid, group=clusters[sid], run=run_index,
                         aar=float(correct.mean()),
                         caar=float(correct[contacts].mean()) if contacts.any() else None,
                         designed_sequence=h3_sequence, native_sequence=native_sequence)
            # TM-score on the H3 loop, matching the additional-metric table.
            write_pdb(tmp / 'model.pdb', [('A', h3_sequence,
                                           parsed['loop_bb'].numpy(), parsed['loop_mask'].numpy())])
            write_pdb(tmp / 'native.pdb', [('A', native_sequence, n['y']['bb'].numpy(),
                                            n['y']['mask'].numpy())])
            entry['tm_h3'] = tmscore(tmp / 'model.pdb', tmp / 'native.pdb', tmp)
            records.append(entry)

    per_run = {}
    for record in records:
        per_run.setdefault(record['run'], []).append(record)
    run_means = {key: [float(np.mean([r[key] for r in part if r.get(key) is not None]))
                       for part in per_run.values()] for key in list(MEASURES) + ['aar', 'caar', 'tm_h3']}
    summary = dict(n=len(per_run[0]), runs=len(per_run), source='GeoAB-Designer, authors released checkpoint',
                   designs=str(args.designs), measures=list(MEASURES) + ['aar', 'caar', 'tm_h3'])
    for key in summary['measures']:
        values = run_means[key]
        summary[key] = dict(mean=float(np.mean(values)),
                            sd=float(np.std(values, ddof=1)) if len(values) > 1 else 0.,
                            per_run=values, n=summary['n'])
    # Paired contrasts against RAAD, using the reference diagnostics.
    reference = json.loads(args.reference.read_text())
    raad = reference['per_complex']['raad']['saved']
    from research_audit.statistics import interval
    groups = [clusters[sid] for sid in ids]
    lower = {'cn_internal', 'planarity', 'clash_internal', 'clash_external', 'interface_fape',
             'loop_aligned_rmsd', 'antigen_aligned_h3_rmsd'}
    summary['contrasts_vs_raad'] = {}
    averaged = {key: {sid: float(np.mean([r[key] for r in records
                                          if r['sample_id'] == sid and r.get(key) is not None]))
                      for sid in ids} for key in MEASURES}
    for key in MEASURES:
        delta = np.array([[(-1. if key in lower else 1.) *
                           (averaged[key][sid] - raad[key][i]) for i, sid in enumerate(ids)]])
        stats = interval(delta, groups, args.replicates)
        summary['contrasts_vs_raad'][key] = dict(
            improvement=stats['mean'], ci95=stats['conditional_group_ci95'],
            significant=bool(stats['conditional_group_ci95'][0] *
                             stats['conditional_group_ci95'][1] > 0))
    (out / 'geoab_scores.json').write_text(json.dumps(dict(summary=summary, per_complex=records),
                                                      indent=2) + '\n')
    for key in summary['measures']:
        block = summary[key]
        print('%-26s %.4f +- %.4f (n=%d)' % (key, block['mean'], block['sd'], block['n']))


if __name__ == '__main__':
    main()
