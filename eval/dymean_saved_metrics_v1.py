"""Metrics for the saved historical dyMEAN structures, computed with the same functions as
the matched comparison, so the numbers can be reported next to it.

What is available
-----------------
Each saved structure contains the predicted H3 loop (one chain, N/CA/C/O) and a predicted
antigen chain. There is **no framework**, so the two flanking framework connections cannot
be evaluated; only in-loop C--N is defined. The saved relative antibody--antigen pose is
dyMEAN's own prediction, not the native framework pose used by the current model, so raw
coordinates are not comparable and raw RMSD is not reported.

What is computed
----------------
* in-loop C--N MAE (consecutive loop residues only, ideal 1.329 A)
* frame-aligned FAPE, using dyMEAN's own antigen backbone against the native antigen points
  in the corresponding H3 frames (pose-invariant by construction)
* hard 8 A CA contact F1 against the native contact set, in dyMEAN's own pose
* C-alpha RMSD of the loop after Kabsch superposition on the loop itself
* amino-acid recovery against the native loop sequence

DockQ, Fnat, iRMSD and LRMSD for this artifact were already computed in
``outputs/dockq_dymean_saved_diagnostic_v1``; they are read from there rather than
recomputed.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from orientation_interface_refiner import interface_fape
from structure_refinement_ablation import residue_geometry
from research_audit.statistics import interval

WORKSPACE = Path('/media/xd/code1/dyGF')
CANONICAL = 'ACDEFGHIKLMNPQRSTVWY'
THREE = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
         'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
         'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
         'TYR': 'Y', 'VAL': 'V'}
BACKBONE = ('N', 'CA', 'C', 'O')


def read_structure(path):
    """Backbone tensor for the loop chain and the antigen chain."""
    from Bio.PDB import PDBParser
    structure = PDBParser(QUIET=True).get_structure('d', str(path))[0]
    chains = []
    for chain in structure:
        residues = [r for r in chain if r.id[0] == ' ' and r.resname in THREE]
        if residues:
            chains.append(residues)
    if len(chains) != 2:
        raise ValueError(f'{path.name}: expected two chains, found {len(chains)}')
    chains.sort(key=len)                     # loop is the short chain
    loop, antigen = chains
    def pack(residues):
        coords, mask, sequence = [], [], []
        for residue in residues:
            row = []
            for atom in BACKBONE:
                if atom in residue:
                    row.append(residue[atom].coord)
                    mask.append(True)
                else:
                    row.append(np.zeros(3, dtype=np.float32))
                    mask.append(False)
            coords.append(row)
            sequence.append(THREE[residue.resname])
        return (torch.tensor(np.asarray(coords), dtype=torch.float32),
                torch.tensor(np.asarray(mask).reshape(len(residues), 4)),
                ''.join(sequence))
    loop_bb, loop_mask, loop_seq = pack(loop)
    antigen_bb, antigen_mask, antigen_seq = pack(antigen)
    return dict(loop_bb=loop_bb, loop_mask=loop_mask, loop_seq=loop_seq,
                antigen_bb=antigen_bb, antigen_mask=antigen_mask, antigen_seq=antigen_seq)


def cn_internal(prediction):
    """Mean |C-N - 1.329| over consecutive loop residues (no framework flanks available)."""
    c, n = prediction[:, 2], prediction[:, 0]
    bonds = (c[:-1] - n[1:]).norm(dim=-1)
    return float((bonds - 1.329).abs().mean())


def contact_f1(prediction, x, y, threshold=8.):
    context = x['context_bb'][x['antigen_mask'] & x['context_mask'][:, 1], 1]
    if len(context) < 1 or len(prediction) < 1:
        return float('nan')
    native_ca = y['bb'][y['mask'][:, 1], 1]
    distances = torch.cdist(prediction[:, 1], context)
    truth = torch.cdist(native_ca, context)
    contacts = truth <= threshold
    predicted = distances <= threshold
    return float(2 * (predicted & contacts).sum() /
                 (predicted.sum() + contacts.sum()).clamp_min(1))


def kabsch_rmsd(first, second):
    first, second = first.double(), second.double()
    first = first - first.mean(0, keepdim=True)
    second = second - second.mean(0, keepdim=True)
    u, _, vt = torch.linalg.svd(first.T @ second)
    d = torch.sign(torch.linalg.det(vt.T @ u.T))
    rotation = vt.T @ torch.diag(torch.tensor([1., 1., d], dtype=torch.double)) @ u.T
    return float((first @ rotation.T - second).square().sum(-1).mean().sqrt())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--structures', type=Path,
                        default=WORKSPACE / 'outputs/dockq_dymean_saved_diagnostic_v1/structures')
    parser.add_argument('--dockq-dir', type=Path,
                        default=WORKSPACE / 'outputs/dockq_dymean_saved_diagnostic_v1')
    parser.add_argument('--replicates', type=int, default=10000)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    rows = torch.load(WORKSPACE / 'data/abgaze_reconstructed_v1/tensors/test.pt',
                      map_location='cpu', weights_only=False)
    ids = [row[0] for row in rows]
    clusters = {json.loads(line)['pdb']: f"cluster_{json.loads(line)['cluster']}"
                for line in (WORKSPACE / "data/abgaze_reconstructed_v1/splits/test.json")
                .read_text().splitlines() if line.strip()}
    dockq = {}
    for line in (args.dockq_dir / 'per_complex.csv').read_text().splitlines()[1:]:
        parts = line.split(',')
        if len(parts) < 9 or parts[2] != 'dymean':
            continue
        def number(index):
            # Unscorable targets carry empty fields rather than zeros.
            return float(parts[index]) if parts[index].strip() else None
        dockq[parts[0]] = dict(scorable=parts[4] == 'True', DockQ=number(5),
                               iRMSD=number(6), LRMSD=number(7), fnat=number(8))

    per_complex, skipped = [], []
    for sid, x, y in rows:
        path = args.structures / f'{sid}_dymean.pdb'
        if not path.is_file():
            skipped.append(dict(sample_id=sid, reason='missing structure'))
            continue
        parsed = read_structure(path)
        if len(parsed['loop_bb']) != len(y['bb']):
            skipped.append(dict(sample_id=sid, reason='loop length mismatch',
                                dyMEAN=len(parsed['loop_bb']), native=len(y['bb'])))
            continue
        native_sequence = ''.join(CANONICAL[i] for i in y['sequence'].tolist())
        fake_x = dict(context_bb=parsed['antigen_bb'],
                      context_mask=parsed['antigen_mask'],
                      antigen_mask=torch.ones(len(parsed['antigen_bb']), dtype=torch.bool))
        fape = float(interface_fape(parsed['loop_bb'], fake_x, y))
        valid = y['mask'][:, 1] & parsed['loop_mask'][:, 1]
        entry = dict(
            sample_id=sid, group=clusters[sid],
            loop_length=int(len(parsed['loop_bb'])),
            cn_internal=cn_internal(parsed['loop_bb']),
            interface_fape=fape,
            contact_f1=contact_f1(parsed['loop_bb'], fake_x, y),
            aligned_rmsd=float(kabsch_rmsd(parsed['loop_bb'][valid, 1], y['bb'][valid, 1])),
            aar=float(sum(a == b for a, b in zip(parsed['loop_seq'], native_sequence))
                      / len(native_sequence)),
            dockq_scorable=bool(dockq.get(sid, {}).get('scorable', False)),
            dockq=(dockq[sid]['DockQ'] if sid in dockq else None),
            fnat=(dockq[sid]['fnat'] if sid in dockq else None))
        per_complex.append(entry)

    keys = ('cn_internal', 'interface_fape', 'contact_f1', 'aligned_rmsd', 'aar')
    summary = dict(
        source='outputs/dockq_dymean_saved_diagnostic_v1',
        n=len(per_complex), skipped=skipped,
        conditioning=('dyMEAN saves its own predicted antibody-antigen pose and no framework, '
                      'so flank C-N bonds are undefined and raw RMSD is not comparable'),
        metrics={key: dict(mean=statistics.mean(row[key] for row in per_complex),
                           median=statistics.median(row[key] for row in per_complex),
                           n=len(per_complex)) for key in keys},
        dockq=dict(mean=statistics.mean(row['dockq'] for row in per_complex
                                        if row['dockq'] is not None),
                   n=sum(1 for row in per_complex if row['dockq'] is not None)),
        fnat=dict(mean=statistics.mean(row['fnat'] for row in per_complex
                                       if row['fnat'] is not None)),
        notes=['in-loop C-N excludes the two framework flanks because dyMEAN saves no framework',
               'FAPE uses dyMEAN own antigen backbone against native antigen points in the '
               'corresponding H3 frames, so it is pose-invariant',
               'contact F1 is evaluated in dyMEAN own predicted pose and is not matched to the '
               'native framework pose used by the current model',
               'aligned RMSD superposes the loop on itself, so it is pose-invariant',
               'AAR compares the saved loop sequence with the native loop sequence'])
    # intervals over dependence groups
    for key in keys:
        values = np.array([[row[key] for row in per_complex]])
        stats = interval(values, [row['group'] for row in per_complex], args.replicates)
        summary['metrics'][key]['mean_ci95'] = stats['conditional_group_ci95']
    with (out / 'dymean_metrics.json').open('x') as handle:
        json.dump(summary, handle, indent=2)
        handle.write('\n')
    import csv
    with (out / 'dymean_per_complex.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_complex[0]))
        writer.writeheader()
        writer.writerows(per_complex)
    print(json.dumps({key: round(summary['metrics'][key]['mean'], 4) for key in keys}, indent=1))
    print(json.dumps(dict(dockq=round(summary['dockq']['mean'], 4),
                          dockq_n=summary['dockq']['n'], skipped=len(skipped)), indent=1))


if __name__ == '__main__':
    main()
