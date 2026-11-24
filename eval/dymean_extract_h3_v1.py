"""Convert locally generated dyMEAN complexes into the two-chain scoring format.

The released ``generate.py`` writes full complexes (antigen plus heavy and light chains,
with side chains), whereas the matched comparison scores a two-chain file containing only the
H3 loop and the antigen, N/CA/C/O only. The H3 is located by matching the native H3 sequence
inside the heavy chain of the ``_original`` complex, which shares the numbering with the
generated complex; the same residue positions are then taken from the generated heavy chain.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from Bio.PDB import PDBParser, PDBIO, Select

PACKAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, '/media/xd/code1/dyGF')
WORKSPACE = Path('/media/xd/code1/dyGF')
CANONICAL = 'ACDEFGHIKLMNPQRSTVWY'
THREE = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
         'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
         'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
         'TYR': 'Y', 'VAL': 'V'}
BACKBONE = ('N', 'CA', 'C', 'O')


def residues(chain):
    return [r for r in chain if r.id[0] == ' ' and r.resname in THREE]


def find_unique_subsequence(haystack, needle):
    hits = [i for i in range(len(haystack) - len(needle) + 1)
            if haystack[i:i + len(needle)] == needle]
    if len(hits) != 1:
        raise ValueError(f'H3 sequence match is not unique: {len(hits)} hits')
    return hits[0]


class BackboneSelect(Select):
    def accept_residue(self, residue):
        return residue.id[0] == ' '


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2026, 2027, 2028])
    parser.add_argument('--runs', type=Path, default=WORKSPACE / 'outputs')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    rows = torch.load(WORKSPACE / 'data/abgaze_reconstructed_v1/tensors/test.pt',
                      map_location='cpu', weights_only=False)
    meta = {}
    with open('/media/xd/code2/dyMEAN-main/all_data/RAbD/test.json') as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                meta[record['pdb']] = record
    report = {}
    for seed in args.seeds:
        directory = args.runs / f'dymean_generated_seed{seed}'
        target = out / f'seed{seed}'
        target.mkdir()
        written = []
        for sid, x, y in rows:
            native_h3 = ''.join(CANONICAL[i] for i in y['sequence'].tolist())
            record = meta[sid]
            generated = PDBParser(QUIET=True).get_structure('g', str(directory / f'{sid}.pdb'))[0]
            original = PDBParser(QUIET=True).get_structure('o',
                                                           str(directory / f'{sid}_original.pdb'))[0]
            heavy_id = record['heavy_chain']
            heavy_original = residues(original[heavy_id])
            heavy_generated = residues(generated[heavy_id])
            sequence = ''.join(THREE[r.resname] for r in heavy_original)
            start = find_unique_subsequence(sequence, native_h3)
            if len(heavy_generated) != len(heavy_original):
                raise ValueError(f'{sid}: heavy chain length changed')
            h3 = heavy_generated[start:start + len(native_h3)]
            antigen = []
            for chain_id in record['antigen_chains']:
                antigen.extend(residues(generated[chain_id]))
            lines = []
            serial = 1
            for chain_label, group in (('H', h3), ('A', antigen)):
                for index, residue in enumerate(group, 1):
                    for atom in BACKBONE:
                        if atom not in residue:
                            continue
                        coord = residue[atom].coord
                        # Fixed-column PDB record: atom 13-16, altLoc 17, resName 18-20,
                        # chain 22, resSeq 23-26, x/y/z 31-54, element 77-78.
                        lines.append(
                            'ATOM  %5d %4s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s\n'
                            % (serial, atom, residue.resname, chain_label, index,
                               coord[0], coord[1], coord[2], 1.0, 0.0, atom[0]))
                        serial += 1
                lines.append('TER\n')
            (target / f'{sid}_dymean.pdb').write_text(''.join(lines))
            designed = ''.join(THREE[r.resname] for r in h3)
            written.append(dict(sample_id=sid, h3_length=len(h3), antigen_length=len(antigen),
                                heavy_start=start, designed_h3=designed, native_h3=native_h3,
                                aar=sum(a == b for a, b in zip(designed, native_h3))
                                    / len(native_h3)))
        report[str(seed)] = written
        first = written[0]
        print(f'seed{seed}: {len(written)} structures, first {first}')
    (out / 'extraction_report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
