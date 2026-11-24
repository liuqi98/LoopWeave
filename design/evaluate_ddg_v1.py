"""Score the designed CDRH3 sequences by Rosetta interface ddG.

Protocol is the archived one (``cal_AAR/cal_dG_pyrosetta.py``): REF2015
``get_score_function()`` with ``InterfaceAnalyzerMover(1)``, ``set_pack_input`` and
``set_pack_separated`` both on, reporting ``get_separated_interface_energy()`` and
``get_interface_delta_sasa()``.

What is being compared
----------------------
Each design keeps the experimental complex exactly as measured and only its H3
sequence is threaded onto the native loop backbone. Therefore

    ddG = dG(design, native loop conformation) - dG(native sequence, same conformation)

isolates the **sequence** contribution to the interface at the crystallographic
pose. It is not a binding-affinity prediction: no docking, no refolding, no
backbone relaxation. The model's own predicted loop backbone is deliberately not
used here, because its peptide bonds deviate by 0.4-1.4 A (versus 0.27 A
in-distribution), which would make the Rosetta energy a measure of broken geometry
rather than of interface complementarity; that comparison would need backbone repair
first and answers a different question.

Interpretation of ddG (Rosetta energy units, REU): negative = more favourable
interface than the native loop; positive = less favourable.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pyrosetta
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
from pyrosetta.rosetta.core.scoring import get_score_function

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ['gp120_G36', 'cMET_1_7', 'PRV_Nb36', 'PRV_Nb1']
ONE_TO_THREE = {
    'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE', 'G': 'GLY',
    'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU', 'M': 'MET', 'N': 'ASN',
    'P': 'PRO', 'Q': 'GLN', 'R': 'ARG', 'S': 'SER', 'T': 'THR', 'V': 'VAL',
    'W': 'TRP', 'Y': 'TYR'}
THREE_TO_ONE = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V',
    'TRP': 'W', 'TYR': 'Y'}


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def interface_energy(pose, iam_factory):
    iam = iam_factory()
    iam.apply(pose)
    return dict(dG=float(iam.get_separated_interface_energy()),
                dSASA=float(iam.get_interface_delta_sasa()))


def h3_pose_indices(pose, chain='H', low=105, high=117):
    """Pose indices of the IMGT 105-117 loop on the antibody chain."""
    info = pose.pdb_info()
    indices = []
    for index in range(1, pose.total_residue() + 1):
        if info.chain(index) != chain:
            continue
        if low <= info.number(index) <= high:
            indices.append(index)
    return indices


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--designs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--targets', nargs='+', default=TARGETS)
    args = parser.parse_args()
    pyrosetta.init('-mute all -ex1 -ex2aro -out:level 300')
    get_score_function()                                   # force REF2015 load
    iam_factory = lambda: InterfaceAnalyzerMover(1)

    root = args.designs.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    summary = {}
    for target in args.targets:
        directory = root / target
        document = json.loads((directory / 'designs.json').read_text())
        pdb = directory / 'complex_imgt.pdb'
        if not pdb.is_file():
            summary[target] = dict(status='missing complex')
            continue

        native_pose = pyrosetta.pose_from_pdb(str(pdb))
        loop = h3_pose_indices(native_pose)
        native_sequence = ''.join(
            THREE_TO_ONE[native_pose.residue(i).name3()] for i in loop)
        if native_sequence != document['native_h3']:
            raise ValueError(f'{target}: H3 sequence mismatch {native_sequence} vs '
                             f"{document['native_h3']}")
        start = time.monotonic()
        baseline = interface_energy(native_pose, iam_factory)
        baseline_seconds = time.monotonic() - start

        rows = []
        for rank, design in enumerate(document['design_sequences'], 1):
            sequence = design['sequence']
            if len(sequence) != len(loop):
                raise ValueError('design length differs from the native loop')
            pose = native_pose.clone()
            for index, amino_acid in zip(loop, sequence):
                # This PyRosetta build core-dumps on the one-letter form, so the
                # mutation is always issued with the three-letter residue name.
                MutateResidue(index, ONE_TO_THREE[amino_acid]).apply(pose)
            start = time.monotonic()
            result = interface_energy(pose, iam_factory)
            rows.append(dict(
                rank_in_model=rank, sequence=sequence, seed=design['seed'],
                source=design['source'], model_logprob=design['mean_logprob'],
                dG=result['dG'], dSASA=result['dSASA'],
                ddG=result['dG'] - baseline['dG'],
                native_recovery=sum(a == b for a, b in zip(sequence, native_sequence))
                / len(native_sequence),
                seconds=round(time.monotonic() - start, 2)))
            print(json.dumps(dict(target=target, rank=rank, ddG=round(rows[-1]['ddG'], 2))),
                  flush=True)

        rows.sort(key=lambda row: row['ddG'])
        better = [row for row in rows if row['ddG'] < 0]
        summary[target] = dict(
            native_sequence=native_sequence, native_dG=baseline['dG'],
            native_dSASA=baseline['dSASA'], loop_length=len(loop),
            n_designs=len(rows),
            n_better_than_native=len(better),
            best_ddG=rows[0]['ddG'], best_sequence=rows[0]['sequence'],
            median_ddG=float(sorted(row['ddG'] for row in rows)[len(rows) // 2]),
            worst_ddG=rows[-1]['ddG'],
            spearman_logprob_vs_ddg=_spearman(
                [row['model_logprob'] for row in rows], [row['ddG'] for row in rows]),
            baseline_seconds=round(baseline_seconds, 2),
            designs=rows)
        dump(out / f'ddg_{target}.json', summary[target])
        print(json.dumps(dict(target=target, native_dG=round(baseline['dG'], 2),
                              n_better=len(better), best=round(rows[0]['ddG'], 2))), flush=True)

    dump(out / 'ddg_summary.json', dict(
        protocol='REF2015 get_score_function; InterfaceAnalyzerMover(1) with pack_input and '
                 'pack_separated; dG = separated interface energy',
        comparison='design sequence threaded onto the native loop backbone at the '
                   'crystallographic pose; ddG = dG(design) - dG(native)',
        units='Rosetta energy units (REU); negative ddG means a more favourable interface',
        targets=summary,
        limitations=['Sequence-only comparison: the backbone and the pose are the experimental ones, '
                     'so this measures interface complementarity of the sequence, not the folded '
                     'structure of the design.',
                     'No docking, refolding or backbone relaxation; not a binding-affinity prediction.',
                     'Rosetta interface energies are a coarse proxy; ordering of similar designs is '
                     'not resolvable.',
                     'Single repack per pose (one stochastic packing trajectory), so absolute ddG '
                     'carries repack noise of order 1 REU.']))
    print(json.dumps({t: {k: v[k] for k in
                          ('native_dG', 'n_better_than_native', 'best_ddG', 'median_ddG') if k in v}
                      for t, v in summary.items() if 'native_dG' in v}, indent=1))


def _spearman(first, second):
    if len(first) < 3:
        return None
    import numpy as np
    from scipy.stats import spearmanr
    value = spearmanr(first, second).correlation
    return None if value != value else float(value)


if __name__ == '__main__':
    main()
