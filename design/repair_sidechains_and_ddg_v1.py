"""Repair sidechains with Rosetta, then score design ddG on the repaired complex.

Motivation: the raw ddG pass produced nonphysical baselines for some targets
(native dG +29035 REU for PRV). A geometric scan showed 67-97 heavy-atom pairs below
1.5 A, i.e. the deposited/modelled complexes interpenetrate. Rosetta sidechain
repair is the first thing to try, and it is what this script does:

  stage A (repair, once per target)
      full-structure repack (``RestrictToRepacking`` on every residue) followed by a
      sidechain-only L-BFGS minimisation (``MoveMap`` with ``set_bb(False)``).
      Backbone coordinates are never moved, so the experimental pose is preserved
      and nothing is invented.

  stage B (ddG)
      baseline = repaired native complex;
      each design = the same repaired pose with its H3 sidechains rebuilt by
      ``MutateResidue``, then the same repack + sidechain minimisation, then
      ``InterfaceAnalyzerMover`` (REF2015, ``pack_input`` and ``pack_separated``).
      ddG = dG(design) - dG(native), both measured after identical treatment.

Acceptance criteria (fixed before looking at the numbers)
--------------------------------------------------------
A target is only reported as evaluable if, after repair:

  A1  zero non-adjacent backbone-backbone heavy-atom pairs below 1.5 A,
  A2  native separated interface dG < 0,
  A3  native interface dSASA in 200-3000 A^2 (a physically plausible antibody
      interface; below that there is essentially no interface).

If a target fails, sidechain repair cannot fix it and its ddG is reported as not
evaluable rather than as a result.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyrosetta
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepacking
from pyrosetta.rosetta.core.scoring import get_score_function
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.protocols.minimization_packing import MinMover, PackRotamersMover
from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ['gp120_G36', 'cMET_1_7', 'PRV_Nb36', 'PRV_Nb1']
ONE_TO_THREE = {'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE', 'G': 'GLY',
                'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU', 'M': 'MET', 'N': 'ASN',
                'P': 'PRO', 'Q': 'GLN', 'R': 'ARG', 'S': 'SER', 'T': 'THR', 'V': 'VAL',
                'W': 'TRP', 'Y': 'TYR'}
THREE_TO_ONE = {value: key for key, value in ONE_TO_THREE.items()}
BACKBONE = {'N', 'CA', 'C', 'O'}
CLASH_CUTOFF = 1.5


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def clash_profile(pose, cutoff=CLASH_CUTOFF):
    """Non-adjacent heavy-atom pairs below the cutoff, split by atom class."""
    xyz, meta = [], []
    for index in range(1, pose.total_residue() + 1):
        residue = pose.residue(index)
        for atom in range(1, residue.natoms() + 1):
            if residue.atom_is_hydrogen(atom):
                continue
            xyz.append(residue.xyz(atom))
            meta.append((index, residue.atom_name(atom).strip()))
    xyz = np.array(xyz)
    pairs = cKDTree(xyz).query_pairs(cutoff, output_type='ndarray')
    counts = dict(bb_bb=0, bb_sc=0, sc_sc=0)
    for first, second in pairs:
        if abs(meta[first][0] - meta[second][0]) <= 1:
            continue
        a = meta[first][1] in BACKBONE
        b = meta[second][1] in BACKBONE
        if a and b:
            counts['bb_bb'] += 1
        elif a or b:
            counts['bb_sc'] += 1
        else:
            counts['sc_sc'] += 1
    counts['total'] = counts['bb_bb'] + counts['bb_sc'] + counts['sc_sc']
    counts['atoms'] = int(len(xyz))
    return counts


def interface_energy(pose, score_function):
    mover = InterfaceAnalyzerMover(1)
    mover.set_pack_input(True)
    mover.set_pack_separated(True)
    mover.apply(pose)
    return dict(dG=float(mover.get_separated_interface_energy()),
                dSASA=float(mover.get_interface_delta_sasa()))


def repair_sidechains(pose, score_function):
    """Repack every residue, then minimise chi only (backbone fixed)."""
    factory = TaskFactory()
    factory.push_back(RestrictToRepacking())
    PackRotamersMover(score_function,
                      factory.create_task_and_apply_taskoperations(pose)).apply(pose)
    move_map = MoveMap()
    move_map.set_bb(False)
    move_map.set_chi(True)
    move_map.set_jump(False)
    MinMover(move_map, score_function, 'lbfgs_armijo_nonmonotone', 1e-4, True).apply(pose)
    return pose


def h3_indices(pose, chain='H', low=105, high=117):
    info = pose.pdb_info()
    return [i for i in range(1, pose.total_residue() + 1)
            if info.chain(i) == chain and low <= info.number(i) <= high]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--designs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repaired-dir', type=Path, required=True)
    parser.add_argument('--targets', nargs='+', default=TARGETS)
    args = parser.parse_args()
    pyrosetta.init('-mute all -ex1 -ex2aro -out:level 300')
    score_function = get_score_function()

    root = args.designs.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    repaired_dir = args.repaired_dir.resolve()
    repaired_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for target in args.targets:
        directory = root / target
        document = json.loads((directory / 'designs.json').read_text())
        pdb = directory / 'complex_imgt.pdb'
        if not pdb.is_file():
            summary[target] = dict(status='missing complex')
            continue

        pose = pyrosetta.pose_from_pdb(str(pdb))
        before = clash_profile(pose)
        native_raw = interface_energy(pose.clone(), score_function)
        tick = time.monotonic()
        repair_sidechains(pose, score_function)
        repair_seconds = time.monotonic() - tick
        after = clash_profile(pose)
        native = interface_energy(pose.clone(), score_function)
        repaired_path = repaired_dir / f'{target}_repaired.pdb'
        pose.dump_pdb(str(repaired_path))

        evaluable = (after['bb_bb'] == 0 and native['dG'] < 0
                     and 200 <= native['dSASA'] <= 3000)
        reasons = []
        if after['bb_bb'] != 0:
            reasons.append(f"{after['bb_bb']} backbone-backbone clashes below "
                           f'{CLASH_CUTOFF} A remain after sidechain repair')
        if native['dG'] >= 0:
            reasons.append(f'native separated interface dG is {native["dG"]:.1f} REU, not negative')
        if not 200 <= native['dSASA'] <= 3000:
            reasons.append(f'native interface dSASA is {native["dSASA"]:.0f} A^2, outside the '
                           f'plausible antibody range')

        loop = h3_indices(pose)
        native_sequence = ''.join(THREE_TO_ONE[pose.residue(i).name3()] for i in loop)
        rows = []
        for rank, design in enumerate(document['design_sequences'], 1):
            sequence = design['sequence']
            if len(sequence) != len(loop):
                raise ValueError('design length differs from the native loop')
            mutant = pose.clone()
            for index, amino_acid in zip(loop, sequence):
                MutateResidue(index, ONE_TO_THREE[amino_acid]).apply(mutant)
            repair_sidechains(mutant, score_function)
            result = interface_energy(mutant, score_function)
            rows.append(dict(rank_in_model=rank, sequence=sequence, seed=design['seed'],
                             source=design['source'], model_logprob=design['mean_logprob'],
                             dG=result['dG'], dSASA=result['dSASA'],
                             ddG=result['dG'] - native['dG'],
                             native_recovery=sum(a == b for a, b in
                                                 zip(sequence, native_sequence))
                             / len(native_sequence)))
            print(json.dumps(dict(target=target, rank=rank,
                                  ddG=round(rows[-1]['ddG'], 2))), flush=True)

        rows.sort(key=lambda row: row['ddG'])
        summary[target] = dict(
            repair_seconds=round(repair_seconds, 1),
            clashes_before=before, clashes_after=after,
            native_dG_before_repair=native_raw['dG'], native_dSASA_before_repair=native_raw['dSASA'],
            native_dG=native['dG'], native_dSASA=native['dSASA'],
            native_sequence=native_sequence, loop_length=len(loop),
            evaluable=bool(evaluable), not_evaluable_reasons=reasons,
            n_designs=len(rows),
            n_better_than_native=None if not evaluable else len([r for r in rows if r['ddG'] < 0]),
            best_ddG=rows[0]['ddG'], best_sequence=rows[0]['sequence'],
            median_ddG=float(sorted(row['ddG'] for row in rows)[len(rows) // 2]),
            designs=rows)
        dump(out / f'ddg_repaired_{target}.json', summary[target])
        print(json.dumps(dict(target=target, evaluable=bool(evaluable),
                              native_dG=round(native['dG'], 2),
                              bb_bb=after['bb_bb'], dSASA=round(native['dSASA']),
                              best_ddG=round(rows[0]['ddG'], 2))), flush=True)

    dump(out / 'ddg_repaired_summary.json', dict(
        protocol='stage A: RestrictToRepacking full-structure repack + chi-only L-BFGS '
                 'minimisation (backbone fixed); stage B: same treatment on each mutant, then '
                 'REF2015 InterfaceAnalyzerMover(1) with pack_input and pack_separated',
        acceptance=['zero non-adjacent backbone-backbone heavy-atom pairs below 1.5 A',
                    'native separated interface dG < 0',
                    'native interface dSASA in 200-3000 A^2'],
        targets=summary,
        limitations=['Only sidechains are repaired; the backbone stays at the deposited/modelled '
                     'coordinates, so a complex whose peptide geometry is broken cannot be rescued.',
                     'Sequence-only comparison at a fixed pose: not a binding-affinity prediction.',
                     'One repack trajectory per point; absolute ddG carries repack noise.']))
    print(json.dumps({t: {k: v.get(k) for k in ('evaluable', 'native_dG', 'best_ddG',
                                                'n_better_than_native')}
                      for t, v in summary.items() if 'evaluable' in v}, indent=1))


if __name__ == '__main__':
    main()
