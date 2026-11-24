"""All-atom lDDT for our model versus RAAD on the dev-60 cohort.

Why sidechains have to be rebuilt
---------------------------------
Both methods predict only the CDRH3 backbone (N, CA, C, O). An all-atom lDDT needs
sidechain atoms, so they are rebuilt with Rosetta while the backbone is held fixed
(``RestrictToRepacking`` on the loop, ``PreventRepacking`` elsewhere). This is the
same situation as any design pipeline that folds a designed sequence before scoring.

Why each method gets its own reference
--------------------------------------
lDDT matches atoms by name, so it is only defined for identical atom sets. The two
methods emit different sequences, therefore for each method we build **two**
structures from the same native complex:

    reference : native loop backbone, loop sequence = the method's sequence, repacked
    model     : the method's loop backbone, loop sequence = the method's sequence, repacked

The difference between them is purely geometric, and sequence effects cancel. A
backbone-only lDDT (no packing needed) is reported alongside as a pack-free control.

lDDT definition
---------------
For every scored atom in the reference: take all atoms with reference distance < 15 A
(excluding the same residue), and for each threshold in {0.5, 1, 2, 4} A count the
fraction whose model distance is within that threshold; average over thresholds and
pairs, then over atoms. Scored atoms are the heavy atoms of the loop.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings('ignore')

ORIG = Path('/media/xd/code1/dyGF')
CANONICAL = 'ACDEFGHIKLMNPQRSTVWY'
ONE_TO_THREE = {
    'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE', 'G': 'GLY', 'H': 'HIS',
    'I': 'ILE', 'K': 'LYS', 'L': 'LEU', 'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN',
    'R': 'ARG', 'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'}
BACKBONE_ATOMS = ('N', 'CA', 'C', 'O')
METHODS = {
    'ours': PACKAGE / 'results/aligned_shape_v1/test_w0_original/generated.json',
    'raad': ORIG / 'outputs/raad_abgaze_reconstructed_test_v1/generated.json',
    'mean': ORIG / 'outputs/mean_reconstructed_test_v1/generated.json',
    'diffab': ORIG / 'outputs/diffab_reconstructed_seed2026_v1/generated.json',
}
INCLUSION_RADIUS = 15.0
THRESHOLDS = (0.5, 1.0, 2.0, 4.0)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def lddt(reference: dict, model: dict, atom_selection=None, radius=INCLUSION_RADIUS):
    """lDDT between two atom collections keyed by (chain, resseq, icode, atom name).

    ``atom_selection`` restricts which atoms are scored; all atoms still act as
    neighbours, which is the standard lDDT behaviour.
    """
    keys = [k for k in reference if k in model]
    if len(keys) < 2:
        return None, 0
    ref_xyz = np.array([reference[k] for k in keys])
    mdl_xyz = np.array([model[k] for k in keys])
    residue_of = np.array(['%s%d%s' % (k[0], k[1], k[2]) for k in keys])
    scores, n_pairs = [], 0
    for index, key in enumerate(keys):
        if atom_selection is not None and key not in atom_selection:
            continue
        distances = np.linalg.norm(ref_xyz - ref_xyz[index], axis=1)
        neighbours = np.where((distances < radius) &
                              (residue_of != residue_of[index]))[0]
        if neighbours.size < 2:            # too few pairs to be informative
            continue
        delta = np.abs(np.linalg.norm(mdl_xyz[neighbours] - mdl_xyz[index], axis=1)
                       - distances[neighbours])
        scores.append(np.mean([(delta < threshold).mean() for threshold in THRESHOLDS]))
        n_pairs += int(neighbours.size)
    if not scores:
        return None, 0
    return float(np.mean(scores)), n_pairs


def loop_residues(structure, heavy_chain, start, end):
    """Loop residues by **chain order index** (``cdrh3_pos`` is a sequence index)."""
    from Bio.SeqUtils import seq1
    chain = structure[heavy_chain]
    residues = [r for r in chain
                if r.id[0] == ' ' and seq1(r.resname) in CANONICAL]
    if end >= len(residues):
        raise ValueError('cdrh3_pos runs past the heavy chain')
    return residues[start:end + 1]


def read_complex(path, heavy_chain, loop_ids):
    """Heavy atoms of the complex, plus the loop atom keys."""
    from Bio.PDB import PDBParser
    structure = PDBParser(QUIET=True).get_structure('c', str(path))[0]
    atoms, loop = {}, set()
    for chain in structure:
        for residue in chain:
            if residue.id[0] != ' ':
                continue
            for atom in residue:
                if atom.element == 'H':
                    continue
                key = (chain.id, residue.id[1], residue.id[2].strip(), atom.name.strip())
                atoms[key] = atom.coord
                if chain.id == heavy_chain and (residue.id[1],
                                                residue.id[2].strip()) in loop_ids:
                    loop.add(key)
    return atoms, loop


def write_variant(source, destination, heavy_chain, start, end, sequence,
                  backbone=None):
    """Rewrite the complex with the loop renamed (and optionally re-coordinated).

    Works on Bio.PDB objects rather than raw text so insertion codes survive. Loop
    sidechains are dropped; Rosetta rebuilds them during packing.
    """
    from Bio.PDB import PDBParser, PDBIO
    from Bio.PDB.Atom import Atom
    from Bio.PDB.Residue import Residue
    structure = PDBParser(QUIET=True).get_structure('v', str(source))[0]
    residues = loop_residues(structure, heavy_chain, start, end)
    if len(residues) != len(sequence):
        raise ValueError(f'loop has {len(residues)} residues but sequence has {len(sequence)}')
    chain = structure[heavy_chain]
    for offset, (residue, amino_acid) in enumerate(zip(residues, sequence)):
        replacement = Residue(residue.id, ONE_TO_THREE[amino_acid], residue.segid)
        for atom_index, atom_name in enumerate(BACKBONE_ATOMS):
            if backbone is None:
                if atom_name not in residue:
                    continue
                template = residue[atom_name]
                coord = template.coord.copy()
                element = template.element
            else:
                coord = np.asarray(backbone, dtype=float)[offset, atom_index]
                element = atom_name[0]
            replacement.add(Atom(atom_name, coord, 1.0, 1.0, ' ', atom_name,
                                 len(replacement), element=element))
        position = chain.child_list.index(residue)
        chain.child_list[position] = replacement
        replacement.parent = chain
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(destination))
    return destination


def pack_loop(pdb_in, pdb_out, heavy_chain, loop_ids, score_function, reuse=False):
    """Rebuild loop sidechains with Rosetta, keeping every backbone atom fixed."""
    if reuse and Path(pdb_out).is_file():
        return pdb_out
    import pyrosetta
    from pyrosetta.rosetta.core.pack.task import TaskFactory
    from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepacking
    from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
    pose = pyrosetta.pose_from_pdb(str(pdb_in))
    info = pose.pdb_info()
    task = TaskFactory()
    task.push_back(RestrictToRepacking())
    allowed = []
    for index in range(1, pose.total_residue() + 1):
        key = (info.number(index), str(info.icode(index)).strip())
        if info.chain(index) == heavy_chain and key in loop_ids:
            allowed.append(index)
    if not allowed:
        raise ValueError('no loop residues found in the pose')
    packer_task = task.create_task_and_apply_taskoperations(pose)
    keep = set(allowed)
    for index in range(1, pose.total_residue() + 1):
        if index not in keep:
            packer_task.nonconst_residue_task(index).prevent_repacking()
    PackRotamersMover(score_function, packer_task).apply(pose)
    pose.dump_pdb(str(pdb_out))
    return pdb_out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--methods', nargs='+', default=list(METHODS))
    parser.add_argument('--methods-json', type=Path, default=None,
                        help='JSON mapping method name to generated.json, overriding METHODS')
    parser.add_argument('--replicates', type=int, default=10000)
    parser.add_argument('--reuse', action='store_true',
                        help='reuse already packed structures in the work directory')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    work = ORIG / 'outputs/lddt_allatom_v1_work'
    work.mkdir(parents=True, exist_ok=True)

    dataset = ORIG / 'data/abgaze_reconstructed_v1'
    records = [json.loads(line) for line in
               (dataset / 'splits/test.json').read_text().splitlines() if line.strip()]
    if args.limit:
        records = records[:args.limit]
    groups = {r['pdb']: f"cluster_{r['cluster']}" for r in records}
    table = dict(METHODS)
    if args.methods_json is not None:
        overrides = json.loads(args.methods_json.read_text())
        table.update({name: Path(path) for name, path in overrides.items()})
    predictions = {name: {row['sample_id']: row for row in
                          json.loads(Path(table[name]).read_text())}
                   for name in args.methods}

    import pyrosetta
    from pyrosetta.rosetta.core.scoring import get_score_function
    pyrosetta.init('-mute all -ex1 -ex2aro -out:level 300')
    score_function = get_score_function()

    per_sample = []
    for position, record in enumerate(records, 1):
        sample_id = record['pdb']
        source = dataset / 'pdb' / f'{sample_id}.pdb'
        if not source.is_file():
            print(json.dumps(dict(skip=sample_id, reason='no pdb')), flush=True)
            continue
        heavy = record['heavy_chain']
        start, end = record['cdrh3_pos']
        from Bio.PDB import PDBParser
        native_structure = PDBParser(QUIET=True).get_structure('n', str(source))[0]
        native_loop_residues = loop_residues(native_structure, heavy, start, end)
        loop_ids = {(r.id[1], r.id[2].strip()) for r in native_loop_residues}
        loop_numbers = sorted(r.id[1] for r in native_loop_residues)
        row = dict(sample_id=sample_id, group=groups[sample_id],
                   n_loop=len(native_loop_residues))
        for name in args.methods:
            entry = predictions[name].get(sample_id)
            if entry is None:
                continue
            # DiffAb stores joint samples under 'draws'; the others expose 'greedy'.
            if 'greedy' in entry:
                sequence = entry['greedy']
                backbone = np.asarray(entry['backbone'], dtype=float)
            else:
                draw = entry['draws'][0]
                sequence = draw['sequence']
                backbone = np.asarray(draw['backbone'], dtype=float)
            if backbone.shape[0] != len(native_loop_residues):
                raise ValueError(f'{sample_id}/{name}: loop length mismatch '
                                 f'{backbone.shape[0]} vs {len(native_loop_residues)}')
            tag = f'{sample_id}_{name}'
            reference_pdb = write_variant(source, work / f'{tag}_ref.pdb', heavy,
                                          start, end, sequence)
            model_pdb = write_variant(source, work / f'{tag}_mdl.pdb', heavy,
                                      start, end, sequence, backbone)
            reference_packed = pack_loop(reference_pdb, work / f'{tag}_ref_packed.pdb',
                                         heavy, loop_ids, score_function, args.reuse)
            model_packed = pack_loop(model_pdb, work / f'{tag}_mdl_packed.pdb',
                                     heavy, loop_ids, score_function, args.reuse)
            ref_atoms, ref_loop = read_complex(reference_packed, heavy, loop_ids)
            mdl_atoms, _ = read_complex(model_packed, heavy, loop_ids)
            value, pairs = lddt(ref_atoms, mdl_atoms, atom_selection=ref_loop)
            ref_bb = {k: v for k, v in ref_atoms.items() if k[3] in BACKBONE_ATOMS}
            mdl_bb = {k: v for k, v in mdl_atoms.items() if k[3] in BACKBONE_ATOMS}
            loop_bb = {k for k in ref_loop if k[3] in BACKBONE_ATOMS}
            value_bb, pairs_bb = lddt(ref_bb, mdl_bb, atom_selection=loop_bb)
            row[f'{name}_lddt_allatom'] = value
            row[f'{name}_lddt_backbone'] = value_bb
            row[f'{name}_pairs'] = pairs
            row[f'{name}_pairs_bb'] = pairs_bb
        if row.get('raad_lddt_allatom') is not None and row.get('ours_lddt_allatom') is not None:
            row['delta_allatom'] = row['ours_lddt_allatom'] - row['raad_lddt_allatom']
            row['delta_backbone'] = row['ours_lddt_backbone'] - row['raad_lddt_backbone']
        per_sample.append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in row.items()
                          if k.startswith('ours') or k.startswith('raad') or k == 'sample_id'}),
              flush=True)

    from research_audit.statistics import interval
    summary = dict(
        metric='lDDT over the loop, inclusion radius 15 A, thresholds 0.5/1/2/4 A',
        allatom_scope='all heavy atoms of the loop; neighbours are all complex atoms',
        reference='native loop backbone with the method\'s own sequence, sidechains repacked',
        model='the method\'s loop backbone with its own sequence, sidechains repacked',
        sidechain_protocol='Rosetta REF2015 RestrictToRepacking on the loop, PreventRepacking '
                           'elsewhere, backbone fixed',
        n=len(per_sample), methods=args.methods,
        means={}, comparisons={})
    ids = [r['sample_id'] for r in per_sample]
    labels = [groups[s] for s in ids]
    for name in args.methods:
        for metric in ('lddt_allatom', 'lddt_backbone'):
            values = [r.get(f'{name}_{metric}') for r in per_sample]
            clean = [v for v in values if v is not None]
            summary['means'][f'{name}_{metric}'] = float(np.mean(clean)) if clean else None
    for metric in ('allatom', 'backbone'):
        delta = [r.get(f'delta_{metric}') for r in per_sample]
        usable = [(d, g) for d, g in zip(delta, labels) if d is not None]
        if len(usable) > 2:
            # interval() expects a seed-by-sample matrix; a single deterministic
            # measurement per sample is one row.
            stats = interval(np.asarray([[d for d, _ in usable]]),
                             [g for _, g in usable], args.replicates)
            summary['comparisons'][f'delta_{metric}'] = dict(
                mean=stats['mean'], ci95=stats['conditional_group_ci95'],
                per_seed_mean=stats['per_seed_mean'],
                significant=bool(stats['conditional_group_ci95'][0] *
                                 stats['conditional_group_ci95'][1] > 0))
    dump(out / 'lddt_summary.json', summary)
    import csv
    if per_sample:
        with (out / 'per_sample.csv').open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_sample[0]))
            writer.writeheader()
            writer.writerows(per_sample)
    print(json.dumps(summary['means'], indent=1))
    print(json.dumps(summary['comparisons'], indent=1))


if __name__ == '__main__':
    main()
