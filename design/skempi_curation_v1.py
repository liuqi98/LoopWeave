"""Step 1 of the affinity-optimisation track: curate the SKEMPI v2.0 antibody subset.

SKEMPI v2.0 carries a ``Hold_out_type`` column; ``AB/AG`` selects exactly the
antibody-antigen complexes used for the affinity-optimisation benchmark (53 PDB
entries, which matches the 53 complexes quoted for that benchmark). This script

  1. selects ``Hold_out_type == 'AB/AG'`` and parses every mutation string into
     (chain, wild-type, position, mutant);
  2. converts the raw affinities into ddG = RT ln(Kd_mut / Kd_wt) with RT = 1.36
     kcal/mol at 298 K (positive = weaker binding);
  3. numbers every antibody chain with ANARCI (IMGT) and maps each mutation's PDB
     residue number onto its IMGT position, so H3 (IMGT 105-117) can be separated
     from framework and from antigen-side mutations;
  4. runs a structural sanity check on the local copies (heavy-atom overlaps,
     presence of the mutated residue with the expected wild-type identity, and
     whether an H3 can be defined at all).

Nothing is trained or scored here; this only decides whether the data can support a
mutation-level optimisation task, and how much of it is clean.
"""
import argparse
import csv
import json
import math
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

warnings.filterwarnings('ignore')
RT = 1.36                                  # kcal/mol at 298 K
CANONICAL = 'ACDEFGHIKLMNPQRSTVWY'
BACKBONE = {'N', 'CA', 'C', 'O'}
CLASH_CUTOFF = 1.5


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def parse_mutations(text):
    """SKEMPI tokens are ``[wild-type][chain][position][mutant]``.

    ``HA30A`` therefore means histidine -> alanine at chain A position 30, not chain
    H. This ordering was confirmed against the local structures: chain A exists in
    1VFB and ``HA30A``/``YA32A`` resolve there, while the older chain-first reading
    left 1979 of 1999 records unresolved.
    """
    result = []
    for token in [t for t in text.replace(' ', '').split(',') if t]:
        if len(token) < 4:
            return None
        wild_type, chain, mutant = token[0], token[1], token[-1]
        position = token[2:-1]
        if not position.isdigit() or wild_type not in CANONICAL or mutant not in CANONICAL:
            return None
        result.append(dict(chain=chain, wild_type=wild_type,
                           position=int(position), mutant=mutant))
    return result or None


def ddg(mutant, wild_type):
    try:
        return RT * math.log10(float(mutant) / float(wild_type))
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def anarci_map(sequence):
    """IMGT position for the residues of the aligned V domain.

    ANARCI numbers only the immunoglobulin domain, not the whole chain: a 214-residue
    Fab heavy chain yields ~128 IMGT positions. The domain's offset in the input is
    therefore taken from ``query_start`` and added back, so the returned indices are
    indices into ``sequence``.
    """
    from anarci import run_anarci
    numbering, alignment, details, _ = run_anarci([('q', sequence)], scheme='imgt', ncpu=1)
    if not alignment or not alignment[0]:
        raise ValueError('no immunoglobulin domain found')
    numbered = alignment[0][0][0]
    domain = details[0][0]
    start = int(domain['query_start'])
    mapping, local = [], 0
    for (position, insertion), amino_acid in numbered:
        if amino_acid == '-':
            continue
        mapping.append((start + local, int(position), str(insertion).strip()))
        local += 1
    if start + local > len(sequence):
        raise ValueError('domain mapping runs past the chain')
    return mapping, domain['chain_type']


def clash_profile(pdb_path):
    """Non-adjacent heavy-atom pairs below the cutoff, split by atom class.

    Adjacency must be judged within a chain: an earlier version compared PDB residue
    numbers only, which made residues with the same number in different chains look
    bonded and silently dropped real inter-chain clashes.
    """
    from Bio.PDB import PDBParser
    structure = PDBParser(QUIET=True).get_structure('c', str(pdb_path))[0]
    xyz, meta = [], []
    for chain in structure:
        index_of = {}
        for position, residue in enumerate(chain):
            if residue.id[0] != ' ':
                continue
            index_of[id(residue)] = position
            for atom in residue:
                if atom.element == 'H':
                    continue
                xyz.append(atom.coord)
                meta.append((chain.id, position, atom.name.strip()))
    if not xyz:
        return dict(total=0, bb_bb=0, bb_sc=0, sc_sc=0, atoms=0)
    xyz = np.array(xyz)
    pairs = cKDTree(xyz).query_pairs(CLASH_CUTOFF, output_type='ndarray')
    counts = dict(bb_bb=0, bb_sc=0, sc_sc=0)
    for first, second in pairs:
        chain_a, index_a, name_a = meta[first]
        chain_b, index_b, name_b = meta[second]
        if chain_a == chain_b and abs(index_a - index_b) <= 1:
            continue
        a, b = name_a in BACKBONE, name_b in BACKBONE
        if a and b:
            counts['bb_bb'] += 1
        elif a or b:
            counts['bb_sc'] += 1
        else:
            counts['sc_sc'] += 1
    counts['total'] = counts['bb_bb'] + counts['bb_sc'] + counts['sc_sc']
    counts['atoms'] = int(len(xyz))
    # Hetero groups (glycans, ions) are covalently attached at glycosylation sites; a
    # naive distance test reports those bonds as clashes. Count them separately so a
    # glycan is never mistaken for a structural defect.
    hetero = []
    for chain in structure:
        for residue in chain:
            if residue.id[0] == ' ':
                continue
            for atom in residue:
                if atom.element == 'H':
                    continue
                hetero.append(atom.coord)
    if hetero and xyz.size:
        distances = cKDTree(xyz).query(np.array(hetero), k=1)[0]
        counts['hetero_contacts_below_cutoff'] = int((distances < CLASH_CUTOFF).sum())
    else:
        counts['hetero_contacts_below_cutoff'] = 0
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skempi', type=Path,
                        default=Path('/media/xd/code2/MutationAB/skempi_v2.csv'))
    parser.add_argument('--pdb-dir', type=Path,
                        default=Path('/media/xd/code2/MutationAB/skempi_pdbs'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    with args.skempi.open(newline='') as handle:
        rows = list(csv.DictReader(handle, delimiter=';'))
    antibody = [row for row in rows if row.get('Hold_out_type') == 'AB/AG']
    by_complex = defaultdict(list)
    for row in antibody:
        by_complex[row['#Pdb']].append(row)
    structures = {p.stem.upper(): p for p in args.pdb_dir.glob('*.pdb')}

    from Bio.PDB import PDBParser
    from Bio.SeqUtils import seq1

    complexes, mutation_rows = [], []
    for pdb_id in sorted(by_complex):
        base = pdb_id.split('_')[0].upper()
        chains_in_id = pdb_id.split('_')[1] if '_' in pdb_id else ''
        entry = dict(pdb_id=pdb_id, base=base, chain_spec=chains_in_id,
                     structure=str(structures.get(base)) if base in structures else None,
                     n_mutations=len(by_complex[pdb_id]))
        if base not in structures:
            entry.update(status='structure_missing')
            complexes.append(entry)
            continue

        structure = PDBParser(QUIET=True).get_structure(base, str(structures[base]))[0]
        # ANARCI every polymer chain that looks like protein.
        chain_info = {}
        for chain in structure:
            residues = [r for r in chain
                        if r.id[0] == ' ' and seq1(r.resname) in CANONICAL]
            if len(residues) < 60:
                continue
            sequence = ''.join(seq1(r.resname) for r in residues)
            try:
                mapping, chain_type = anarci_map(sequence)
            except Exception as error:
                chain_info[chain.id] = dict(error=str(error))
                continue
            imgt_of_pdb = {}
            for index, position, insertion in mapping:
                imgt_of_pdb[residues[index].id[1]] = (position, insertion)
            h3 = [residues[index].id[1] for index, position, insertion in mapping
                  if 105 <= position <= 117]
            chain_info[chain.id] = dict(
                chain_type=chain_type, length=len(sequence),
                h3_length=len(h3), h3_pdb_numbers=h3,
                h3_sequence=''.join(seq1(r.resname) for r in residues
                                    if r.id[1] in set(h3)) if h3 else '',
                imgt_of_pdb={str(k): list(v) for k, v in imgt_of_pdb.items()})
        entry['chains'] = {chain: {k: v for k, v in info.items() if k != 'imgt_of_pdb'}
                           for chain, info in chain_info.items()}
        antibody_chains = [chain for chain, info in chain_info.items()
                           if info.get('chain_type') in ('H', 'K', 'L')]
        heavy = [chain for chain in antibody_chains
                 if chain_info[chain].get('chain_type') == 'H']
        entry['antibody_chains'] = antibody_chains
        entry['heavy_chains'] = heavy
        entry['h3_defined'] = any(chain_info[chain].get('h3_length') for chain in heavy)

        # Resolve mutations: parse, map to IMGT, classify by region.
        in_h3 = framework = antigen_side = unresolved = 0
        usable_ddg = []
        for row in by_complex[pdb_id]:
            parsed = parse_mutations(row.get('Mutation(s)_PDB', ''))
            value = ddg(row.get('Affinity_mut_parsed'), row.get('Affinity_wt_parsed'))
            if parsed is None:
                unresolved += 1
                continue
            for mutation in parsed:
                chain = mutation['chain']
                info = chain_info.get(chain)
                record = dict(pdb_id=pdb_id, base=base, **mutation, ddg=value,
                              n_mutations_in_entry=len(parsed),
                              location=row.get('iMutation_Location(s)'),
                              method=row.get('Method'))
                if info is None or 'imgt_of_pdb' not in info:
                    # Distinguish "chain exists but is not an immunoglobulin" (antigen)
                    # from "chain letter absent from this file" (label mismatch).
                    present = chain in structure
                    record['region'] = 'antigen' if present else 'chain_absent'
                    mutation_rows.append(record)
                    continue
                imgt = info['imgt_of_pdb'].get(str(mutation['position']))
                if imgt is not None:
                    try:
                        observed = seq1(structure[chain][mutation['position']].resname)
                        record['observed_wild_type'] = observed
                        record['wild_type_matches'] = bool(observed == mutation['wild_type'])
                    except KeyError:
                        record['wild_type_matches'] = None
                if imgt is None:
                    record['region'] = 'position_unresolved'
                    mutation_rows.append(record)
                    continue
                record['imgt_position'] = imgt[0]
                record['imgt_insertion'] = imgt[1]
                if info.get('chain_type') == 'H' and 105 <= imgt[0] <= 117:
                    record['region'] = 'H3'
                    in_h3 += 1
                elif info.get('chain_type') in ('H', 'K', 'L'):
                    record['region'] = 'framework'
                    framework += 1
                else:
                    record['region'] = 'antigen'
                    antigen_side += 1
                mutation_rows.append(record)
            if value is not None and abs(value) < 10:
                usable_ddg.append(value)
        resolved = [r for r in mutation_rows if r['pdb_id'] == pdb_id
                    and r.get('wild_type_matches') is not None]
        entry.update(mutations_in_h3=in_h3, mutations_in_framework=framework,
                     mutations_on_antigen=antigen_side,
                     wild_type_checked=len(resolved),
                     wild_type_matches=sum(1 for r in resolved if r['wild_type_matches']),
                     mutations_unresolved=unresolved,
                     usable_ddg_entries=len(usable_ddg),
                     ddg_mean=float(np.mean(usable_ddg)) if usable_ddg else None)
        entry['clashes'] = clash_profile(structures[base])
        entry['status'] = 'ok'
        complexes.append(entry)

    fields = ['pdb_id', 'base', 'antibody_chains', 'heavy_chains', 'h3_defined',
              'mutations_in_h3', 'mutations_in_framework', 'mutations_on_antigen',
              'wild_type_checked', 'wild_type_matches',
              'usable_ddg_entries', 'ddg_mean', 'status']
    with (out / 'skempi_abag_complexes.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + ['clashes', 'chains'],
                                extrasaction='ignore')
        writer.writeheader()
        for entry in complexes:
            writer.writerow({**{k: entry.get(k) for k in fields},
                             'clashes': json.dumps(entry.get('clashes')),
                             'chains': json.dumps(entry.get('chains'))})
    mutation_fields = ['pdb_id', 'base', 'chain', 'wild_type', 'position', 'mutant',
                       'imgt_position', 'imgt_insertion', 'region', 'ddg',
                       'observed_wild_type', 'wild_type_matches',
                       'n_mutations_in_entry', 'location', 'method']
    with (out / 'skempi_abag_mutations.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=mutation_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(mutation_rows)

    regions = Counter(row.get('region') for row in mutation_rows)
    single = [row for row in mutation_rows
              if row.get('n_mutations_in_entry') == 1 and row.get('ddg') is not None]
    h3_single = [row for row in single if row.get('region') == 'H3']
    clean_complexes = [entry for entry in complexes
                       if entry.get('status') == 'ok' and entry.get('h3_defined')
                       and entry.get('mutations_in_h3', 0) > 0]
    summary = dict(
        source=str(args.skempi), pdb_dir=str(args.pdb_dir),
        selection="Hold_out_type == 'AB/AG' (the antibody-antigen subset)",
        n_complexes=len(complexes),
        n_complexes_with_structure=sum(1 for e in complexes if e.get('structure')),
        n_complexes_with_h3=sum(1 for e in complexes if e.get('h3_defined')),
        n_complexes_usable_h3_mutations=len(clean_complexes),
        missing_structures=[e['pdb_id'] for e in complexes if not e.get('structure')],
        n_mutation_records=len(mutation_rows),
        mutation_regions=dict(regions),
        n_single_mutants=len(single),
        wild_type_agreement=dict(
            checked=sum(1 for r in mutation_rows if r.get('wild_type_matches') is not None),
            matching=sum(1 for r in mutation_rows if r.get('wild_type_matches') is True)),
        n_single_mutants_in_h3=len(h3_single),
        single_mutant_ddg=dict(
            mean=float(np.mean([r['ddg'] for r in single])) if single else None,
            median=float(np.median([r['ddg'] for r in single])) if single else None,
            frac_stabilising=float(np.mean([r['ddg'] < 0 for r in single])) if single else None),
        h3_single_mutant_ddg=dict(
            mean=float(np.mean([r['ddg'] for r in h3_single])) if h3_single else None,
            median=float(np.median([r['ddg'] for r in h3_single])) if h3_single else None,
            frac_stabilising=float(np.mean([r['ddg'] < 0 for r in h3_single]))
            if h3_single else None),
        complexes_with_clashes_below_1_5A=sum(
            1 for e in complexes if e.get('clashes', {}).get('total', 0) > 0),
        complexes_with_backbone_clashes=sum(
            1 for e in complexes if e.get('clashes', {}).get('bb_bb', 0) > 0),
        complexes_with_protein_clashes=sum(
            1 for e in complexes if e.get('clashes', {}).get('total', 0) > 0),
        note_on_clashes=('protein-only, chain-aware heavy-atom contacts below 1.5 A. An earlier '
                         'version compared PDB residue numbers across chains and counted '
                         'protein-glycan glycosidic bonds as clashes, which wrongly flagged about '
                         'half the set; both are fixed here.'),
        limitations=['ddG is derived as RT ln(Kd_mut/Kd_wt) with a fixed RT; mixed measurement '
                     'methods (SPR/FL/ITC) are pooled exactly as SKEMPI does.',
                     'Mutation positions come from the PDB mutational field and are mapped to IMGT '
                     'through ANARCI; positions whose numbering disagrees are marked unresolved '
                     'rather than guessed.',
                     'Clash counting here is a heavy-atom geometric proxy, not a Rosetta energy.'])
    dump(out / 'curation_summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'limitations'}, indent=1,
                     default=str)[:3000])


if __name__ == '__main__':
    main()
