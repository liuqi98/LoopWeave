"""Design CDRH3 loops for VHH targets with the current full model.

For each target this builds exactly the inputs the trained model expects, without
fabricating any tensor:

  1. number the nanobody chain to IMGT with ANARCI so the H3 occupies 105-117;
  2. write an antigen-only PDB and run the archived binding-feature pipeline
     (PDB2PQR -> APBS + MSMS) to obtain the six ``binding_*`` tensors;
  3. build the target-free context with the archived ``scratch_inputs`` path and the
     frame-relation graph;
  4. run the frozen generator to obtain ``base_bb``, then the trained full refiner;
  5. sample sequences from the model's own logits and report them with the
     predicted loop geometry.

Honest scope
------------
The current model was trained only on Fab complexes (heavy + light chain, light
chain ~101-117 aa in every training record). Every target here is a single-domain
VHH with no light chain, so this is an out-of-distribution application. Results are
exploratory redesigns of a known binder's loop, not affinity predictions and not
validated designs. No docking, folding or energy calculation is performed.

Only the native complex is used to define the framework placement and the two
flanking anchor residues; the native H3 sequence and coordinates are never passed to
the model (the archived target-leakage assertion is re-run here).
"""
import argparse
import json
import shutil
import subprocess
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT.parent
# Package code first, then the source workspace's data helpers only. Putting the
# whole source workspace on the path would shadow the package modules with their
# older copies.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, '/media/xd/code1/dyGF/data')
warnings.filterwarnings('ignore')

from Bio.PDB import PDBParser, PDBIO, Select, Structure, Model
from Bio.SeqUtils import seq1
from anarci import run_anarci

from frame_relation_refiner import FrameRelationRefiner, build_frame_relation_graph
from persistent_loop_pair_refiner import PersistentLoopPairRefiner
from prepare_raad_protocol_frame_data import candidate, NATIVE_FIELDS
from prepare_abgaze_reconstructed import scratch_inputs, backbone, residue_aa

ORIG = Path('/media/xd/code1/dyGF')
CHECKPOINTS = ORIG / 'deliverables/dyGF_scoped_research_snapshot_20261015_v1/checkpoints/original_seed{seed}/pair.pt'
GENERATOR = ORIG / 'deliverables/dyGF_scoped_research_snapshot_20261015_v1/checkpoints/original_seed{seed}/generator.pt'
PQR_PYTHON = ORIG / '.tools/binding-pqr-v1/bin/python'
MSMS = Path('/home/xd/anaconda3/envs/openfold/bin/msms')
APBS = Path('/usr/bin/apbs')
ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'

TARGETS = {
    'gp120_G36': dict(path='/media/xd/code2/GenCDRH3_final/hiv/hiv_g36_complex.pdb',
                      antibody='J', antigen=['E'], note='HIV-1 gp120 / VHH G36'),
    'cMET_1_7': dict(path='/media/xd/code2/GenCDRH3_final/cmet/cmet_1_7_complex.pdb',
                     antibody='H', antigen=['A'], note='c-MET / VHH 1_7'),
    'PRV_Nb36': dict(path='/media/xd/code2/GenCDRH3_final/PRV/nb36_complex.pdb',
                     antibody='H', antigen=['A'], note='PRV / nanobody Nb36'),
    'PRV_Nb1': dict(path='/media/xd/code2/GenCDRH3_final/PRV/Nb1_complex.pdb',
                    antibody='H', antigen=['A'], note='PRV / nanobody Nb1'),
}


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def imgt_numbering(sequence):
    """Map each chain residue index to (IMGT position, insertion code)."""
    result = run_anarci([('q', sequence)], scheme='imgt', ncpu=1)
    numbered = result[1][0][0][0]
    mapping, index = [], 0
    for (position, insertion), amino_acid in numbered:
        if amino_acid == '-':
            continue
        if position is None:
            raise ValueError('unassigned position in IMGT alignment')
        mapping.append((index, int(position), str(insertion).strip()))
        index += 1
    if index != len(sequence):
        raise ValueError(f'IMGT mapping covers {index} of {len(sequence)} residues')
    return mapping


def renumber_target(spec, out_dir):
    """Write a complex PDB whose antibody chain carries IMGT numbering."""
    structure = PDBParser(QUIET=True).get_structure('t', spec['path'])[0]
    antibody_residues = [r for r in structure[spec['antibody']]
                         if r.id[0] == ' ' and seq1(r.resname) in ALPHABET]
    sequence = ''.join(seq1(r.resname) for r in antibody_residues)
    mapping = imgt_numbering(sequence)
    h3 = [(index, position, insertion, seq1(antibody_residues[index].resname))
          for index, position, insertion in mapping if 105 <= position <= 117]
    if not h3:
        raise ValueError('no IMGT 105-117 residues found')

    output = Structure.Structure('design')
    model_out = Model.Model(0)
    output.add(model_out)
    # Rebuild the antibody chain so every residue gets a unique IMGT id. Biopython
    # refuses in-place renumbering because the id setter checks siblings, so the
    # residues are copied with no parent, renumbered, then attached.
    from Bio.PDB.Chain import Chain
    antibody_chain = Chain('H')
    imgt_of = {index: (position, insertion) for index, position, insertion in mapping}
    cursor = 0
    for residue in structure[spec['antibody']]:
        if residue.id[0] != ' ' or seq1(residue.resname) not in ALPHABET:
            continue
        position, insertion = imgt_of[cursor]
        copy = residue.copy()
        copy.parent = None
        copy.id = (' ', position, insertion if insertion else ' ')
        antibody_chain.add(copy)
        cursor += 1
    if cursor != len(mapping):
        raise ValueError('renumbered residue count differs from the IMGT mapping')
    model_out.add(antibody_chain)
    for index, antigen_chain in enumerate(spec['antigen']):
        antigen = structure[antigen_chain].copy()
        antigen.id = 'ABCDEFGH'[index]
        model_out.add(antigen)
    path = out_dir / 'complex_imgt.pdb'
    io = PDBIO()
    io.set_structure(output)
    io.save(str(path))
    return dict(pdb=path, sequence=sequence,
                antigen_chains=['ABCDEFGH'[i] for i in range(len(spec['antigen']))],
                h3_sequence=''.join(r[3] for r in h3),
                h3_pdb_resseq=[antibody_residues[r[0]].id[1] for r in h3],
                h3_imgt=[(r[1], r[2]) for r in h3],
                anchor_n=seq1(antibody_residues[h3[0][0] - 1].resname) if h3[0][0] > 0 else None,
                anchor_c=seq1(antibody_residues[h3[-1][0] + 1].resname))


def antigen_only(spec, renumbered, out_dir):
    """Extract antigen chains into their own PDB for the feature pipeline."""
    class AntigenOnly(Select):
        def __init__(self, chains):
            self.chains = set(chains)

        def accept_chain(self, chain):
            return chain.id in self.chains

    structure = PDBParser(QUIET=True).get_structure('a', renumbered['pdb'])
    path = out_dir / 'antigen.pdb'
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(path), AntigenOnly(renumbered['antigen_chains']))
    return path


def binding_features(pdb_dir, target, out_root):
    """Run the archived PDB2PQR/APBS/MSMS pipeline unchanged."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'prepare_binding_features', ORIG / 'data/prepare_binding_features.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = types.SimpleNamespace(ph=7.4, salt=.15, pqr_python=PQR_PYTHON,
                                 msms=MSMS, apbs=APBS)
    record = dict(pdb=target, antigen_chains=['A'], heavy_chain='H', light_chain='L')
    directory = out_root / 'features' / target
    shutil.rmtree(directory, ignore_errors=True)
    metadata = module.process(record, pdb_dir, directory, args)
    loaded = torch.load(directory / 'features.pt', map_location='cpu', weights_only=False)
    if not loaded['metadata']['antigen_only'] or loaded['metadata']['native_h3_used']:
        raise ValueError('binding features are not antigen-only')
    return loaded['features'], directory, metadata


def read_complex_vhh(record, pdb_dir, split='design'):
    """``read_complex`` adapted for a single-domain VHH (no light chain).

    Identical in every other respect: same backbone extraction, same missing-atom
    handling, same absence of any native H3 information in the context tensors.
    """
    model = PDBParser(QUIET=True).get_structure(
        record['pdb'], Path(pdb_dir) / f"{record['pdb']}.pdb")[0]

    def residues(chain):
        return [r for r in model[chain] if r.id[0] == ' ' and residue_aa(r) in ALPHABET]

    heavy = residues(record['heavy_chain'])
    antigen = [r for chain in record['antigen_chains'] for r in residues(chain)]
    hs = ''.join(map(residue_aa, heavy))
    if hs != record['heavy_chain_seq']:
        raise ValueError('heavy chain sequence mismatch')
    h3 = [i for i, r in enumerate(heavy) if 105 <= r.id[1] <= 117]
    if not h3 or h3 != list(range(h3[0], h3[-1] + 1)):
        raise ValueError('H3 residues are not contiguous IMGT 105-117')
    if h3[0] == 0 or h3[-1] == len(heavy) - 1:
        raise ValueError('H3 lacks a flanking anchor residue')
    sequence = ''.join(hs[i] for i in h3)
    if sequence != record['cdrh3_seq']:
        raise ValueError('H3 sequence mismatch')
    hb, hm = backbone(heavy)
    ab, am = backbone(antigen)

    def locations(rows, bb, mask):
        return np.stack([bb[i, 1] if mask[i, 1]
                         else np.mean([a.coord for a in r], axis=0)
                         for i, r in enumerate(rows)]).astype(np.float32)

    hc = locations(heavy, hb, hm)
    ac = locations(antigen, ab, am)
    keep = np.ones(len(heavy), bool)
    keep[h3] = False
    hcontext = np.where(hm[..., None], hb, hc[:, None])
    acontext = np.where(am[..., None], ab, ac[:, None])
    return dict(sample_id=record['pdb'], split=split, target_length=len(h3),
                framework_seq=''.join(np.asarray(list(hs))[keep]),
                framework_ca=hc[keep],
                framework_backbone=hcontext[keep],
                framework_backbone_mask=hm[keep],
                heavy_sequence_without_h3=''.join(np.asarray(list(hs))[keep]),
                heavy_h3_indices=np.array(h3),
                h3_n_anchor=hc[h3[0] - 1], h3_c_anchor=hc[h3[-1] + 1],
                h3_n_anchor_frame=hcontext[h3[0] - 1],
                h3_c_anchor_frame=hcontext[h3[-1] + 1],
                antigen_seq=''.join(map(residue_aa, antigen)), antigen_ca=ac,
                antigen_backbone=acontext, antigen_backbone_mask=am,
                antigen_epitope_mask=np.ones(len(antigen), bool),
                native_h3_sequence=sequence, cdr3_backbone=hb[h3],
                cdr3_backbone_mask=hm[h3])


def run_design(target, spec, work, out_root, seeds, samples, temperature):
    directory = out_root / target
    directory.mkdir(parents=True, exist_ok=True)

    renumbered = renumber_target(spec, directory)
    # read_complex expects <dir>/<sample_id>.pdb
    shutil.copyfile(renumbered['pdb'], directory / f'{target}.pdb')
    antigen_pdb_dir = work / target / 'pdb'
    antigen_pdb_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(antigen_only(spec, renumbered, directory),
                    antigen_pdb_dir / f'{target}.pdb')

    # process() resolves dataset/'pdb'/'<id>.pdb', so pass the dataset root.
    features, feature_dir, feature_meta = binding_features(antigen_pdb_dir.parent, target, work)

    record = dict(pdb=target, heavy_chain='H', light_chain=None,
                  antigen_chains=renumbered['antigen_chains'],
                  heavy_chain_seq=renumbered['sequence'], light_chain_seq='',
                  cdrh3_seq=renumbered['h3_sequence'], numbering='imgt')
    item = read_complex_vhh(record, directory, 'design')

    # Target-leakage assertion, exactly as in the archived converter: perturb every
    # native H3 field and require the recomputed inputs to be unchanged. Only the
    # native H3 is perturbed; the framework sequence is legitimate context.
    clean = {k: v for k, v in item.items() if k not in NATIVE_FIELDS}
    x = scratch_inputs(dict(item))
    altered = dict(item)
    for key in NATIVE_FIELDS:
        if key in altered:
            value = altered[key]
            altered[key] = 'W' * len(value) if isinstance(value, str) else np.asarray(value) + 123
    check = scratch_inputs(altered)
    leaked = [k for k in x if not torch.equal(x[k], check[k])]
    if leaked:
        raise ValueError(f'native H3 leaked into inputs: {leaked}')
    leakage = {k: True for k in x}

    graph = build_frame_relation_graph(clean, candidate(clean))
    x.update({k: v for k, v in features.items()})

    predictions, per_seed = [], {}
    for seed in seeds:
        generator_state = torch.load(Path(str(GENERATOR).format(seed=seed)),
                                     map_location='cpu', weights_only=False)
        generator = FrameRelationRefiner(**generator_state['model_config'])
        generator.load_state_dict(generator_state['model'])
        generator.eval()
        with torch.no_grad():
            raw = generator(graph)
        base_bb = raw['h3_backbone']
        inputs = dict(x)
        inputs['base_bb'] = base_bb
        inputs['base_logits'] = torch.zeros_like(inputs['base_logits'])

        state = torch.load(Path(str(CHECKPOINTS).format(seed=seed)),
                           map_location='cpu', weights_only=False)
        model = PersistentLoopPairRefiner(**state['model_config'])
        model.load_state_dict(state['model'])
        model.eval()
        with torch.no_grad():
            output = model(inputs)
        logits = output['sequence_logits']
        backbone = output['backbone']
        if not (torch.isfinite(logits).all() and torch.isfinite(backbone).all()):
            raise ValueError(f'nonfinite output for {target}')
        log_probs = logits.log_softmax(-1)
        greedy = ''.join(ALPHABET[i] for i in logits.argmax(-1).tolist())
        generator_rng = torch.Generator().manual_seed(2026 + seed)
        draws = torch.multinomial((logits / temperature).softmax(-1), samples,
                                  replacement=True, generator=generator_rng).T
        candidates = []
        for draw in draws.tolist():
            sequence = ''.join(ALPHABET[i] for i in draw)
            score = float(sum(log_probs[i, a] for i, a in enumerate(draw)) / len(draw))
            candidates.append(dict(sequence=sequence, mean_logprob=score,
                                   seed=seed, source='sampled'))
        candidates.append(dict(sequence=greedy,
                               mean_logprob=float(log_probs.gather(
                                   1, logits.argmax(-1, keepdim=True)).mean()),
                               seed=seed, source='greedy'))
        per_seed[seed] = candidates
        predictions.extend(candidates)
        per_seed[f'{seed}_backbone'] = backbone.tolist()
    # Geometry QC of the predicted loop, using the project's own proxy metrics.
    from structure_refinement_ablation import geometry_quality
    quality = {}
    for seed in seeds:
        backbone = torch.tensor(per_seed[f'{seed}_backbone'])
        inputs_cpu = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in x.items()}
        with torch.no_grad():
            qc = geometry_quality(backbone, inputs_cpu)
        quality[str(seed)] = {k: float(v) for k, v in qc.items()}
        write_loop_pdb(directory / f'designed_backbone_seed{seed}.pdb',
                       backbone, inputs_cpu, renumbered, seed)
    ranked = sorted(predictions, key=lambda row: -row['mean_logprob'])
    dump(directory / 'designs.json', dict(
        target=target, note=spec['note'], native_h3=renumbered['h3_sequence'],
        native_h3_length=len(renumbered['h3_sequence']),
        anchors=dict(n=renumbered['anchor_n'], c=renumbered['anchor_c']),
        model='PersistentLoopPairRefiner (full model), seeds ' + '/'.join(map(str, seeds)),
        temperature=temperature, samples_per_seed=samples,
        target_free_leakage_check=leakage,
        design_sequences=ranked,
        geometry_quality=quality,
        per_seed=per_seed,
        feature_metadata=feature_meta,
        provenance=dict(complex_pdb=str(spec['path']), complex_sha256=sha(spec['path']),
                        renumbered_pdb=sha(renumbered['pdb']),
                        features_sha256=sha(feature_dir / 'features.pt'),
                        generator_sha256={s: sha(Path(str(GENERATOR).format(seed=s))) for s in seeds},
                        model_sha256={s: sha(Path(str(CHECKPOINTS).format(seed=s))) for s in seeds}),
        limitations=['The current model was trained only on Fab complexes (heavy + light chain); '
                     'these targets are single-domain VHH with no light chain, so this is an '
                     'out-of-distribution application.',
                     'Only the framework placement and the two flanking anchors come from the native '
                     'complex; native H3 sequence and coordinates are excluded from the model inputs '
                     '(asserted above).',
                     'No docking, refolding, energy or affinity calculation was performed, so these are '
                     'not ranked by predicted affinity.',
                     'Sequences are ranked by the model mean log-probability, a deployable score that '
                     'uses no native labels.']))
    return directory / 'designs.json'


def write_loop_pdb(path, backbone, inputs, renumbered, seed):
    """Write the predicted H3 backbone plus its two flanking anchors as PDB."""
    lines = [f'REMARK designed H3 backbone, model seed {seed}, target-free generation']
    anchors = inputs['anchors']
    chain_residues = []
    for index in range(len(backbone)):
        chain_residues.append((100 + index, backbone[index]))
    for offset, (number, coords) in enumerate(chain_residues):
        for atom, name in enumerate(('N', 'CA', 'C', 'O')):
            xyz = coords[atom].tolist()
            lines.append(
                f'ATOM  {offset * 4 + atom + 1:5d} {name:<4s}GLY H{number:4d}    '
                f'{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00           {name[0]:>2s}')
    lines.append('TER')
    Path(path).write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True,
                        help='Persistent scratch root for feature intermediates')
    parser.add_argument('--targets', nargs='+', default=None)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2026, 2027, 2028])
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=.5)
    args = parser.parse_args()
    torch.set_num_threads(4)
    out_root = args.output.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    work = args.work.resolve()
    names = args.targets or list(TARGETS)
    produced = {}
    for name in names:
        spec = TARGETS[name]
        if not Path(spec['path']).is_file():
            produced[name] = 'missing complex structure'
            continue
        try:
            path = run_design(name, spec, work, out_root, args.seeds, args.samples,
                              args.temperature)
            produced[name] = str(path)
            print(json.dumps(dict(target=name, status='ok')), flush=True)
        except Exception as error:
            import traceback
            produced[name] = f'{type(error).__name__}: {error}'
            print(json.dumps(dict(target=name, status='failed', error=str(error))), flush=True)
            traceback.print_exc()
    dump(out_root / 'status.json', dict(phase='complete', targets=produced))
    print(json.dumps(produced, indent=1))


if __name__ == '__main__':
    main()
