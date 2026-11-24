"""Two diagnostics requested for the three saved method outputs.

Diagnostic 1 — antigen-aligned H3 RMSD
--------------------------------------
Fit one rigid transform from the **antigen** atoms of the prediction onto the native
antigen, apply that single transform to the whole predicted complex, and then measure the
H3 RMSD without any further fit. Reported next to the loop-aligned RMSD:
  * loop-aligned: mainly how close the loop shape is;
  * antigen-aligned: loop shape plus its placement relative to the antigen.
For the current model and RAAD the antigen is the native one (they predict the loop in the
native pose), so the antigen transform is the identity and the antigen-aligned value
coincides with the raw H3 RMSD. That identity is verified numerically rather than assumed.
No H3-to-native superposition is ever used before computing contacts or FAPE, which would
leak the target structure into those readouts.

Diagnostic 2 — reconstruction before/after for all three methods
Numerator: the deterministic intra-residue geometry reconstruction (``residue_geometry``).
Because dyMEAN saves no framework, the shared readouts avoid framework-dependent terms:
in-loop C-N, internal planarity, internal clash (loop against itself) plus external clash
against each method's own antigen, interface FAPE and contact F1. The last two are evaluated
in the method's own pose, which is the only pose dyMEAN provides.

Wording: dyMEAN entries are called *saved outputs*, not raw network outputs, because the
historical artifact may already contain upstream processing that cannot be traced.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from frame_relation_refiner import FrameRelationGraph, FrameRelationRefiner
from persistent_loop_pair_refiner import PersistentLoopPairRefiner
from orientation_interface_refiner import interface_fape
from structure_refinement_ablation import residue_geometry
from train.train_abgaze_reconstructed import move
from research_audit.statistics import interval

WORKSPACE = Path('/media/xd/code1/dyGF')
PACKAGE = ROOT.parent
SEEDS = (2026, 2027, 2028)
THREE = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
         'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
         'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
         'TYR': 'Y', 'VAL': 'V'}
BACKBONE = ('N', 'CA', 'C', 'O')
MEASURES = ('cn_internal', 'planarity', 'clash_internal', 'clash_external',
            'interface_fape', 'contact_f1', 'loop_aligned_rmsd', 'antigen_aligned_h3_rmsd')
RECONSTRUCTION_MEASURES = ('cn_internal', 'planarity', 'clash_internal', 'clash_external',
                           'interface_fape', 'contact_f1')
# contact_f1 is higher-is-better; everything else here is lower-is-better.
LOWER_BETTER = {'cn_internal', 'planarity', 'clash_internal', 'clash_external',
                'interface_fape', 'loop_aligned_rmsd', 'antigen_aligned_h3_rmsd'}


def kabsch(prediction, reference):
    """Rotation and translation mapping ``prediction`` onto ``reference``."""
    prediction = prediction.double()
    reference = reference.double()
    p_centre, r_centre = prediction.mean(0), reference.mean(0)
    p_centred, r_centred = prediction - p_centre, reference - r_centre
    u, _, vt = torch.linalg.svd(p_centred.T @ r_centred)
    d = torch.sign(torch.linalg.det(vt.T @ u.T))
    rotation = vt.T @ torch.diag(torch.tensor([1., 1., d], dtype=torch.double)) @ u.T
    return rotation, r_centre - rotation @ p_centre


def rmsd_after(rotation, translation, prediction, reference):
    moved = prediction.double() @ rotation.T + translation
    return float((moved - reference.double()).square().sum(-1).mean().sqrt())


def loop_aligned_rmsd(prediction, native):
    rotation, translation = kabsch(prediction, native)
    return rmsd_after(rotation, translation, prediction, native)


def internal_planarity(bb):
    """Same formula as ``geometry_quality`` but over loop-internal bonds only."""
    n, ca, c, o = bb.unbind(1)
    normal = torch.nn.functional.normalize(torch.cross(ca[:-1] - c[:-1], n[1:] - c[:-1],
                                                       dim=-1), dim=-1)
    first = (torch.nn.functional.normalize(o[:-1] - c[:-1], dim=-1) * normal).sum(-1).square()
    axis = torch.nn.functional.normalize(n[1:] - c[:-1], dim=-1)
    a = ca[:-1] - c[:-1]
    b = ca[1:] - n[1:]
    a = torch.nn.functional.normalize(a - (a * axis).sum(-1, keepdim=True) * axis, dim=-1)
    b = torch.nn.functional.normalize(b - (b * axis).sum(-1, keepdim=True) * axis, dim=-1)
    second = 1 - (a * b).sum(-1).square()
    return float((first + second).mean())


def cn_internal(bb):
    bonds = (bb[:-1, 2] - bb[1:, 0]).norm(dim=-1)
    return float((bonds - 1.329).abs().mean())


def clash(bb, antigen):
    atoms = bb.flatten(0, 1)
    positions = torch.arange(len(bb)).repeat_interleave(4)
    nonneighbor = (positions[:, None] - positions[None, :]).abs() > 1
    internal = float((torch.nn.functional.relu(2. - torch.cdist(atoms, atoms)).square()
                      [nonneighbor]).mean())
    if antigen is not None and len(antigen):
        external = float((torch.nn.functional.relu(
            2. - torch.cdist(atoms, antigen.flatten(0, 1))).square()).mean())
    else:
        external = float('nan')
    return internal, external


def contact_f1(prediction, antigen, native_ca, native_contacts, threshold=8.):
    predicted = torch.cdist(prediction[:, 1], antigen[:, 1])
    return float(2 * ((predicted <= threshold) & native_contacts).sum() /
                 ((predicted <= threshold).sum() + native_contacts.sum()).clamp_min(1))


def read_saved_structure(path):
    """Loop chain and antigen chain from a saved two-chain structure."""
    from Bio.PDB import PDBParser
    structure = PDBParser(QUIET=True).get_structure('s', str(path))[0]
    chains = []
    for chain in structure:
        residues = [r for r in chain if r.id[0] == ' ' and r.resname in THREE]
        if residues:
            chains.append(residues)
    chains.sort(key=len)
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
    loop_bb, loop_mask, loop_seq = pack(chains[0])
    antigen_bb, antigen_mask, antigen_seq = pack(chains[1])
    return dict(loop_bb=loop_bb, loop_mask=loop_mask, loop_seq=loop_seq,
                antigen_bb=antigen_bb, antigen_mask=antigen_mask, antigen_seq=antigen_seq)


def measure(loop_bb, antigen_bb, antigen_mask, native_loop, native_antigen,
            native_antigen_mask, native_ca, native_contacts):
    """All shared readouts for one prediction in its own pose."""
    fake_x = dict(context_bb=antigen_bb, context_mask=antigen_mask,
                  antigen_mask=torch.ones(len(antigen_bb), dtype=torch.bool))
    internal, external = clash(loop_bb, antigen_bb)
    native_sequence = None
    return dict(
        cn_internal=cn_internal(loop_bb),
        planarity=internal_planarity(loop_bb),
        clash_internal=internal,
        clash_external=external,
        interface_fape=float(interface_fape(loop_bb, fake_x,
                                            dict(bb=native_loop,
                                                 mask=torch.ones_like(native_loop[..., 0],
                                                                      dtype=torch.bool)))),
        contact_f1=contact_f1(loop_bb, antigen_bb, native_ca, native_contacts),
        loop_aligned_rmsd=loop_aligned_rmsd(loop_bb[:, 1], native_loop[:, 1]),
        antigen_aligned_h3_rmsd=antigen_aligned(loop_bb, antigen_bb, antigen_mask,
                                                native_loop, native_antigen,
                                                native_antigen_mask))


def antigen_aligned(loop_bb, antigen_bb, antigen_mask, native_loop, native_antigen,
                    native_antigen_mask):
    """H3 RMSD after ONE transform fitted on the shared antigen backbone atoms.

    Only atoms present in both the prediction and the native context take part, so a
    missing native atom cannot silently contribute a zero coordinate.
    """
    joint = antigen_mask.bool() & native_antigen_mask.bool()
    if int(joint.sum()) < 4:
        raise ValueError('too few shared antigen atoms')
    prediction = antigen_bb[joint]
    reference = native_antigen[joint]
    rotation, translation = kabsch(prediction, reference)
    return rmsd_after(rotation, translation, loop_bb[:, 1], native_loop[:, 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--replicates', type=int, default=10000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    dataset = WORKSPACE / 'data/abgaze_reconstructed_v1'
    pairs = torch.load(WORKSPACE / 'data/raad_abgaze_reconstructed_v1/test.pt',
                       map_location='cpu', weights_only=False)
    graphs = {item['sample_id']: graph for item, graph in pairs}
    items = {item['sample_id']: item for item, _ in pairs}
    rows = torch.load(dataset / 'tensors/test.pt', map_location='cpu', weights_only=False)
    ids = [row[0] for row in rows]
    clusters = {json.loads(line)['pdb']: f"cluster_{json.loads(line)['cluster']}"
                for line in (dataset / 'splits/test.json').read_text().splitlines()
                if line.strip()}
    raad_files = [
        WORKSPACE / 'outputs/raad_abgaze_reconstructed_test_v1/generated.json',
        WORKSPACE / 'outputs/raad_reconstructed_seed2027_test_v2/generated.json',
        WORKSPACE / 'outputs/raad_reconstructed_seed2028_test_v2/generated.json']
    raad_by_seed = []
    for path in raad_files:
        if path.is_file():
            raad_by_seed.append({row['sample_id']: row for row in
                                 json.loads(path.read_text())})

    # Native context per complex: antigen backbone, native loop, native contacts.
    native = {}
    for sid, x, y in rows:
        # Keep every residue the graph retained as antigen, including those whose CA is missing
        # in the native context; the atom-level mask excludes the missing atoms instead of
        # silently dropping the residue and breaking the index correspondence.
        mask = x['antigen_mask']
        antigen = x['context_bb'][mask]
        antigen_atom_mask = x['context_mask'][mask]
        retained = torch.as_tensor(graphs[sid]['antigen_original_indices'])
        if len(antigen) != len(retained):
            raise ValueError(f'{sid}: retained antigen count {len(antigen)} does not match the '
                             f'graph indices {len(retained)}')
        native_ca = y['bb'][y['mask'][:, 1], 1]
        native_contacts = torch.cdist(native_ca, antigen[:, 1]) <= 8.
        native[sid] = dict(x=x, y=y, antigen=antigen,
                           antigen_mask=antigen_atom_mask,
                           antigen_mask_atoms=antigen_atom_mask,
                           native_ca=native_ca, native_contacts=native_contacts)

    outputs = {name: {when: {} for when in ('saved', 'reconstructed')}
               for name in ('current', 'raad', 'dymean')}

    def store(name, sid, loop_bb, antigen_bb, antigen_mask):
        info = native[sid]
        entry = measure(loop_bb, antigen_bb, antigen_mask, info['y']['bb'],
                        info['antigen'], info['antigen_mask_atoms'],
                        info['native_ca'], info['native_contacts'])
        outputs[name]['saved'][sid] = entry
        loop_rebuilt = residue_geometry(loop_bb)
        outputs[name]['reconstructed'][sid] = measure(
            loop_rebuilt, antigen_bb, antigen_mask, info['y']['bb'], info['antigen'],
            info['antigen_mask_atoms'], info['native_ca'], info['native_contacts'])

    # current model: aligned-loss checkpoints, three seeds
    current_by_seed = {}
    for seed in SEEDS:
        generator_state = torch.load(PACKAGE / f'checkpoints/original_seed{seed}/generator.pt',
                                     map_location='cpu', weights_only=False)
        generator = FrameRelationRefiner(**generator_state['model_config']).to(device)
        generator.load_state_dict(generator_state['model'])
        generator.eval()
        pair_path = (WORKSPACE / f'outputs/aligned_shape_w0.3_seed{seed}/pair/best.pt')
        pair_state = torch.load(pair_path, map_location='cpu', weights_only=False)
        model = PersistentLoopPairRefiner(**pair_state['model_config']).to(device)
        model.load_state_dict(pair_state['model'])
        model.eval()
        captured = {}
        handle = model.loop_pair_layers[-1].register_forward_hook(
            lambda module, inputs, output: captured.__setitem__(
                'raw', output[1].detach().cpu().clone()))
        current_by_seed[seed] = {}
        with torch.no_grad():
            for sid, x, y in rows:
                feature = torch.load(WORKSPACE / 'outputs/binding_interface_test_v1/features'
                                     / sid / 'features.pt', map_location='cpu',
                                     weights_only=False)
                raw = generator(FrameRelationGraph(**graphs[sid]).to(device))
                inputs = move({**x, **feature['features']}, device)
                inputs.update(base_bb=raw['h3_backbone'],
                              base_logits=torch.zeros_like(inputs['base_logits']))
                captured.clear()
                result = model(dict(inputs))
                # The saved output is already reconstructed, so the pre-reconstruction state is
                # the last pair-layer output, not the saved file.
                raw_state = captured['raw']
                if not torch.allclose(residue_geometry(raw_state),
                                      result['backbone'].detach().cpu(), atol=1e-5):
                    raise ValueError(f'{sid}: saved output is not residue_geometry(raw state)')
                current_by_seed[seed][sid] = (raw_state, result['backbone'].detach().cpu())
        handle.remove()
        del generator, model
        torch.cuda.empty_cache()
    for sid in ids:
        info = native[sid]
        seed_entries = []
        for seed in SEEDS:
            raw_state, saved = current_by_seed[seed][sid]
            tail = (info['y']['bb'], info['antigen'], info['antigen_mask_atoms'],
                    info['native_ca'], info['native_contacts'])
            seed_entries.append((
                measure(raw_state, info['antigen'], info['antigen_mask'], *tail),
                measure(saved, info['antigen'], info['antigen_mask'], *tail)))
        outputs['current']['saved'][sid] = {k: statistics.mean(e[0][k] for e in seed_entries)
                                            for k in MEASURES}
        outputs['current']['reconstructed'][sid] = {
            k: statistics.mean(e[1][k] for e in seed_entries) for k in MEASURES}

    # RAAD saved outputs, averaged over the available three runs so the row matches Table 1 block A
    for sid in ids:
        info = native[sid]
        saved_entries, rebuilt_entries = [], []
        for table in raad_by_seed:
            loop_bb = torch.tensor(table[sid]['backbone'], dtype=torch.float32)
            fake = dict(loop_bb=loop_bb)
            saved_entries.append(measure(loop_bb, info['antigen'], info['antigen_mask'],
                                         info['y']['bb'], info['antigen'],
                                         info['antigen_mask_atoms'], info['native_ca'],
                                         info['native_contacts']))
            rebuilt_entries.append(measure(residue_geometry(loop_bb), info['antigen'],
                                           info['antigen_mask'], info['y']['bb'],
                                           info['antigen'], info['antigen_mask_atoms'],
                                           info['native_ca'], info['native_contacts']))
        outputs['raad']['saved'][sid] = {k: statistics.mean(e[k] for e in saved_entries)
                                         for k in MEASURES}
        outputs['raad']['reconstructed'][sid] = {
            k: statistics.mean(e[k] for e in rebuilt_entries) for k in MEASURES}

    # dyMEAN saved structures: own pose and own antigen
    for sid in ids:
        info = native[sid]
        path = WORKSPACE / ('outputs/dockq_dymean_saved_diagnostic_v1/structures'
                            f'/{sid}_dymean.pdb')
        parsed = read_saved_structure(path)
        full_sequence = items[sid]['antigen_seq']
        if parsed['antigen_seq'] != full_sequence:
            raise ValueError(f'{sid}: dyMEAN antigen sequence differs from the native antigen')
        indices = torch.as_tensor(graphs[sid]['antigen_original_indices'])
        if len(parsed['antigen_bb']) != len(full_sequence):
            raise ValueError(f'{sid}: dyMEAN antigen length differs from the native antigen')
        # The native context keeps the nearest subset of antigen residues; map dyMEAN onto the
        # same subset so the two antigen atom sets correspond one to one.
        parsed = dict(loop_bb=parsed['loop_bb'], loop_mask=parsed['loop_mask'],
                      loop_seq=parsed['loop_seq'],
                      antigen_bb=parsed['antigen_bb'][indices],
                      antigen_mask=parsed['antigen_mask'][indices],
                      antigen_seq=''.join(full_sequence[i] for i in indices.tolist()))
        if len(parsed['antigen_bb']) != len(info['antigen']):
            raise ValueError(f'{sid}: retained antigen length mismatch')
        store('dymean', sid, parsed['loop_bb'], parsed['antigen_bb'],
              parsed['antigen_mask'])

    summary = dict(cohort='dev-60', n=len(ids), seeds=list(SEEDS),
                   measures=list(MEASURES),
                   terminology='dyMEAN entries are saved outputs, not raw network outputs',
                   notes=[
                       'antigen-aligned H3 RMSD fits one transform on antigen backbone atoms and '
                       'applies it unchanged to the loop; no H3 superposition is used',
                       'the current model and RAAD keep the native antigen, so their antigen '
                       'transform is the identity and the antigen-aligned value equals the raw H3 RMSD',
                       'readouts avoid framework-dependent terms because dyMEAN saves no framework',
                       'FAPE and contact F1 are evaluated in each method own pose',
                       'clash is the project proxy: relu(2 A - distance)^2, internal is loop against '
                       'itself, external is loop against the method own antigen'],
                   means={}, reconstruction_effects={}, cross_method={})
    for name in ('current', 'raad', 'dymean'):
        summary['means'][name] = {}
        for when in ('saved', 'reconstructed'):
            summary['means'][name][when] = {
                key: float(statistics.mean(outputs[name][when][sid][key] for sid in ids))
                for key in MEASURES}
        summary['reconstruction_effects'][name] = {}
        for key in RECONSTRUCTION_MEASURES:
            matrix = np.array([[outputs[name]['reconstructed'][sid][key] -
                                outputs[name]['saved'][sid][key] for sid in ids]])
            sign = -1. if key in LOWER_BETTER else 1.
            stats = interval(sign * matrix, [clusters[sid] for sid in ids], args.replicates)
            summary['reconstruction_effects'][name][key] = dict(
                improvement=stats['mean'], ci95=stats['conditional_group_ci95'],
                significant=bool(stats['conditional_group_ci95'][0] *
                                 stats['conditional_group_ci95'][1] > 0))

    # Cross-method paired contrasts at the saved state and after the same reconstruction.
    for key in MEASURES:
        for when in ('saved', 'reconstructed'):
            for label, first, second in (
                    ('current_minus_raad', 'current', 'raad'),
                    ('current_minus_dymean', 'current', 'dymean'),
                    ('raad_minus_dymean', 'raad', 'dymean')):
                matrix = np.array([[outputs[first][when][sid][key] -
                                    outputs[second][when][sid][key] for sid in ids]])
                sign = -1. if key in LOWER_BETTER else 1.
                stats = interval(sign * matrix, [clusters[sid] for sid in ids],
                                 args.replicates)
                summary['cross_method'][f'{key}:{when}:{label}'] = dict(
                    improvement=stats['mean'], ci95=stats['conditional_group_ci95'],
                    significant=bool(stats['conditional_group_ci95'][0] *
                                     stats['conditional_group_ci95'][1] > 0))
    summary['per_complex'] = {
        name: {when: {measure: [outputs[name][when][sid][measure] for sid in ids]
                      for measure in MEASURES}
               for when in ('saved', 'reconstructed')}
        for name in ('current', 'raad', 'dymean')}
    with (out / 'three_method_diagnostics.json').open('x') as handle:
        json.dump(summary, handle, indent=2)
        handle.write('\n')

    print('%-24s %10s %10s %10s' % ('measure', 'current', 'RAAD', 'dyMEAN'))
    for key in MEASURES:
        print('%-24s %10.4f %10.4f %10.4f' % (
            key, summary['means']['current']['saved'][key],
            summary['means']['raad']['saved'][key],
            summary['means']['dymean']['saved'][key]))
    print()
    print('%-24s %10s %10s %10s' % ('measure (after rebuild)', 'current', 'RAAD', 'dyMEAN'))
    for key in MEASURES:
        print('%-24s %10.4f %10.4f %10.4f' % (
            key, summary['means']['current']['reconstructed'][key],
            summary['means']['raad']['reconstructed'][key],
            summary['means']['dymean']['reconstructed'][key]))
    print()
    for name in ('current', 'raad', 'dymean'):
        print(name, json.dumps({key: round(value['improvement'], 4)
                                for key, value in
                                summary['reconstruction_effects'][name].items()}))
    print()
    for key in ('cn_internal', 'planarity', 'clash_internal', 'interface_fape', 'contact_f1',
                'loop_aligned_rmsd', 'antigen_aligned_h3_rmsd'):
        row = summary['cross_method'][f'{key}:reconstructed:current_minus_raad']
        print('%-26s current-RAAD %+.4f %s' % (
            f'{key} (after)', row['improvement'],
            '[%+.4f,%+.4f]%s' % (row['ci95'][0], row['ci95'][1],
                                 '*' if row['significant'] else '')))


if __name__ == '__main__':
    main()
