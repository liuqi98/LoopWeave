"""TM-score and contact-restricted amino-acid recovery for every method and seed.

TM-score is computed with the TMscore binary on backbone-only bodies. Two scopes are
reported: the H3 loop alone, which is pose-invariant because the loop is superposed on
itself, and the H3-plus-antigen complex, which is only meaningful for methods that keep
the native framework pose. CAAR is the amino-acid recovery restricted to H3 positions
whose native C-alpha lies within 8 A of an antigen C-alpha, so it measures sequence
accuracy at the interface rather than over the whole loop.
"""
import argparse
import json
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

PACKAGE = Path(__file__).resolve().parents[2]
WORKSPACE = Path('/media/xd/code1/dyGF')
ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
THREE = {'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE', 'G': 'GLY', 'H': 'HIS',
         'I': 'ILE', 'K': 'LYS', 'L': 'LEU', 'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN',
         'R': 'ARG', 'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'}
ARMS = {
    'final3seed': ['results/final3seed/generated_run1.json',
                   'results/final3seed/generated_run2.json',
                   'results/final3seed/generated_run3.json'],
    'loopweave': ['results/aligned_shape_v1/test_w0.3_converged_seed2026/generated.json',
                  'results/aligned_shape_v1/test_w0.3_converged_seed2027/generated.json',
                  'results/aligned_shape_v1/test_w0.3_converged_seed2028/generated.json'],
    'raad': ['outputs/raad_abgaze_reconstructed_test_v1/generated.json',
             'outputs/raad_reconstructed_seed2027_test_v2/generated.json',
             'outputs/raad_reconstructed_seed2028_test_v2/generated.json'],
    'mean': ['outputs/mean_reconstructed_test_v1/generated.json',
             'outputs/mean_reconstructed_seed2027_test/generated.json',
             'outputs/mean_reconstructed_seed2028_test/generated.json'],
    'diffab': ['outputs/diffab_reconstructed_seed2026_v1/generated.json',
               'outputs/diffab_reconstructed_seed2027_v1/generated.json',
               'outputs/diffab_reconstructed_seed2028_v1/generated.json'],
    'dymean': ['dymean_generated_seed2026', 'dymean_generated_seed2027',
               'dymean_generated_seed2028'],
}
NATIVE_POSE = ('loopweave', 'raad', 'mean', 'diffab', 'final3seed')


def locate(relative):
    for base in (PACKAGE, WORKSPACE):
        candidate = base / relative
        if candidate.exists():
            return candidate
    raise FileNotFoundError(relative)


def load_predictions(path):
    path = Path(path)
    if path.is_dir():
        # dyMEAN raw output: full complexes; take the extracted two-chain files
        return None
    document = json.loads(path.read_text())
    if 'draws' in document[0]:
        return {row['sample_id']: dict(backbone=row['draws'][0]['backbone'],
                                       sequence=row['draws'][0]['sequence'])
                for row in document}
    return {row['sample_id']: dict(backbone=row['backbone'],
                                   sequence=row.get('greedy') or row.get('sequence'))
            for row in document}


def write_pdb(path, chains):
    lines, serial = [], 1
    for chain, sequence, backbone, mask in chains:
        for index, (aa, atoms, observed) in enumerate(zip(sequence, backbone, mask), 1):
            for name, xyz, valid in zip(('N', 'CA', 'C', 'O'), atoms, observed):
                if not valid:
                    continue
                # PDB column convention: atom name occupies 13-16 with a one-character
                # element left-justified from column 14, which TMscore requires.
                lines.append('ATOM  %5d  %-3s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s\n'
                             % (serial, name, THREE[aa], chain, index, xyz[0], xyz[1], xyz[2],
                                1., 0., name[0]))
                serial += 1
        lines.append('TER\n')
    path.write_text(''.join(lines) + 'END\n')


def tmscore(model, native, workdir, multi_chain=False):
    """TM-score via the TMscore binary.

    Multi-chain inputs are written as one concatenated chain with continuous
    numbering, because this TMscore build rejects ``-c`` whenever either file has
    more than one TER record.
    """
    result = subprocess.run(['TMscore', str(model), str(native)], capture_output=True, text=True)
    match = re.search(r'TM-score\s*=\s*([\d.]+)', result.stdout)
    if match is None:
        raise RuntimeError(f'TMscore failed for {model}: {result.stdout[:200]}')
    return float(match.group(1))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--contact-cutoff', type=float, default=8.0)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    data = WORKSPACE / 'data/abgaze_reconstructed_v1'
    pairs = torch.load(WORKSPACE / 'data/raad_abgaze_reconstructed_v1/test.pt',
                       map_location='cpu', weights_only=False)
    items = {item['sample_id']: item for item, _ in pairs}
    graphs = {item['sample_id']: graph for item, graph in pairs}
    rows = torch.load(data / 'tensors/test.pt', map_location='cpu', weights_only=False)
    ids = [row[0] for row in rows]
    clusters = {json.loads(line)['pdb']: f"cluster_{json.loads(line)['cluster']}"
                for line in (data / 'splits/test.json').read_text().splitlines() if line.strip()}

    native = {}
    for sid, x, y in rows:
        antigen_mask = x['antigen_mask']
        antigen = x['context_bb'][antigen_mask]
        antigen_atom_mask = x['context_mask'][antigen_mask]
        native_ca = y['bb'][y['mask'][:, 1], 1]
        contacts = torch.cdist(native_ca, antigen[:, 1]) <= args.contact_cutoff
        native[sid] = dict(sequence=''.join(ALPHABET[i] for i in y['sequence'].tolist()),
                           backbone=y['bb'].numpy(), mask=y['mask'].numpy(),
                           antigen=antigen.numpy(), antigen_mask=antigen_atom_mask.numpy(),
                           contact_positions=contacts.any(1).numpy())

    records = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # native H3 and native complex references
        for sid in ids:
            n = native[sid]
            write_pdb(tmp / f'{sid}_native_h3.pdb', [('A', n['sequence'], n['backbone'], n['mask'])])
            write_pdb(tmp / f'{sid}_native_complex.pdb',
                      [('A', n['sequence'] + items[sid]['antigen_seq'],
                        np.concatenate([n['backbone'], n['antigen']]),
                        np.concatenate([n['mask'], n['antigen_mask']]))])
        for arm, paths in ARMS.items():
            for run_index, path in enumerate(paths):
                predictions = None if path.startswith('dymean_generated') \
                    else load_predictions(locate(path))
                for sid in ids:
                    n = native[sid]
                    if arm == 'dymean':
                        from Bio.PDB import PDBParser
                        import warnings
                        warnings.filterwarnings('ignore')
                        structure = PDBParser(QUIET=True).get_structure(
                            'd', str(WORKSPACE / 'outputs/dymean_extracted_v1'
                                    / f'seed{2026 + run_index}' / f'{sid}_dymean.pdb'))[0]
                        chains = {}
                        for chain in structure:
                            residues = [r for r in chain if r.id[0] == ' ']
                            if residues:
                                chains[chain.id] = residues
                        h3 = chains['H']
                        pred_sequence = ''.join(
                            {v: k for k, v in THREE.items()}[r.resname] for r in h3)
                        pred_backbone = np.array([
                            [[r[a].coord if a in r else (0., 0., 0.) for a in ('N', 'CA', 'C', 'O')]
                             for r in h3]])[0]
                        pred_mask = np.array([[a in r for a in ('N', 'CA', 'C', 'O')] for r in h3])
                        antigen_residues = chains['A']
                        antigen_backbone = np.array([
                            [[r[a].coord if a in r else (0., 0., 0.) for a in ('N', 'CA', 'C', 'O')]
                             for r in antigen_residues]])[0]
                        antigen_mask = np.array([[a in r for a in ('N', 'CA', 'C', 'O')]
                                                 for r in antigen_residues])
                        antigen_sequence = ''.join(
                            {v: k for k, v in THREE.items()}[r.resname] for r in antigen_residues)
                        retained = torch.as_tensor(graphs[sid]['antigen_original_indices']).numpy()
                        pred_antigen = antigen_backbone[retained]
                        pred_antigen_mask = antigen_mask[retained]
                        pred_antigen_sequence = ''.join(antigen_sequence[i] for i in retained)
                        native_pose = False
                    else:
                        record = predictions[sid]
                        pred_sequence = record['sequence']
                        pred_backbone = np.asarray(record['backbone'])
                        pred_mask = n['mask']
                        pred_antigen = n['antigen']
                        pred_antigen_mask = n['antigen_mask']
                        pred_antigen_sequence = items[sid]['antigen_seq']
                        native_pose = True

                    write_pdb(tmp / 'model_h3.pdb',
                              [('A', pred_sequence, pred_backbone, pred_mask)])
                    h3_tm = tmscore(tmp / 'model_h3.pdb', tmp / f'{sid}_native_h3.pdb', tmp)
                    complex_tm = None
                    if native_pose:
                        write_pdb(tmp / 'model_complex.pdb',
                                  [('A', pred_sequence + pred_antigen_sequence,
                                    np.concatenate([pred_backbone, pred_antigen]),
                                    np.concatenate([pred_mask, pred_antigen_mask]))])
                        complex_tm = tmscore(tmp / 'model_complex.pdb',
                                             tmp / f'{sid}_native_complex.pdb', tmp,
                                             multi_chain=True)
                    correct = np.array([a == b for a, b in
                                        zip(pred_sequence, n['sequence'])], dtype=float)
                    positions = n['contact_positions']
                    caar = float(correct[positions].mean()) if positions.any() else None
                    records.append(dict(sample_id=sid, group=clusters[sid], arm=arm,
                                        run=run_index, tm_h3=h3_tm, tm_complex=complex_tm,
                                        caar=caar, aar=float(correct.mean()),
                                        n_contact_positions=int(positions.sum())))

    summaries = {}
    for arm in ARMS:
        per_run = {}
        for record in records:
            if record['arm'] != arm:
                continue
            per_run.setdefault(record['run'], []).append(record)
        def run_mean(key):
            values = []
            for run in per_run.values():
                part = [r[key] for r in run if r[key] is not None]
                if part:
                    values.append(statistics.mean(part))
            return values
        entry = dict(runs=len(per_run))
        for key in ('tm_h3', 'tm_complex', 'caar', 'aar'):
            values = run_mean(key)
            entry[key] = dict(mean=float(statistics.mean(values)) if values else None,
                              sd=float(statistics.stdev(values)) if len(values) > 1 else 0.,
                              run_means=values)
        summaries[arm] = entry

    (out / 'tmscore_caar.json').write_text(json.dumps(
        dict(protocol=dict(tm='TMscore binary on backbone-only bodies, N/CA/C/O',
                           tm_h3='H3 superposed on itself, pose-invariant',
                           tm_complex='H3 plus antigen concatenated into one chain and numbered '
                                      'continuously; native pose only, not defined for dyMEAN',
                           caar='amino-acid recovery at H3 positions whose native CA is within '
                                f'{args.contact_cutoff} A of an antigen CA',
                           selection='first joint draw for DiffAb, greedy sequence elsewhere'),
             summaries=summaries, per_sample=records), indent=2) + '\n')
    print('%-10s %14s %14s %14s %14s' % ('arm', 'TM H3', 'TM complex', 'CAAR', 'AAR'))
    for arm, block in summaries.items():
        def fmt(key):
            value = block[key]
            return '--' if value['mean'] is None else '%.4f+-%.3f' % (value['mean'], value['sd'])
        print('%-10s %14s %14s %14s %14s' % (arm, fmt('tm_h3'), fmt('tm_complex'),
                                             fmt('caar'), fmt('aar')))


if __name__ == '__main__':
    main()
