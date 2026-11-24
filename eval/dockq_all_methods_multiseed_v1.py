"""Official DockQ for every method under one identical backbone-restricted definition.

The manuscript previously mixed DockQ values produced by different script versions. This
script rebuilds every arm with the same code path: the predicted H3 is placed on the shared
observed-atom mask, the antigen and sequence come from the matched data, and the official
DockQ package scores the H3-versus-antigen interface on N/CA/C/O only. No alignment, no
native side chains, no candidate reselection, and unscorable targets are reported rather
than zeroed.
"""
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from Bio.SeqUtils import seq3
from DockQ.DockQ import load_PDB, run_on_all_native_interfaces, run_on_chains

PACKAGE = Path(__file__).resolve().parents[2]
WORKSPACE = Path('/media/xd/code1/dyGF')
ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'

ARMS = {
    'current': ['results/aligned_shape_v1/test_w0.3_converged_seed2026/generated.json',
                'results/aligned_shape_v1/test_w0.3_converged_seed2027/generated.json',
                'results/aligned_shape_v1/test_w0.3_converged_seed2028/generated.json'],
    'raad': ['outputs/raad_abgaze_reconstructed_test_v1/generated.json',
             'outputs/raad_reconstructed_seed2027_test_v2/generated.json',
             'outputs/raad_reconstructed_seed2028_test_v2/generated.json'],
    'mean': ['outputs/mean_reconstructed_test_v1/generated.json',
             'outputs/mean_reconstructed_seed2027_test/generated.json',
             'outputs/mean_reconstructed_seed2028_test/generated.json'],
    'final3seed': ['results/final3seed/generated_run1.json',
                   'results/final3seed/generated_run2.json',
                   'results/final3seed/generated_run3.json'],
    'geoab': ['results/geoab_design_v1/generated_run1.json',
              'results/geoab_design_v1/generated_run2.json',
              'results/geoab_design_v1/generated_run3.json'],
    'diffab': ['outputs/diffab_reconstructed_seed2026_v1/generated.json',
               'outputs/diffab_reconstructed_seed2027_v1/generated.json',
               'outputs/diffab_reconstructed_seed2028_v1/generated.json'],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def locate(relative):
    for base in (PACKAGE, WORKSPACE):
        candidate = base / relative
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(relative)


def build_pdb(path, bodies):
    lines, serial = [], 1
    for chain, sequence, bb, mask in bodies:
        for index, (aa, atoms, observed) in enumerate(zip(sequence, bb, mask), 1):
            for name, xyz, valid in zip(('N', 'CA', 'C', 'O'), atoms, observed):
                if not valid:
                    continue
                lines.append('ATOM  %5d %4s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s\n'
                             % (serial, name, seq3(aa).upper(), chain, index,
                                xyz[0], xyz[1], xyz[2], 1., 0., name[0]))
                serial += 1
        lines.append('TER\n')
    path.write_text(''.join(lines) + 'END\n')


def measure(model, native):
    ms, ns = load_PDB(str(model)), load_PDB(str(native))
    result, _ = run_on_all_native_interfaces(ms, ns, chain_map={'A': 'A', 'B': 'B'}, no_align=True)
    run_on_chains.cache_clear()
    return result.get('AB')


def load_predictions(path):
    document = json.loads(Path(path).read_text())
    if 'draws' in document[0]:
        return {row['sample_id']: dict(backbone=row['draws'][0]['backbone'],
                                       sequence=row['draws'][0]['sequence'])
                for row in document}
    return {row['sample_id']: dict(backbone=row['backbone'],
                                   sequence=row.get('greedy') or row.get('sequence'))
            for row in document}


def main():
    out = (PACKAGE / 'results/dockq_all_methods_multiseed_v1')
    out.mkdir(parents=True, exist_ok=False)
    structures = out / 'structures'
    structures.mkdir()
    data = WORKSPACE / 'data/abgaze_reconstructed_v1'
    pairs = torch.load(WORKSPACE / 'data/raad_abgaze_reconstructed_v1/test.pt',
                       map_location='cpu', weights_only=False)
    items = {item['sample_id']: item for item, _ in pairs}
    rows = {sid: (x, y) for sid, x, y in
            torch.load(data / 'tensors/test.pt', map_location='cpu', weights_only=False)}
    metadata = [json.loads(line) for line in
                (data / 'splits/test.json').read_text().splitlines() if line]
    ids = [row['pdb'] for row in metadata]
    groups = {row['pdb']: row['cluster'] for row in metadata}

    references = {}
    for sid in ids:
        x, y = rows[sid]
        item = items[sid]
        sequence = ''.join(ALPHABET[i] for i in y['sequence'].tolist())
        antigen = ('B', item['antigen_seq'], np.asarray(item['antigen_backbone']),
                   np.asarray(item['antigen_backbone_mask'], dtype=bool))
        path = structures / f'{sid}_native.pdb'
        build_pdb(path, [('A', sequence, y['bb'].numpy(), y['mask'].numpy()), antigen])
        references[sid] = path

    results = []
    for arm, paths in ARMS.items():
        for seed_index, path in enumerate(paths):
            predictions = load_predictions(locate(path))
            for sid in ids:
                x, y = rows[sid]
                item = items[sid]
                record = predictions[sid]
                prediction = np.asarray(record['backbone'])
                sequence = ''.join(ALPHABET[i] for i in y['sequence'].tolist())
                antigen = ('B', item['antigen_seq'], np.asarray(item['antigen_backbone']),
                           np.asarray(item['antigen_backbone_mask'], dtype=bool))
                if prediction.shape != y['bb'].shape:
                    raise ValueError(f'{arm} {sid}: backbone shape mismatch')
                model = structures / f'{sid}_{arm}_{seed_index}.pdb'
                build_pdb(model, [('A', record['sequence'], prediction, y['mask'].numpy()),
                                  antigen])
                scored = measure(model, references[sid])
                # Distinct runs may share a directory (per-run JSON files), so key the run on
                # the file stem as well as the parent directory.
                run_label = Path(path).parent.name + '/' + Path(path).stem
                entry = dict(sample_id=sid, group=groups[sid], arm=arm,
                             run=run_label, scorable=scored is not None)
                if scored is not None:
                    entry.update({k: float(scored[k]) for k in ('DockQ', 'iRMSD', 'LRMSD', 'fnat')})
                results.append(entry)

    summaries = {}
    for arm in ARMS:
        per_run = {}
        for entry in results:
            if entry['arm'] != arm:
                continue
            per_run.setdefault(entry['run'], []).append(entry)
        run_means = []
        for run, entries in per_run.items():
            eligible = [e for e in entries if e['scorable']]
            run_means.append(statistics.mean(e['DockQ'] for e in eligible))
        per_complex = {}
        for sid in ids:
            values = [e['DockQ'] for run in per_run.values() for e in run
                      if e['sample_id'] == sid and e['scorable']]
            if values:
                per_complex[sid] = float(np.mean(values))
        summaries[arm] = dict(
            runs=list(per_run), n_runs=len(per_run),
            n_scorable=len(per_complex),
            mean=float(np.mean(list(per_complex.values()))),
            seed_sd=float(statistics.stdev(run_means)) if len(run_means) > 1 else 0.,
            run_means=run_means)
    protocol = dict(scope='H3 versus all supplied antigen residues, N/CA/C/O only',
                    dockq='official package, no_align=True, chain_map A->A B->B',
                    mask='shared observed-atom mask; no native side chains',
                    selection='first joint draw for DiffAb; greedy sequence elsewhere; no reselection',
                    unscorable='reported, not zeroed')
    (out / 'result.json').write_text(json.dumps(dict(protocol=protocol, summaries=summaries,
                                                     per_complex=results), indent=2) + '\n')
    print('%-10s %8s %8s %6s %6s' % ('arm', 'DockQ', 'seed_sd', 'runs', 'scorable'))
    for arm, block in summaries.items():
        print('%-10s %8.4f %8.4f %6d %6d' % (arm, block['mean'], block['seed_sd'],
                                             block['n_runs'], block['n_scorable']))


if __name__ == '__main__':
    main()
