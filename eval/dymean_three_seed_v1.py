"""Score the three self-generated dyMEAN runs with the same functions as the matched comparison.

dyMEAN was re-run locally from the released checkpoint (``checkpoints/cdrh3_design.ckpt``) on
the same 60 targets with three sampling seeds, using the restored upstream
``generate.py`` (the file was missing from the code2 copy). Each run writes
``<sample_id>.pdb`` and ``<sample_id>_original.pdb``; the generated file contains the
predicted H3 loop and the antigen chain in dyMEAN's own pose.

Readouts and their definitions are taken verbatim from ``three_method_diagnostics_v1`` so the
numbers are comparable with the current model and RAAD: in-loop C-N, internal planarity,
internal and external clash, frame-aligned FAPE, contact F1, loop-aligned CA RMSD and
antigen-aligned H3 RMSD. The antigenaligned value uses one rigid transform fitted on the
shared observed antigen atoms.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

PACKAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE / 'code'))
sys.path.insert(0, '/media/xd/code1/dyGF')
from eval.three_method_diagnostics_v1 import (MEASURES, measure, read_saved_structure)
from structure_refinement_ablation import residue_geometry
from research_audit.statistics import interval

WORKSPACE = Path('/media/xd/code1/dyGF')
SEEDS = (2026, 2027, 2028)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runs', type=Path,
                        default=WORKSPACE / 'outputs/dymean_extracted_v1')
    parser.add_argument('--reference', type=Path,
                        default=PACKAGE / 'results/three_method_diagnostics_v1'
                                  / 'three_method_diagnostics.json')
    parser.add_argument('--replicates', type=int, default=10000)
    args = parser.parse_args()
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
    groups = [clusters[sid] for sid in ids]

    native = {}
    for sid, x, y in rows:
        mask = x['antigen_mask']
        antigen = x['context_bb'][mask]
        antigen_atom_mask = x['context_mask'][mask]
        retained = torch.as_tensor(graphs[sid]['antigen_original_indices'])
        if len(antigen) != len(retained):
            raise ValueError(f'{sid}: retained antigen count mismatch')
        native_ca = y['bb'][y['mask'][:, 1], 1]
        native[sid] = dict(x=x, y=y, antigen=antigen, antigen_atom_mask=antigen_atom_mask,
                           native_ca=native_ca,
                           native_contacts=torch.cdist(native_ca, antigen[:, 1]) <= 8.)

    reference = json.loads(args.reference.read_text())
    per_seed, per_seed_rebuilt = {}, {}
    for seed in SEEDS:
        directory = args.runs / f'seed{seed}'
        entries, rebuilt_entries = {}, {}
        for sid in ids:
            path = directory / f'{sid}_dymean.pdb'
            if not path.is_file():
                raise FileNotFoundError(path)
            parsed = read_saved_structure(path)
            info = native[sid]
            retained = torch.as_tensor(graphs[sid]['antigen_original_indices'])
            if len(parsed['antigen_bb']) != len(items[sid]['antigen_seq']):
                raise ValueError(f'{sid}: dyMEAN antigen length differs from the native antigen')
            antigen = parsed['antigen_bb'][retained]
            antigen_mask = parsed['antigen_mask'][retained]
            tail = (info['y']['bb'], info['antigen'], info['antigen_atom_mask'],
                    info['native_ca'], info['native_contacts'])
            entries[sid] = measure(parsed['loop_bb'], antigen, antigen_mask, *tail)
            rebuilt_entries[sid] = measure(residue_geometry(parsed['loop_bb']), antigen,
                                           antigen_mask, *tail)
        per_seed[seed] = entries
        per_seed_rebuilt[seed] = rebuilt_entries

    summary = dict(seeds=list(SEEDS), n=len(ids), measures=list(MEASURES),
                   per_seed_reconstructed=None,
                   protocol='dyMEAN rerun locally from the released checkpoint with three '
                            'sampling seeds through the restored upstream generate.py',
                   per_seed={}, seed_statistics={}, contrasts_vs_raad={})
    for seed in SEEDS:
        summary['per_seed'][str(seed)] = {
            key: float(statistics.mean(per_seed[seed][sid][key] for sid in ids))
            for key in MEASURES}
    summary['seed_statistics'] = {
        key: dict(mean=float(statistics.mean(summary['per_seed'][str(s)][key] for s in SEEDS)),
                  sd=float(statistics.stdev([summary['per_seed'][str(s)][key] for s in SEEDS]))
                  if len(SEEDS) > 1 else 0.)
        for key in MEASURES}
    rebuilt_seed_means = {
        str(seed): {key: float(statistics.mean(per_seed_rebuilt[seed][sid][key]
                                               for sid in ids)) for key in MEASURES}
        for seed in SEEDS}
    summary['per_seed_reconstructed'] = rebuilt_seed_means
    summary['seed_statistics_reconstructed'] = {
        key: dict(mean=float(statistics.mean(rebuilt_seed_means[str(s)][key] for s in SEEDS)),
                  sd=float(statistics.stdev([rebuilt_seed_means[str(s)][key] for s in SEEDS])))
        for key in MEASURES}

    raad = reference['per_complex']['raad']['saved']
    for key in MEASURES:
        matrix = np.array([[per_seed[seed][sid][key] - raad[key][index]
                            for index, sid in enumerate(ids)] for seed in SEEDS]).mean(0)[None]
        sign = -1. if key in ('cn_internal', 'planarity', 'clash_internal', 'clash_external',
                              'interface_fape', 'loop_aligned_rmsd', 'antigen_aligned_h3_rmsd') else 1.
        stats = interval(sign * matrix, groups, args.replicates)
        summary['contrasts_vs_raad'][key] = dict(
            improvement=stats['mean'], ci95=stats['conditional_group_ci95'],
            significant=bool(stats['conditional_group_ci95'][0] *
                             stats['conditional_group_ci95'][1] > 0))

    (out / 'dymean_three_seed.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('%-24s %8s %8s %8s | %8s %7s' % ('measure', *[f'seed{s}' for s in SEEDS],
                                           'mean', 'sd'))
    for key in MEASURES:
        block = summary['seed_statistics'][key]
        print('%-24s %8.4f %8.4f %8.4f | %8.4f %7.4f' % (
            key, *[summary['per_seed'][str(s)][key] for s in SEEDS], block['mean'], block['sd']))
    print()
    print('%-24s %8s %8s %8s | %8s %7s' % ('after reconstruction',
                                           *[f'seed{s}' for s in SEEDS], 'mean', 'sd'))
    for key in MEASURES:
        block = summary['seed_statistics_reconstructed'][key]
        print('%-24s %8.4f %8.4f %8.4f | %8.4f %7.4f' % (
            key, *[rebuilt_seed_means[str(s)][key] for s in SEEDS], block['mean'], block['sd']))
    print()
    print('dyMEAN vs RAAD (positive = dyMEAN better):')
    for key, block in summary['contrasts_vs_raad'].items():
        print('  %-24s %+8.4f [%+.4f,%+.4f] %s' % (
            key, block['improvement'], block['ci95'][0], block['ci95'][1],
            'sig' if block['significant'] else 'ns'))


if __name__ == '__main__':
    main()
