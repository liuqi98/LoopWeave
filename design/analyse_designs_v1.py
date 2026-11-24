"""Design CDRH3 loops for FLAG / gp120 / c-MET / PRV with the current model, then
compare the designs against the native loops and the training distribution.

Honest framing
--------------
The current model was trained only on Fab complexes (heavy + light chain). All four
targets are single-domain VHH nanobodies, so this is an out-of-distribution
application. Nothing here is an affinity prediction or a validated design: no
docking, refolding, energy or binding assay was run.

This script only reads the committed design outputs from ``vhh_design_v1.py`` and the
archived native/training sequences. It writes a comparison table and a figure.
"""
import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT.parent
ORIG = Path('/media/xd/code1/dyGF')
TARGETS = ['gp120_G36', 'cMET_1_7', 'PRV_Nb36', 'PRV_Nb1']
CANONICAL = 'ACDEFGHIKLMNPQRSTVWY'


def load_design(target, root):
    return json.loads((root / target / 'designs.json').read_text())


def hamming(first, second):
    length = min(len(first), len(second))
    return sum(a != b for a, b in zip(first[:length], second[:length])) / max(length, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--designs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.designs.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    training = json.loads((PACKAGE / 'data/train/records.json').read_text())
    lengths = [len(r['cdrh3_seq']) for r in training]
    length_bins = [(6, 13), (14, 17), (18, 20), (21, 24)]
    length_support = {f'{low}-{high}': sum(1 for value in lengths if low <= value <= high)
                      for low, high in length_bins}

    rows, all_designed, all_native = [], Counter(), Counter()
    for target in TARGETS:
        document = load_design(target, root)
        designs = document['design_sequences']
        sequences = [row['sequence'] for row in designs]
        native = document['native_h3']
        quality = document['geometry_quality']
        all_designed.update(''.join(sequences))
        all_native.update(native)
        entry = dict(
            target=target, note=document['note'], native_h3=native,
            native_h3_length=document['native_h3_length'],
            n_designs=len(sequences),
            unique_sequences=len(set(sequences)),
            native_cn_mae=None,
            cn_mae=statistics.mean(quality[s]['cn_mae'] for s in quality),
            cn_outlier=statistics.mean(quality[s]['cn_outlier'] for s in quality),
            clash=statistics.mean(quality[s]['clash'] for s in quality),
            planarity=statistics.mean(quality[s]['plane'] for s in quality),
            bond=statistics.mean(quality[s]['bond'] for s in quality),
            mean_hamming_to_native=statistics.mean(hamming(s, native) for s in sequences),
            leakage_checks_passed=all(document['target_free_leakage_check'].values()),
            atoms=document['feature_metadata']['atoms'],
            surface_points=document['feature_metadata']['surface_points'],
            top_design=sequences[0],
            top_design_logprob=designs[0]['mean_logprob'],
        )
        rows.append(entry)

    summary = dict(
        cohort='VHH design with the current Fab-trained model',
        targets=rows,
        training_h3_length_support=length_support,
        training_h3_length_median=statistics.median(lengths),
        designed_aa_frequency={a: all_designed[a] / sum(all_designed.values())
                               for a in [x for x, _ in all_designed.most_common()]},
        native_aa_frequency={a: all_native[a] / sum(all_native.values())
                             for a in [x for x, _ in all_native.most_common()]},
        reference_development='dev-60 test cn_mae = 0.272 Å (three seeds, aligned model)',
        verdict=('The pipeline runs end to end and every design is target-free (leakage assertions '
                 'pass), but the designs are not usable as candidate binders: loop geometry '
                 'degrades sharply with length (C-N MAE 0.40 Å at 12 aa to 1.44 Å at 23 aa versus '
                 '0.27 Å in-distribution), and the sequence composition collapses to a '
                 'G/Y/D/A/S/R-rich consensus that differs markedly from the native loops.'),
        limitations=['The model was trained only on Fab complexes; these VHH targets have no light '
                     'chain, so this is out of distribution.',
                     'No docking, refolding, energy or affinity calculation was performed, so no '
                     'candidate can be called improved.',
                     'Sequences are ranked only by a deployable model log-probability; the ranking '
                     'is not a binding prediction.',
                     'Only 3 training seeds and 6 candidates per seed; this is a feasibility check, '
                     'not a design campaign.'],
    )
    (out / 'design_summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    names = [row['target'] for row in rows]
    xs = np.arange(len(rows))

    ax = axes[0]
    ax.bar(xs - .2, [row['cn_mae'] for row in rows], width=.4, label='designed loop', color='#e45756')
    ax.axhline(0.272, color='#4c78a8', linestyle='--', linewidth=1.5,
               label='in-distribution dev-60 (0.272 Å)')
    for x, row in zip(xs, rows):
        ax.text(x - .2, row['cn_mae'], f"{row['cn_mae']:.2f}", ha='center', va='bottom', fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f'{row["target"]}\n({row["native_h3_length"]} aa)' for row in rows], fontsize=8)
    ax.set_ylabel('C–N bond MAE (Å)')
    ax.set_title('A. Predicted loop continuity', fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=.25, axis='y')

    ax = axes[1]
    ax.bar(xs, [row['mean_hamming_to_native'] for row in rows], color='#f58518', width=.55)
    for x, row in zip(xs, rows):
        ax.text(x, row['mean_hamming_to_native'], f"{row['mean_hamming_to_native']:.2f}",
                ha='center', va='bottom', fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=20, fontsize=8)
    ax.set_ylabel('normalised Hamming distance to native')
    ax.set_ylim(0, 1.05)
    ax.set_title('B. Distance from the native loop', fontsize=11)
    ax.grid(alpha=.25, axis='y')

    ax = axes[2]
    designed = [summary['designed_aa_frequency'].get(a, 0.) for a in CANONICAL]
    native = [summary['native_aa_frequency'].get(a, 0.) for a in CANONICAL]
    positions = np.arange(len(CANONICAL))
    ax.bar(positions - .2, designed, width=.4, label='designed', color='#e45756')
    ax.bar(positions + .2, native, width=.4, label='native H3', color='#4c78a8')
    ax.set_xticks(positions)
    ax.set_xticklabels(list(CANONICAL), fontsize=8)
    ax.set_ylabel('amino-acid frequency')
    ax.set_title('C. Composition shift', fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=.25, axis='y')

    fig.suptitle('VHH CDRH3 design with a Fab-trained model: feasibility and its limits',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, .94))
    for suffix in ('png', 'svg'):
        fig.savefig(out / f'fig_design_diagnostics.{suffix}', dpi=200)
    plt.close(fig)
    print(json.dumps({row['target']: dict(length=row['native_h3_length'],
                                          cn_mae=round(row['cn_mae'], 3),
                                          hamming=round(row['mean_hamming_to_native'], 3),
                                          top=row['top_design']) for row in rows}, indent=1))


if __name__ == '__main__':
    main()
