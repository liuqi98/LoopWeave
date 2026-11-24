"""Figure for the full-cohort four-cell post-processing comparison."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CONDITIONS = ('ours_before', 'ours_after', 'raad_raw', 'raad_after')
LABELS = ('ours\nbefore', 'ours\nafter', 'RAAD\nraw', 'RAAD\nafter')
COLORS = ('#c9d7ee', '#4c78a8', '#f2c9c1', '#e45756')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    document = json.loads(args.summary.read_text())
    means = document['means']
    panel_specs = (('cn_mae', 'C–N bond MAE (Å)', 'A. Bond continuity'), 
                   ('interface_fape', 'interface FAPE', 'B. Interface-frame error'),
                   ('contact_f1', 'contact F1', 'C. Interface contacts'))
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    for ax, (key, ylabel, title) in zip(axes, panel_specs):
        values = [means[condition][key]['mean'] for condition in CONDITIONS]
        errors = [means[condition][key]['sample_std'] for condition in CONDITIONS]
        positions = range(len(CONDITIONS))
        ax.bar(positions, values, color=COLORS, edgecolor='#333333', linewidth=.7,
               yerr=errors, capsize=4, error_kw=dict(ecolor='#555555', lw=1))
        for position, value in zip(positions, values):
            ax.text(position, value, f'{value:.3f}', ha='center', va='bottom', fontsize=8.5)
        ax.set_xticks(list(positions))
        ax.set_xticklabels(LABELS, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=.22, axis='y')
        # bracket the projection effect for each method
        for first, second, colour, text in ((0, 1, '#4c78a8', 'ours'),
                                            (2, 3, '#e45756', 'RAAD')):
            top = max(values[first], values[second])
            span = max(values) * .12
            ax.annotate('', xy=(second, top + span), xytext=(first, top + span),
                        arrowprops=dict(arrowstyle='->', color=colour, lw=1.6))
            delta = document['deltas'][f'{key}:projection_effect_{text.lower()}']
            ax.text((first + second) / 2, top + span * 1.6,
                    f'Δ {delta["difference"]:+.3f}' + (' *' if delta['significant'] else ''),
                    ha='center', fontsize=8.5, color=colour)
        ax.set_ylim(0, max(values) * 1.32)
    fig.suptitle('Four-cell post-processing comparison on the full RAbD test set (n=60, ours = 3 '
                 'seeds)\nthe deterministic intra-residue rebuild is CA-preserving and performs no '
                 'inter-residue closure, so FAPE and contact F1 cannot move; only C–N changes '
                 '(Δ, * = CI excludes 0)',
                 fontsize=11.5, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, .87))
    for suffix in ('png', 'svg'):
        fig.savefig(args.output.with_suffix(f'.{suffix}'), dpi=200)
    plt.close(fig)
    print(args.output.with_suffix('.png'))


if __name__ == '__main__':
    main()
