"""Figure for the mechanism-arm comparison: paired deltas against the full arm."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PRIMARY = ('cn_mae', 'interface_fape', 'contact_f1')
SECONDARY = ('rmsd', 'aligned_rmsd', 'clash', 'precision', 'recall', 'aar')
LABELS = dict(cn_mae='C-N MAE', interface_fape='interface FAPE', contact_f1='contact F1',
              rmsd='RMSD', aligned_rmsd='aligned RMSD', clash='clash', precision='precision',
              recall='recall', aar='AAR')
ARMS = ('reset_z', 'pair_mlp', 'feature_only')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evaluation = args.evaluation.resolve()
    primary = json.loads((evaluation / 'mechanism_summary.json').read_text())
    tradeoff = json.loads((evaluation / 'tradeoffs/tradeoff_summary.json').read_text())
    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    metrics = PRIMARY + SECONDARY
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), sharey=True)
    for ax, arm in zip(axes, ARMS):
        positions, values, lows, highs, significant = [], [], [], [], []
        for index, key in enumerate(metrics):
            source = primary['contrasts'][arm] if key in primary['contrasts'][arm] else \
                tradeoff['contrasts'][arm]
            entry = source.get(key)
            if entry is None:
                continue
            positions.append(index)
            values.append(entry['improvement'])
            lows.append(entry['ci95'][0])
            highs.append(entry['ci95'][1])
            significant.append(entry['significant'])
        values = np.array(values)
        lows = np.array(lows)
        highs = np.array(highs)
        colors = ['#e45756' if flag else '#9ecae1' for flag in significant]
        ax.axhline(0, color='#333333', linewidth=1)
        ax.errorbar(positions, values, yerr=[values - lows, highs - values], fmt='none',
                    ecolor='#666666', capsize=3, linewidth=1.1)
        ax.scatter(positions, values, c=colors, s=55, zorder=3,
                   edgecolor='#333333', linewidth=.6)
        for boundary in (2.5,):
            ax.axvline(boundary, color='#bbbbbb', linestyle=':', linewidth=1)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([LABELS[key] for key in metrics], rotation=45, ha='right',
                           fontsize=8)
        ax.set_title(arm.replace('_', ' '), fontsize=11)
        ax.grid(alpha=.22, axis='y')
        ax.text(.02, .95, 'primary', transform=ax.transAxes, fontsize=8, color='#555555')
        ax.text(.55, .95, 'trade-offs', transform=ax.transAxes, fontsize=8, color='#555555')
    axes[0].set_ylabel('full − arm   (positive = full better)')
    fig.suptitle('Mechanism controls: paired difference from the full arm '
                 '(3 seeds, group bootstrap by H3 cluster; red = CI excludes 0)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, .93))
    for suffix in ('png', 'svg'):
        fig.savefig(out.with_suffix(f'.{suffix}'), dpi=200)
    plt.close(fig)
    print(out.with_suffix('.png'))


if __name__ == '__main__':
    main()
