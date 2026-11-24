"""Figure for the stage-by-stage geometry diagnostic."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STAGES = ('initial', 'before_pair', 'layer1', 'layer2', 'layer3', 'layer4', 'final')
LABELS = ('initial\n(generator)', 'before\npair', 'pair L1', 'pair L2', 'pair L3', 'pair L4',
          'final\n(intra-residue\nrebuild)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    document = json.loads(args.diagnostic.read_text())
    metrics = document['metrics']
    positions = list(range(len(STAGES)))
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))

    ax = axes[0]
    for key, label, color, marker in (
            ('cn_internal', 'in-loop bond', '#4c78a8', 'o'),
            ('cn_left_anchor', 'left flank (N-term)', '#f58518', 's'),
            ('cn_right_anchor', 'right flank (C-term)', '#54a24b', '^'),
            ('cn_mae', 'all bonds', '#333333', 'D')):
        values = [metrics[key][stage]['mean'] for stage in STAGES]
        spread = [metrics[key][stage]['sample_std'] for stage in STAGES]
        ax.errorbar(positions, values, yerr=spread, label=label, color=color, marker=marker,
                    linewidth=1.8, markersize=5, capsize=2.5)
    ax.axvline(5.5, color='#bbbbbb', linestyle=':', linewidth=1.2)
    ax.set_xticks(positions)
    ax.set_xticklabels(LABELS, fontsize=7.5)
    ax.set_ylabel('C–N bond length MAE (Å)')
    ax.set_title('A. Bond continuity, split by bond type', fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=.22, axis='y')

    ax = axes[1]
    values = [metrics['interface_fape'][stage]['mean'] for stage in STAGES]
    spread = [metrics['interface_fape'][stage]['sample_std'] for stage in STAGES]
    ax.errorbar(positions, values, yerr=spread, color='#e45756', marker='o', linewidth=1.9,
                markersize=5, capsize=2.5)
    ax.axvline(5.5, color='#bbbbbb', linestyle=':', linewidth=1.2)
    ax.set_xticks(positions)
    ax.set_xticklabels(LABELS, fontsize=7.5)
    ax.set_ylabel('interface FAPE (lower is better)')
    ax.set_title('B. Interface-frame error', fontsize=11)
    ax.grid(alpha=.22, axis='y')
    ax.annotate('improvement happens in L4\n(−0.15)', xy=(5, values[5]), xytext=(2.2, values[5] + .05),
                fontsize=8, arrowprops=dict(arrowstyle='->', color='#666666', lw=1))

    ax = axes[2]
    values = [metrics['contact_f1'][stage]['mean'] for stage in STAGES]
    spread = [metrics['contact_f1'][stage]['sample_std'] for stage in STAGES]
    ax.errorbar(positions, values, yerr=spread, color='#4c78a8', marker='o', linewidth=1.9,
                markersize=5, capsize=2.5)
    ax.axvline(5.5, color='#bbbbbb', linestyle=':', linewidth=1.2)
    ax.set_xticks(positions)
    ax.set_xticklabels(LABELS, fontsize=7.5)
    ax.set_ylabel('contact F1 (higher is better)')
    ax.set_title('C. Interface contacts', fontsize=11)
    ax.grid(alpha=.22, axis='y')

    fig.suptitle('Where the geometry comes from: four pair layers trade bond continuity for '
                 'interface accuracy, and a deterministic intra-residue rebuild restores the '
                 'bonds\n(dev-60, three seeds; the rebuild keeps CA fixed and performs no '
                 'inter-residue closure, so panels B and C are unchanged between pair L4 and '
                 'final)',
                 fontsize=11.5, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, .88))
    for suffix in ('png', 'svg'):
        fig.savefig(args.output.with_suffix(f'.{suffix}'), dpi=200)
    plt.close(fig)
    print(args.output.with_suffix('.png'))


if __name__ == '__main__':
    main()
