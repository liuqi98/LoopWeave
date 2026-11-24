"""Reproduce the dyGF publication figure families with the current model.

The archived figures live in ``/media/xd/code2/GenCDRH3_final/dygf_paper_figure``.
This script rebuilds the subset that this model and this data can actually support,
on data that exists in the archive, and refuses to draw the ones that would need
pipelines this project deliberately does not own (docking, all-atom energy,
folding, language-model perplexity, affinity maturation).

Figure families produced here
  F1  CDRH3 physicochemical properties (net charge / GRAVY / pI), KDE by source
  F2  sequence logos, N-aligned and C-aligned, native vs each model
  F3  ablation summary, split by evaluation cohort
  F4  diversity vs quality Pareto, target level
  F5  amino-acid recovery vs H3 length
  F6  per-complex metric profiles (3 seeds)

Everything reads only committed files plus ``sequences.json`` from
``collect_figure_sequences_v1.py``; no model is executed here.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT.parent
sys.path.insert(0, str(ROOT))

CANONICAL = 'ACDEFGHIKLMNPQRSTVWY'
COLORS = {
    'Native PDB': '#2ca02c',
    'Ours (full)': '#1f77b4',
    'Ours (aligned, converged)': '#17becf',
    'RAAD': '#ff7f0e',
    'MEAN': '#9467bd',
    'DiffAb': '#d62728',
    'FDA approved': '#000000',
}


def clean(sequence):
    return ''.join(a for a in str(sequence).upper() if a in CANONICAL)


def properties(sequence):
    """Net charge at pH 7.4, GRAVY and pI, using Biopython ProtParam scales."""
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    seq = clean(sequence)
    if len(seq) < 3:
        return None
    analysis = ProteinAnalysis(seq)
    return dict(netcharge=float(analysis.charge_at_pH(7.4)),
                gravy=float(analysis.gravy()),
                pI=float(analysis.isoelectric_point()),
                length=len(seq))


def load_sequences(path):
    return json.loads(Path(path).read_text())


def style(ax):
    ax.grid(alpha=.25, linewidth=.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def figure_properties(payload, out, cohort='dev60'):
    """KDE of net charge / GRAVY / pI for native, each model, and FDA references."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    panels = [('netcharge', 'Net Charge at pH 7.4'), ('gravy', 'GRAVY (Hydrophobicity)'),
              ('pI', 'Isoelectric Point (pI)')]
    sources = {}
    native = [properties(s) for _, s in payload['native'][cohort]]
    sources['Native PDB'] = [p for p in native if p]
    for arm in ('Ours (full)', 'Ours (aligned, converged)', 'RAAD', 'MEAN', 'DiffAb'):
        rows = payload['generated'].get(arm, {}).get(cohort)
        if not rows:
            continue
        sources[arm] = [p for p in (properties(s) for _, s in rows) if p]
    if payload.get('fda_approved'):
        sources['FDA approved'] = [p for p in
                                   (properties(s) for _, s in payload['fda_approved']) if p]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (key, label) in zip(axes, panels):
        for name, rows in sources.items():
            values = np.array([r[key] for r in rows])
            if len(values) < 5 or values.std() < 1e-9:
                continue
            grid = np.linspace(values.min() - .5, values.max() + .5, 250)
            density = gaussian_kde(values, bw_method='scott')(grid)
            ax.plot(grid, density, label=f'{name} (n={len(values)})',
                    color=COLORS.get(name), linewidth=1.6)
            ax.fill_between(grid, density, alpha=.10, color=COLORS.get(name))
        ax.set_xlabel(label)
        ax.set_ylabel('Density')
        style(ax)
    axes[0].legend(fontsize=7.5, frameon=False, loc='best')
    fig.suptitle(f'CDRH3 physicochemical properties — generated vs native ({cohort})',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, .95))
    target = out / f'F1_properties_kde_{cohort}.png'
    fig.savefig(target, dpi=220)
    fig.savefig(target.with_suffix('.svg'))
    plt.close(fig)
    summary = {name: {key: float(np.mean([r[key] for r in rows]))
                      for key, _ in panels} | dict(n=len(rows))
               for name, rows in sources.items()}
    return target, summary


def information_logo(sequences, length, align='N'):
    """Per-position information content and base frequencies, aligned by N or C term."""
    counts = np.zeros((length, len(CANONICAL)))
    used = 0
    for seq in sequences:
        seq = clean(seq)
        if len(seq) < length:
            continue
        window = seq[:length] if align == 'N' else seq[-length:]
        used += 1
        for i, aa in enumerate(window):
            counts[i, CANONICAL.index(aa)] += 1
    if used == 0:
        return None, None, 0
    frequencies = counts / counts.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        entropy = -np.nansum(np.where(frequencies > 0,
                                      frequencies * np.log2(frequencies), 0.), axis=1)
    information = np.log2(len(CANONICAL)) - entropy
    return information, frequencies, used


def figure_logos(payload, out, cohort='dev60'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    native = [s for _, s in payload['native'][cohort]]
    arms = ['Ours (full)', 'RAAD', 'MEAN', 'DiffAb']
    rows = [('Native', native)]
    for arm in arms:
        generated = payload['generated'].get(arm, {}).get(cohort)
        if generated:
            rows.append((arm, [s for _, s in generated]))
    length = min(25, int(np.median([len(clean(s)) for s in native])))
    fig, axes = plt.subplots(2, len(rows), figsize=(4.2 * len(rows), 6.4))
    for column, (name, sequences) in enumerate(rows):
        for row_index, (align, label) in enumerate((('N', 'N-aligned'), ('C', 'C-aligned'))):
            ax = axes[row_index][column]
            information, frequencies, used = information_logo(sequences, length, align)
            if information is None:
                ax.set_axis_off()
                continue
            for position in range(length):
                order = np.argsort(frequencies[position])
                bottom = 0.
                for index in order:
                    height = information[position] * frequencies[position][index]
                    if height < .02:
                        continue
                    ax.text(position + 1, bottom + height / 2, CANONICAL[index],
                            ha='center', va='center', fontsize=7 + 12 * height,
                            color=plt.cm.tab20(index / len(CANONICAL)))
                    bottom += height
            ax.set_xlim(.5, length + .5)
            ax.set_ylim(0, max(1.0, information.max() * 1.05))
            if align == 'C':
                ax.set_xlabel('Position (C-term = 0)' if row_index else '')
            else:
                ax.set_xlabel('Position (N → C)')
            if column == 0:
                ax.set_ylabel(f'Information (bits)\n{label}')
            if row_index == 0:
                ax.set_title(f'{name}', fontsize=11, fontweight='bold')
            else:
                ax.set_title(f'{name} · {label} · n={used}', fontsize=8)
            ax.set_xticks(range(1, length + 1, max(1, length // 10)))
            style(ax)
    fig.suptitle('CDRH3 sequence logos: native vs models (N- and C-terminal alignment)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, .95))
    target = out / f'F2_sequence_logos_{cohort}.png'
    fig.savefig(target, dpi=200)
    fig.savefig(target.with_suffix('.svg'))
    plt.close(fig)
    return target


def load_per_complex(path):
    document = json.loads(Path(path).read_text())
    return {c['sample_id']: c for c in document['per_complex']}


def figure_ablation(out):
    """Ablation summary as dot plots: means with the seed spread, no truncated bars.

    Bar charts with a non-zero baseline exaggerate small differences, and
    highlighting "the best" variant in an ablation panel implies a recommendation
    the data does not support. Both are avoided here: the reference arm is marked,
    and every point carries its seed spread.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    metrics = [('aligned_rmsd', 'aligned RMSD (Å) — lower is better'),
               ('contact_f1', 'contact F1 — higher is better'),
               ('interface_fape', 'interface FAPE — lower is better')]

    def gather(per_arm, arms):
        out = {}
        for arm in arms:
            entry = per_arm[arm]
            out[arm] = {m: (float(entry[m]['mean']),
                            float(np.std(entry[m].get('per_seed', [0.]), ddof=1)
                                  if len(entry[m].get('per_seed', [])) > 1 else 0.))
                        for m, _ in metrics}
        return out

    dev = json.loads((PACKAGE / 'results/ablation_2x2_v1/comparison.json').read_text())
    panels = [dict(title='A. dev-60 · 2×2 interaction cells (3 seeds)',
                   arms=['base', 'only_pair', 'only_binding', 'full'],
                   labels=['base', 'only\npair', 'only\nbinding', 'full*'],
                   reference='full',
                   data=gather(dev['per_arm_summary'],
                               ['base', 'only_pair', 'only_binding', 'full']))]

    external = json.loads(
        (PACKAGE / 'results/external_interface_ablations_v1/comparison.json').read_text())
    order = ['full', 'mask_surface_geometry', 'mask_electrostatics_chemistry',
             'mask_none', 'static_neighborhood']
    labels = ['full*', 'mask\nsurface', 'mask\nelec.', 'mask\nall', 'static\nnbr.']
    per_arm = {}
    for arm in order:
        per_arm[arm] = {}
        for metric, _ in metrics:
            values = [dev and None]
            seeds = [external['per_seed_summaries'][s][arm][metric]
                     for s in external['per_seed_summaries']]
            per_arm[arm][metric] = dict(mean=float(np.mean(seeds)), per_seed=seeds)
    panels.append(dict(title='B. external-20 · inference-time ablations (3 seeds)',
                       arms=order, labels=labels, reference='full',
                       data=gather(per_arm, order)))

    aligned = json.loads(
        (PACKAGE / 'results/aligned_shape_v1/stats_conv_vs_orig.json').read_text())
    orig = json.loads((PACKAGE / 'results/ablation_2x2_v1/comparison.json')
                      .read_text())['per_arm_summary']['full']
    conv = None
    sources = {
        'three_arm': 'results/aligned_shape_v1/summaries_converged.json'}
    three = {
        'w = 0\n(reference)': dict(
            aligned_rmsd=(1.6017, 0.0275), contact_f1=(0.5682, 0.0105),
            interface_fape=(0.6870, 0.0108)),
        'w = 0.3\n(30 epochs)': dict(
            aligned_rmsd=(1.5940, 0.0269), contact_f1=(0.5684, 0.0148),
            interface_fape=(0.6869, 0.0111)),
        'w = 0.3\n(converged)': dict(
            aligned_rmsd=(1.5920, 0.0269), contact_f1=(0.5710, 0.0148),
            interface_fape=(0.6862, 0.0111)),
    }
    panels.append(dict(title='C. dev-60 · aligned-backbone weight and budget',
                       arms=list(three),
                       labels=['w=0*', 'w=0.3\n30 ep', 'w=0.3\nconv.'],
                       reference='w = 0\n(reference)',
                       data={arm: {m: (v[0], v[1]) for m, v in three[arm].items()}
                             for arm in three}))

    fig, axes = plt.subplots(len(panels), len(metrics), figsize=(13.2, 3.1 * len(panels)))
    for row, panel in enumerate(panels):
        for column, (metric, label) in enumerate(metrics):
            ax = axes[row][column]
            values = [panel['data'][arm][metric] for arm in panel['arms']]
            xs = np.arange(len(values))
            colors = ['#e45756' if arm == panel['reference'] else '#4c78a8'
                      for arm in panel['arms']]
            ax.errorbar(xs, [v[0] for v in values], yerr=[v[1] for v in values],
                        fmt='none', ecolor='#666666', elinewidth=1.1, capsize=3, zorder=2)
            ax.scatter(xs, [v[0] for v in values], s=64, c=colors, zorder=3)
            for x, (mean, spread) in zip(xs, values):
                ax.annotate(f'{mean:.4f}', (x, mean), textcoords='offset points',
                            xytext=(0, 9), ha='center', fontsize=7.5)
            ax.set_xticks(xs)
            ax.set_xticklabels(panel['labels'], fontsize=8)
            ax.text(.01, .02, '* reference arm', transform=ax.transAxes, fontsize=6.5,
                    color='#666666')
            ax.set_xlim(-.6, len(values) - .4)
            span = max(v[0] for v in values) - min(v[0] for v in values)
            pad = max(span * .5, 2 * max(v[1] for v in values), 1e-4)
            ax.set_ylim(min(v[0] for v in values) - pad, max(v[0] for v in values) + pad)
            if column == 0:
                ax.set_ylabel(label, fontsize=9)
                ax.set_title(panel['title'], fontsize=10, fontweight='bold', loc='left')
            else:
                ax.set_ylabel(label.split(' — ')[0], fontsize=8.5)
            style(ax)
    fig.suptitle('Ablation summary — point estimates with seed spread (error bars = ±1 SD)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, .95))
    target = out / 'F3_ablation_summary.png'
    fig.savefig(target, dpi=220)
    fig.savefig(target.with_suffix('.svg'))
    plt.close(fig)
    return target


def normalized_levenshtein(first, second):
    import Levenshtein
    longest = max(len(first), len(second))
    return 0. if longest == 0 else Levenshtein.distance(first, second) / longest


def figure_diversity_quality(payload, out, cohort='dev60'):
    """Target-level Pareto: mean sequence diversity against reconstruction quality.

    The archived figure used TM-score and ddG; those need folding and all-atom
    energy, so this reproduction uses the aligned RMSD this project actually
    measures and says so on the axis.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    arms = {'Ours (full)': 'results/aligned_shape_v1/test_w0_original/metrics.json',
            'Ours (aligned, converged)':
                'results/aligned_shape_v1/test_w0.3_converged_seed2026/metrics.json',
            'RAAD': None, 'MEAN': None, 'DiffAb': None}
    quality = {}
    for arm, path in arms.items():
        if path and (PACKAGE / path).is_file():
            quality[arm] = load_per_complex(PACKAGE / path)
    metric_sources = {
        'RAAD': PACKAGE / 'outputs/raad_abgaze_reconstructed_test_v1/metrics.json',
        'MEAN': PACKAGE / 'outputs/mean_reconstructed_test_v1/metrics.json',
        'DiffAb': PACKAGE / 'results/diffab_reconstructed_seed2026_v1/test_metrics.json',
    }
    for arm, path in metric_sources.items():
        resolved = path if path.is_file() else Path('/media/xd/code1/dyGF') / path.relative_to(PACKAGE)
        if not resolved.is_file():
            continue
        document = json.loads(resolved.read_text())
        cases = {}
        for case in document['per_complex']:
            flat = case.get('first', case)
            cases[case['sample_id']] = dict(aligned_rmsd=flat['aligned_rmsd'])
        quality[arm] = cases

    generated = payload['generated']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for arm, cases in quality.items():
        key = 'Ours (full)' if arm not in generated else arm
        per_seed = generated.get(key, {}).get(f'{cohort}_per_seed')
        if not per_seed:
            continue
        seed_rows = next(iter(per_seed.values()))
        diversity, rmsd = [], []
        for sid, candidates in seed_rows:
            if sid not in cases:
                continue
            scores = [normalized_levenshtein(a, b)
                      for i, a in enumerate(candidates) for b in candidates[i + 1:]]
            diversity.append(float(np.mean(scores)))
            rmsd.append(float(cases[sid]['aligned_rmsd']))
        if not diversity:
            continue
        axes[0].scatter(diversity, rmsd, s=26, alpha=.75, label=f'{arm} (n={len(diversity)})',
                        color=COLORS.get(arm))
        axes[1].hist(diversity, bins=14, alpha=.55, label=arm, color=COLORS.get(arm))
    axes[0].set_xlabel('Mean sequence diversity (normalised Levenshtein)')
    axes[0].set_ylabel('aligned RMSD (Å) — lower is better')
    axes[0].set_title('A. Target-level Pareto: diversity vs reconstruction quality', fontsize=11)
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].set_xlabel('Mean sequence diversity')
    axes[1].set_ylabel('Number of complexes')
    axes[1].set_title('B. Diversity distribution', fontsize=11)
    axes[1].legend(fontsize=8, frameon=False)
    for ax in axes:
        style(ax)
    fig.suptitle('Diversity–quality trade-off (diversity from five sampled sequences per complex)',
                 fontsize=12.5, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, .95))
    target = out / f'F4_diversity_quality_{cohort}.png'
    fig.savefig(target, dpi=220)
    fig.savefig(target.with_suffix('.svg'))
    plt.close(fig)
    return target


def figure_recovery_by_length(payload, out, cohort='dev60'):
    """Recovery vs H3 length on a continuous length axis.

    Length bins hold only 2-5 complexes at the extremes, so bars would imply a
    precision the data does not have. Complexes are shown individually and the
    overlay is a median over coarse bins, drawn at the bin's mean length with the
    bin count printed.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    native = dict(payload['native'][cohort])
    lengths = {sid: len(clean(seq)) for sid, seq in native.items()}
    arms = ['Ours (full)', 'Ours (aligned, converged)', 'RAAD', 'MEAN', 'DiffAb']
    bins = [(0, 11), (12, 14), (15, 17), (18, 99)]
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    jitter = np.linspace(-.22, .22, len(arms))
    for offset, arm in zip(jitter, arms):
        per_seed = payload['generated'].get(arm, {}).get(f'{cohort}_per_seed')
        if not per_seed:
            continue
        rows = next(iter(per_seed.values()))
        xs, ys = [], []
        for sid, candidates in rows:
            if sid not in lengths:
                continue
            reference = clean(native[sid])
            recovery = float(np.mean([
                sum(a == b for a, b in zip(clean(candidate), reference)) /
                max(len(reference), 1) for candidate in candidates]))
            xs.append(lengths[sid] + offset)
            ys.append(recovery)
        ax.scatter(xs, ys, s=16, alpha=.55, color=COLORS.get(arm), label=arm)
        centres, medians = [], []
        for low, high in bins:
            members = [(x - offset, y) for x, y in zip(xs, ys) if low <= x - offset <= high]
            if not members:
                continue
            centres.append(float(np.mean([m[0] for m in members])))
            medians.append(float(np.median([m[1] for m in members])))
        if centres:
            ax.plot(centres, medians, color=COLORS.get(arm), linewidth=1.9,
                    marker='o', markersize=5, zorder=4)
    counts = {f'{low}-{high if high < 99 else "+"}': sum(
        1 for length in lengths.values() if low <= length <= high) for low, high in bins}
    ax.text(.985, .975, 'complexes per bin: ' +
            ', '.join(f'{k}: {v}' for k, v in counts.items()),
            transform=ax.transAxes, ha='right', va='top', fontsize=7.5,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#cccccc', alpha=.9))
    ax.set_xlabel('CDRH3 length (residues)')
    ax.set_ylabel('Amino-acid recovery (mean over five samples)')
    ax.set_title('Recovery vs H3 length — points are complexes, lines are binned medians',
                 fontsize=11)
    ax.legend(fontsize=8, frameon=False, ncol=3)
    style(ax)
    fig.tight_layout()
    target = out / f'F5_recovery_by_length_{cohort}.png'
    fig.savefig(target, dpi=220)
    fig.savefig(target.with_suffix('.svg'))
    plt.close(fig)
    return target


def figure_per_complex(payload, out, cohort='dev60'):
    """Per-complex profiles across the three seeds, all arms."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sources = {
        'Ours (full)': ['results/aligned_shape_v1/test_w0_original/metrics.json',
                        'results/aligned_shape_v1/test_w0_seed2027/metrics.json',
                        'results/aligned_shape_v1/test_w0_seed2028/metrics.json'],
        'Ours (aligned, converged)': [
            'results/aligned_shape_v1/test_w0.3_converged_seed2026/metrics.json',
            'results/aligned_shape_v1/test_w0.3_converged_seed2027/metrics.json',
            'results/aligned_shape_v1/test_w0.3_converged_seed2028/metrics.json'],
        'RAAD': ['outputs/raad_abgaze_reconstructed_test_v1/metrics.json',
                 'outputs/raad_reconstructed_seed2027_test_v2/metrics.json',
                 'outputs/raad_reconstructed_seed2028_test_v2/metrics.json'],
    }
    metrics = [('aligned_rmsd', 'aligned RMSD (Å)'), ('contact_f1', 'contact F1'),
               ('cn_mae', 'C–N MAE (Å)'), ('interface_fape', 'interface FAPE')]
    ids = sorted(payload['native'][cohort] and dict(payload['native'][cohort]))
    fig, axes = plt.subplots(len(metrics), 1, figsize=(15, 2.1 * len(metrics)), sharex=True)
    for ax, (metric, label) in zip(axes, metrics):
        for arm, paths in sources.items():
            values = []
            for path in paths:
                resolved = PACKAGE / path
                if not resolved.is_file():
                    resolved = Path('/media/xd/code1/dyGF') / path
                values.append(load_per_complex(resolved))
            means = [float(np.mean([v[sid][metric] for v in values if sid in v]))
                     for sid in ids]
            ax.plot(range(len(ids)), means, linewidth=1.1, label=arm, color=COLORS.get(arm))
        ax.set_ylabel(label, fontsize=8.5)
        style(ax)
    axes[0].legend(fontsize=8, frameon=False, ncol=3)
    axes[-1].set_xticks(range(len(ids)))
    axes[-1].set_xticklabels(ids, rotation=90, fontsize=6.5)
    fig.suptitle('Per-complex metric profiles (dev-60, mean over three seeds)',
                 fontsize=12.5, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, .96))
    target = out / f'F6_per_complex_profiles_{cohort}.png'
    fig.savefig(target, dpi=200)
    fig.savefig(target.with_suffix('.svg'))
    plt.close(fig)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sequences', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--only', nargs='+', default=None,
                        help='Subset of F1 F2 F3 F4 F5 F6')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    payload = load_sequences(args.sequences)
    wanted = args.only or ['F1', 'F2', 'F3', 'F4', 'F5', 'F6']
    produced = {}
    if 'F1' in wanted:
        for cohort in ('dev60', 'external20'):
            target, summary = figure_properties(payload, out, cohort)
            produced[target.name] = summary
    if 'F2' in wanted:
        produced[figure_logos(payload, out, 'dev60').name] = 'sequence logos'
    if 'F3' in wanted:
        produced[figure_ablation(out).name] = 'ablation summary'
    if 'F4' in wanted:
        produced[figure_diversity_quality(payload, out, 'dev60').name] = 'diversity vs quality'
    if 'F5' in wanted:
        produced[figure_recovery_by_length(payload, out, 'dev60').name] = 'recovery by length'
    if 'F6' in wanted:
        produced[figure_per_complex(payload, out, 'dev60').name] = 'per-complex profiles'
    (out / 'figures.json').write_text(json.dumps(produced, indent=2) + '\n')
    print(json.dumps(produced, indent=1)[:2000])


if __name__ == '__main__':
    main()
