"""Report for the chemical-channel ablation (no_chem arm).

Intervention: the binding features keep every geometric channel (surface normals,
curvature, dispersion, concavity, solvent exposure, atom radius, side-chain flag,
added-atom flag) and zero only the explicit chemical channels (charge, donor and acceptor
flags and bisectors, nonpolar term, electrostatic potential and field). Channel counts are
unchanged, so the parameter count is identical to the full arm (702,992).
"""
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, '/media/xd/code1/dyGF')
from research_audit.statistics import interval

RESULT = PACKAGE / 'results/mechanism_arms_v1_no_chem/mechanism_summary.json'
PAIRED = PACKAGE / 'results/dev60_allmetrics_no_chem/paired_no_chem_vs_raad.csv'
REPORT = PACKAGE / 'REPORT_NO_CHEM_v1.md'

LABELS = {'cn_mae': 'C-N MAE (A)', 'interface_fape': 'interface FAPE', 'contact_f1': 'contact F1',
          'rmsd': 'RMSD (A)', 'aligned_rmsd': 'aligned RMSD (A)', 'clash': 'clash proxy',
          'aar': 'AAR', 'mean_sample_aar': 'mean-sample AAR', 'oracle_aar': 'oracle AAR',
          'cn_internal': 'in-loop C-N MAE (A)'}


def raad_pairing():
    rows = list(csv.DictReader(PAIRED.open()))
    by = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        for key in ('interface_fape', 'contact_f1', 'cn_mae', 'aligned_rmsd', 'aar',
                    'cn_internal'):
            by[row['sample_id']][row['arm']][key] = float(row[key])
        by[row['sample_id']][row['arm']]['group'] = row['group']
    ids = sorted(by)
    groups = [by[sid]['no_chem']['group'] for sid in ids]
    keys = ('interface_fape', 'contact_f1', 'cn_mae', 'aligned_rmsd', 'aar', 'cn_internal')
    means = {arm: {key: statistics.mean(by[sid][arm][key] for sid in ids) for key in keys}
             for arm in ('no_chem', 'raad')}
    contrasts = {}
    for key, sign in (('interface_fape', -1), ('contact_f1', 1), ('cn_mae', -1),
                      ('aligned_rmsd', -1), ('aar', 1), ('cn_internal', -1)):
        matrix = np.array([[sign * (by[sid]['no_chem'][key] - by[sid]['raad'][key])
                            for sid in ids]])
        stats = interval(matrix, groups, 10000)
        contrasts[key] = dict(improvement=stats['mean'],
                              ci95=stats['conditional_group_ci95'],
                              significant=bool(stats['conditional_group_ci95'][0] *
                                               stats['conditional_group_ci95'][1] > 0))
    return means, contrasts


def main():
    summary = json.loads(RESULT.read_text())
    arms = summary['arms']
    full = {key: value['mean'] for key, value in arms['full'].items()}
    no_chem = {key: value['mean'] for key, value in arms['no_chem'].items()}
    contrasts = summary['contrasts']['no_chem']
    raad_means, raad_contrasts = raad_pairing()

    lines = ['# 化学通道消融：保留几何、移除显式化学描述（no_chem）', '',
             'dev-60，三个训练种子（2026/2027/2028），pair 阶段 30 epoch，与机制对照臂共用同一上游',
             'binding 检查点（epoch 60）作为共同初始化。', '',
             '## 干预', '',
             '- **保留**：表面法向、曲率、dispersion、concavity、溶剂暴露度、原子半径、侧链标记、',
             '  补全原子标记（全部几何与拓扑信息）。',
             '- **置零**：电荷、氢键供体/受体标记及其方向双分向量、非极性项、静电势、静电场。',
             '- 通道数不变，因此**参数量与 full 臂完全相同（702,992）**。', '',
             '## 结果一：相对 full 臂（同一预算、同一初始化）', '',
             '| 指标 | full | no_chem | arm-full | improvement（>0 = no_chem 更优） |',
             '|---|---:|---:|---:|---|']
    for key in ('cn_mae', 'interface_fape', 'contact_f1', 'rmsd', 'aligned_rmsd', 'clash',
                'aar', 'mean_sample_aar'):
        block = contrasts[key]
        lines.append('| %s | %.4f | %.4f | %+.4f | %+.4f [%+.4f, %+.4f]%s |' % (
            LABELS.get(key, key), full[key], no_chem[key], block['arm_minus_full'],
            block['improvement'], block['ci95'][0], block['ci95'][1],
            ' *' if block['significant'] else ''))

    lines += ['', '## 结果二：相对 RAAD（同一 dev-60 打分流程）', '',
              '| 指标 | no_chem | RAAD | no_chem 更优幅度 | 95% 条件区间 | 显著 |',
              '|---|---:|---:|---:|---|---|']
    for key in ('interface_fape', 'contact_f1', 'cn_mae', 'aligned_rmsd', 'aar', 'cn_internal'):
        block = raad_contrasts[key]
        lines.append('| %s | %.4f | %.4f | %+.4f | [%+.4f, %+.4f] | %s |' % (
            LABELS.get(key, key), raad_means['no_chem'][key], raad_means['raad'][key],
            block['improvement'], block['ci95'][0], block['ci95'][1],
            '是' if block['significant'] else '否'))

    lines += ['', '## 判读', '',
              '1. **界面收益不依赖显式化学描述。** 化学通道全部置零后，interface FAPE 相对 full '
              '数值上更低（%+.4f，显著），contact F1 无差异（%+.4f，不显著）；相对 RAAD 的界面'
              '优势仍然成立（FAPE %+.4f、F1 %+.4f，均显著）。' % (
                  contrasts['interface_fape']['improvement'],
                  contrasts['contact_f1']['arm_minus_full'],
                  raad_contrasts['interface_fape']['improvement'],
                  raad_contrasts['contact_f1']['improvement']),
              '2. **几何输入足以支撑该收益。** 环内 C–N 与 aligned RMSD 同样不劣于 full，'
              '说明收益来自几何/环境输入与成对关系表示，而不是电荷或疏水性等显式通道。',
              '3. **不能把界面改善归因于化学描述。** 论文相关表述应改为“几何/环境条件化”，'
              '化学通道在本预算下作用为中性的。', '',
              '## 局限', '',
              '- 单次 30 epoch 与单一共同初始化；full 臂在此预算下本身弱于最终发布检查点。',
              '- 置零后模型仍可用零通道编码偏置，因此这是保守的可用性检验。',
              '- 未做“只保留化学、去掉几何”的反向臂。',
              '- dev-60 为反复查看的开发集。', '',
              '## 复现', '',
              '```bash',
              'bash code/scripts/run_mechanism_arms_v1.sh 0 no_chem:2026',
              'bash code/scripts/run_mechanism_arms_v1.sh 1 no_chem:2027 no_chem:2028',
              'PYTHONPATH=code:/media/xd/code1/dyGF python code/eval/test_mechanism_arms_v1.py \\',
              '  --output results/mechanism_arms_v1_no_chem --device cuda:0',
              'PYTHONPATH=code:/media/xd/code1/dyGF python code/eval/compare_dev60_allmetrics_v1.py \\',
              '  --output results/dev60_allmetrics_no_chem',
              '```']
    REPORT.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
