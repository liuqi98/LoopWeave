"""Compute the Table 1 metric set for the GeoAB designs.

Table 1 reports aligned and raw H3 C-alpha RMSD, backbone-restricted DockQ, contact F1, all-bond
C-N MAE and normalized interface FAPE. DockQ comes from ``dockq_all_methods_multiseed_v1``; the
remaining five are computed here with the same objective function as the matched arms.
"""
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

PACKAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE / 'code'))
sys.path.insert(0, '/media/xd/code1/dyGF')
from abgaze_reconstructed_objective import objective
from eval.test_loop_pair_replicates_v1 import aligned_rmsd

WORKSPACE = Path('/media/xd/code1/dyGF')


def main():
    rows = torch.load(WORKSPACE / 'data/abgaze_reconstructed_v1/tensors/test.pt',
                      map_location='cpu', weights_only=False)
    out = PACKAGE / 'results/geoab_design_v1'
    per_run = {}
    for run in (1, 2, 3):
        document = {r['sample_id']: r for r in
                    json.loads((out / f'generated_run{run}.json').read_text())}
        values = {k: [] for k in ('aligned_rmsd', 'rmsd', 'contact_f1', 'cn_mae',
                                  'interface_fape', 'clash')}
        for sid, x, y in rows:
            backbone = torch.tensor(document[sid]['backbone'])
            sequence = document[sid]['greedy']
            onehot = torch.nn.functional.one_hot(
                torch.tensor(['ACDEFGHIKLMNPQRSTVWY'.index(a) for a in sequence]), 20
            ).float() * 10
            prediction = dict(backbone=backbone, sequence_logits=onehot)
            _, metrics = objective(prediction, x, y)
            valid = y['mask'][:, 1]
            values['aligned_rmsd'].append(float(aligned_rmsd(backbone[valid, 1], y['bb'][valid, 1])))
            for key in ('rmsd', 'contact_f1', 'cn_mae', 'interface_fape', 'clash'):
                values[key].append(metrics[key])
        per_run[run] = {k: statistics.mean(v) for k, v in values.items()}
    summary = {}
    for key in ('aligned_rmsd', 'rmsd', 'contact_f1', 'cn_mae', 'interface_fape', 'clash'):
        run_means = [per_run[r][key] for r in (1, 2, 3)]
        summary[key] = dict(mean=statistics.mean(run_means),
                            sd=statistics.stdev(run_means), per_run=run_means)
    (out / 'table1_metrics.json').write_text(json.dumps(dict(summary=summary, per_run=per_run),
                                                        indent=2) + '\n')
    for key, block in summary.items():
        print('%-16s %.4f +- %.4f' % (key, block['mean'], block['sd']))


if __name__ == '__main__':
    main()
