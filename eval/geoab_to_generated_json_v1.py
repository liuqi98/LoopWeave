"""Convert the scored GeoAB designs into the arm format used by the comparison pipeline."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

PACKAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE / 'code'))
from eval.three_method_diagnostics_v1 import read_saved_structure

WORKSPACE = Path('/media/xd/code1/dyGF')
STRUCT = PACKAGE / 'results/geoab_design_v1/structures'


def main():
    rows = torch.load(WORKSPACE / 'data/abgaze_reconstructed_v1/tensors/test.pt',
                      map_location='cpu', weights_only=False)
    ids = [row[0] for row in rows]
    scores = json.loads((PACKAGE / 'results/geoab_design_v1/geoab_scores.json').read_text())
    sequences = {}
    for record in scores['per_complex']:
        sequences.setdefault(record['sample_id'], {})[record['run']] = record['designed_sequence']
    out = PACKAGE / 'results/geoab_design_v1'
    for run in (0, 1, 2):
        records = []
        for sid in ids:
            # The converter writes one file per run; run 0 uses the unsuffixed copy.
            candidate = STRUCT / (f'{sid}_geoab.pdb' if run == 0 else f'{sid}_geoab_run{run}.pdb')
            if not candidate.is_file():
                raise FileNotFoundError(candidate)
            parsed = read_saved_structure(candidate)
            records.append(dict(sample_id=sid, backbone=parsed['loop_bb'].numpy().tolist(),
                                greedy=sequences[sid][run], sequence=sequences[sid][run]))
        (out / f'generated_run{run + 1}.json').write_text(json.dumps(records) + '\n')
        print('wrote', out / f'generated_run{run + 1}.json', len(records))


if __name__ == '__main__':
    main()
