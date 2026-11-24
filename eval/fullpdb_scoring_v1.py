"""Score arm predictions directly against the original RAbD60 PDB structures.

The reconstructed benchmark tensors hold the *same* H3 coordinates as the source PDBs (verified:
zero missing H3 atoms and zero coordinate mismatch across all 60 targets), so C-alpha RMSD is
already computed on the original structures. What the reconstruction *does* change is the antigen
context: the model sees only the nearest retained antigen residues, whereas the deposited complex
contains the full antigen. This script therefore re-scores every arm against the complete
crystallographic complex:

    * raw and aligned H3 C-alpha RMSD against the PDB H3 (identical by construction, a control),
    * hard 8 A C-alpha contact F1 against ALL antigen residues in the PDB,
    * the same contact F1 against only the retained antigen subset (the current definition).

Nothing is re-inferred; the delivered backbones are read from the saved JSON files.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

PACKAGE = Path(__file__).resolve().parents[2]
WORKSPACE = Path('/media/xd/code1/dyGF')
sys.path.insert(0, '/media/xd/code1/dyGF')

ARMS = {
    'baseline': WORKSPACE / 'outputs/layer4_baseline_seed2026_backbones.json',
    'init_state': WORKSPACE / 'outputs/layer4_initstate_seed2026_backbones.json',
    'anchor': WORKSPACE / 'outputs/layer4_anchor_seed2026_backbones.json',
    'two_scale': WORKSPACE / 'outputs/layer4_twoscale_seed2026_backbones.json',
}


def kabsch_rmsd(prediction, reference):
    prediction, reference = prediction.double(), reference.double()
    p_centre, r_centre = prediction.mean(0), reference.mean(0)
    u, _, vt = torch.linalg.svd((prediction - p_centre).T @ (reference - r_centre))
    d = torch.sign(torch.linalg.det(vt.T @ u.T))
    rotation = vt.T @ torch.diag(torch.tensor([1., 1., d], dtype=torch.double)) @ u.T
    moved = (prediction - p_centre) @ rotation.T + r_centre
    return float((moved - reference).square().sum(-1).mean().sqrt())


def contact_f1(prediction_ca, antigen_ca, native_ca, threshold=8.):
    predicted = torch.cdist(prediction_ca, antigen_ca) <= threshold
    native = torch.cdist(native_ca, antigen_ca) <= threshold
    return float(2 * (predicted & native).sum() /
                 (predicted.sum() + native.sum()).clamp_min(1))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    from Bio.PDB import PDBParser
    import warnings
    warnings.filterwarnings('ignore')
    dataset = WORKSPACE / 'data/abgaze_reconstructed_v1'
    records = [json.loads(line) for line in
               (dataset / 'splits/test.json').read_text().splitlines() if line]
    meta = {row['pdb']: row for row in records}
    rows = torch.load(dataset / 'tensors/test.pt', map_location='cpu', weights_only=False)
    ids = [row[0] for row in rows]

    native = {}
    for sid, x, y in rows:
        info = meta[sid]
        structure = PDBParser(QUIET=True).get_structure('p', info['pdb_data_path'])[0]
        heavy = [r for r in structure[info['heavy_chain']] if r.id[0] == ' ']
        start, end = info['cdrh3_pos']
        h3 = heavy[start:end + 1]
        h3_ca = np.array([r['CA'].coord for r in h3], dtype=np.float32)
        full_antigen = []
        for chain_id in info['antigen_chains']:
            for residue in structure[chain_id]:
                if residue.id[0] == ' ' and 'CA' in residue:
                    full_antigen.append(residue['CA'].coord)
        native[sid] = dict(h3_ca=torch.tensor(h3_ca),
                           full_antigen=torch.tensor(np.asarray(full_antigen, dtype=np.float32)),
                           retained_antigen=x['context_bb'][x['antigen_mask']][:, 1])

    summary = {}
    per_arm = {}
    for name, path in ARMS.items():
        if not path.is_file():
            print('skip (missing):', name)
            continue
        document = json.loads(path.read_text())['2026']
        by_id = {row['sample_id']: row for row in document}
        values = {k: [] for k in ('raw_rmsd', 'aligned_rmsd', 'f1_full', 'f1_retained',
                                  'n_full_antigen', 'n_retained_antigen')}
        for sid in ids:
            backbone = torch.tensor(by_id[sid]['backbone'])
            info = native[sid]
            prediction = backbone[:, 1]
            values['raw_rmsd'].append(float((prediction - info['h3_ca']).square()
                                            .sum(-1).mean().sqrt()))
            values['aligned_rmsd'].append(kabsch_rmsd(prediction, info['h3_ca']))
            values['f1_full'].append(contact_f1(prediction, info['full_antigen'], info['h3_ca']))
            values['f1_retained'].append(contact_f1(prediction, info['retained_antigen'],
                                                   info['h3_ca']))
            values['n_full_antigen'].append(len(info['full_antigen']))
            values['n_retained_antigen'].append(len(info['retained_antigen']))
        per_arm[name] = {k: v for k, v in values.items()}
        summary[name] = {k: float(np.mean(v)) for k, v in values.items()}
    (out / 'fullpdb_scores.json').write_text(json.dumps(dict(summary=summary,
                                                             per_arm=per_arm), indent=2) + '\n')
    print('%-10s %9s %9s %9s %9s' % ('arm', 'raw', 'aligned', 'F1 full', 'F1 retained'))
    for name, block in summary.items():
        print('%-10s %9.4f %9.4f %9.4f %9.4f' % (name, block['raw_rmsd'], block['aligned_rmsd'],
                                                 block['f1_full'], block['f1_retained']))
    print()
    print('抗原残基数：完整 %.1f / 保留 %.1f' % (
        statistics.mean(summary[list(summary)[0] ]['n_full_antigen'] for _ in [0]),
        statistics.mean(summary[list(summary)[0]]['n_retained_antigen'] for _ in [0])))


if __name__ == '__main__':
    main()
