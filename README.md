# LoopWeave

LoopWeave is a frame-relation, two-scale refinement framework for antibody CDR–H3
sequence–structure co-design. It refines a backbone candidate under a fixed
framework/antigen pose and jointly samples the H3 sequence.

## Architecture

```
framework + antigen context (fixed pose)
        │
        ▼
FrameRelationRefiner (frozen generator, 913k params)
  - N/CA/C backbone frames as graph nodes
  - edges carry invariant local-frame directions, relative rotations,
    sequence separation, spatial kNN and interface relations
  - relation graph is rebuilt each round from candidate coordinates
        │  base backbone
        ▼
OrientationInterfaceRefiner        (stage 2, 30 epochs)
        │
        ▼
BindingInterfaceRefiner            (stage 3, 60 epochs, surface features)
        │
        ▼
TwoScalePairRefiner                (stage 4, 30 epochs)
  Δx = Δx_coordinated + Δx_local
  - coordinated: 3 cosine modes over the loop index, bounded (±0.8 Å)
  - local:       per-atom displacement, bounded (±0.2 Å)
  - zero-initialised heads, bounded branches are independently ablatable
```

Outputs per residue: 4 backbone atoms (N, CA, C, O) and 20-way amino-acid logits.

## Repository layout

```
frame_relation_refiner.py       frame-relation generator
orientation_interface_refiner.py
binding_interface_refiner.py    surface/electrostatics feature fusion
persistent_loop_pair_refiner.py loop-pair refiner backbone
two_scale_pair_refiner.py       two-scale coordinate update (core mechanism)
aligned_shape_objective.py      alignment-invariant shape objective
abgaze_reconstructed_objective.py training objective (geometry + FAPE + CE)
dygf.py                         backbone-frame utilities
scripts/                        training / sampling / audit entry points
eval/                           metric implementations and comparison scripts
design/                         design-evaluation utilities (ddG, side chains)
baselines/                      baseline adapters (RAAD/MEAN/DiffAb)
tests/                          unit tests for geometry and metrics
```

## Installation

```bash
pip install torch numpy biopython scipy easydict anarci matplotlib
# optional for evaluation: DockQ, PyRosetta
```

## Data format

Training consumes per-complex tensors produced by `scripts/` data preparation:
native H3 backbone (L×4×3), H3 sequence, framework backbone/sequence, antigen
backbone and context features, plus a precomputed relation graph. Binding-stage
features (surface normals, electrostatics, H-bond directions) are optional for
the first two stages and required for the binding/pair stages.

## Training pipeline

```bash
# 1) stage 1: frozen generator features are produced by the data script
# 2) orientation/interface refiner
python scripts/train_...py --stage refiner ...
# 3) binding interface refiner
python scripts/train_...py --stage binding ...
# 4) two-scale pair refiner
python scripts/train_...py --stage pair ...
```

See `scripts/` for the exact snapshot commands used in the study.

## Evaluation

`eval/` contains the metric implementations used in the paper:

- `interface_fape` — interface-aligned FAPE
- `cn_details` — peptide-bond C–N geometry audit
- `contact_f1` — 8 Å CA contact F1 against the native H3–antigen interface
- `aligned_loss_twoscale_table_v1.py` — two-scale ablation table

## Notes

- This repository contains the model/training/evaluation code only.
  Datasets, cached features and checkpoints are not included.
- Paths in a few scripts point to the internal cluster layout used for the
  study; adapt the `ROOT`/`WS` constants before running.
