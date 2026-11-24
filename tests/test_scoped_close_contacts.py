import torch
from eval.compare_peptide_candidates_scoped_v1 import close_contacts


def test_counts_nonadjacent_pairs_once_and_masks_missing_atoms():
    bb = torch.zeros(3, 4, 3)
    bb[0, 0, 0], bb[1, 0, 0], bb[2, 0, 0] = 0., 10., 1.
    mask = torch.zeros(3, 4, dtype=torch.bool)
    mask[:, 0] = True
    x = dict(context_bb=torch.full((1, 4, 3), 100.),
             context_mask=torch.ones(1, 4, dtype=torch.bool),
             antigen_mask=torch.tensor([True]), anchors=torch.full((2, 4, 3), 200.))
    result = close_contacts(bb, mask, x)
    assert result == dict(short_internal_pairs=1, short_context_pairs=0,
                          short_antigen_pairs=0, any_short_pair=1.)
    mask[2, 0] = False
    assert close_contacts(bb, mask, x)['any_short_pair'] == 0


def test_context_and_antigen_counts():
    bb = torch.zeros(1, 4, 3)
    mask = torch.tensor([[True, False, False, False]])
    context = torch.full((1, 4, 3), 100.)
    context[0, 0] = torch.tensor([.5, 0., 0.])
    x = dict(context_bb=context, context_mask=torch.tensor([[True, False, False, False]]),
             antigen_mask=torch.tensor([True]), anchors=torch.full((2, 4, 3), 200.))
    result = close_contacts(bb, mask, x)
    assert result['short_context_pairs'] == result['short_antigen_pairs'] == 1
    x['antigen_mask'][:] = False
    result = close_contacts(bb, mask, x)
    assert result['short_context_pairs'] == 1 and result['short_antigen_pairs'] == 0
