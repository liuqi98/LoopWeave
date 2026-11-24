"""Refiner with a decoupled sequence prior, for the stage-1 sequence-prior pilot.

The original model has a single ``base_logits`` tensor that serves **two** roles:

* it is the input-side prior: ``query(cat([base_logits.softmax(-1), ...]))`` conditions the
  H3 input embedding;
* it is the output-side residual: ``sequence_logits = base_logits + native_head(h)``.

The pilot needs to separate these, so this subclass adds an optional
``base_logits_output`` key that is used **only** by the final residual decode:

    condition            base_logits (input)   base_logits_output (residual)
    baseline             zeros                 zeros
    probability_init     B = alpha * S^0 / T   zeros
    full                 B                     B

``B`` is the stage-1 sequence prior ``S^0`` scaled by ``alpha / T``. When
``base_logits_output`` is absent the subclass behaves exactly like the original model, so
existing checkpoints and results are unaffected.

The per-layer feedback path is unchanged: the loop-pair layers already compute
``softmax(base_logits + native_head(h))``, which for the prior conditions equals
``P^l = softmax[B + g_seq(H^l)]`` from the protocol.
"""
import torch

from persistent_loop_pair_refiner import PersistentLoopPairRefiner


class SequencePriorRefiner(PersistentLoopPairRefiner):
    """Same architecture and weights, with the input prior and output residual separated."""

    def forward(self, x):
        out = super().forward(x)
        base = x['base_logits']
        output_prior = x.get('base_logits_output', base)
        if output_prior is not base:
            # super() computed base + native_head(h); swap the base for the output prior.
            out['sequence_logits'] = output_prior + (out['sequence_logits'] - base)
        return out


def make_inputs(x, prior=None, output_prior=False):
    """Attach the prior tensors for a condition.

    ``prior=None`` reproduces the baseline (zero logits, zero residual). Otherwise the
    prior tensor becomes the input-side ``base_logits`` and, when ``output_prior`` is
    true, also the residual that the decoded sequence is added to.
    """
    inputs = dict(x)
    if prior is None:
        return inputs
    inputs['base_logits'] = prior
    inputs['base_logits_output'] = prior if output_prior else torch.zeros_like(prior)
    return inputs
