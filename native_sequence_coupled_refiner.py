"""Native final-state sequence prediction, with no external language model."""
import torch
from torch import nn
from torch.nn import functional as F
from relation_coupled_refiner import RelationCoupledRefiner, objective as geometry_objective


class NativeSequenceCoupledRefiner(RelationCoupledRefiner):
    def __init__(self,width=64,layers=4,variant='constrained',relation_mode='coupled',final_native_sequence=True):
        if not final_native_sequence:raise ValueError('Native sequence head must be enabled')
        super().__init__(width,layers,variant,relation_mode)
        self.native_head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,20))
        nn.init.zeros_(self.native_head[-1].weight);nn.init.zeros_(self.native_head[-1].bias)

    def load_coupled(self,state):
        result=self.load_state_dict(state,strict=False)
        if result.unexpected_keys or any(not k.startswith('native_head.') for k in result.missing_keys):
            raise ValueError(result)

    def forward(self,x):
        out=super().forward(x)
        # No detach: CE trains final H3 representations when the backbone is unfrozen.
        out['sequence_logits']=x['base_logits']+self.native_head(out['h3_hidden'])
        return out


def objective(out,x,y):
    geometry,metrics=geometry_objective(out,x,y,'coupled')
    targets=y['sequence']
    if targets.shape!=(len(out['backbone']),) or targets.dtype!=torch.long:
        raise ValueError('Expected explicit native sequence labels, separate from model inputs')
    ce=F.cross_entropy(out['sequence_logits'],targets)
    loss=geometry+ce
    metrics.update(geometry_loss=float(geometry.detach()),sequence_ce=float(ce.detach()),
                   aar=float((out['sequence_logits'].argmax(-1)==targets).float().mean()),loss=float(loss.detach()))
    return loss,metrics


def geometry_guard(metrics,baseline):
    return (all(metrics[k]<=baseline[k]+.02 for k in ('rmsd','aligned_rmsd','contact_position_rmsd'))
            and all(metrics[k]>=baseline[k]-.002 for k in ('contact_f1','contact_precision','contact_recall'))
            and all(metrics[k]<=baseline[k]+.01 for k in ('cn_mae','endpoint_cn_mae'))
            and metrics['clash']<=baseline['clash']*1.1+1e-5)
