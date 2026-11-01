"""Educational LoRA-style adapter on a small linear layer.

Focus: frozen base parameters, trainable low-rank matrices, and gradients.
No quantization, pretrained weights, datasets, or network access required.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """Low-rank residual of a frozen linear layer.

    For input width D, output width O and rank r, train r*(D+O) params
    instead of O*D (ignoring biases). Forward O(B*(D*O+r*(D+O))).
    """
    def __init__(self,in_features:int,out_features:int,rank:int=2,alpha:float=4):
        super().__init__()
        if rank<=0:raise ValueError('rank must be positive')
        self.base=nn.Linear(in_features,out_features,bias=False)
        self.base.weight.requires_grad_(False)
        self.a=nn.Parameter(torch.randn(rank,in_features)*.01)
        self.b=nn.Parameter(torch.zeros(out_features,rank))
        self.scaling=alpha/rank

    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.base(x)+self.scaling*F.linear(F.linear(x,self.a),self.b)


def demo_step(seed:int=9,steps:int=120)->dict[str,float|int|bool]:
    """Fit an artificial rank-2 correction while frozen base weights stay fixed.

    O(steps*B*D*O) time, O(B*(D+O)+model params) working storage.
    Test correctness of freezing, not external-task accuracy.
    """
    torch.manual_seed(seed)
    model=LoRALinear(8,6,rank=2)
    frozen_before=model.base.weight.detach().clone()
    x=torch.randn(96,8)
    with torch.no_grad():
        target=model.base(x)+1.5*x[:,:2]@torch.randn(2,6)
    optim=torch.optim.Adam([model.a,model.b],lr=.04)
    with torch.no_grad():initial=F.mse_loss(model(x),target).item()
    for _ in range(steps):
        optim.zero_grad(set_to_none=True)
        loss=F.mse_loss(model(x),target)
        loss.backward();optim.step()
    with torch.no_grad():final=F.mse_loss(model(x),target).item()
    return {'initial_mse':float(initial),'final_mse':float(final),
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'base_frozen':bool(torch.equal(model.base.weight,frozen_before))}
