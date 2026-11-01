"""Small runnable PyTorch experiments: autograd, SFT mechanics and profiling.

No pretrained weights, network calls, synthetic benchmark claims, or GPU needed.
"""
from __future__ import annotations
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


def gradient_check(x:float=0.7,epsilon:float=1e-4)->tuple[float,float]:
    """Autograd versus symmetric finite difference of f(x)=x^3+sin(x).
    O(1) arithmetic/time/space; finite differences are approximate.
    """
    v=torch.tensor(x,dtype=torch.float64,requires_grad=True)
    y=v**3+torch.sin(v);y.backward()
    def f(z):return z**3+torch.sin(torch.tensor(z,dtype=torch.float64)).item()
    finite=(f(x+epsilon)-f(x-epsilon))/(2*epsilon)
    return v.grad.item(),finite


def tiny_train_eval(seed:int=7,epochs:int=60,save_path:str|None=None)->dict[str,float]:
    """Synthetic regression with held-out data. O(epochs*N*D*H) operations.
    Illustrates seed, train()/eval(), no_grad(), checkpoint save/load. CPU runnable.
    Does not claim real-world generalization performance.
    """
    torch.manual_seed(seed)
    inputs=torch.randn(160,3)
    targets=(2*inputs[:,0]-inputs[:,1]+0.5*inputs[:,2]).unsqueeze(1)
    train_x,test_x=inputs[:120],inputs[120:]
    train_y,test_y=targets[:120],targets[120:]
    model=nn.Sequential(nn.Linear(3,8),nn.ReLU(),nn.Linear(8,1))
    opt=torch.optim.Adam(model.parameters(),lr=0.04)
    for _ in range(epochs):
        model.train();opt.zero_grad(set_to_none=True)
        loss=F.mse_loss(model(train_x),train_y)
        loss.backward();opt.step()
    model.eval()
    with torch.no_grad():held_out=F.mse_loss(model(test_x),test_y).item()
    if save_path:
        torch.save(model.state_dict(),save_path)
        clone=nn.Sequential(nn.Linear(3,8),nn.ReLU(),nn.Linear(8,1))
        clone.load_state_dict(torch.load(save_path,map_location='cpu',weights_only=True))
        clone.eval()
        with torch.no_grad():assert torch.allclose(clone(test_x),model(test_x))
    return {'train_mse':float(loss.item()),'held_out_mse':float(held_out)}


def attention_shape_demo(batch:int=2,heads:int=4,length:int=16,dim:int=8)->dict[str,tuple[int,...]|int]:
    """Causal self-attention with explicit Q/K/V shapes; O(BHL^2D) compute.
    Attention weights alone occupy O(BHL^2) elements; not all kernels materialize them.
    """
    torch.manual_seed(0)
    q=torch.randn(batch,heads,length,dim)
    k=torch.randn(batch,heads,length,dim)
    v=torch.randn(batch,heads,length,dim)
    out=F.scaled_dot_product_attention(q,k,v,is_causal=True)
    return {'query':tuple(q.shape),'output':tuple(out.shape),'naive_score_elements':batch*heads*length**2}


def amp_profile_demo(steps:int=3)->dict[str,int|str]:
    """Autocast and torch.profiler exercise, portable to CPU.
    Training cost O(steps*batch*dim^2); trace collection adds measurement overhead.
    """
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=nn.Linear(32,16).to(device)
    x=torch.randn(64,32,device=device)
    optimizer=torch.optim.SGD(model.parameters(),lr=.01)
    dtype=torch.float16 if device=='cuda' else torch.bfloat16
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device,dtype=dtype):
                loss=model(x).square().mean()
            loss.backward();optimizer.step()
    return {'device':device,'steps':steps,'profile_events':len(prof.events())}
