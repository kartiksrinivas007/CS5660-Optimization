from torch.optim import Optimizer
import torch

class SMD(Optimizer):
    def __init__(self, params, q=1, lr=1e-3):
        defaults = dict(lr=lr, q=q)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                grad = p.grad
                param = p.data
                next_param = torch.pow(torch.abs(param), group['q'] - 1)*torch.sign(param)-group['lr']*grad
                p.data = torch.abs(next_param).pow(1/(group['q'] - 1))*torch.sign(next_param)
        return loss