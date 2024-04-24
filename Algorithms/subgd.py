from torch.optim import Optimizer
import torch

class SUBGD(Optimizer):
    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
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
                p.data = param - group['lr'] * grad
        return loss