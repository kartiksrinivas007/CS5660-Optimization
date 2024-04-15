from torch.optim import Optimizer
import torch

class ACCELERATEDPROXSMD(Optimizer):
    def __init__(self, params, num_calls, q=1, lr=1e-3, reg=1.0, momentum=0.9):
        defaults = dict(lr=lr, q=q, reg=reg, momentum=momentum, num_calls=num_calls)
        super().__init__(params, defaults)
        self.z = [] # velocity in dual space
        self.dual_param = [] # parameter in dual space
        self.param = [] # parameter in primal space
        self.count = 1

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        index = 0
        for group in self.param_groups:
            for p in group['params']:
                grad = p.grad
                param = p.data

                if self.count == 1:
                    prev_param = param
                else:
                    prev_param = self.param[index]

                if self.count == 1:
                    self.z.append(-group['lr']*grad)
                    self.dual_param.append(torch.pow(torch.abs(param), group['q'] - 1)*torch.sign(param) + self.z[index])
                    self.param.append(torch.pow((torch.maximum((torch.abs(self.dual_param[index]) - group['reg'] * group['lr']), torch.zeros_like(self.dual_param[index]))), 1 / (group['q'] - 1)) * torch.sign(self.dual_param[index]))
                else:
                    self.z[index] = group['momentum']*self.z[index] - group['lr']*grad
                    self.dual_param[index] = self.dual_param[index] + self.z[index]
                    self.param[index] = torch.pow((torch.maximum((torch.abs(self.dual_param[index]) - group['reg'] * group['lr']), torch.zeros_like(self.dual_param[index]))), 1 / (group['q'] - 1)) * torch.sign(self.dual_param[index])

                if self.count == group['num_calls']:
                    p.data = self.param[index]
                elif self.count < group['num_calls']:
                    p.data = self.param[index] + group['lr'] * (self.param[index] - prev_param)
                else:
                    raise Exception("Calls exceeded the previously specified limit.")

                index+=1

        self.count += 1
        return loss