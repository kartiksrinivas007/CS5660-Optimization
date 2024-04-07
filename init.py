import torch.nn as nn
import torch



def custom_weight_init(m):
    # make cases if m is either a conv layer or not!
    # breakpoint()
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight, 0.0001)
        if m.bias is not None:
            nn.init.normal_(m.bias, 0.0001)
    elif isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, a=-0.01, b=0.01)
        if m.bias is not None:
            nn.init.uniform_(m.bias, a=-0.01, b=0.01)
    pass