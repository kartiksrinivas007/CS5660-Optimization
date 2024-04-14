from Algorithms import smd, prox_smd, accelerated_prox_smd
from Dataset import data
import torchvision as tv
from utils.train import train
from utils.parser import argument_parser
from utils.test import test
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from init import custom_weight_init
import torch.nn as nn

def main(args=None):
    args_copy = args # this is not a copy, you should deep copy it if you want a copy, I think changing args_copy will change args as well
    if args is None:
        args = argument_parser()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    num_classes = 0
    print(f"Using {device}")

    if args.model == "resnet18":
        model = tv.models.resnet18(num_classes=10)
        num_classes = 10
    elif args.model == "linear":
        model = nn.Sequential(
            nn.Linear(1010, 2),
            # nn.Linear(200, 2),
        )
        num_classes = 2
    else:
        assert False, "Unknown model"
    
    if args.algorithm == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.algorithm == "smd":
        optimizer = smd.SMD(model.parameters(), lr=args.lr, q=args.q_norm)
    elif args.algorithm == "prox_smd":
        optimizer = prox_smd.PROXSMD(model.parameters(), lr=args.lr, q=args.q_norm, reg=args.reg)
    elif args.algorithm == "accelerated_prox_smd":
        optimizer = accelerated_prox_smd.ACCELERATEDPROXSMD(model.parameters(), lr=args.lr, q=args.q_norm, reg=args.reg)
    else:
        assert False, "Unknown algorithm"
        
    model = model.to(device)
    train_loader, test_loader, val_loader = data.return_loader(args)
    # model.apply(custom_weight_init)

    # print(model.state_dict())

    model, train_hist, val_hist = train(model, train_loader, val_loader, optimizer, args)
    print(test(model, test_loader, args, num_classes))
    if args_copy is None:
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(val_hist, label="Validation per batch", color="red")
        ax[1].plot(train_hist, label="Training per sample")
        fig.legend()
        fig.savefig(f"plot_{args.lr}_{args.algorithm}_{args.dataset}.png")
        fig.show()
    return model, test_loader

if __name__ == "__main__":
    main() # passing no arguments here for some reason
