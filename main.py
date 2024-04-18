import numpy as np

from Algorithms import smd, prox_smd, accelerated_prox_smd
from Dataset import data
import torchvision as tv
from utils.train import train
from utils.parser import argument_parser
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from init import custom_weight_init
from torchmetrics import Accuracy
import torch.nn as nn
import time

def main(args=None):
    args_copy = args # this is not a copy, you should deep copy it if you want a copy, I think changing args_copy will change args as well
    if args is None:
        args = argument_parser()

    print(f"Using {args.device}")

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

    train_loader, test_loader, val_loader = data.return_loader(args)

    if args.algorithm == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.algorithm == "smd":
        optimizer = smd.SMD(model.parameters(), lr=args.lr, q=args.q_norm)
    elif args.algorithm == "prox_smd":
        optimizer = prox_smd.PROXSMD(model.parameters(), lr=args.lr, q=args.q_norm, reg=args.reg)
    elif args.algorithm == "accelerated_prox_smd":
        num_calls = args.num_epochs * len(train_loader)
        optimizer = accelerated_prox_smd.ACCELERATEDPROXSMD(model.parameters(), lr=args.lr, q=args.q_norm, reg=args.reg, momentum=args.momentum, num_calls=num_calls)
    else:
        assert False, "Unknown algorithm"
        
    model = model.to(args.device)
    metrics = [Accuracy(task="multiclass", num_classes=num_classes).to(device=args.device)]
    # model.apply(custom_weight_init)

    # print(model.state_dict())
    start_time = time.time()
    train_hist, val_hist, test_hist, computed_metrics = train(model, train_loader, val_loader, test_loader, optimizer, args, metrics)
    end_time = time.time()

    if args_copy is None:
        fig, ax = plt.subplots(3, 1)
        ax[0].plot(train_hist, label="Training loss vs epoch", color="red")
        ax[1].plot(val_hist, label="Validation loss vs epoch", color="blue")
        ax[2].plot(test_hist, label="Test loss vs epoch", color="green")
        fig.legend()
        fig.savefig(f"Plots/plot_{args.lr}_{args.algorithm}_{args.dataset}_loss_vs_epoch.png")
        # fig.show()

        fig, ax = plt.subplots(3, 1)
        ax[0].plot(np.linspace(0, end_time-start_time, len(computed_metrics[0][0])), computed_metrics[0][0], label="Training Accuracy vs time", color="red")
        ax[1].plot(np.linspace(0, end_time-start_time, len(computed_metrics[0][1])), computed_metrics[0][1], label="Validation Accuracy vs time", color="blue")
        ax[2].plot(np.linspace(0, end_time-start_time, len(computed_metrics[0][2])), computed_metrics[0][2], label="Test Accuracy vs time", color="green")
        fig.legend()
        fig.savefig(f"Plots/plot_{args.lr}_{args.algorithm}_{args.dataset}_accuracy_vs_time.png")
        print(f"Time taken: {end_time-start_time}")
        print(f"Final test accuracy: {computed_metrics[0][2][-1]}")
    return computed_metrics[0][2][-1]

if __name__ == "__main__":
    main() # passing no arguments here for some reason
