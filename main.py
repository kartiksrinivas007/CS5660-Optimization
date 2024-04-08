from Algorithms import smd
from Dataset import data
import torchvision as tv
from utils.train import train
from utils.parser import argument_parser
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from init import custom_weight_init

def main(args=None):
    args_copy = args
    if args is None:
        args = argument_parser()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    print(f"Using {device}")

    model = tv.models.resnet18()
    model = model.to(device)
    if args.algorithm == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.algorithm == "smd":
        optimizer = smd.SMD(model.parameters(), lr=args.lr, q=args.q_norm)
    else:
        assert False, "Unknown algorithm"

    train_loader, test_loader, val_loader = data.return_loader(args)
    model.apply(custom_weight_init)

    # print(model.state_dict())

    model, train_hist, val_hist = train(model, train_loader, val_loader, optimizer, args)

    if args_copy is None:
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(val_hist, label="Validation per batch")
        ax[1].plot(train_hist, label="Training per sample")
        plt.legend()
        plt.savefig(f"plot_{args.lr}.png")
        plt.show()
    return model, test_loader

if __name__ == "__main__":
    main()
