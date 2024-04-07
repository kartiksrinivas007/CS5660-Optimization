from Algorithms import smd
from Dataset import data
import argparse
import torchvision as tv
from train import train
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Run the algorithms')
    parser.add_argument('--algorithm', type=str, default='sgd', help='The algorithm to run')
    parser.add_argument('--dataset', type=str, default='cifar10', help='The dataset to use')
    parser.add_argument('--batch_size', type=int, default=32, help='The batch size')
    parser.add_argument('--num_epochs', type=int, default=3, help='The number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='The learning rate')
    parser.add_argument('--device', type=str, default="cuda", help='The device to run on')
    parser.add_argument('--q_norm', type=float, default=2, help='The q norm')
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    
    model, train_hist, val_hist = train(model, train_loader, val_loader, optimizer, args)
    
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(val_hist, label="Validation per batch")
    ax[1].plot(train_hist, label = "Training per sample")
    plt.legend()
    plt.savefig(f"plot_{args.lr}.png")
    plt.show()
    
    
    