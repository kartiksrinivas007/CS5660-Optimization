import torchvision as tv
import torchvision.transforms as transforms
import torch
import torch.utils.data as tud
import torch.utils.data.dataset as dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from .MAGIC import *

def return_loader(args):
    if (args.dataset == "cifar10"):
        cifar10_mean = [0.4914, 0.4822, 0.4465]
        cifar10_std = [0.247, 0.243, 0.261]
        
        cifar10_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cifar10_mean, std=cifar10_std), transforms.CenterCrop(32)]) # need to normalize
        cifar10_trainset = tv.datasets.CIFAR10(root = './data', train=True, download=True, transform=cifar10_transform)
        cifar10_trainloader = tud.DataLoader(cifar10_trainset, batch_size = args.batch_size, shuffle=True, num_workers=2)
        cifar10_testset = tv.datasets.CIFAR10(root = './data', train=False, download=True, transform=cifar10_transform)
        cifar10_testloader = tud.DataLoader(cifar10_trainset, batch_size = args.batch_size, shuffle=False, num_workers=2)
        cifar10_valset, cifar10_testset = train_test_split(cifar10_testset, test_size=0.5)
        cifar10_valoader = tud.DataLoader(cifar10_valset, batch_size = args.batch_size, shuffle=False, num_workers=2)

        return cifar10_trainloader, cifar10_testloader, cifar10_valoader
    if (args.dataset == "magic04s"):
        # note that this dataset by default is normalized even the test set!
        
        magic04s_X, magic04s_y = magic04s()
        magic04s_train_X, magic04s_test_X, magic04s_val_X, magic04s_train_y, magic04s_test_y, magic04s_val_y = split_data(magic04s_X, magic04s_y, args)
        magic04s_trainloader = tud.DataLoader(dataset.TensorDataset(torch.tensor(magic04s_train_X), torch.tensor(magic04s_train_y)), batch_size=args.batch_size, shuffle=True)
        magic04s_testloader = tud.DataLoader(dataset.TensorDataset(torch.tensor(magic04s_test_X), torch.tensor(magic04s_test_y)), batch_size=args.batch_size, shuffle=False)
        magic04s_valoader = tud.DataLoader(dataset.TensorDataset(torch.tensor(magic04s_val_X), torch.tensor(magic04s_val_y)), batch_size=args.batch_size, shuffle=False)
        return magic04s_trainloader, magic04s_testloader, magic04s_valoader

    assert False, f"{args.dset} is not a valid dataset"

def test():
    train, test = return_loader({"dset": "cifar10"})
    plt.imshow(train.dataset[0][0].permute(1, 2, 0))
    plt.show()
    pass

