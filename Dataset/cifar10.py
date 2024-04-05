import torchvision as tv
import torchvision.transforms as transforms
import torch
import torch.utils.data.dataloader as dataloader
import torch.utils.data.dataset as dataset


def return_loader(args):
    if (args.dset == "cifar10"):
        cifar10_transform = [transforms.ToTensor()] # need to normalize
        cifar10_trainset = tv.datasets.CIFAR10(root = './data', train=True, download=True, transform=cifar10_transform)
        cifar10_trainloader = dataloader(cifar10_trainset, batch_size = args.batch_size, shuffle=True, num_workers=2)
        pass
    pass