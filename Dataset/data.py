import torchvision as tv
import torchvision.transforms as transforms
import torch
import torch.utils.data as tud
import torch.utils.data.dataset as dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
        # breakpoint()
        return cifar10_trainloader, cifar10_testloader, cifar10_valoader
    
    assert False, f"{args.dset} is not a valid dataset"

def test():
    train, test = return_loader({"dset": "cifar10"})
    plt.imshow(train.dataset[0][0].permute(1, 2, 0))
    plt.show()
    pass

