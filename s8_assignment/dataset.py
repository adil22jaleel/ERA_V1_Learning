import numpy as np
import torch
from torchvision import datasets,transforms
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(1)

def get_loader(train_transform, test_transform, batch_size=128, use_cuda=True):
    """Get instance of tran and test loaders

    Args:
        train_transform (Transform): Instance of transform function for training
        test_transform (Transform): Instance of transform function for validation
        batch_size (int, optional): batch size to be uised in training. Defaults to 64.
        use_cuda (bool, optional): Enable/Disable Cuda Gpu. Defaults to True.

    Returns:
        (DataLoader, DataLoader): Get instance of train and test data loaders
    """
    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

    train = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

    test = datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
    
    sample_transform = transforms.Compose([transforms.ToTensor()])
    
    sample = datasets.CIFAR10(root='./data', train=True, transform=sample_transform)
    
    sampleloader = torch.utils.data.DataLoader(sample, batch_size=10,
                                              shuffle=True,  **kwargs)

    return trainloader, testloader, sampleloader